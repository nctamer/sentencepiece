// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "unigram_continuation_trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstdio>
#include <queue>
#include <set>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "filesystem.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "normalizer.h"
#include "ret_check.h"
#include "unigram_model.h"
#include "unigram_model_trainer.h"
#include "util.h"

ABSL_FLAG(std::string, continuation_fence_strings, "",
          "comma-separated LOGICAL corpus strings (e.g. \"PL:,PR:,Vn:\") that "
          "fence Unigram continuation CANDIDATE GENERATION. Each is normalized "
          "with the prior's own normalizer before matching, so callers pass "
          "PL: and never the internal \xe2\x96\x81PL: form. No new candidate may "
          "overlap a fenced character. The corpus, the E-step, inherited "
          "pieces and BPE are all unaffected. Entries may not be empty and may "
          "not themselves contain a comma.");

ABSL_FLAG(std::string, continuation_spill_dir, "",
          "directory under which candidate-extraction spill runs are created. "
          "Empty means $TMPDIR, or /tmp when that is unset. Each run creates a "
          "private mkdtemp() subdirectory that is removed on every exit path.");

ABSL_FLAG(int32_t, continuation_spill_entries, 2097152,
          "Max in-memory candidate entries before continuation spills a "
          "sorted run to disk. Bounds aggregation memory only; it does NOT "
          "change results, because run counts are summed exactly during the "
          "k-way merge before any candidate is scored or ranked.");

namespace sentencepiece::unigram {
namespace {

// A private, per-invocation scratch directory for candidate-extraction spill
// runs, removed on EVERY exit path by the destructor.
//
// Uniqueness comes from mkdtemp(), not from getpid(): on a shared filesystem
// two nodes can hold the same PID at the same time, and a PID-named directory
// then lets one run delete another's sorted runs in the middle of its merge.
// The base directory is a flag, else $TMPDIR, else /tmp -- never a path
// belonging to one user's scratch space.
class SpillScratch {
 public:
  SpillScratch() = default;
  ~SpillScratch() { Cleanup(); }
  SpillScratch(const SpillScratch&) = delete;
  SpillScratch& operator=(const SpillScratch&) = delete;

  absl::Status Open(absl::string_view base_dir) {
    std::string base(base_dir);
    if (base.empty()) {
      const char* tmp = std::getenv("TMPDIR");
      base = (tmp != nullptr && *tmp != '\0') ? tmp : "/tmp";
    }
    while (base.size() > 1 && base.back() == '/') base.pop_back();
    std::string tmpl = absl::StrCat(base, "/spm_continuation_spill_XXXXXX");
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (::mkdtemp(buf.data()) == nullptr) {
      return absl::InternalError(absl::StrCat(
          "cannot create a continuation spill directory under ", base,
          " (errno ", errno,
          "); set --continuation_spill_dir to a writable location"));
    }
    dir_.assign(buf.data());
    open_ = true;
    return absl::OkStatus();
  }

  // Idempotent, and safe to call from an error path before the destructor.
  void Cleanup() {
    for (const auto& r : runs_) ::remove(r.c_str());
    runs_.clear();
    if (open_) {
      ::rmdir(dir_.c_str());
      open_ = false;
    }
  }

  const std::string& dir() const { return dir_; }
  void AddRun(std::string path) { runs_.push_back(std::move(path)); }
  const std::vector<std::string>& runs() const { return runs_; }

 private:
  std::string dir_;
  std::vector<std::string> runs_;
  bool open_ = false;
};

// ONE FENCED-SPAN MECHANISM, TWO SOURCES.
//
// Candidate enumeration must exclude characters covered by
//   A. inherited TYPED meta symbols (USER_DEFINED / CONTROL / UNKNOWN / BYTE),
//      because the lattice scores those by maximal matching rather than as
//      ordinary Unigram log-probabilities (defect D); and
//   B. explicit curriculum fence STRINGS supplied by
//      --continuation_fence_strings, which are ordinary NORMAL surface text
//      such as the normalized form of "Vn:".
// The mask is the UNION of both; the counters keep them apart, because "no
// candidate crossed a meta symbol" and "no candidate crossed a context switch"
// are different claims.
//
// TWO CORRECTNESS PROPERTIES THIS TYPE OWES, both of which the earlier
// meta-only code got wrong and neither of which a caller can see:
//
//  1. UNION OF ALL MATCHES, not longest-match-then-skip. The old loop did
//     `b += n` after a match, so with fences {abc, bcd} over "abcd" it marked
//     [0,3) and never looked at position 1 again -- [3,4) stayed admissible
//     although "bcd" covers it. Every character boundary is now probed, and a
//     match only ever ADDS to the mask.
//  2. EXACT byte<->character mapping. Character positions are what enumeration
//     indexes, matchers return byte lengths. `char_of_byte` is defined ONLY at
//     real UTF-8 boundaries (-1 elsewhere), matches are attempted only at
//     boundaries, and a match whose end is not a boundary is refused rather
//     than rounded to one.
class FenceMask {
 public:
  void SetInheritedMeta(std::unique_ptr<normalizer::PrefixMatcher> m) {
    meta_ = std::move(m);
  }
  void SetExplicit(std::unique_ptr<normalizer::PrefixMatcher> m) {
    explicit_ = std::move(m);
  }
  bool active() const { return meta_ != nullptr || explicit_ != nullptr; }

  // Fills `fenced` (one entry per Unicode character of `utf8`) or leaves it
  // empty when no fencing is configured. `utf8` and `text` are the same
  // sentence in the two representations enumeration already has in hand, so
  // this adds no corpus copy.
  void Build(absl::string_view utf8, const string_util::UnicodeText& text,
             std::vector<char>* fenced) {
    fenced->clear();
    if (!active()) return;

    // Byte offset -> character index, defined only at character boundaries.
    std::vector<int64_t> char_of_byte(utf8.size() + 1, -1);
    std::vector<size_t> byte_of_char(text.size() + 1, 0);
    size_t b = 0;
    for (size_t c = 0; c < text.size(); ++c) {
      char_of_byte[b] = static_cast<int64_t>(c);
      byte_of_char[c] = b;
      b += string_util::UnicodeCharToUTF8(text[c]).size();
    }
    char_of_byte[b] = static_cast<int64_t>(text.size());
    byte_of_char[text.size()] = b;
    if (b != utf8.size()) {
      // The two representations disagree; fence nothing rather than mark the
      // wrong characters.
      ++malformed_;
      return;
    }

    fenced->assign(text.size(), 0);
    bool any_explicit_here = false;
    // EVERY character boundary is a candidate match start. Never skip ahead by
    // a previous match length.
    for (size_t c = 0; c < text.size(); ++c) {
      Mark(meta_.get(), utf8, byte_of_char[c], c, char_of_byte, fenced,
           &meta_matches_, nullptr);
      Mark(explicit_.get(), utf8, byte_of_char[c], c, char_of_byte, fenced,
           &explicit_matches_, &any_explicit_here);
    }
    if (any_explicit_here) ++records_with_explicit_;
    for (const char f : *fenced) {
      if (f) ++fenced_characters_;
    }
  }

  uint64_t meta_matches() const { return meta_matches_; }
  uint64_t explicit_matches() const { return explicit_matches_; }
  uint64_t fenced_characters() const { return fenced_characters_; }
  uint64_t records_with_explicit() const { return records_with_explicit_; }
  uint64_t malformed() const { return malformed_; }

  // The absolute byte offset is passed in rather than recovered from pointer
  // arithmetic: the match end must be checked against the SENTENCE's boundary
  // table, and a suffix view does not carry that offset.
  void Mark(const normalizer::PrefixMatcher* matcher, absl::string_view utf8,
            size_t start_byte, size_t c0,
            const std::vector<int64_t>& char_of_byte, std::vector<char>* fenced,
            uint64_t* counter, bool* seen_here) {
    if (matcher == nullptr) return;
    bool found = false;
    const int n = matcher->PrefixMatch(utf8.substr(start_byte), &found);
    if (!found || n <= 0) return;
    const size_t end_byte = start_byte + static_cast<size_t>(n);
    // Refuse a match that does not land on a character boundary rather than
    // rounding it onto one: a partial-byte span must never reach the mask.
    if (end_byte >= char_of_byte.size() || char_of_byte[end_byte] < 0) {
      ++malformed_;
      return;
    }
    const size_t c1 = static_cast<size_t>(char_of_byte[end_byte]);
    ++*counter;
    if (seen_here != nullptr) *seen_here = true;
    for (size_t c = c0; c < c1 && c < fenced->size(); ++c) (*fenced)[c] = 1;
  }

 private:
  std::unique_ptr<normalizer::PrefixMatcher> meta_;
  std::unique_ptr<normalizer::PrefixMatcher> explicit_;
  uint64_t meta_matches_ = 0, explicit_matches_ = 0;
  uint64_t fenced_characters_ = 0, records_with_explicit_ = 0, malformed_ = 0;
};





bool Finite(double x) { return std::isfinite(x); }

// log(1e-30). Stands in for the lambda -> -infinity limit when the inherited
// side carries no expected count at all.
//
// At namespace scope rather than inside the function that uses it: a
// constexpr local read by a lambda with no default capture is accepted by
// GCC and Clang and rejected by MSVC (C3493), and the constant does not need
// to be local to be clear.
constexpr double kDegenerateLogBaseMass = -69.07755278982137;

// Two scores that differ only by floating-point accumulation noise rank as a
// tie, so the piece string decides.
//
// This exists because a weighted corpus and its physical repetition are the
// same corpus stated two ways, but not the same sum: n*p and p added n times
// differ in the last bits, and addition is not associative. Without a tie
// band, that noise decides which of two equally-scored pieces gets the lower
// external ID - and an ID is an ABI, not a presentation detail.
//
// The band sits far below any meaningful score difference and far above the
// accumulated noise of a corpus with ~10^6 records.
// Relative-tolerance equality that is SAFE FOR NON-FINITE INPUTS.
//
// The finite-only form below was `|a-b| <= 1e-9 * max(|a|,|b|,1)`, and it is
// wrong in both directions once an infinity reaches it:
//   ScoresTie(+inf, finite) -> TRUE, because |inf - finite| <= 1e-9*inf is
//                              inf <= inf. A "must keep" candidate therefore
//                              tied with every ordinary one and the comparator
//                              fell through to lexical order.
//   ScoresTie(+inf, +inf)   -> FALSE, because the difference is NaN and every
//                              NaN comparison is false, so two must-keeps did
//                              not even tie with each other.
// Measured consequence: the required coverage extension 'V', ranked +inf precisely so
// it could not be pruned, was ordered lexically and pruned out of the emitted
// model, leaving a vocabulary that cannot spell its own training corpus.
//
// Infinities are now handled before the tolerance is applied. NaN never ties
// with anything, including itself, which keeps a corrupt score loud instead of
// letting it masquerade as an ordering.
bool ScoresTie(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return false;
  if (std::isinf(a) || std::isinf(b)) {
    // Equal infinities of the same sign tie; everything else is ordered.
    return a == b;
  }
  const double scale = std::max({std::abs(a), std::abs(b), 1.0});
  return std::abs(a - b) <= 1e-9 * scale;
}

}  // namespace

// Deterministic bracketed bisection for a monotone objective.
//
// Every continuation lambda is the root of a monotone function, so a bracket
// plus halving is enough - and it is reproducible, which a general-purpose
// optimizer would not be. The point of doing it by hand is that each way the
// search can fail becomes a distinct status instead of a returned endpoint:
// a bracket that never changes sign, an objective that leaves the finite
// range, and an interval that stays wide are three different bugs and read as
// three different messages.
absl::Status BisectMonotoneRoot(absl::string_view what,
                                const std::function<double(double)>& f,
                                bool increasing, double seed_lo,
                                double seed_hi, double* root) {
  constexpr int kMaxExpansions = 400;
  constexpr int kMaxRefinements = 200;

  // Work with an increasing view so the bracket logic has one shape.
  const auto oriented = [&](double x) {
    const double value = f(x);
    return increasing ? value : -value;
  };

  double lo = seed_lo;
  double hi = seed_hi;
  double f_lo = oriented(lo);
  double f_hi = oriented(hi);

  int expansions = 0;
  while (Finite(f_lo) && f_lo > 0.0 && expansions < kMaxExpansions) {
    hi = lo;
    f_hi = f_lo;
    lo -= std::max(1.0, std::abs(lo));
    f_lo = oriented(lo);
    ++expansions;
  }
  while (Finite(f_hi) && f_hi < 0.0 && expansions < kMaxExpansions) {
    lo = hi;
    f_lo = f_hi;
    hi += std::max(1.0, std::abs(hi));
    f_hi = oriented(hi);
    ++expansions;
  }

  if (!Finite(f_lo) || !Finite(f_hi)) {
    return absl::FailedPreconditionError(absl::StrCat(
        what, ": objective left the finite range while bracketing (lambda=",
        lo, " gives ", f_lo, ", lambda=", hi, " gives ", f_hi, ")"));
  }
  if (!(f_lo <= 0.0 && f_hi >= 0.0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        what, ": no sign change over [", lo, ", ", hi,
        "] after ", expansions,
        " expansions, so the root is not bracketed and no lambda can be "
        "reported"));
  }

  for (int i = 0; i < kMaxRefinements; ++i) {
    const double mid = 0.5 * (lo + hi);
    // Halving stops meaning anything once mid stops being interior.
    if (!(mid > lo && mid < hi)) break;
    const double f_mid = oriented(mid);
    if (!Finite(f_mid)) {
      return absl::FailedPreconditionError(absl::StrCat(
          what, ": objective became non-finite at lambda=", mid));
    }
    if (f_mid < 0.0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const double width = hi - lo;
  if (!Finite(width) ||
      width > std::max(1e-9, 1e-9 * std::max(std::abs(lo), std::abs(hi)))) {
    return absl::FailedPreconditionError(absl::StrCat(
        what, ": bisection did not converge; bracket [", lo, ", ", hi,
        "] is still ", width, " wide"));
  }
  *root = 0.5 * (lo + hi);
  return absl::OkStatus();
}

double CandidateScore(absl::string_view piece, uint64_t freq) {
  const double len = static_cast<double>(string_util::UTF8Len(piece));
  const double power =
      static_cast<double>(absl::GetFlag(FLAGS_seed_piece_length_power));
  return static_cast<double>(freq) * std::pow(len, power);
}


absl::Status ContinuationTrainer::LoadAndValidatePrior() {
  if (trainer_spec_.unigram_prior_model().empty()) {
    return absl::InvalidArgumentError(
        "Unigram continuation requires --unigram_prior_model");
  }
  if (!trainer_spec_.expansion_spec().empty()) {
    return absl::InvalidArgumentError(
        "unigram_prior_model and expansion_spec are mutually exclusive");
  }
  if (!trainer_spec_.seed_sentencepieces_file().empty()) {
    return absl::InvalidArgumentError(
        "seed_sentencepieces_file is a fresh-training candidate mechanism and "
        "cannot be combined with unigram_prior_model");
  }

  ABSL_RETURN_IF_ERROR(continuation::ReadModelProto(
      trainer_spec_.unigram_prior_model(), &prior_model_, &prior_model_bytes_));
  if (!prior_model_.has_trainer_spec() ||
      prior_model_.trainer_spec().model_type() != TrainerSpec::UNIGRAM) {
    return absl::InvalidArgumentError(
        "unigram_prior_model is not a Unigram ModelProto");
  }
  if (!prior_model_.has_normalizer_spec()) {
    return absl::InvalidArgumentError(
        "unigram prior does not contain an authoritative normalizer spec");
  }
  if (prior_model_.pieces_size() <= 0) {
    return absl::InvalidArgumentError("unigram prior has no pieces");
  }
  if (trainer_spec_.vocab_size() < prior_model_.pieces_size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "vocab_size is smaller than prior piece count: ",
        trainer_spec_.vocab_size(), " < ", prior_model_.pieces_size()));
  }

  extension_target_ = trainer_spec_.vocab_size() - prior_model_.pieces_size();
  inherited_normal_.clear();
  prior_unk_id_ = -1;
  absl::flat_hash_set<std::string> strings;
  int unknown_count = 0;
  for (int id = 0; id < prior_model_.pieces_size(); ++id) {
    const auto& piece = prior_model_.pieces(id);
    if (piece.piece().empty() || !strings.insert(piece.piece()).second) {
      return absl::InvalidArgumentError(
          "prior ModelProto contains empty or duplicate piece state");
    }
    if (!Finite(piece.score())) {
      return absl::InvalidArgumentError("prior ModelProto has non-finite score");
    }
    if (piece.type() == ModelProto::SentencePiece::UNKNOWN) {
      ++unknown_count;
      prior_unk_id_ = id;
    }
    if (piece.type() == ModelProto::SentencePiece::NORMAL) {
      const int len = static_cast<int>(string_util::UTF8Len(piece.piece()));
      if (len <= 0) {
        return absl::InvalidArgumentError(
            "prior NORMAL piece has zero additive length");
      }
      inherited_normal_.push_back(
          InheritedNormal{id, piece.piece(), piece.score(), len});
    }
  }
  if (unknown_count != 1) {
    return absl::InvalidArgumentError(
        "prior SentencePiece Unigram model must contain exactly one UNKNOWN");
  }
  if (inherited_normal_.empty()) {
    return absl::InvalidArgumentError("prior has no NORMAL Unigram pieces");
  }

  // The prior ModelProto is authoritative for normalization: continuation
  // never invents a new normalization regime, because inherited scores are
  // only meaningful over the text the prior was fitted on.
  //
  // Authoritative is not the same as silent. A caller who left normalization
  // alone gets the prior's. A caller who asked for exactly the prior's gets
  // it too. A caller who asked for something else has stated a requirement
  // this trainer cannot honour, and is told so rather than having the request
  // quietly discarded.
  //
  // "Left alone" has to be judged by value, not by field presence: the CLI
  // sets every normalization field on every run, so presence would report
  // "explicit" for a command line that never mentioned normalization.
  ABSL_RETURN_IF_ERROR(ReconcileNormalization());
  return absl::OkStatus();
}

namespace {

// Compares the fields that decide what normalization actually does.
// normalization_rule_tsv is excluded: it is an input to compilation, and is
// represented by precompiled_charsmap once compiled.
bool NormalizationEquivalent(const NormalizerSpec& a, const NormalizerSpec& b) {
  return a.name() == b.name() &&
         a.precompiled_charsmap() == b.precompiled_charsmap() &&
         a.add_dummy_prefix() == b.add_dummy_prefix() &&
         a.remove_extra_whitespaces() == b.remove_extra_whitespaces() &&
         a.escape_whitespaces() == b.escape_whitespaces();
}

// A spec nobody populated. Production callers reach the trainer through
// SentencePieceTrainer::Train, which fills these in, so this is the shape of a
// direct caller who simply did not ask for normalization.
bool NormalizationUnspecified(const NormalizerSpec& spec) {
  return spec.name().empty() && spec.precompiled_charsmap().empty() &&
         spec.normalization_rule_tsv().empty();
}

std::string DescribeNormalization(const NormalizerSpec& spec) {
  return absl::StrCat("name=", spec.name(),
                      " add_dummy_prefix=", spec.add_dummy_prefix(),
                      " remove_extra_whitespaces=",
                      spec.remove_extra_whitespaces(),
                      " escape_whitespaces=", spec.escape_whitespaces(),
                      " charsmap_bytes=", spec.precompiled_charsmap().size());
}

}  // namespace

absl::Status ContinuationTrainer::ReconcileNormalization() {
  const NormalizerSpec& prior_normalizer = prior_model_.normalizer_spec();
  const NormalizerSpec& prior_denormalizer = prior_model_.denormalizer_spec();

  // What a caller who said nothing about normalization ends up with.
  NormalizerSpec default_normalizer;
  ABSL_RETURN_IF_ERROR(
      SentencePieceTrainer::PopulateNormalizerSpec(&default_normalizer));

  if (!NormalizationUnspecified(normalizer_spec_) &&
      !NormalizationEquivalent(normalizer_spec_, prior_normalizer) &&
      !NormalizationEquivalent(normalizer_spec_, default_normalizer)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "continuation normalization conflict: the prior model is "
        "authoritative but the caller asked for a different normalizer. "
        "prior [", DescribeNormalization(prior_normalizer), "] caller [",
        DescribeNormalization(normalizer_spec_),
        "]. Drop the normalization flags to inherit the prior's, or pass "
        "exactly the prior's."));
  }

  // A denormalizer is opt-in, so "unset" is genuinely detectable here.
  const bool caller_set_denormalizer =
      !denormalizer_spec_.normalization_rule_tsv().empty() ||
      !denormalizer_spec_.precompiled_charsmap().empty();
  if (caller_set_denormalizer &&
      !NormalizationEquivalent(denormalizer_spec_, prior_denormalizer)) {
    return absl::FailedPreconditionError(
        "continuation denormalization conflict: the prior model's "
        "denormalizer is authoritative and the caller supplied a different "
        "one");
  }

  // Whitespace placement is part of the same contract: it decides what the
  // inherited pieces mean, so it cannot be re-chosen per continuation run.
  const bool prior_suffix =
      prior_model_.trainer_spec().treat_whitespace_as_suffix();
  if (trainer_spec_.treat_whitespace_as_suffix() != prior_suffix &&
      trainer_spec_.treat_whitespace_as_suffix() !=
          TrainerSpec::default_instance().treat_whitespace_as_suffix()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "continuation conflict: prior model has treat_whitespace_as_suffix=",
        prior_suffix, " but the caller asked for ",
        trainer_spec_.treat_whitespace_as_suffix()));
  }

  normalizer_spec_ = prior_normalizer;
  denormalizer_spec_ = prior_denormalizer;
  trainer_spec_.set_treat_whitespace_as_suffix(prior_suffix);
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::VerifyCorpusCoverage() {
  // ADMIT AS A REQUIRED COVERAGE EXTENSION, DO NOT REFUSE.
  // A continuation corpus routinely contains characters the prior never saw --
  // expanding a piano tokenizer with violin introduces 'V' for the "Vn:"
  // marker, and the prior cannot spell it. Refusing here forced callers to
  // hand-edit the prior, which is worse in three ways: it mutates an artifact
  // that is supposed to be immutable, it silently spends one of the extension
  // slots (1201+299 instead of 1200+300), and the hand-chosen score is a fake
  // inherited probability that no data supports.
  //
  // Instead: collect the missing characters here and admit them as ordinary
  // EXTENSION candidates. They are new pieces, they append after the whole
  // prior, they count against the extension budget, and their scores are
  // learned by the constrained M-step like any other extension. Pruning keeps
  // them automatically -- a character with no alternative segmentation has
  // infinite deletion loss -- so no special-casing is needed there.
  required_coverage_extensions_.clear();
  Model model(prior_model_);
  if (!model.status().ok()) return model.status();

  absl::flat_hash_set<std::string> missing;
  for (const auto& sentence : corpus_.sentences) {
    Lattice lattice;
    lattice.SetSentence(sentence.first);
    model.PopulateNodes(&lattice);
    const auto path = lattice.Viterbi();
    for (const auto* node : path.first) {
      if (node->id == prior_unk_id_) {
        missing.emplace(node->piece.data(), node->piece.size());
      }
    }
  }
  required_coverage_extensions_.assign(missing.begin(), missing.end());
  std::sort(required_coverage_extensions_.begin(), required_coverage_extensions_.end());
  // ASSUMPTION MADE LOAD-BEARING: required coverage extensions are ATOMIC single
  // characters. The initializer records one basis decomposition at lambda=0
  // and reuses it as beta_i + lambda*h_i for every lambda. That is EXACT only
  // because a candidate containing m missing characters must consume exactly
  // those m required atoms in every basis segmentation, so every path shares
  // the same lambda offset (L_x - m) and the argmax cannot move with lambda.
  // Allow multi-character required pieces and h_i becomes path-dependent and
  // the derivation silently stops holding.
  for (const auto& b : required_coverage_extensions_) {
    if (string_util::UTF8Len(b) != 1) {
      return absl::UnimplementedError(absl::StrCat(
          "continuation required coverage extensions must be single-character "
          "only; got '", b, "' of length ", string_util::UTF8Len(b),
          ". A multi-character required piece makes the inherited-length "
          "exposure h_i path-dependent and invalidates the beta_i + "
          "lambda*h_i initialization."));
    }
  }
  required_extensions_.clear();
  required_extensions_.insert(required_coverage_extensions_.begin(),
                              required_coverage_extensions_.end());
  if (!required_coverage_extensions_.empty()) {
    LOG(INFO) << "Unigram continuation required coverage extension: " << required_coverage_extensions_.size()
              << " character(s) absent from the prior will be admitted as "
                 "EXTENSION candidates (they consume extension budget, and "
                 "the prior is not modified): "
              << absl::StrJoin(required_coverage_extensions_, " ");
  }
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::MakeWeightedExtensionCandidates() {
  extension_candidates_.clear();
  if (extension_target_ == 0) return absl::OkStatus();

  // THE CANDIDATE POOL IS THE SEARCH UNIVERSE AND MUST NOT DEPEND ON K.
  // This used to be max(10000, extension_target * 64), with
  // seed_sentencepiece_size able only to SHRINK it. Two things were wrong.
  // First, the universe moved with the OUTPUT budget, so a K=300 run and a
  // K=1200 run searched different candidate sets and their vocabularies were
  // not comparable -- a phrase could be absent either because EM pruned it or
  // because it was never admitted, and nothing distinguished the two.
  // Second, seed_sentencepiece_size could not ENLARGE the pool, so there was
  // no way to ask for a wider search at a small K.
  //
  // Now: seed_sentencepiece_size, when set, IS the pool size; otherwise a
  // fixed default (SentencePiece's own seed default) is used. The pool is only
  // ever raised to extension_target, since a pool smaller than the budget
  // cannot fill it.
  //
  // INDEPENDENCE FROM K IS CONDITIONAL and the condition is stated where the
  // threshold is computed: it holds under the validated default
  // min_freq_alpha = 0. With a nonzero alpha the dynamic frequency floor reads
  // vocab_size, so admission -- and therefore the pool contents, though not
  // this cap -- becomes K-dependent again.
  constexpr size_t kDefaultContinuationCandidatePool = 1000000;
  size_t candidate_limit =
      trainer_spec_.seed_sentencepiece_size() > 0
          ? static_cast<size_t>(trainer_spec_.seed_sentencepiece_size())
          : kDefaultContinuationCandidatePool;
  candidate_limit = std::max<size_t>(candidate_limit,
                                     static_cast<size_t>(extension_target_));

  absl::flat_hash_set<std::string> inherited_strings;
  inherited_strings.reserve(prior_model_.pieces_size());
  for (const auto& piece : prior_model_.pieces()) {
    inherited_strings.insert(piece.piece());
  }

  // EXACT GLOBAL WEIGHTED COUNTS IN BOUNDED MEMORY (spill + k-way merge).
  //
  // WHAT WAS WRONG BEFORE. Two successive implementations were both unfit:
  //   (1) an unbounded flat_hash_map holding EVERY unique valid substring,
  //       duplicated into a second vector, fully sorted, and only then
  //       truncated to candidate_limit -- so seed_sentencepiece_size was not a
  //       memory bound at all;
  //   (2) a bounded map with periodic GC that evicted by CURRENT PARTIAL rank.
  //       That is not equivalent to exact global top-K: a candidate can be
  //       rare in an early corpus prefix, be evicted, and become globally
  //       frequent later. Its count then restarts and its final value is only
  //       a lower bound, so the retained set depends on record ORDER and on
  //       the flush threshold. The claim that "GC uses the same ranking as
  //       selection, so it cannot reorder what selection picks" is false, and
  //       is withdrawn.
  //
  // THIS implementation keeps the ORIGINAL exact semantics:
  //   freq(x) = sum over records of weight * occurrences of x
  // while bounding resident memory. Partial counts are flushed to sorted runs
  // keyed by candidate BYTES (never by score), and a k-way merge sums all runs
  // for each distinct string. A candidate's final score is computed only once
  // its exact global count is known, so ordering never depends on flush
  // boundaries. Determinism is therefore independent of spill threshold and
  // record order.
  struct SpillEntry {
    std::string piece;
    uint64_t count;
  };
  // SPILL SCRATCH IS PRIVATE, PORTABLE AND RAII-CLEANED.
  //
  // It used to be a hard-coded /gscratch path plus getpid(), removed by a
  // lambda that had to be called on every return. Two problems: library code
  // must not contain one user's absolute path, and a PID is not unique on a
  // shared filesystem -- two nodes can and do collide on the same PID, and
  // then one run deletes the other's sorted runs mid-merge. mkdtemp() gives
  // both portability and per-invocation uniqueness, and the destructor makes
  // "every exit path cleans up" a property of the type rather than of the
  // author's diligence.
  SpillScratch scratch;
  ABSL_RETURN_IF_ERROR(
      scratch.Open(absl::GetFlag(FLAGS_continuation_spill_dir)));
  const std::string& spill_dir = scratch.dir();

  // DEFECT D: INHERITED META SYMBOLS FENCE CANDIDATE GENERATION.
  //
  // Standard SentencePiece replaces meta/user-defined symbols with a
  // pretokenization boundary before seed generation, so a learned NORMAL piece
  // can never span one. Continuation normalized the corpus but did not impose
  // the equivalent fence, so an extension could span an inherited
  // USER_DEFINED symbol. That is wrong because USER_DEFINED pieces are scored
  // by the lattice's maximal-matching bonus rather than as ordinary Unigram
  // log-probabilities, while the constrained M-step assumes the learned family
  // is inherited NORMAL + extension NORMAL. Fencing restores that separation.
  //
  // The fence applies to the CANDIDATE-GENERATION VIEW ONLY. The E-step keeps
  // reading the original normalized record and still sees the inherited
  // symbols normally; nothing on disk is mutated.
  std::set<absl::string_view> meta_symbols;
  for (const auto& piece : prior_model_.pieces()) {
    switch (piece.type()) {
      case ModelProto::SentencePiece::USER_DEFINED:
      case ModelProto::SentencePiece::CONTROL:
      case ModelProto::SentencePiece::UNKNOWN:
      case ModelProto::SentencePiece::BYTE:
        meta_symbols.insert(piece.piece());
        break;
      default:
        break;                      // NORMAL and UNUSED are not boundaries
    }
  }
  // Cheap precheck: if no meta symbol occurs anywhere in the corpus, fencing
  // is provably a no-op and the per-position matcher is skipped entirely.
  bool meta_present = false;
  for (const auto& sentence : corpus_.sentences) {
    for (const auto& m : meta_symbols) {
      if (absl::StrContains(sentence.first, m)) { meta_present = true; break; }
    }
    if (meta_present) break;
  }

  FenceMask fence;
  if (meta_present && !meta_symbols.empty()) {
    fence.SetInheritedMeta(
        std::make_unique<normalizer::PrefixMatcher>(meta_symbols));
  }
  LOG(INFO) << "METAFENCE symbols=" << meta_symbols.size()
            << " present_in_corpus=" << (meta_present ? "yes" : "no")
            << " matcher=" << (meta_present && !meta_symbols.empty()
                                   ? "active"
                                   : "skipped(no-op)");

  // EXPLICIT CURRICULUM FENCES (--continuation_fence_strings).
  //
  // The caller supplies LOGICAL corpus strings ("PL:", "PR:", "Vn:"). They are
  // normalized here with the PRIOR's own normalizer -- the same regime that
  // produced the records in corpus_ -- so a caller never types the internal
  // U+2581 form and there is no second normalization convention. With this
  // corpus's normalizer (identity, add_dummy_prefix, escape_whitespaces) the
  // logical "Vn:" becomes the boundary-bearing surface that actually occurs in
  // the normalized text, which is also what keeps it from matching inside
  // unrelated material such as "fooVn:bar".
  std::set<std::string> fence_surface_storage;
  size_t logical_fence_count = 0;
  {
    const std::string spec = absl::GetFlag(FLAGS_continuation_fence_strings);
    if (!spec.empty()) {
      normalizer::Normalizer fence_normalizer(prior_model_.normalizer_spec(),
                                              trainer_spec_);
      ABSL_RETURN_IF_ERROR(fence_normalizer.status());
      for (const auto& logical : absl::StrSplit(spec, ',')) {
        const std::string one(logical);
        if (one.empty()) {
          return absl::InvalidArgumentError(
              "--continuation_fence_strings contains an empty entry; the "
              "syntax is a comma-separated list of non-empty logical strings "
              "and an entry may not itself contain a comma");
        }
        ++logical_fence_count;
        const std::string surface = fence_normalizer.Normalize(one);
        if (surface.empty()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "continuation fence string ", one,
              " normalizes to the empty string under the prior's normalizer"));
        }
        // Deterministic dedup: a std::set of surfaces, so the configured order
        // of the list cannot become a modelling parameter.
        const bool inserted = fence_surface_storage.insert(surface).second;
        LOG(INFO) << "FENCE logical=\"" << one << "\" normalized=\"" << surface
                  << "\"" << (inserted ? "" : " (duplicate, deduplicated)");
      }
    }
  }
  if (!fence_surface_storage.empty()) {
    std::set<absl::string_view> views;
    for (const auto& sfc : fence_surface_storage) views.insert(sfc);
    fence.SetExplicit(std::make_unique<normalizer::PrefixMatcher>(views));
  }
  LOG(INFO) << "FENCE logical_count=" << logical_fence_count
            << " normalized_unique=" << fence_surface_storage.size()
            << " matcher=" << (fence_surface_storage.empty()
                                   ? "none(no-op)"
                                   : "active");

  absl::flat_hash_map<std::string, uint64_t> counts;
  // Entry cap governs resident aggregation memory, NOT the retained pool.
  // Floor is deliberately tiny so tests can FORCE multi-run spilling. A
  // max() against 2^21 here silently clamped small thresholds upward and made
  // the spill path unreachable in testing.
  const size_t kMaxLiveCounts = std::max<size_t>(
      16, static_cast<size_t>(absl::GetFlag(FLAGS_continuation_spill_entries)));
  size_t spill_runs = 0;
  uint64_t spilled_entries = 0;

  auto flush_run = [&]() -> absl::Status {
    if (counts.empty()) return absl::OkStatus();
    std::vector<SpillEntry> live;
    live.reserve(counts.size());
    for (auto& it : counts) live.push_back({it.first, it.second});
    counts.clear();
    // SORT BY BYTES. The merge is keyed on the string, so runs must be in
    // string order; sorting by score here would make the merge wrong.
    std::sort(live.begin(), live.end(),
              [](const SpillEntry& a, const SpillEntry& b) {
                return a.piece < b.piece;
              });
    const std::string path =
        absl::StrCat(spill_dir, "/run", spill_runs, ".bin");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      return absl::InternalError(
          absl::StrCat("cannot open continuation spill run ", path));
    }
    for (const auto& e : live) {
      const uint32_t n = static_cast<uint32_t>(e.piece.size());
      out.write(reinterpret_cast<const char*>(&n), sizeof(n));
      out.write(e.piece.data(), n);
      out.write(reinterpret_cast<const char*>(&e.count), sizeof(e.count));
    }
    out.flush();
    if (!out) {
      return absl::InternalError(
          absl::StrCat("short write on continuation spill run ", path));
    }
    out.close();
    scratch.AddRun(path);
    ++spill_runs;
    // TEST-ONLY fault injection: fail after N runs exist, so the cleanup
    // guarantee can be exercised on an error path rather than asserted.
    if (const char* fail_after = std::getenv("SPM_SPILL_FAIL_AFTER_RUNS")) {
      int n = 0;
      if (absl::SimpleAtoi(fail_after, &n) && n > 0 &&
          static_cast<int>(scratch.runs().size()) >= n) {
        return absl::InternalError(
            "SPM_SPILL_FAIL_AFTER_RUNS: injected spill failure");
      }
    }
    spilled_entries += static_cast<uint64_t>(live.size());
    return absl::OkStatus();
  };

  long double weighted_bytes = 0.0L;
  const size_t max_piece_length =
      static_cast<size_t>(trainer_spec_.max_sentencepiece_length());

  for (const auto& sentence : corpus_.sentences) {
    if (sentence.second <= 0) {
      return absl::InvalidArgumentError(
          "continuation corpus contains a nonpositive sentence weight");
    }
    weighted_bytes += static_cast<long double>(sentence.first.size()) *
                      static_cast<long double>(sentence.second);

    const string_util::UnicodeText text =
        string_util::UTF8ToUnicodeText(sentence.first);

    // Candidate-only fence: mark every character position covered by an
    // inherited meta symbol OR an explicit curriculum fence string, as a
    // UNION. Enumeration may not start inside a fenced character and may not
    // extend across one, so no NEW piece can contain, cross or equal a fenced
    // span. The record itself is untouched and the E-step still reads it whole.
    //
    // The mask is per-sentence and transient: no persistent corpus-wide fence
    // storage, no second copy of the corpus, no extra pass.
    std::vector<char> fenced;
    fence.Build(sentence.first, text, &fenced);

    for (size_t begin = 0; begin < text.size(); ++begin) {
      if (!fenced.empty() && fenced[begin]) continue;   // cannot start inside
      string_util::UnicodeText piece;
      piece.reserve(std::min(max_piece_length, text.size() - begin));
      const size_t stop =
          std::min(text.size(), begin + std::max<size_t>(1, max_piece_length));
      for (size_t end = begin; end < stop; ++end) {
        if (!fenced.empty() && fenced[end]) break;      // cannot cross
        piece.push_back(text[end]);
        if (piece.size() <= 1) continue;
        if (!IsValidSentencePiece(piece)) continue;

        std::string candidate = string_util::UnicodeTextToUTF8(piece);
        if (inherited_strings.contains(candidate)) continue;
        uint64_t& count = counts[candidate];
        const uint64_t delta = static_cast<uint64_t>(sentence.second);
        if (delta > std::numeric_limits<uint64_t>::max() - count) {
          return absl::OutOfRangeError(absl::StrCat(
              "weighted Unigram candidate count overflow for piece: ",
              candidate));
        }
        count += delta;
        if (counts.size() > kMaxLiveCounts) {
          ABSL_RETURN_IF_ERROR(flush_run());
        }
      }
    }
  }

  // CANDIDATE-POOL INDEPENDENCE FROM THE EXTENSION TARGET holds under the
  // validated and default configuration, min_freq_alpha = 0: then
  // global_min_freq is the constant 2 and nothing in admission reads
  // vocab_size, so the pool a corpus produces is the same whatever K is asked
  // for. That is what the K-independence claim means and all it means.
  //
  // With min_freq_alpha != 0 the threshold below divides by vocab*log(vocab),
  // so the FINAL vocabulary size feeds back into which candidates are admitted
  // at all: changing K changes the pool. That is a K-dependent filtering
  // heuristic and it is named as one here rather than papered over. It is not
  // redesigned in this cleanup, and it does not affect the validated
  // experiment, which runs at alpha = 0.
  const double alpha =
      static_cast<double>(absl::GetFlag(FLAGS_min_freq_alpha));
  const double vocab = std::max<double>(1.0, trainer_spec_.vocab_size());
  const double denom = std::max<double>(1.0, vocab * std::log(vocab));
  const uint64_t global_min_freq = std::max<uint64_t>(
      2, static_cast<uint64_t>(
             alpha * static_cast<double>(weighted_bytes) / denom));

  // Final flush, then an exact k-way merge over all runs.
  ABSL_RETURN_IF_ERROR(flush_run());

  const auto by_rank = [](const std::pair<std::string, uint64_t>& a,
                          const std::pair<std::string, uint64_t>& b) {
    const double sa = CandidateScore(a.first, a.second);
    const double sb = CandidateScore(b.first, b.second);
    if (sa != sb) return sa > sb;
    const size_t la = string_util::UTF8Len(a.first);
    const size_t lb = string_util::UTF8Len(b.first);
    if (la != lb) return la > lb;
    return a.first < b.first;
  };

  // Bounded top-K over EXACT counts. priority_queue's top() is the maximum
  // under the comparator, so comparing with `by_rank` (a is better than b)
  // puts the WORST retained candidate on top and makes eviction O(log K).
  std::priority_queue<std::pair<std::string, uint64_t>,
                      std::vector<std::pair<std::string, uint64_t>>,
                      decltype(by_rank)>
      best(by_rank);

  struct RunReader {
    std::ifstream in;
    std::string piece;
    uint64_t count = 0;
    bool ok = false;
    bool Advance() {
      uint32_t n = 0;
      if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) return ok = false;
      piece.resize(n);
      if (n && !in.read(&piece[0], n)) return ok = false;
      if (!in.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return ok = false;
      }
      return ok = true;
    }
  };
  std::vector<std::unique_ptr<RunReader>> readers;
  readers.reserve(scratch.runs().size());
  for (const auto& path : scratch.runs()) {
    auto r = std::make_unique<RunReader>();
    r->in.open(path, std::ios::binary);
    if (!r->in) {
      scratch.Cleanup();
      return absl::InternalError(
          absl::StrCat("cannot reopen continuation spill run ", path));
    }
    r->Advance();
    readers.push_back(std::move(r));
  }

  size_t merged_distinct = 0, meta_rejected_at_merge = 0;
  while (true) {
    const std::string* smallest = nullptr;
    for (const auto& r : readers) {
      if (!r->ok) continue;
      if (smallest == nullptr || r->piece < *smallest) smallest = &r->piece;
    }
    if (smallest == nullptr) break;
    const std::string key = *smallest;
    // Sum every run's contribution for this exact string, with overflow check.
    uint64_t total = 0;
    for (auto& r : readers) {
      while (r->ok && r->piece == key) {
        if (r->count > std::numeric_limits<uint64_t>::max() - total) {
          scratch.Cleanup();
          return absl::OutOfRangeError(absl::StrCat(
              "weighted Unigram candidate count overflow for piece: ", key));
        }
        total += r->count;
        r->Advance();
      }
    }
    ++merged_distinct;
    // UNICODE LENGTH, not UTF-8 byte length. This exemption spares SHORT
    // candidates from the dynamic frequency floor, and "short" everywhere else
    // in candidate handling means characters: the rank comparator uses
    // UTF8Len, max_sentencepiece_length is applied in characters, and the
    // required-coverage counts are per character. Using key.size() made the
    // exemption depend on the encoding -- a 3-character ASCII candidate was
    // exempt while a 2-character candidate containing one 3-byte codepoint was
    // not. Nothing in the filter's purpose distinguishes those.
    const uint64_t effective_min =
        string_util::UTF8Len(key) <= 3 ? 2 : global_min_freq;
    if (total < effective_min) continue;
    best.emplace(key, total);
    if (best.size() > candidate_limit) best.pop();
  }
  readers.clear();
  scratch.Cleanup();

  std::vector<std::pair<std::string, uint64_t>> ranked;
  ranked.reserve(best.size());
  while (!best.empty()) {
    ranked.push_back(best.top());
    best.pop();
  }
  std::sort(ranked.begin(), ranked.end(), by_rank);
  LOG(INFO) << "EXTRACTSPILL runs=" << spill_runs
            << " spilled_entries=" << spilled_entries
            << " max_live_counts=" << kMaxLiveCounts
            << " merged_distinct=" << merged_distinct
            << " retained=" << ranked.size()
            << " meta_fenced_spans=" << fence.meta_matches();
  // Explicit context fences are NOT meta_fenced_spans and are counted apart:
  // "no candidate crossed an inherited meta symbol" and "no candidate crossed
  // a curriculum context switch" are different claims about different sources.
  LOG(INFO) << "FENCE inherited_meta_matches=" << fence.meta_matches()
            << " explicit_string_matches=" << fence.explicit_matches()
            << " fenced_characters=" << fence.fenced_characters()
            << " records_with_explicit_fence=" << fence.records_with_explicit()
            << " malformed_boundary_matches=" << fence.malformed();

  // REQUIRED COVERAGE EXTENSIONS ARE ADMITTED UNCONDITIONALLY, ahead of the frequency
  // filter and the pool cap. Without them the corpus is unrepresentable, so
  // they are not competing on merit with ordinary candidates -- but they are
  // still ORDINARY EXTENSIONS: appended after the prior, scored by the
  // constrained M-step, and counted against the extension budget. Pruning
  // keeps them on its own, because a piece with no alternative segmentation
  // has infinite deletion loss.
  if (!required_coverage_extensions_.empty()) {
    absl::flat_hash_set<std::string> present;
    present.reserve(ranked.size());
    for (const auto& item : ranked) present.insert(item.first);
    // EXACT WEIGHTED OCCURRENCE COUNTS, computed independently of the spill
    // map. Looking the character up in `counts` gave every required extension
    // freq = 1, for two compounding reasons: single-character ordinary
    // candidates are skipped during substring enumeration, and by this point
    // the map has been drained into sorted runs anyway. With exactly one
    // missing character (V, in the validated piano->violin run) a constant is
    // indistinguishable from a count, so nothing was measurably wrong -- but
    // the initializer distributes the required mass in PROPORTION to these
    // numbers, so with two or more missing characters synthetic 1s silently
    // impose an equal split while the comments claim proportionality.
    //
    //   required_frequency[c] = sum over normalized records of
    //                             record_weight * occurrences of c
    //
    // Counted over Unicode characters, matching every other length and
    // identity convention in candidate handling.
    absl::flat_hash_map<std::string, uint64_t> required_freq;
    required_freq.reserve(required_coverage_extensions_.size());
    for (const auto& ch : required_coverage_extensions_) required_freq[ch] = 0;
    if (!required_freq.empty()) {
      for (const auto& sentence : corpus_.sentences) {
        const std::string& w = sentence.first;
        const uint64_t weight = static_cast<uint64_t>(sentence.second);
        for (size_t i = 0; i < w.size();) {
          const size_t clen = std::min<size_t>(
              string_util::OneCharLen(w.data() + i), w.size() - i);
          auto it = required_freq.find(w.substr(i, clen));
          if (it != required_freq.end()) it->second += weight;
          i += clen;
        }
      }
    }

    std::vector<std::pair<std::string, uint64_t>> prepend;
    for (const auto& ch : required_coverage_extensions_) {
      if (present.count(ch)) continue;
      const uint64_t freq = required_freq[ch];
      // A required coverage character is by construction present in the
      // corpus: VerifyCorpusCoverage() found it there. Zero would mean the
      // character set and the corpus disagree, which is a bug, not a datum.
      if (freq == 0) {
        return absl::InternalError(absl::StrCat(
            "required coverage extension ", ch,
            " has weighted corpus frequency 0, but it was admitted because "
            "the corpus contains it"));
      }
      prepend.emplace_back(ch, freq);
    }
    for (const auto& e : prepend) {
      LOG(INFO) << "REQUIREDFREQ piece=" << e.first
                << " weighted_occurrences=" << e.second;
    }
    if (!prepend.empty()) {
      ranked.insert(ranked.begin(), prepend.begin(), prepend.end());
      LOG(INFO) << "Unigram continuation: admitted " << prepend.size()
                << " required coverage extension candidate(s) into the pool";
    }
  }

  extension_candidates_.reserve(ranked.size());
  for (auto& item : ranked) {
    ExtensionCandidate candidate;
    candidate.piece = std::move(item.first);
    candidate.freq = item.second;      // needed to initialize its score
    extension_candidates_.push_back(std::move(candidate));
  }

  LOG(INFO) << "Unigram continuation extracted " << extension_candidates_.size()
            << " bounded weighted candidates (see EXTRACTGC) from "
            << corpus_.sentences.size() << " fenced records";

  // EXTRACTION-ONLY DUMP. Set SPM_DUMP_CANDIDATES=<path> to write
  // "rank\tpiece\tweighted_freq" for the whole pool. (It does not itself stop
  // the run; pair it with SPM_STOP_AFTER_INIT for an extraction-only gate.) This
  // exists because "the phrase is absent" has to be attributable to a stage:
  // never enumerated, rejected at initialization, or out-competed. Extraction
  // is the first of the three and was invisible until now.
  if (const char* dump = std::getenv("SPM_DUMP_CANDIDATES")) {
    auto out = filesystem::NewWritableFile(dump);
    if (out->status().ok()) {
      for (size_t i = 0; i < extension_candidates_.size(); ++i) {
        out->WriteLine(absl::StrCat(i, "\t", extension_candidates_[i].piece,
                                    "\t", extension_candidates_[i].freq));
      }
    }
    LOG(INFO) << "SPM_DUMP_CANDIDATES wrote " << extension_candidates_.size()
              << " candidates to " << dump;
  }
  if (static_cast<int>(extension_candidates_.size()) < extension_target_ &&
      trainer_spec_.hard_vocab_limit()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "only ", extension_candidates_.size(), " extension candidates for ",
        extension_target_, " requested pieces"));
  }
  return absl::OkStatus();
}

double ContinuationTrainer::LogBaseMass(double lambda) const {
  double max_term = -std::numeric_limits<double>::infinity();
  for (const auto& piece : inherited_normal_) {
    max_term = std::max(max_term,
                        piece.prior_score + lambda * piece.additive_length);
  }
  double sum = 0.0;
  for (const auto& piece : inherited_normal_) {
    sum += std::exp(piece.prior_score + lambda * piece.additive_length -
                    max_term);
  }
  return max_term + std::log(sum);
}

double ContinuationTrainer::BaseMass(double lambda,
                                     double* derivative) const {
  const double log_mass = LogBaseMass(lambda);
  if (log_mass > 700.0) {
    if (derivative) *derivative = std::numeric_limits<double>::infinity();
    return std::numeric_limits<double>::infinity();
  }
  const double mass = std::exp(log_mass);
  if (derivative != nullptr) {
    double scaled = 0.0;
    for (const auto& piece : inherited_normal_) {
      scaled += piece.additive_length *
                std::exp(piece.prior_score +
                         lambda * piece.additive_length - log_mass);
    }
    *derivative = mass * scaled;
  }
  return mass;
}

absl::Status ContinuationTrainer::SolveCoveredBasisLambda(
    const std::vector<ExtensionCandidate>& extensions, double required_mass,
    double* lambda) const {
  // Required extensions contribute a CONSTANT `required_mass`: their scores are
  // free parameters fixed at initialization, with no gauge exposure. Ordinary
  // candidates are tied to their covered-basis decomposition, whose only
  // gauge-exposed part is its inherited pieces, hence h_i and not L_i.
  auto log_mass = [&](double lam) {
    double max_term = -std::numeric_limits<double>::infinity();
    for (const auto& piece : inherited_normal_) {
      max_term = std::max(max_term,
                          piece.prior_score + lam * piece.additive_length);
    }
    for (const auto& c : extensions) {
      if (c.is_required) continue;
      max_term = std::max(max_term,
                          c.inherited_best_score + lam * c.basis_inherited_len);
    }
    if (required_mass > 0.0) {
      max_term = std::max(max_term, std::log(required_mass));
    }
    double sum = 0.0;
    for (const auto& piece : inherited_normal_) {
      sum += std::exp(piece.prior_score + lam * piece.additive_length -
                      max_term);
    }
    for (const auto& c : extensions) {
      if (c.is_required) continue;
      sum += std::exp(c.inherited_best_score + lam * c.basis_inherited_len -
                      max_term);
    }
    if (required_mass > 0.0) sum += std::exp(std::log(required_mass) - max_term);
    return max_term + std::log(sum);
  };
  return BisectMonotoneRoot("covered-basis continuation lambda", log_mass,
                            /*increasing=*/true, -1.0, 1.0, lambda);
}

absl::Status ContinuationTrainer::InitializeContinuationScores() {
  if (extension_target_ == 0) {
    lambda_ = 0.0;
    return absl::OkStatus();
  }

  // THE COVERED BASIS = immutable inherited NORMAL pieces + REQUIRED coverage extension
  // extensions, and nothing else.
  //
  // The basis must contain the required extensions or every candidate that
  // needs a character the prior cannot spell is rejected here -- with 'V'
  // absent from rnnt_1200 that is the entire Vn: family, 231,737 retained
  // candidates in the correct grammar. It must contain NOTHING ELSE: if
  // ordinary candidates could decompose through other ordinary candidates,
  // initialization order and model population would decide the result, which
  // is not a definition.
  //
  // Required extensions are NOT relabelled as inherited. They stay extensions:
  // appended after the whole prior, counted against the extension budget,
  // never pruned. What changes is only that they are admissible spelling
  // material for ordinary candidates.
  //
  // NORMALIZATION. The basis is itself a complete model over
  // (inherited + required), so its free mass is fully assigned to the required
  // pieces: q_r = epsilon(lambda_basis) * freq_r / sum_r freq_r, with
  // lambda_basis solving BaseMass(lambda) + sum_r q_r = 1. Because a required
  // piece carries a FREE score rather than a gauge-shifted inherited one, the
  // old identity "candidate score == B_x + lambda*L_x" no longer holds for any
  // V-containing decomposition. Its basis score is
  //   sum_j (s_j + lambda*l_j) + sum_r q_r
  // so the initializer below reads the basis Viterbi path score DIRECTLY
  // instead of reconstructing it from B_x and L_x.
  TrainerModel::SentencePieces normal_pieces;
  normal_pieces.reserve(inherited_normal_.size() + required_extensions_.size());
  absl::flat_hash_set<std::string> normal_strings;   // admissible basis symbols
  double base_mass_at_zero = 0.0;
  for (const auto& piece : inherited_normal_) {
    normal_pieces.emplace_back(piece.piece,
                               static_cast<float>(piece.prior_score));
    normal_strings.insert(piece.piece);
    base_mass_at_zero += std::exp(piece.prior_score);
  }

  // Required extensions, scored from the basis free mass by frequency share.
  std::vector<std::pair<std::string, uint64_t>> required_freq;
  for (const auto& c : extension_candidates_) {
    if (required_extensions_.contains(c.piece)) {
      required_freq.emplace_back(c.piece, std::max<uint64_t>(1, c.freq));
    }
  }
  double lambda_basis = 0.0;
  if (!required_freq.empty()) {
    // BaseMass is increasing in lambda; find the lambda leaving exactly the
    // mass the required pieces need. With required mass free to be small, the
    // basis lambda is near the value where BaseMass == 1 - required_share.
    double deriv = 0.0;
    const double eps0 = 1.0 - BaseMass(0.0, &deriv);
    // The covered-basis rule "give the required atoms the residual while
    // holding lambda_basis = 0" needs a residual to exist. A prior whose
    // NORMAL mass already fills the simplex cannot host a required atom this
    // way, and there is no defined fallback; fail explicitly rather than
    // silently producing a non-normalized basis.
    if (!(eps0 > 0.0)) {
      return absl::UnimplementedError(absl::StrCat(
          "continuation required coverage extension needs unallocated prior mass: Z_B(0)=",
          BaseMass(0.0, &deriv), " leaves residual ", eps0,
          " (<= 0), so required coverage extensions cannot be given probability "
          "at lambda_basis=0. This prior/coverage-extension combination is "
          "unsupported."));
    }
    // lambda_basis = 0 and the required pieces take ALL of the residual, so the
    // covered basis is a normalized model over (inherited + required) with
    // every inherited score still exactly at its prior value. The previous
    // "half of epsilon0" was an unjustified heuristic and left the basis
    // sub-normalized. These scores are initialization only; the constrained
    // M-step estimates the extension probabilities afterwards.
    // EXACT eps0. This was max(1e-12, eps0), which OVERALLOCATES required mass
    // whenever 0 < eps0 < 1e-12 and so breaks the covered-basis normalization
    // it is supposed to satisfy. eps0 > 0 has already been established above,
    // so the floor protected nothing and only introduced the error.
    const double required_total = eps0;
    double fsum = 0.0;
    for (const auto& r : required_freq) fsum += static_cast<double>(r.second);
    for (const auto& r : required_freq) {
      const double q = required_total * (static_cast<double>(r.second) / fsum);
      normal_pieces.emplace_back(r.first, static_cast<float>(std::log(q)));
      normal_strings.insert(r.first);
    }
    // OPT-IN INITIALIZATION DUMP (SPM_DUMP_REQUIRED_INIT=<path>), so a test can
    // check the required-mass split at INITIALIZATION. Final EM re-estimation
    // would otherwise hide an initialization bug completely: the piece would
    // still be present with a plausible score.
    if (const char* dump = std::getenv("SPM_DUMP_REQUIRED_INIT")) {
      std::ofstream out(dump, std::ios::trunc);
      if (out) {
        for (const auto& r : required_freq) {
          const double q =
              required_total * (static_cast<double>(r.second) / fsum);
          out << r.first << "\t" << r.second << "\t" << std::setprecision(17)
              << q << "\n";
        }
      }
    }
    LOG(INFO) << "GATE2 basis: inherited_normal=" << inherited_normal_.size()
              << " required=" << required_freq.size()
              << " base_mass_at_lambda0=" << base_mass_at_zero
              << " epsilon0=" << eps0
              << " required_extension_mass=" << required_total
              << " lambda_basis=" << lambda_basis;
  }

  TrainerModel normal_model(trainer_spec_, prior_model_.normalizer_spec());
  ABSL_RETURN_IF_ERROR(normal_model.SetSentencePieces(std::move(normal_pieces)));
  if (!normal_model.status().ok()) return normal_model.status();

  // CLASS E: rejected here, before initialization, for not being spellable by
  // the inherited-only basis. This filter is why a required coverage extension and
  // EVERYTHING THAT CONTAINS IT never reach EM or pruning: with 'V' absent
  // from the prior, "V", "\xe2\x96\x81Vn:" and every Vn:+pitch piece are
  // rejected at this line, not out-competed later. Such candidates must not be
  // classified as pruning failures.
  size_t in_count = extension_candidates_.size();
  size_t in_with_V = 0;
  for (const auto& c : extension_candidates_) {
    if (absl::StrContains(c.piece, "V")) ++in_with_V;
  }

  std::vector<ExtensionCandidate> valid;
  valid.reserve(extension_candidates_.size());
  for (auto candidate : extension_candidates_) {
    if (required_extensions_.contains(candidate.piece)) {
      // A required extension is a basis symbol; it cannot be asked to
      // decompose into the basis without trivially matching itself. Its score
      // is the FIXED r_k assigned above and carries no gauge exposure.
      Lattice rl;
      rl.SetSentence(candidate.piece);
      normal_model.PopulateNodes(&rl);
      candidate.inherited_best_score = rl.Viterbi().second;
      candidate.basis_inherited_len = 0;
      candidate.is_required = true;
      valid.push_back(std::move(candidate));
      continue;
    }
    Lattice lattice;
    lattice.SetSentence(candidate.piece);
    normal_model.PopulateNodes(&lattice);
    const auto path = lattice.Viterbi();
    bool inherited_only = !path.first.empty();
    std::string offender;
    for (const auto* node : path.first) {
      if (!normal_strings.contains(std::string(node->piece))) {
        inherited_only = false;
        offender.assign(node->piece.data(), node->piece.size());
        break;
      }
    }
    if (!inherited_only) {
      LOG(INFO) << "INITREJECT piece=" << candidate.piece
                << " freq=" << candidate.freq << " reason="
                << (path.first.empty()
                        ? "no basis decomposition at all"
                        : absl::StrCat("decomposition uses non-basis piece '",
                                       offender, "'"));
      continue;
    }
    // The basis path score ALREADY carries the gauge on inherited pieces and
    // the free score on required extensions, so it is used directly. Do not
    // reconstruct it as B_x + lambda*L_x: that identity holds only when every
    // symbol in the decomposition is a gauge-shifted inherited piece, which is
    // false for any V-containing candidate.
    candidate.inherited_best_score = path.second;
    // h_i: additive length of the INHERITED pieces only. Required atoms carry
    // free scores, so they contribute no lambda exposure.
    int h = 0;
    for (const auto* node : path.first) {
      const std::string sym(node->piece.data(), node->piece.size());
      if (!required_extensions_.contains(sym)) {
        h += static_cast<int>(string_util::UTF8Len(sym));
      }
    }
    candidate.basis_inherited_len = h;
    candidate.is_required = false;
    valid.push_back(std::move(candidate));
  }
  extension_candidates_.swap(valid);

  size_t out_with_V = 0;
  for (const auto& c : extension_candidates_) {
    if (absl::StrContains(c.piece, "V")) ++out_with_V;
  }
  LOG(INFO) << "INITFILTER before_init_candidates=" << in_count
            << " before_init_containing_V=" << in_with_V
            << " after_init_candidates=" << extension_candidates_.size()
            << " after_init_containing_V=" << out_with_V
            << " rejected=" << (in_count - extension_candidates_.size());


  if (static_cast<int>(extension_candidates_.size()) < extension_target_ &&
      trainer_spec_.hard_vocab_limit()) {
    return absl::FailedPreconditionError(
        "insufficient extension candidates have an inherited NORMAL decomposition");
  }

  double required_mass_total = 0.0;
  for (const auto& c : extension_candidates_) {
    if (c.is_required) required_mass_total += std::exp(c.inherited_best_score);
  }
  ABSL_RETURN_IF_ERROR(SolveCoveredBasisLambda(
      extension_candidates_, required_mass_total, &lambda_));

  // Candidate and its inherited decomposition are EXACTLY TIED here, and that
  // is not an accident: the decomposition's pieces partition the same surface,
  // so sum_j l_j = L_x and sum_j (s_j + lambda*l_j) = B_x + lambda*L_x, which
  // is precisely this score. lambda cancels. Whether that systematic tie is
  // benign or is what breaks Viterbi-based pruning is being measured; do not
  // "fix" it by adding corpus frequency until that measurement exists.
  for (auto& candidate : extension_candidates_) {
    // EXACTLY the term the root solver summed. Required pieces keep their
    // fixed r_k (no gauge exposure); ordinary pieces get beta_i + lambda*h_i.
    candidate.score = candidate.is_required
                          ? candidate.inherited_best_score
                          : candidate.inherited_best_score +
                                lambda_ * candidate.basis_inherited_len;
  }
  // GATE 2 raw report.
  {
    static const char* kG2[] = {
        "\xe2\x96\x81" "a4" "\xe2\x96\x81" "G4",
        "\xe2\x96\x81" "b4" "\xe2\x96\x81" "A4",
        "\xe2\x96\x81" "d4" "\xe2\x96\x81" "C4",
        "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "d5",
        "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "a4",
        "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "e5",
        "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "c5",
        "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "g4",
        "V"};
    absl::flat_hash_set<std::string> present;
    present.reserve(extension_candidates_.size());
    for (const auto& c : extension_candidates_) present.insert(c.piece);
    for (const char* t : kG2) {
      LOG(INFO) << "GATE2TRACK " << t << " survives_init="
                << (present.contains(std::string(t)) ? "yes" : "NO");
    }
    double deriv = 0.0;
    const double bm = BaseMass(lambda_, &deriv);
    double req_mass = 0.0, ord_mass = 0.0;
    for (const auto& c : extension_candidates_) {
      const double m = std::exp(c.score);
      if (required_extensions_.contains(c.piece)) req_mass += m;
      else ord_mass += m;
    }
    // INVARIANT: required and ordinary are disjoint, and each required piece
    // occurs exactly once in the candidate table (hence exactly once in the
    // final working piece table).
    size_t req_seen = 0;
    absl::flat_hash_set<std::string> req_dupe;
    for (const auto& c : extension_candidates_) {
      if (!c.is_required) continue;
      ++req_seen;
      if (!req_dupe.insert(c.piece).second) {
        return absl::InternalError(absl::StrCat(
            "required extension appears more than once: ", c.piece));
      }
      if (!required_extensions_.contains(c.piece)) {
        return absl::InternalError("is_required set on a non-required piece");
      }
    }
    if (req_seen != required_extensions_.size()) {
      return absl::InternalError(absl::StrCat(
          "required extension count mismatch: table has ", req_seen,
          " but ", required_extensions_.size(), " are required"));
    }
    const double residual = std::abs((bm + req_mass + ord_mass) - 1.0);
    LOG(INFO) << "GATE2INV required_disjoint=yes required_once=yes"
              << " required_count=" << req_seen
              << " normalization_residual=" << residual;
    LOG(INFO) << "GATE2NORM BaseMass=" << bm
              << " required_extension_mass=" << req_mass
              << " ordinary_extension_mass=" << ord_mass
              << " total_mass=" << (bm + req_mass + ord_mass)
              << " lambda=" << lambda_
              << " all_scores_finite="
              << (std::all_of(extension_candidates_.begin(),
                              extension_candidates_.end(),
                              [](const ExtensionCandidate& c) {
                                return std::isfinite(c.score);
                              })
                      ? "yes" : "NO");
    if (const char* d2 = std::getenv("SPM_STOP_AFTER_INIT")) {
      (void)d2;
      LOG(INFO) << "SPM_STOP_AFTER_INIT set; stopping before EM.";
      return absl::CancelledError("stop-after-init");
    }
  }
  return absl::OkStatus();
}

ModelProto ContinuationTrainer::BuildWorkingModel(
    const std::vector<ExtensionCandidate>& extensions, double lambda) const {
  ModelProto model = prior_model_;
  for (const auto& piece : inherited_normal_) {
    model.mutable_pieces(piece.external_id)
        ->set_score(static_cast<float>(piece.prior_score +
                                      lambda * piece.additive_length));
  }
  for (const auto& candidate : extensions) {
    auto* piece = model.add_pieces();
    piece->set_piece(candidate.piece);
    piece->set_score(static_cast<float>(candidate.score));
    piece->set_type(ModelProto::SentencePiece::NORMAL);
  }
  model.mutable_trainer_spec()->set_vocab_size(model.pieces_size());
  return model;
}

absl::Status ContinuationTrainer::RebuildWorkingModel(
    const std::vector<ExtensionCandidate>& extensions, double lambda,
    WorkingModelState* state) const {
  // Destroy the Model FIRST: it points into state->proto, and the next step
  // mutates that proto structurally.
  state->model.reset();

  state->proto = prior_model_;
  for (const auto& piece : inherited_normal_) {
    state->proto.mutable_pieces(piece.external_id)
        ->set_score(static_cast<float>(piece.prior_score +
                                       lambda * piece.additive_length));
  }
  for (const auto& candidate : extensions) {
    auto* piece = state->proto.add_pieces();
    piece->set_piece(candidate.piece);
    piece->set_score(static_cast<float>(candidate.score));
    piece->set_type(ModelProto::SentencePiece::NORMAL);
  }
  state->proto.mutable_trainer_spec()->set_vocab_size(
      state->proto.pieces_size());

  const size_t expect =
      static_cast<size_t>(prior_model_.pieces_size()) + extensions.size();
  if (static_cast<size_t>(state->proto.pieces_size()) != expect) {
    return absl::InternalError(absl::StrCat(
        "working support size mismatch: proto has ", state->proto.pieces_size(),
        " expected ", expect));
  }

  state->model = std::make_unique<Model>(state->proto);
  if (!state->model->status().ok()) return state->model->status();
  ++state->support_generation;
  ++state->support_builds;
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::UpdateWorkingScores(
    const std::vector<ExtensionCandidate>& extensions, double lambda,
    WorkingModelState* state) const {
  const size_t base_n = static_cast<size_t>(prior_model_.pieces_size());
  const size_t expect = base_n + extensions.size();
  if (static_cast<size_t>(state->proto.pieces_size()) != expect) {
    return absl::InternalError(absl::StrCat(
        "score-only update against a stale support: proto has ",
        state->proto.pieces_size(), " expected ", expect,
        ". A support change requires RebuildWorkingModel()."));
  }
  if (state->model == nullptr) {
    return absl::FailedPreconditionError("no working model to update");
  }

  for (const auto& piece : inherited_normal_) {
    state->proto.mutable_pieces(piece.external_id)
        ->set_score(static_cast<float>(piece.prior_score +
                                       lambda * piece.additive_length));
  }
  for (size_t i = 0; i < extensions.size(); ++i) {
    // Strings/ids/types are untouched; only the score field moves.
    state->proto.mutable_pieces(static_cast<int>(base_n + i))
        ->set_score(static_cast<float>(extensions[i].score));
  }
#ifndef NDEBUG
  for (size_t i = 0; i < extensions.size(); ++i) {
    if (state->proto.pieces(static_cast<int>(base_n + i)).piece() !=
        extensions[i].piece) {
      return absl::InternalError(
          "score-only update found a support drift at the extension region");
    }
  }
#endif
  // min_score_ is derived from scores and would otherwise go stale.
  ABSL_RETURN_IF_ERROR(state->model->RefreshScoreCache());
  ++state->score_refreshes;
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::RunEStep(
    const Model& model, size_t support_size, std::vector<float>* expected,
    double* objective) const {
  // Consumes the persistent model. Builds nothing.
  expected->assign(support_size, 0.0f);
  *objective = 0.0;
  long double total_weight = 0.0L;
  for (const auto& sentence : corpus_.sentences) {
    total_weight += sentence.second;
    Lattice lattice;
    lattice.SetSentence(sentence.first);
    model.PopulateNodes(&lattice);
    const float z = lattice.PopulateMarginal(
        static_cast<float>(sentence.second), expected);
    if (std::isnan(z)) {
      return absl::InternalError(
          "Unigram continuation E-step produced NaN likelihood");
    }
    *objective -= static_cast<double>(z);
  }
  if (total_weight > 0) *objective /= static_cast<double>(total_weight);
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::SolveMStepLambda(
    const std::vector<float>& expected, size_t extension_count,
    double* lambda) const {
  double l0 = 0.0;
  for (const auto& piece : inherited_normal_) {
    l0 += static_cast<double>(expected[piece.external_id]) *
          piece.additive_length;
  }
  double ca = 0.0;
  const size_t base_n = static_cast<size_t>(prior_model_.pieces_size());
  for (size_t i = 0; i < extension_count; ++i) {
    ca += expected[base_n + i];
  }

  if (!Finite(l0) || !Finite(ca) || l0 < 0.0 || ca < 0.0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "constrained M-step received invalid expected counts: inherited "
        "additive length ", l0, ", extension mass ", ca));
  }

  // The upper boundary: the lambda at which inherited NORMAL mass is exactly
  // one, leaving nothing for extensions. Every admissible lambda is below it.
  const auto log_base_mass = [this](double x) { return LogBaseMass(x); };
  double boundary = 0.0;
  ABSL_RETURN_IF_ERROR(BisectMonotoneRoot(
      "inherited-mass boundary lambda", log_base_mass, /*increasing=*/true,
      -1.0, 1.0, &boundary));

  // If no extension token was used, the constrained optimum is the upper
  // boundary where inherited NORMAL mass is one. Conversely, if extensions
  // were used but inherited expected additive length is zero, the objective is
  // monotone toward lambda -> -infinity: inherited mass must go to zero rather
  // than to one. Represent that limiting solution by a deterministic finite
  // lambda whose inherited mass is <= 1e-30.
  if (ca <= 1e-30) {
    *lambda = boundary;
    return absl::OkStatus();
  }
  if (l0 <= 1e-30) {
    // Extensions carry every count and the inherited side carries none, so the
    // objective is monotone toward lambda -> -infinity. Stand in for that
    // limit with the deterministic finite lambda whose inherited mass is 1e-30.
    const auto shifted = [this](double x) {
      return LogBaseMass(x) - kDegenerateLogBaseMass;
    };
    return BisectMonotoneRoot("degenerate continuation lambda", shifted,
                              /*increasing=*/true, boundary - 1.0, boundary,
                              lambda);
  }

  auto derivative = [&](double lambda) {
    double zprime = 0.0;
    const double z = BaseMass(lambda, &zprime);
    if (!(z < 1.0) || !Finite(zprime)) {
      return -std::numeric_limits<double>::infinity();
    }
    return l0 - ca * zprime / std::max(1e-300, 1.0 - z);
  };

  // The stationary point of the constrained objective. The derivative falls
  // with lambda, so the root is bracketed from below.
  const double hi = boundary - 1e-10;
  return BisectMonotoneRoot("constrained M-step lambda", derivative,
                            /*increasing=*/false, hi - 1.0, hi, lambda);
}

absl::Status ContinuationTrainer::RunConstrainedMStep(
    const std::vector<float>& expected,
    std::vector<ExtensionCandidate>* extensions, double* lambda) const {
  if (expected.size() !=
      static_cast<size_t>(prior_model_.pieces_size()) + extensions->size()) {
    return absl::InternalError("continuation expected-count shape mismatch");
  }

  ABSL_RETURN_IF_ERROR(
      SolveMStepLambda(expected, extensions->size(), lambda));
  double derivative = 0.0;
  const double base_mass = BaseMass(*lambda, &derivative);
  if (!Finite(base_mass) || !(base_mass > 0.0) || !(base_mass <= 1.0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "constrained M-step produced an invalid inherited mass ", base_mass,
        " at lambda=", *lambda, "; it must lie in (0, 1]"));
  }
  const double epsilon = std::max(1e-300, 1.0 - base_mass);

  const size_t base_n = static_cast<size_t>(prior_model_.pieces_size());
  // TWO DENOMINATORS, DELIBERATELY.
  //   ca       = sum of RAW expected counts. This is the observed posterior
  //              mass C = sum_j d_j and is the coefficient of log(epsilon(l))
  //              in the constrained M-step derivation, so the lambda solve
  //              must keep using it.
  //   ca_floor = sum of FLOORED numerators, sum_j max(d_j, 1e-30). The floor is
  //              a finite-support numerical convention for the relative shares,
  //              not observed mass. Allocating floored numerators over the raw
  //              denominator makes the assigned extension mass exceed epsilon
  //              by epsilon*(ca_floor - ca)/ca. Bounded by ~1e-24 for 1e6
  //              extensions and thus far below the 1e-12 residual checked at
  //              initialization, but it is an inconsistency and costs nothing
  //              to remove.
  double ca = 0.0;
  double ca_floor = 0.0;
  size_t floored = 0;
  for (size_t i = 0; i < extensions->size(); ++i) {
    const double d = expected[base_n + i];
    ca += d;
    const double f = std::max<double>(1e-30, d);
    if (f > d) ++floored;
    ca_floor += f;
  }
  if (!(ca > 0.0) || !Finite(ca) || !(epsilon > 0.0)) {
    return absl::FailedPreconditionError(
        "extension received no finite probability mass in constrained M-step");
  }
  if (floored > 0) {
    LOG(INFO) << "MSTEPFLOOR extensions=" << extensions->size()
              << " ca_raw=" << ca << " ca_floored=" << ca_floor
              << " floored_counts=" << floored
              << " mass_residual_if_unfixed="
              << (epsilon * (ca_floor - ca) / ca);
  }

  for (size_t i = 0; i < extensions->size(); ++i) {
    const double count = std::max<double>(1e-30, expected[base_n + i]);
    (*extensions)[i].score = std::log(epsilon) + std::log(count / ca_floor);
  }
  return absl::OkStatus();
}

// STAGED CANDIDATE TRACKER.
//
// Reports, per tracked candidate, the quantities that distinguish four
// mutually exclusive failure modes. It does NOT interpret them.
//   A  Viterbi=0 but posterior substantial -> Viterbi pruning misclassifies.
//   B  Viterbi=0 and posterior negligible before pruning -> competition/EM.
//   C  posterior substantial then collapses at the M-step -> M-step dynamics.
//   D  posterior substantial until pruning, then removed -> pruning criterion.
// Viterbi frequency alone never establishes any of them: the E-step is
// forward-backward (PopulateMarginal), and every candidate is initialized
// EXACTLY tied with its inherited-only decomposition (B_x + lambda*L_x on both
// sides), so ties are systematic and a deterministic Viterbi tie-break can
// report 0 for a candidate carrying real posterior mass.
absl::Status ContinuationTrainer::ReportTracked(
    absl::string_view stage, const std::vector<float>* expected_in,
    const std::vector<double>* loss_in) const {
  // OPT-IN. Each call builds a full working ModelProto (prior + every
  // candidate) and makes a complete corpus pass. At a 10^6 candidate pool that
  // is several GB and minutes per call, and calling it at three stages plus
  // every pruning round exhausted a 16 GB cgroup before the first prune. The
  // diagnostic is worth its cost on small pools and during Gate 2; it must not
  // be paid by a production run.
  if (std::getenv("SPM_TRACK") == nullptr) return absl::OkStatus();
  static const char* kTracked[] = {
      "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "d5",
      "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "a4",
      "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "e5",
      "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "c5",
      "\xe2\x96\x81" "Vn:" "\xe2\x96\x81" "g4",
      "\xe2\x96\x81" "a4" "\xe2\x96\x81" "G4",
      "\xe2\x96\x81" "b4" "\xe2\x96\x81" "A4",
      "\xe2\x96\x81" "d5" "\xe2\x96\x81" "C5",
      "\xe2\x96\x81" "e5" "\xe2\x96\x81" "D5",
      "\xe2\x96\x81" "c5" "\xe2\x96\x81" "B4",
      "\xe2\x96\x81" "g5" "\xe2\x96\x81" "F5",
      "\xe2\x96\x81" "f5" "\xe2\x96\x81" "E5",
      "\xe2\x96\x81" "a5" "\xe2\x96\x81" "G5",
      "\xe2\x96\x81" "b5" "\xe2\x96\x81" "A5",
      "\xe2\x96\x81" "d4" "\xe2\x96\x81" "C4",
      "\xe2\x96\x81" "A4.",
      "\xe2\x96\x81" "E5.",
      "\xe2\x96\x81" "B4.",
      "\xe2\x96\x81" "D5.",
      "\xe2\x96\x81" "G4.",
      "/128",
      "R:",
      "L:",
      "12/8k",
      "/4k",
      "V"};

  const ModelProto working = BuildWorkingModel(extension_candidates_, lambda_);
  Model model(working);
  if (!model.status().ok()) return model.status();
  Model prior_only(prior_model_);
  if (!prior_only.status().ok()) return prior_only.status();
  const size_t base_n = static_cast<size_t>(prior_model_.pieces_size());

  std::vector<float> viterbi(working.pieces_size(), 0.0f);
  std::vector<float> expected_local(working.pieces_size(), 0.0f);
  const bool need_local = (expected_in == nullptr);
  for (const auto& sentence : corpus_.sentences) {
    Lattice lat;
    lat.SetSentence(sentence.first);
    model.PopulateNodes(&lat);
    if (need_local) {
      lat.PopulateMarginal(static_cast<float>(sentence.second), &expected_local);
    }
    for (const auto* node : lat.Viterbi().first) {
      if (node->id >= 0) viterbi[node->id] += static_cast<float>(sentence.second);
    }
  }
  const std::vector<float>& expected = need_local ? expected_local : *expected_in;

  absl::flat_hash_map<std::string, size_t> index;
  for (size_t i = 0; i < extension_candidates_.size(); ++i) {
    index[extension_candidates_[i].piece] = i;
  }

  LOG(INFO) << "TRACK[" << stage << "] piece|occ|cand_score|decomp_score|delta"
               "|viterbi|posterior|post_per_occ|loss|self_1tok|alt_seg";
  for (const char* t : kTracked) {
    const std::string key(t);
    auto it = index.find(key);
    if (it == index.end()) {
      LOG(INFO) << "TRACK[" << stage << "] " << key << " | ABSENT_FROM_CANDIDATES";
      continue;
    }
    const size_t i = it->second;
    const auto& c = extension_candidates_[i];
    const int len = static_cast<int>(string_util::UTF8Len(c.piece));
    Lattice pl;
    pl.SetSentence(c.piece);
    prior_only.PopulateNodes(&pl);
    const double decomp = pl.Viterbi().second + lambda_ * len;

    // Is the candidate its own 1-token Viterbi segmentation in isolation, and
    // what is the best full-model alternative?
    Lattice fl;
    fl.SetSentence(c.piece);
    model.PopulateNodes(&fl);
    const auto nb = fl.NBest(2, false, 0.0);
    const bool self_1tok = !nb.empty() && nb[0].first.size() == 1;
    std::string alt = "(none)";
    double alt_score = 0.0;
    if (nb.size() > 1) {
      alt.clear();
      for (const auto* n : nb[1].first) {
        alt.append(std::string(n->piece)).append(" ");
      }
      alt_score = nb[1].second;
    }
    const double occ = static_cast<double>(c.freq);
    const double post = expected[base_n + i];
    LOG(INFO) << "TRACK[" << stage << "] " << c.piece << " | " << occ << " | "
              << c.score << " | " << decomp << " | " << (c.score - decomp)
              << " | " << viterbi[base_n + i] << " | " << post << " | "
              << (occ > 0 ? post / occ : 0.0) << " | "
              << (loss_in ? absl::StrCat((*loss_in)[i]) : std::string("-"))
              << " | " << (self_1tok ? "yes" : "no") << " | " << alt << "("
              << alt_score << ")";
  }

  // Aggregates over ALL candidates.
  size_t n = 0, zero_vit = 0, zv_post_gt1 = 0, zv_post_gt100 = 0, post_le1 = 0;
  double post_sum = 0.0;
  for (size_t i = 0; i < extension_candidates_.size(); ++i) {
    const double v = viterbi[base_n + i];
    const double p = expected[base_n + i];
    ++n; post_sum += p;
    if (v <= 0.0) {
      ++zero_vit;
      if (p > 1.0) ++zv_post_gt1;
      if (p > 100.0) ++zv_post_gt100;
    }
    if (p <= 1.0) ++post_le1;
  }
  LOG(INFO) << "TRACKAGG[" << stage << "] cands=" << n
            << " zero_viterbi=" << zero_vit
            << " posterior_sum=" << post_sum
            << " zeroVit_post>1=" << zv_post_gt1
            << " zeroVit_post>100=" << zv_post_gt100
            << " post<=1=" << post_le1;
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::RunContinuationEM() {
  ABSL_RETURN_IF_ERROR(ReportTracked("1-post-init", nullptr, nullptr));
  if (extension_target_ == 0) return absl::OkStatus();
  if (extension_candidates_.empty()) {
    return absl::FailedPreconditionError("no Unigram extension candidates");
  }

  int prune_round = 0;
  bool reported_first_em = false;

  // ONE support build before the first E-step. Thereafter the trie is rebuilt
  // only when pruning changes the support; score-only M-step results are
  // written into the live proto and the score cache refreshed in place.
  WorkingModelState working;
  ABSL_RETURN_IF_ERROR(
      RebuildWorkingModel(extension_candidates_, lambda_, &working));

  while (static_cast<int>(extension_candidates_.size()) > extension_target_) {
    std::vector<float> expected;
    double objective = 0.0;
    for (int sub = 0; sub < trainer_spec_.num_sub_iterations(); ++sub) {
      ABSL_RETURN_IF_ERROR(RunEStep(*working.model,
                                    static_cast<size_t>(working.proto.pieces_size()),
                                    &expected, &objective));
      if (!reported_first_em) {
        ABSL_RETURN_IF_ERROR(
            ReportTracked("2-post-Estep-pre-Mstep", &expected, nullptr));
      }
      ABSL_RETURN_IF_ERROR(
          RunConstrainedMStep(expected, &extension_candidates_, &lambda_));
      // SCORE-ONLY: no trie rebuild.
      ABSL_RETURN_IF_ERROR(
          UpdateWorkingScores(extension_candidates_, lambda_, &working));
      if (!reported_first_em) {
        ABSL_RETURN_IF_ERROR(
            ReportTracked("3-post-Mstep-pre-prune", nullptr, nullptr));
        reported_first_em = true;
      }
    }

    const size_t base_n = static_cast<size_t>(prior_model_.pieces_size());

    // RANK OPTIONAL EXTENSION CANDIDATES BY FORWARD-BACKWARD POSTERIOR
    // EXPECTED OCCUPANCY. That is the active and only pruning rule here.
    //
    // Deletion loss is NOT active. An earlier revision switched to it on
    // evidence that is now known to be confounded: those runs passed
    // split_by_whitespace=true, so IsValidSentencePiece() rejected any piece
    // containing an internal U+2581 -- every multi-word candidate, including
    // the bare note bigrams that define the acceptance criterion and every
    // Vn:+pitch piece, was structurally impossible to enumerate. Expected-count
    // pruning had never been tested on the intended candidate universe.
    //
    // `expected` is forward-backward POSTERIOR occupancy, never Viterbi
    // counts, so this ranking is unaffected by the zero-Viterbi /
    // positive-posterior discrepancy measured separately (/128: Viterbi 0,
    // posterior 363.8).

    // PROVENANCE (opt-in via SPM_TRACK; see ReportTracked for why).
    // Without this, "the phrase is absent" cannot be attributed to a cause: it
    // may never have been a candidate, may have been ranked out, or may have
    // been starved of mass.
    //
    // The deletion-loss and Viterbi-frequency statistics that used to be
    // printed here were read out of two vectors that this function declared
    // and never filled -- `viterbi_freq[bn + i]` on an empty vector, i.e.
    // undefined behaviour on every SPM_TRACK run. They are removed rather than
    // repaired: reconstructing them costs a full extra corpus pass and a
    // million-piece model rebuild, and ReportTracked() already computes the
    // Viterbi/posterior diagnostics on request.
    if (std::getenv("SPM_TRACK") != nullptr) {
      size_t n_prefixed = 0, n_zero_posterior = 0;
      double ext_mass = 0.0;
      const size_t bn = base_n;
      for (size_t i = 0; i < extension_candidates_.size(); ++i) {
        if (absl::StartsWith(extension_candidates_[i].piece, "\xe2\x96\x81")) {
          ++n_prefixed;
        }
        if (expected[bn + i] <= 0.0) ++n_zero_posterior;
        ext_mass += expected[bn + i];
      }
      double deriv_dbg = 0.0;
      const double base_mass_dbg = BaseMass(lambda_, &deriv_dbg);
      LOG(INFO) << "continuation provenance: cands="
                << extension_candidates_.size() << " word_initial=" << n_prefixed
                << " zero_posterior=" << n_zero_posterior
                << " extension_expected_mass=" << ext_mass
                << " base_mass=" << base_mass_dbg
                << " epsilon=" << (1.0 - base_mass_dbg);
    }

    // REQUIRED EXTENSIONS DO NOT COMPETE.
    // Corpus characters the prior cannot spell are genuinely new final
    // vocabulary entries -- they count against extension_target_ -- but they
    // are not optional, so they must not be ranked at all. Encoding them as
    // +inf inside the ranking was wrong twice over: it made correctness depend
    // on the comparator's handling of non-finite values (which was broken, and
    // silently pruned the required character 'V' by falling through to lexical
    // order), and it conflated "must keep" with "ranked first", which are
    // different statements. The reference implementation likewise removes
    // always-keep pieces from competition and fills the remaining capacity
    // from ranked candidates.
    ABSL_RETURN_IF_ERROR(ReportTracked(
        absl::StrCat("4-pre-prune-round", ++prune_round), &expected, nullptr));

    std::vector<size_t> required_idx, optional_idx;
    for (size_t i = 0; i < extension_candidates_.size(); ++i) {
      if (required_extensions_.contains(extension_candidates_[i].piece)) {
        required_idx.push_back(i);
      } else {
        optional_idx.push_back(i);
      }
    }
    if (static_cast<int>(required_idx.size()) > extension_target_) {
      return absl::FailedPreconditionError(absl::StrCat(
          "continuation needs ", required_idx.size(),
          " required extension pieces to cover the corpus but the extension "
          "budget is only ", extension_target_));
    }

    std::sort(optional_idx.begin(), optional_idx.end(),
              [&](size_t a, size_t b) {
                const double ea = expected[base_n + a];
                const double eb = expected[base_n + b];
                if (!ScoresTie(ea, eb)) return ea > eb;
                // Deterministic tie-break, as before.
                return extension_candidates_[a].piece <
                       extension_candidates_[b].piece;
              });

    // Required first, then the ranked optional tail. `order` keeps the rest of
    // the loop unchanged.
    std::vector<size_t> order;
    order.reserve(extension_candidates_.size());
    order.insert(order.end(), required_idx.begin(), required_idx.end());
    order.insert(order.end(), optional_idx.begin(), optional_idx.end());

    size_t keep = std::max<size_t>(
        extension_target_, static_cast<size_t>(
                               extension_candidates_.size() *
                               trainer_spec_.shrinking_factor()));
    keep = std::min(keep, extension_candidates_.size() - 1);
    keep = std::max<size_t>(keep, extension_target_);
    // Never shrink into the required prefix.
    keep = std::max(keep, required_idx.size());

    std::vector<ExtensionCandidate> next;
    next.reserve(keep);
    // Same convention as the M-step: the warm-start denominator must be the
    // sum of the FLOORED numerators actually used below, or the kept set is
    // allocated slightly more than epsilon.
    double kept_count = 0.0;
    for (size_t i = 0; i < keep; ++i) {
      next.push_back(extension_candidates_[order[i]]);
      kept_count += std::max<double>(1e-30, expected[base_n + order[i]]);
    }

    double deriv = 0.0;
    const double epsilon = std::max(1e-300, 1.0 - BaseMass(lambda_, &deriv));
    for (size_t i = 0; i < keep; ++i) {
      const double count =
          std::max<double>(1e-30, expected[base_n + order[i]]);
      next[i].score = std::log(epsilon) +
                      std::log(count / std::max(1e-300, kept_count));
    }
    extension_candidates_.swap(next);
    // SUPPORT CHANGED -> exactly one rebuild.
    ABSL_RETURN_IF_ERROR(
        RebuildWorkingModel(extension_candidates_, lambda_, &working));
    LOG(INFO) << "Unigram continuation pruned extension candidates to "
              << extension_candidates_.size() << " objective=" << objective
              << " lambda=" << lambda_;
  }

  if (static_cast<int>(extension_candidates_.size()) < extension_target_) {
    if (trainer_spec_.hard_vocab_limit()) {
      return absl::FailedPreconditionError(
          "hard Unigram continuation capacity cannot be reached");
    }
    extension_target_ = static_cast<int>(extension_candidates_.size());
  }

  std::vector<float> expected;
  double objective = 0.0;
  const int final_iters = std::max(2, trainer_spec_.num_sub_iterations());
  // The loop above already rebuilt for the final support after its last prune;
  // if the target was met without entering the loop, `working` still holds the
  // initial support, which is the correct one. Either way: no rebuild here.
  if (working.model == nullptr ||
      static_cast<size_t>(working.proto.pieces_size()) !=
          static_cast<size_t>(prior_model_.pieces_size()) +
              extension_candidates_.size()) {
    ABSL_RETURN_IF_ERROR(
        RebuildWorkingModel(extension_candidates_, lambda_, &working));
  }
  for (int sub = 0; sub < final_iters; ++sub) {
    ABSL_RETURN_IF_ERROR(RunEStep(*working.model,
                                  static_cast<size_t>(working.proto.pieces_size()),
                                  &expected, &objective));
    ABSL_RETURN_IF_ERROR(
        RunConstrainedMStep(expected, &extension_candidates_, &lambda_));
    ABSL_RETURN_IF_ERROR(
        UpdateWorkingScores(extension_candidates_, lambda_, &working));
  }
  LOG(INFO) << "WORKINGMODEL support_builds=" << working.support_builds
            << " score_refreshes=" << working.score_refreshes
            << " support_generation=" << working.support_generation;

  const size_t base_n = static_cast<size_t>(prior_model_.pieces_size());
  int zero_count = 0;
  for (size_t i = 0; i < extension_candidates_.size(); ++i) {
    if (expected[base_n + i] <= 0.0f) ++zero_count;
  }
  if (zero_count != 0 && trainer_spec_.hard_vocab_limit()) {
    return absl::FailedPreconditionError(absl::StrCat(
        zero_count, " requested extension pieces have zero expected count"));
  }

  std::sort(extension_candidates_.begin(), extension_candidates_.end(),
            [](const ExtensionCandidate& a, const ExtensionCandidate& b) {
              if (!ScoresTie(a.score, b.score)) return a.score > b.score;
              return a.piece < b.piece;
            });
  return absl::OkStatus();
}

absl::Status VerifyPriorPrefixInvariant(const ModelProto& prior_model,
                                        const ModelProto& output,
                                        double lambda) {
  const int prior_n = prior_model.pieces_size();
  if (output.pieces_size() < prior_n) {
    return absl::FailedPreconditionError(absl::StrCat(
        "continuation output has ", output.pieces_size(),
        " pieces, fewer than the prior's ", prior_n));
  }

  // Identity first: an inherited ID keeps its index, its bytes and its type.
  // Anything else silently repoints an embedding row.
  for (int id = 0; id < prior_n; ++id) {
    const auto& prior = prior_model.pieces(id);
    const auto& out = output.pieces(id);
    if (out.piece() != prior.piece()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "inherited piece at ID ", id, " changed from \"", prior.piece(),
          "\" to \"", out.piece(), "\""));
    }
    if (out.type() != prior.type()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "inherited piece \"", prior.piece(), "\" (ID ", id,
          ") changed type from ", static_cast<int>(prior.type()), " to ",
          static_cast<int>(out.type())));
    }
    if (prior.type() != ModelProto::SentencePiece::NORMAL &&
        out.score() != prior.score()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "inherited non-NORMAL piece \"", prior.piece(), "\" (ID ", id,
          ") had its score changed from ", prior.score(), " to ",
          out.score()));
    }
  }

  // Geometry second: every inherited NORMAL score moved by the SAME gauge,
  // lambda * length. That single degree of freedom is what lets inherited
  // segmentation decisions survive continuation, so a per-piece drift here is
  // a silent re-estimation, not a rounding detail.
  for (int id = 0; id < prior_n; ++id) {
    if (prior_model.pieces(id).type() !=
        ModelProto::SentencePiece::NORMAL) {
      continue;
    }
    const std::string& piece = prior_model.pieces(id).piece();
    const int additive_length = static_cast<int>(string_util::UTF8Len(piece));
    const double prior_score = prior_model.pieces(id).score();
    const double expected_shift = lambda * additive_length;
    const double actual_shift =
        static_cast<double>(output.pieces(id).score()) - prior_score;
    // The score is stored as float32, so one rounding of (prior + shift) is
    // unavoidable and is the whole budget.
    const double magnitude =
        std::max({std::abs(prior_score), std::abs(expected_shift),
                  std::abs(prior_score + expected_shift), 1.0});
    const double tolerance =
        8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
        magnitude;
    if (!Finite(actual_shift) ||
        std::abs(actual_shift - expected_shift) > tolerance) {
      return absl::FailedPreconditionError(absl::StrCat(
          "inherited NORMAL piece \"", piece, "\" (ID ", id,
          ") violates the additive-length gauge: expected shift ",
          expected_shift, " (lambda=", lambda,
          " * length=", additive_length, ") but the score moved by ",
          actual_shift, ", which exceeds the float32 tolerance ", tolerance));
    }
  }

  // Extensions append, never interleave.
  for (int id = prior_n; id < output.pieces_size(); ++id) {
    if (output.pieces(id).type() != ModelProto::SentencePiece::NORMAL) {
      return absl::FailedPreconditionError(absl::StrCat(
          "extension piece at ID ", id, " is not NORMAL"));
    }
  }
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::VerifyPriorPrefix(
    const ModelProto& output) const {
  return VerifyPriorPrefixInvariant(prior_model_, output, lambda_);
}

absl::Status ContinuationTrainer::VerifyFinalCoverage() const {
  // THE LAST GATE BEFORE ANYTHING IS WRITTEN.
  // A continuation model that cannot spell its own training corpus is not a
  // tokenizer, and it must never reach disk. This is not hypothetical: a
  // 1500-piece artifact was emitted whose required coverage extension had
  // been pruned by a comparator bug, so every violin row fell back to <unk>.
  // Nothing downstream noticed, because every other check -- inherited IDs,
  // types, gauge error, piece count -- still passed.
  const ModelProto final_model = BuildWorkingModel(extension_candidates_,
                                                   lambda_);
  Model model(final_model);
  if (!model.status().ok()) return model.status();

  const int unk = final_model.trainer_spec().unk_id();
  for (const auto& sentence : corpus_.sentences) {
    Lattice lattice;
    lattice.SetSentence(sentence.first);
    model.PopulateNodes(&lattice);
    for (const auto* node : lattice.Viterbi().first) {
      if (node->id == unk) {
        return absl::InternalError(absl::StrCat(
            "continuation produced a model that cannot represent its own "
            "corpus without <unk>; representative input: ", sentence.first,
            ". Refusing to write artifacts."));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::FinalizeArtifacts() {
  ABSL_RETURN_IF_ERROR(VerifyFinalCoverage());
  ExpansionResult result;
  result.set_schema_version(1);
  result.set_model_type(EXPANSION_UNIGRAM);
  result.set_first_new_external_id(prior_model_.pieces_size());
  result.set_requested_new_pieces(
      trainer_spec_.vocab_size() - prior_model_.pieces_size());
  result.set_actual_new_pieces(static_cast<int>(extension_candidates_.size()));
  result.set_unreachable_pieces(0);
  result.set_rank_prepend(false);
  result.set_prior_model_sha256(continuation::Sha256Hex(prior_model_bytes_));
  result.set_prior_piece_count(prior_model_.pieces_size());
  result.set_unigram_lambda(lambda_);

  // THE GAUGED MODEL IS BUILT FIRST, because the sidecars must describe IT.
  //
  // base_pieces used to be filled from prior.score(), i.e. the PRE-GAUGE
  // score, while the emitted .model carried prior_score + lambda*length. The
  // three artifacts then disagreed: .model held final inherited scores while
  // .expansion and .vocab held pre-gauge ones, and only the learned pieces
  // matched everywhere. ExpansionResult is documented as authoritative
  // continuation state, so that made it ambiguous read on its own.
  //
  // AUTHORITATIVE UNIGRAM SEMANTICS, from here on:
  //     base_pieces.score    = the score in the FINAL emitted tokenizer
  //     learned_pieces.score = the score in the FINAL emitted tokenizer
  // so inherited NORMAL carries the final gauged float32, inherited non-NORMAL
  // carries its original bit-identical score (the gauge never touches it), and
  // extensions carry their learned score. The pre-gauge prior stays fully
  // recoverable from prior_model_sha256, prior_piece_count and unigram_lambda.
  ModelProto output = prior_model_;
  double max_gauge_error = 0.0;
  for (const auto& inherited : inherited_normal_) {
    const double new_score =
        inherited.prior_score + lambda_ * inherited.additive_length;
    output.mutable_pieces(inherited.external_id)
        ->set_score(static_cast<float>(new_score));
    const double actual_shift =
        static_cast<double>(output.pieces(inherited.external_id).score()) -
        inherited.prior_score;
    max_gauge_error =
        std::max(max_gauge_error,
                 std::abs(actual_shift -
                          lambda_ * inherited.additive_length));
  }
  result.set_unigram_score_gauge_max_error(max_gauge_error);

  std::vector<ExpansionPiece> all_pieces;
  all_pieces.reserve(prior_model_.pieces_size() + extension_candidates_.size());
  for (int id = 0; id < prior_model_.pieces_size(); ++id) {
    const auto& final_piece = output.pieces(id);
    ExpansionPiece piece;
    piece.set_external_id(id);
    piece.set_piece(final_piece.piece());
    piece.set_type(final_piece.type());
    piece.set_score(final_piece.score());   // FINAL, not pre-gauge
    piece.set_mergeable(false);
    piece.set_atomic(string_util::UTF8Len(final_piece.piece()) == 1);
    *result.add_base_pieces() = piece;
    all_pieces.push_back(piece);
  }

  for (size_t i = 0; i < extension_candidates_.size(); ++i) {
    const auto& candidate = extension_candidates_[i];
    const int id = prior_model_.pieces_size() + static_cast<int>(i);
    auto* out = output.add_pieces();
    out->set_piece(candidate.piece);
    out->set_score(static_cast<float>(candidate.score));
    out->set_type(ModelProto::SentencePiece::NORMAL);

    ExpansionPiece piece;
    piece.set_external_id(id);
    piece.set_piece(candidate.piece);
    piece.set_type(ModelProto::SentencePiece::NORMAL);
    piece.set_score(static_cast<float>(candidate.score));
    piece.set_mergeable(false);
    piece.set_atomic(false);
    *result.add_learned_pieces() = piece;
    all_pieces.push_back(piece);
  }

  if (!output.has_trainer_spec()) *output.mutable_trainer_spec() = trainer_spec_;
  output.mutable_trainer_spec()->set_vocab_size(output.pieces_size());
  output.mutable_trainer_spec()->set_unigram_prior_model(
      trainer_spec_.unigram_prior_model());
  output.mutable_trainer_spec()->set_model_prefix(trainer_spec_.model_prefix());
  *output.mutable_normalizer_spec() = prior_model_.normalizer_spec();
  *output.mutable_denormalizer_spec() = prior_model_.denormalizer_spec();
  *output.mutable_expansion_result() = result;

  // Nothing is written until the prior prefix has been proven intact. A
  // recorded max error is a report; this is the gate.
  ABSL_RETURN_IF_ERROR(VerifyPriorPrefix(output));

  const std::string result_path = !trainer_spec_.expansion_result().empty()
                                      ? trainer_spec_.expansion_result()
                                      : trainer_spec_.model_prefix() + ".expansion";

  if (output_model_proto_ != nullptr) {
    *output_model_proto_ = output;
  } else {
    if (trainer_spec_.model_prefix().empty()) {
      return absl::InvalidArgumentError(
          "Unigram continuation requires model_prefix when no output proto is supplied");
    }
    ABSL_RETURN_IF_ERROR(continuation::WriteModelProto(
        trainer_spec_.model_prefix() + ".model", output));
    ABSL_RETURN_IF_ERROR(continuation::WriteExpansionVocab(
        trainer_spec_.model_prefix() + ".vocab", all_pieces));
  }

  // As with BPE, the success sidecar is emitted only after the model/vocab
  // artifacts have been constructed successfully.
  if (!result_path.empty()) {
    ABSL_RETURN_IF_ERROR(
        continuation::WriteExpansionResult(result_path, result));
  }

  LOG(INFO) << "Unigram continuation: prior=" << prior_model_.pieces_size()
            << " extension=" << extension_candidates_.size()
            << " first_new_id=" << prior_model_.pieces_size()
            << " lambda=" << lambda_
            << " gauge_max_error=" << max_gauge_error;
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::Train() {
  ABSL_RETURN_IF_ERROR(status());
  RET_CHECK_EQ(TrainerSpec::UNIGRAM, trainer_spec_.model_type());

  prior_model_.Clear();
  prior_model_bytes_.clear();
  inherited_normal_.clear();
  extension_candidates_.clear();
  sentences_.clear();
  final_pieces_.clear();
  lambda_ = 0.0;

  // All malformed configuration/prior-state failures happen before candidate
  // extraction or EM.
  ABSL_RETURN_IF_ERROR(LoadAndValidatePrior());
  ABSL_RETURN_IF_ERROR(continuation::LoadPreparedCorpus(
      trainer_spec_, prior_model_.normalizer_spec(), components_, &corpus_));
  // NOT `sentences_ = corpus_.sentences;`.
  // That kept a second full copy of the ~1.34M-record corpus for the whole
  // run. Continuation's coverage check, candidate extraction and E-step all
  // read corpus_.sentences; nothing in this path reads TrainerInterface's
  // sentences_. Verified by inspection: the only other mention is the clear()
  // above. If an inherited API is ever found to need it, populate it at that
  // call site rather than holding a duplicate for the whole run.
  ABSL_RETURN_IF_ERROR(VerifyCorpusCoverage());

  if (extension_target_ > 0) {
    ABSL_RETURN_IF_ERROR(MakeWeightedExtensionCandidates());
    ABSL_RETURN_IF_ERROR(InitializeContinuationScores());
    ABSL_RETURN_IF_ERROR(RunContinuationEM());
  }

  return FinalizeArtifacts();
}

}  // namespace sentencepiece::unigram
