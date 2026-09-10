// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "unigram_continuation_trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ret_check.h"
#include "unigram_model.h"
#include "unigram_model_trainer.h"
#include "util.h"

namespace sentencepiece::unigram {
namespace {

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
bool ScoresTie(double a, double b) {
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
  // BOOTSTRAP, DO NOT REFUSE.
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
  bootstrap_pieces_.clear();
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
  bootstrap_pieces_.assign(missing.begin(), missing.end());
  std::sort(bootstrap_pieces_.begin(), bootstrap_pieces_.end());
  if (!bootstrap_pieces_.empty()) {
    LOG(INFO) << "Unigram continuation bootstrap: " << bootstrap_pieces_.size()
              << " character(s) absent from the prior will be admitted as "
                 "EXTENSION candidates (they consume extension budget, and "
                 "the prior is not modified): "
              << absl::StrJoin(bootstrap_pieces_, " ");
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
  // Now: seed_sentencepiece_size, when set, IS the pool size, independent of
  // K; otherwise a fixed default (SentencePiece's own seed default) is used.
  // The pool is only ever raised to extension_target, since a pool smaller
  // than the budget cannot fill it.
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

  // Exact global weighted occurrence counts. A TSV row (x, N) contributes
  // exactly the same candidate occurrences as N physical copies of x, while
  // the record boundary remains a hard fence. Unlike the earlier bounded
  // suffix-array queue, no candidate loses accumulated mass through GC and no
  // chunk-local threshold changes the result.
  absl::flat_hash_map<std::string, uint64_t> counts;
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
    for (size_t begin = 0; begin < text.size(); ++begin) {
      string_util::UnicodeText piece;
      piece.reserve(std::min(max_piece_length, text.size() - begin));
      const size_t stop =
          std::min(text.size(), begin + std::max<size_t>(1, max_piece_length));
      for (size_t end = begin; end < stop; ++end) {
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
      }
    }
  }

  const double alpha =
      static_cast<double>(absl::GetFlag(FLAGS_min_freq_alpha));
  const double vocab = std::max<double>(1.0, trainer_spec_.vocab_size());
  const double denom = std::max<double>(1.0, vocab * std::log(vocab));
  const uint64_t global_min_freq = std::max<uint64_t>(
      2, static_cast<uint64_t>(
             alpha * static_cast<double>(weighted_bytes) / denom));

  std::vector<std::pair<std::string, uint64_t>> ranked;
  ranked.reserve(counts.size());
  for (auto& item : counts) {
    const uint64_t effective_min = item.first.size() <= 3 ? 2 : global_min_freq;
    if (item.second >= effective_min) {
      ranked.emplace_back(item.first, item.second);
    }
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
    const double sa = CandidateScore(a.first, a.second);
    const double sb = CandidateScore(b.first, b.second);
    if (sa != sb) return sa > sb;
    const size_t la = string_util::UTF8Len(a.first);
    const size_t lb = string_util::UTF8Len(b.first);
    if (la != lb) return la > lb;
    return a.first < b.first;
  });
  if (ranked.size() > candidate_limit) ranked.resize(candidate_limit);

  // BOOTSTRAP CHARACTERS ARE ADMITTED UNCONDITIONALLY, ahead of the frequency
  // filter and the pool cap. Without them the corpus is unrepresentable, so
  // they are not competing on merit with ordinary candidates -- but they are
  // still ORDINARY EXTENSIONS: appended after the prior, scored by the
  // constrained M-step, and counted against the extension budget. Pruning
  // keeps them on its own, because a piece with no alternative segmentation
  // has infinite deletion loss.
  if (!bootstrap_pieces_.empty()) {
    absl::flat_hash_set<std::string> present;
    present.reserve(ranked.size());
    for (const auto& item : ranked) present.insert(item.first);
    std::vector<std::pair<std::string, uint64_t>> prepend;
    for (const auto& ch : bootstrap_pieces_) {
      if (present.count(ch)) continue;
      auto it = counts.find(ch);
      // A bootstrap character always occurs; fall back to 1 if the counter
      // never saw it as a standalone substring.
      prepend.emplace_back(ch, it == counts.end()
                                   ? static_cast<uint64_t>(1)
                                   : it->second);
    }
    if (!prepend.empty()) {
      ranked.insert(ranked.begin(), prepend.begin(), prepend.end());
      LOG(INFO) << "Unigram continuation: admitted " << prepend.size()
                << " bootstrap character candidate(s) into the pool";
    }
  }

  extension_candidates_.reserve(ranked.size());
  for (auto& item : ranked) {
    ExtensionCandidate candidate;
    candidate.piece = std::move(item.first);
    extension_candidates_.push_back(std::move(candidate));
  }

  LOG(INFO) << "Unigram continuation extracted " << extension_candidates_.size()
            << " exact globally weighted candidates from "
            << corpus_.sentences.size() << " fenced records";
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

absl::Status ContinuationTrainer::SolveInitialLambda(
    const std::vector<ExtensionCandidate>& extensions, double* lambda) const {
  auto log_mass = [&](double lambda) {
    double max_term = -std::numeric_limits<double>::infinity();
    for (const auto& piece : inherited_normal_) {
      max_term = std::max(
          max_term, piece.prior_score + lambda * piece.additive_length);
    }
    for (const auto& piece : extensions) {
      const int len = static_cast<int>(string_util::UTF8Len(piece.piece));
      max_term = std::max(
          max_term, piece.inherited_best_score + lambda * len);
    }
    double sum = 0.0;
    for (const auto& piece : inherited_normal_) {
      sum += std::exp(piece.prior_score + lambda * piece.additive_length -
                      max_term);
    }
    for (const auto& piece : extensions) {
      const int len = static_cast<int>(string_util::UTF8Len(piece.piece));
      sum += std::exp(piece.inherited_best_score + lambda * len - max_term);
    }
    return max_term + std::log(sum);
  };

  // Total mass over inherited and candidate pieces rises with lambda, and the
  // initial gauge is the lambda that makes it exactly one.
  return BisectMonotoneRoot("initial continuation lambda", log_mass,
                            /*increasing=*/true, -1.0, 1.0, lambda);
}

absl::Status ContinuationTrainer::InitializeContinuationScores() {
  if (extension_target_ == 0) {
    lambda_ = 0.0;
    return absl::OkStatus();
  }

  TrainerModel::SentencePieces normal_pieces;
  normal_pieces.reserve(inherited_normal_.size());
  absl::flat_hash_set<std::string> normal_strings;
  for (const auto& piece : inherited_normal_) {
    normal_pieces.emplace_back(piece.piece,
                               static_cast<float>(piece.prior_score));
    normal_strings.insert(piece.piece);
  }
  TrainerModel normal_model(trainer_spec_, prior_model_.normalizer_spec());
  ABSL_RETURN_IF_ERROR(normal_model.SetSentencePieces(std::move(normal_pieces)));
  if (!normal_model.status().ok()) return normal_model.status();

  std::vector<ExtensionCandidate> valid;
  valid.reserve(extension_candidates_.size());
  for (auto candidate : extension_candidates_) {
    Lattice lattice;
    lattice.SetSentence(candidate.piece);
    normal_model.PopulateNodes(&lattice);
    const auto path = lattice.Viterbi();
    bool inherited_only = !path.first.empty();
    for (const auto* node : path.first) {
      if (!normal_strings.contains(std::string(node->piece))) {
        inherited_only = false;
        break;
      }
    }
    if (!inherited_only) continue;
    candidate.inherited_best_score = path.second;
    valid.push_back(std::move(candidate));
  }
  extension_candidates_.swap(valid);

  if (static_cast<int>(extension_candidates_.size()) < extension_target_ &&
      trainer_spec_.hard_vocab_limit()) {
    return absl::FailedPreconditionError(
        "insufficient extension candidates have an inherited NORMAL decomposition");
  }

  ABSL_RETURN_IF_ERROR(
      SolveInitialLambda(extension_candidates_, &lambda_));
  for (auto& candidate : extension_candidates_) {
    const int len = static_cast<int>(string_util::UTF8Len(candidate.piece));
    candidate.score = candidate.inherited_best_score + lambda_ * len;
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

absl::Status ContinuationTrainer::RunEStep(
    const std::vector<ExtensionCandidate>& extensions, double lambda,
    std::vector<float>* expected, double* objective) const {
  const ModelProto working = BuildWorkingModel(extensions, lambda);
  Model model(working);
  if (!model.status().ok()) return model.status();

  expected->assign(working.pieces_size(), 0.0f);
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

absl::Status ContinuationTrainer::ComputeExtensionDeletionLoss(
    const std::vector<ExtensionCandidate>& extensions, double lambda,
    std::vector<double>* loss, std::vector<float>* viterbi_freq) const {
  // Mirrors Trainer::PruneSentencePieces in unigram_model_trainer.cc. The only
  // differences are structural, and both follow from the continuation
  // contract: the piece table is inherited+extensions rather than a single
  // trained vocabulary, and only the extension region is scored, because
  // inherited pieces can never be pruned.
  const ModelProto working = BuildWorkingModel(extensions, lambda);
  Model model(working);
  if (!model.status().ok()) return model.status();

  const size_t base_n = static_cast<size_t>(prior_model_.pieces_size());
  const size_t total = static_cast<size_t>(working.pieces_size());

  // Viterbi token frequencies over the whole corpus. Ordinary Unigram pruning
  // uses the Viterbi path, not the marginal, because the loss below is stated
  // in terms of "every occurrence is replaced by its alternative".
  viterbi_freq->assign(total, 0.0f);
  {
    Lattice lattice;
    for (const auto& sentence : corpus_.sentences) {
      lattice.SetSentence(sentence.first);
      model.PopulateNodes(&lattice);
      for (const auto* node : lattice.Viterbi().first) {
        if (node->id >= 0 && static_cast<size_t>(node->id) < total) {
          (*viterbi_freq)[node->id] += static_cast<float>(sentence.second);
        }
      }
    }
  }

  double sum = 0.0;
  for (size_t i = 0; i < total; ++i) sum += (*viterbi_freq)[i];
  if (!(sum > 0.0)) {
    return absl::FailedPreconditionError(
        "Unigram continuation pruning: empty Viterbi frequency mass");
  }
  const double logsum = std::log(sum);

  loss->assign(extensions.size(), 0.0);
  Lattice lattice;
  for (size_t k = 0; k < extensions.size(); ++k) {
    const size_t id = base_n + k;
    const double f = static_cast<double>((*viterbi_freq)[id]);

    lattice.SetSentence(extensions[k].piece);
    model.PopulateNodes(&lattice);
    const auto nbests = lattice.NBest(2, false, 0.0);

    if (nbests.empty()) {                       // unreachable; drop it
      (*loss)[k] = -std::numeric_limits<double>::infinity();
      continue;
    }
    if (nbests.size() == 1) {
      // No second best: this piece is the only way to spell itself, so
      // removing it would make its string unrepresentable. Must keep.
      (*loss)[k] = std::numeric_limits<double>::infinity();
      continue;
    }
    if (nbests[0].first.size() >= 2) {
      // Its own Viterbi path already prefers a split, so the piece is never
      // used and costs nothing to remove.
      (*loss)[k] = -std::numeric_limits<double>::infinity();
      continue;
    }
    if (f <= 0.0) {                             // never on a Viterbi path
      (*loss)[k] = -std::numeric_limits<double>::infinity();
      continue;
    }

    std::vector<int> alt;
    alt.reserve(nbests[1].first.size());
    for (const auto* node : nbests[1].first) {
      if (node->id >= 0 && static_cast<size_t>(node->id) < total) {
        alt.push_back(node->id);
      }
    }
    if (alt.empty()) {                          // no usable alternative: keep
      (*loss)[k] = std::numeric_limits<double>::infinity();
      continue;
    }

    const double logprob_sp = std::log(f) - logsum;
    // Removing the piece re-assigns its f occurrences to |alt| pieces each.
    const double logsum_alt =
        std::log(sum + f * (static_cast<double>(alt.size()) - 1.0));
    double logprob_alt = 0.0;
    for (const int n : alt) {
      logprob_alt +=
          std::log(static_cast<double>((*viterbi_freq)[n]) + f) - logsum_alt;
    }
    const double F = f / sum;
    (*loss)[k] = F * (logprob_sp - logprob_alt);
  }
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
  double ca = 0.0;
  for (size_t i = 0; i < extensions->size(); ++i) {
    ca += expected[base_n + i];
  }
  if (!(ca > 0.0) || !Finite(ca) || !(epsilon > 0.0)) {
    return absl::FailedPreconditionError(
        "extension received no finite probability mass in constrained M-step");
  }

  for (size_t i = 0; i < extensions->size(); ++i) {
    const double count = std::max<double>(1e-30, expected[base_n + i]);
    (*extensions)[i].score = std::log(epsilon) + std::log(count / ca);
  }
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::RunContinuationEM() {
  if (extension_target_ == 0) return absl::OkStatus();
  if (extension_candidates_.empty()) {
    return absl::FailedPreconditionError("no Unigram extension candidates");
  }

  while (static_cast<int>(extension_candidates_.size()) > extension_target_) {
    std::vector<float> expected;
    double objective = 0.0;
    for (int sub = 0; sub < trainer_spec_.num_sub_iterations(); ++sub) {
      ABSL_RETURN_IF_ERROR(
          RunEStep(extension_candidates_, lambda_, &expected, &objective));
      ABSL_RETURN_IF_ERROR(
          RunConstrainedMStep(expected, &extension_candidates_, &lambda_));
    }

    const size_t base_n = static_cast<size_t>(prior_model_.pieces_size());

    // RANK BY DELETION LOSS, NOT BY EXPECTED COUNT.
    // Until 2026-09-10 this sorted on expected[base_n + i], i.e. it kept the
    // most FREQUENT extensions. That is not Unigram pruning and it selected a
    // materially wrong vocabulary: expanding a 1200-piece piano tokenizer with
    // 300 violin slots produced 241 single pitches, 28 time signatures and 20
    // durations, and ZERO note bigrams -- while the phrase pieces the corpus
    // overwhelmingly supports (Vn: is followed by a note 922,977 times) were
    // discarded. Frequency alone cannot express that a piece is worth keeping
    // BECAUSE its fallback is expensive; deletion loss can.
    std::vector<double> loss;
    std::vector<float> viterbi_freq;
    ABSL_RETURN_IF_ERROR(ComputeExtensionDeletionLoss(
        extension_candidates_, lambda_, &loss, &viterbi_freq));

    std::vector<size_t> order(extension_candidates_.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      const double la = loss[a];
      const double lb = loss[b];
      if (!ScoresTie(la, lb)) return la > lb;
      // Deterministic tie-break, as before.
      return extension_candidates_[a].piece < extension_candidates_[b].piece;
    });

    size_t keep = std::max<size_t>(
        extension_target_, static_cast<size_t>(
                               extension_candidates_.size() *
                               trainer_spec_.shrinking_factor()));
    keep = std::min(keep, extension_candidates_.size() - 1);
    keep = std::max<size_t>(keep, extension_target_);

    std::vector<ExtensionCandidate> next;
    next.reserve(keep);
    double kept_count = 0.0;
    for (size_t i = 0; i < keep; ++i) {
      next.push_back(extension_candidates_[order[i]]);
      kept_count += expected[base_n + order[i]];
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
  for (int sub = 0; sub < final_iters; ++sub) {
    ABSL_RETURN_IF_ERROR(
        RunEStep(extension_candidates_, lambda_, &expected, &objective));
    ABSL_RETURN_IF_ERROR(
        RunConstrainedMStep(expected, &extension_candidates_, &lambda_));
  }

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

absl::Status ContinuationTrainer::FinalizeArtifacts() {
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

  std::vector<ExpansionPiece> all_pieces;
  all_pieces.reserve(prior_model_.pieces_size() + extension_candidates_.size());
  for (int id = 0; id < prior_model_.pieces_size(); ++id) {
    const auto& prior = prior_model_.pieces(id);
    ExpansionPiece piece;
    piece.set_external_id(id);
    piece.set_piece(prior.piece());
    piece.set_type(prior.type());
    piece.set_score(prior.score());
    piece.set_mergeable(false);
    piece.set_atomic(string_util::UTF8Len(prior.piece()) == 1);
    *result.add_base_pieces() = piece;
    all_pieces.push_back(piece);
  }

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
  sentences_ = corpus_.sentences;
  ABSL_RETURN_IF_ERROR(VerifyCorpusCoverage());

  if (extension_target_ > 0) {
    ABSL_RETURN_IF_ERROR(MakeWeightedExtensionCandidates());
    ABSL_RETURN_IF_ERROR(InitializeContinuationScores());
    ABSL_RETURN_IF_ERROR(RunContinuationEM());
  }

  return FinalizeArtifacts();
}

}  // namespace sentencepiece::unigram
