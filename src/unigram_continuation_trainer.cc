// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "unigram_continuation_trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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

uint64_t SaturatingAdd(uint64_t a, uint64_t b) {
  return b > std::numeric_limits<uint64_t>::max() - a
             ? std::numeric_limits<uint64_t>::max()
             : a + b;
}

double CandidateScore(absl::string_view piece, uint64_t freq) {
  const double len = static_cast<double>(string_util::UTF8Len(piece));
  const double power =
      static_cast<double>(absl::GetFlag(FLAGS_seed_piece_length_power));
  return static_cast<double>(freq) * std::pow(len, power);
}

}  // namespace

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

  // The prior ModelProto is authoritative for normalization. This avoids the
  // CLI trap where default normalization flags accidentally disagree with a
  // custom pretrained tokenizer even though the user supplied the exact prior
  // model. Continuation does not silently invent a new normalization regime.
  normalizer_spec_ = prior_model_.normalizer_spec();
  denormalizer_spec_ = prior_model_.denormalizer_spec();
  trainer_spec_.set_treat_whitespace_as_suffix(
      prior_model_.trainer_spec().treat_whitespace_as_suffix());
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::VerifyCorpusCoverage() const {
  Model model(prior_model_);
  if (!model.status().ok()) return model.status();

  for (const auto& sentence : corpus_.sentences) {
    Lattice lattice;
    lattice.SetSentence(sentence.first);
    model.PopulateNodes(&lattice);
    const auto path = lattice.Viterbi();
    for (const auto* node : path.first) {
      if (node->id == prior_unk_id_) {
        return absl::InvalidArgumentError(absl::StrCat(
            "continuation corpus is not representable by the inherited model "
            "without <unk>; representative input: ", sentence.first));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::MakeWeightedExtensionCandidates() {
  extension_candidates_.clear();
  if (extension_target_ == 0) return absl::OkStatus();

  size_t candidate_limit = std::max<size_t>(
      10000, static_cast<size_t>(extension_target_) * 64);
  if (trainer_spec_.seed_sentencepiece_size() > 0) {
    candidate_limit = std::min<size_t>(
        candidate_limit,
        static_cast<size_t>(trainer_spec_.seed_sentencepiece_size()));
  }
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
        count = SaturatingAdd(count, static_cast<uint64_t>(sentence.second));
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

double ContinuationTrainer::SolveInitialLambda(
    const std::vector<ExtensionCandidate>& extensions) const {
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

  double hi = 0.0;
  for (int i = 0; i < 200 && log_mass(hi) < 0.0; ++i) hi += 1.0;
  double lo = hi - 1.0;
  for (int i = 0; i < 400 && log_mass(lo) > 0.0; ++i) lo -= 1.0;
  for (int i = 0; i < 120; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (log_mass(mid) > 0.0) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return 0.5 * (lo + hi);
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

  lambda_ = SolveInitialLambda(extension_candidates_);
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

double ContinuationTrainer::SolveMStepLambda(
    const std::vector<float>& expected, size_t extension_count) const {
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

  double boundary_lo = -100.0;
  double boundary_hi = 100.0;
  for (int i = 0; i < 20 && LogBaseMass(boundary_lo) > 0.0; ++i) {
    boundary_lo *= 2.0;
  }
  for (int i = 0; i < 20 && LogBaseMass(boundary_hi) < 0.0; ++i) {
    boundary_hi *= 2.0;
  }
  for (int i = 0; i < 120; ++i) {
    const double mid = 0.5 * (boundary_lo + boundary_hi);
    if (LogBaseMass(mid) > 0.0) {
      boundary_hi = mid;
    } else {
      boundary_lo = mid;
    }
  }
  const double boundary = 0.5 * (boundary_lo + boundary_hi);
  if (ca <= 1e-30 || l0 <= 1e-30) return boundary;

  auto derivative = [&](double lambda) {
    double zprime = 0.0;
    const double z = BaseMass(lambda, &zprime);
    if (!(z < 1.0) || !Finite(zprime)) {
      return -std::numeric_limits<double>::infinity();
    }
    return l0 - ca * zprime / std::max(1e-300, 1.0 - z);
  };

  double hi = boundary - 1e-10;
  double lo = hi - 1.0;
  double step = 1.0;
  for (int i = 0; i < 200 && derivative(lo) < 0.0; ++i) {
    step *= 2.0;
    lo = hi - step;
  }
  for (int i = 0; i < 120; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (derivative(mid) > 0.0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

absl::Status ContinuationTrainer::RunConstrainedMStep(
    const std::vector<float>& expected,
    std::vector<ExtensionCandidate>* extensions, double* lambda) const {
  if (expected.size() !=
      static_cast<size_t>(prior_model_.pieces_size()) + extensions->size()) {
    return absl::InternalError("continuation expected-count shape mismatch");
  }

  *lambda = SolveMStepLambda(expected, extensions->size());
  double derivative = 0.0;
  const double base_mass = BaseMass(*lambda, &derivative);
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
    std::vector<size_t> order(extension_candidates_.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      const float ea = expected[base_n + a];
      const float eb = expected[base_n + b];
      if (ea != eb) return ea > eb;
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
              if (a.score != b.score) return a.score > b.score;
              return a.piece < b.piece;
            });
  return absl::OkStatus();
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
