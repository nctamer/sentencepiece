// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "unigram_refit.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "continuation_io.h"
#include "filesystem.h"
#include "unigram_model.h"
#include "util.h"

namespace sentencepiece::unigram_refit {
namespace {

using ::sentencepiece::continuation::PreparedCorpus;

bool IsNormal(const ModelProto::SentencePiece& sp) {
  return sp.type() == ModelProto::SentencePiece::NORMAL;
}

// REPRESENTABILITY IS A PATH PROPERTY, NOT A PER-CHARACTER ONE.
//
// A first version of this check asked whether every corpus character exists as
// a single-character piece, mirroring the `has_single_node` test inside
// PopulateNodes. That is wrong, and its own test caught it: a USER_DEFINED
// piece like "<X>" covers a span whose individual characters are not pieces at
// all, and PopulateNodes inserts an UNK node beside it regardless. The
// question that actually matters is whether SOME segmentation of the record
// avoids UNK entirely. So this is a reachability DP over the lattice with UNK
// nodes removed.
absl::Status VerifyCorpusCoverage(const unigram::Model& model,
                                  const ModelProto& proto,
                                  const PreparedCorpus& corpus) {
  // REFIT MAY NOT REPAIR COVERAGE. Adding a missing character is continuation's
  // required-coverage-extension behaviour and it mutates support, which is the
  // one thing this operation promises never to do. So this fails.
  if (proto.trainer_spec().byte_fallback()) return absl::OkStatus();

  int unk_index = -1;
  for (int i = 0; i < proto.pieces_size(); ++i) {
    if (proto.pieces(i).type() == ModelProto::SentencePiece::UNKNOWN) {
      unk_index = i;
      break;
    }
  }

  std::vector<std::string> examples;
  int64_t unrepresentable = 0;
  std::set<std::string> uncovered_chars;
  unigram::Lattice lattice;
  for (const auto& sentence : corpus.sentences) {
    const std::string& w = sentence.first;
    lattice.SetSentence(w);
    model.PopulateNodes(&lattice);
    const int len = lattice.size();
    std::vector<bool> reachable(len + 1, false);
    reachable[0] = true;
    for (int pos = 0; pos < len; ++pos) {
      if (!reachable[pos]) continue;
      for (const auto* node : lattice.begin_nodes(pos)) {
        if (node->id == unk_index) continue;
        reachable[pos + node->length] = true;
      }
    }
    if (reachable[len]) continue;
    ++unrepresentable;
    if (examples.size() < 5) examples.push_back(w);
    // Diagnostic only: which characters have no covering piece at all. This is
    // usually the actual cause, but it is not the criterion.
    for (size_t i = 0; i < w.size();) {
      const size_t clen = std::min<size_t>(
          string_util::OneCharLen(w.data() + i), w.size() - i);
      const std::string ch = w.substr(i, clen);
      bool covered = false;
      for (const auto& sp : proto.pieces()) {
        if (sp.piece().find(ch) != std::string::npos &&
            sp.type() != ModelProto::SentencePiece::UNKNOWN) {
          covered = true;
          break;
        }
      }
      if (!covered) uncovered_chars.insert(ch);
      i += clen;
    }
  }
  if (unrepresentable == 0) return absl::OkStatus();
  return absl::FailedPreconditionError(absl::StrCat(
      "fixed-vocabulary refit cannot represent ", unrepresentable,
      " corpus record(s) without <unk> under the model's fallback policy "
      "(byte_fallback=false), and refit may not add pieces. Characters with "
      "no covering piece: ", absl::StrJoin(uncovered_chars, " "),
      ". Example record(s): ", absl::StrJoin(examples, " | "),
      ". Adding coverage is continuation's job, not refit's."));
}

}  // namespace

absl::Status VerifyFixedSupportContract(const ModelProto& input,
                                        const ModelProto& output) {
  if (input.pieces_size() != output.pieces_size()) {
    return absl::InternalError(absl::StrCat(
        "refit changed the piece count: ", input.pieces_size(), " -> ",
        output.pieces_size()));
  }
  for (int i = 0; i < input.pieces_size(); ++i) {
    const auto& a = input.pieces(i);
    const auto& b = output.pieces(i);
    if (a.piece() != b.piece()) {
      return absl::InternalError(absl::StrCat(
          "refit changed the piece at id ", i, ": ", a.piece(), " -> ",
          b.piece()));
    }
    if (a.type() != b.type()) {
      return absl::InternalError(
          absl::StrCat("refit changed the type of id ", i, " (", a.piece(),
                       "): ", a.type(), " -> ", b.type()));
    }
    if (!IsNormal(a)) {
      // Bit-identical, compared as raw bits so a NaN payload cannot slip
      // through an == that is false for NaN against itself.
      static_assert(sizeof(float) == sizeof(uint32_t), "float is not 32-bit");
      uint32_t abits, bbits;
      const float af = a.score(), bf = b.score();
      std::memcpy(&abits, &af, sizeof(abits));
      std::memcpy(&bbits, &bf, sizeof(bbits));
      if (abits != bbits) {
        return absl::InternalError(absl::StrCat(
            "refit changed a non-NORMAL score at id ", i, " (", a.piece(),
            "): ", a.score(), " -> ", b.score()));
      }
    } else if (std::isnan(b.score()) || std::isinf(b.score())) {
      return absl::InternalError(absl::StrCat(
          "refit produced a non-finite NORMAL score at id ", i, " (",
          a.piece(), "): ", b.score()));
    }
  }
  if (input.normalizer_spec().SerializeAsString() !=
      output.normalizer_spec().SerializeAsString()) {
    return absl::InternalError("refit changed the normalizer spec");
  }
  if (input.denormalizer_spec().SerializeAsString() !=
      output.denormalizer_spec().SerializeAsString()) {
    return absl::InternalError("refit changed the denormalizer spec");
  }
  return absl::OkStatus();
}

absl::Status RefitFixedVocabulary(const ModelProto& input,
                                  const TrainerSpec& corpus_spec,
                                  const TrainerComponents& components,
                                  const RefitOptions& options,
                                  ModelProto* output, RefitStats* stats) {
  if (output == nullptr || stats == nullptr) {
    return absl::InvalidArgumentError("output and stats must not be null");
  }
  *stats = RefitStats();

  if (input.trainer_spec().model_type() != TrainerSpec::UNIGRAM) {
    return absl::InvalidArgumentError(
        "fixed-vocabulary refit requires a UNIGRAM input model");
  }
  if (input.pieces_size() == 0) {
    return absl::InvalidArgumentError("input model has no pieces");
  }
  if (options.num_iterations <= 0) {
    return absl::InvalidArgumentError("num_iterations must be >= 1");
  }
  if (!(options.count_floor > 0.0)) {
    return absl::InvalidArgumentError("count_floor must be > 0");
  }

  int64_t normal_count = 0;
  for (const auto& sp : input.pieces()) {
    if (IsNormal(sp)) ++normal_count;
  }
  if (normal_count == 0) {
    return absl::InvalidArgumentError(
        "input model has no NORMAL pieces; there is nothing to refit");
  }

  // THE INPUT MODEL IS AUTHORITATIVE FOR NORMALIZATION. A caller normalizer is
  // never substituted: the corpus must be normalized exactly the way this
  // model's tokenizer will normalize text at inference time.
  TrainerSpec load_spec = corpus_spec;
  load_spec.set_model_type(TrainerSpec::UNIGRAM);
  PreparedCorpus corpus;
  ABSL_RETURN_IF_ERROR(continuation::LoadPreparedCorpus(
      load_spec, input.normalizer_spec(), components, &corpus));

  // The working proto is the output. Support is copied once and never touched
  // again; only NORMAL score fields are assigned.
  *output = input;

  auto model = std::make_unique<unigram::Model>(*output);
  ABSL_RETURN_IF_ERROR(model->status());

  // Checked against the real lattice, so it must come after the model exists.
  // On failure `output` is cleared: no artifact may survive a refused refit.
  {
    const absl::Status coverage = VerifyCorpusCoverage(*model, *output, corpus);
    if (!coverage.ok()) {
      output->Clear();
      return coverage;
    }
  }

  const size_t n = static_cast<size_t>(output->pieces_size());
  int64_t all_freq = 0;
  for (const auto& s : corpus.sentences) all_freq += s.second;
  if (all_freq <= 0) return absl::InvalidArgumentError("corpus has no weight");

  const int32_t num_threads = std::max(1, options.num_threads);

  // E-STEP: forward-backward POSTERIOR occupancy over the fixed lattice.
  // Never Viterbi counts. `expected` is indexed by proto id because lattice
  // node ids are proto ids.
  auto run_estep = [&](std::vector<double>* expected) -> double {
    std::vector<std::vector<double>> parts(num_threads);
    std::vector<double> objs(num_threads, 0.0);
    {
      auto pool = std::make_unique<ThreadPool>(num_threads);
      for (int t = 0; t < num_threads; ++t) {
        pool->Schedule([&, t]() {
          parts[t].assign(n, 0.0);
          std::vector<float> local(n, 0.0f);
          unigram::Lattice lattice;
          for (size_t i = t; i < corpus.sentences.size(); i += num_threads) {
            const std::string& w = corpus.sentences[i].first;
            const int64_t freq = corpus.sentences[i].second;
            lattice.SetSentence(w);
            model->PopulateNodes(&lattice);
            std::fill(local.begin(), local.end(), 0.0f);
            const float Z = lattice.PopulateMarginal(
                static_cast<float>(freq), &local);
            // Accumulate in double per record: a float accumulator over a
            // million records loses the small counts that keep rare pieces
            // alive, and rare pieces are exactly what must not vanish here.
            for (size_t k = 0; k < n; ++k) parts[t][k] += local[k];
            objs[t] -= static_cast<double>(Z) / all_freq;
          }
        });
      }
    }
    expected->assign(n, 0.0);
    double obj = 0.0;
    for (int t = 0; t < num_threads; ++t) {
      obj += objs[t];
      for (size_t k = 0; k < n; ++k) (*expected)[k] += parts[t][k];
    }
    return obj;
  };

  // M-STEP, support preserving. For every trainable NORMAL piece i:
  //
  //     d~_i  = max(d_i, floor)
  //     denom = sum over NORMAL j of d~_j
  //     score_i = log(d~_i / denom)
  //
  // The floor is in BOTH numerator and denominator, so the NORMAL scores stay
  // a proper simplex. This is NOT Trainer::RunMStep: that one drops every
  // piece with expected count < 0.5 (changing support) and uses the
  // Bayesianified Digamma update (a sparsity prior). Both are wrong here --
  // support is fixed and no sparsity is wanted.
  auto run_mstep = [&](const std::vector<double>& expected) -> int64_t {
    double denom = 0.0;
    int64_t floored = 0;
    for (size_t i = 0; i < n; ++i) {
      if (!IsNormal(output->pieces(static_cast<int>(i)))) continue;
      const double d = expected[i];
      if (!(d > options.count_floor)) ++floored;
      denom += std::max(d, options.count_floor);
    }
    for (size_t i = 0; i < n; ++i) {
      auto* sp = output->mutable_pieces(static_cast<int>(i));
      if (!IsNormal(*sp)) continue;
      const double d = std::max(expected[i], options.count_floor);
      sp->set_score(static_cast<float>(std::log(d) - std::log(denom)));
    }
    return floored;
  };

  std::vector<double> expected;
  stats->initial_objective = run_estep(&expected);
  LOG(INFO) << "REFIT iteration=0 objective=" << stats->initial_objective
            << " (pre-refit, input scores)";

  double objective = stats->initial_objective;
  int32_t performed = 0;
  for (int32_t it = 0; it < options.num_iterations; ++it) {
    const int64_t floored = run_mstep(expected);
    // The only score-dependent cache in unigram::Model. Support did not
    // change, so the trie stays valid and is not rebuilt.
    ABSL_RETURN_IF_ERROR(model->RefreshScoreCache());

    const double prev = objective;
    objective = run_estep(&expected);
    ++performed;
    stats->per_iteration.push_back({objective, floored});
    LOG(INFO) << "REFIT iteration=" << performed << " objective=" << objective
              << " delta=" << (objective - prev)
              << " floored_normal=" << floored;
    if (options.objective_tolerance > 0.0 &&
        prev - objective < options.objective_tolerance) {
      LOG(INFO) << "REFIT early stop: improvement " << (prev - objective)
                << " < tolerance " << options.objective_tolerance;
      break;
    }
  }

  stats->final_objective = objective;
  stats->em_iterations = performed;
  stats->piece_count = output->pieces_size();
  stats->normal_piece_count = normal_count;
  stats->special_piece_count = output->pieces_size() - normal_count;
  stats->distinct_record_count = corpus.sentences.size();
  stats->weighted_record_count = corpus.weighted_sentence_count;
  for (int i = 0; i < input.pieces_size(); ++i) {
    if (IsNormal(input.pieces(i)) &&
        input.pieces(i).score() != output->pieces(i).score()) {
      ++stats->changed_normal_scores;
    }
  }

  // PRODUCTION GUARD, not a test. Nothing is written if this fails.
  ABSL_RETURN_IF_ERROR(VerifyFixedSupportContract(input, *output));

  LOG(INFO) << "REFIT piece_count=" << stats->piece_count
            << " NORMAL=" << stats->normal_piece_count
            << " special=" << stats->special_piece_count
            << " distinct_records=" << stats->distinct_record_count
            << " weighted_records=" << stats->weighted_record_count
            << " em_iterations=" << stats->em_iterations
            << " initial_objective=" << stats->initial_objective
            << " final_objective=" << stats->final_objective
            << " changed_NORMAL_scores=" << stats->changed_normal_scores;
  return absl::OkStatus();
}

absl::Status RefitFixedVocabularyToFiles(absl::string_view input_model_path,
                                         const TrainerSpec& corpus_spec,
                                         const RefitOptions& options,
                                         absl::string_view model_prefix) {
  ModelProto input;
  std::string raw;
  ABSL_RETURN_IF_ERROR(
      continuation::ReadModelProto(input_model_path, &input, &raw));
  LOG(INFO) << "REFIT input_model=" << input_model_path
            << " sha256=" << continuation::Sha256Hex(raw);

  ModelProto output;
  RefitStats stats;
  TrainerComponents components;
  ABSL_RETURN_IF_ERROR(RefitFixedVocabulary(input, corpus_spec, components,
                                            options, &output, &stats));

  const std::string model_path = absl::StrCat(model_prefix, ".model");
  const std::string vocab_path = absl::StrCat(model_prefix, ".vocab");
  ABSL_RETURN_IF_ERROR(continuation::WriteModelProto(model_path, output));

  auto vocab = filesystem::NewWritableFile(vocab_path);
  ABSL_RETURN_IF_ERROR(vocab->status());
  for (const auto& sp : output.pieces()) {
    std::ostringstream os;
    os << sp.piece() << "\t" << sp.score();
    if (!vocab->WriteLine(os.str())) {
      return absl::InternalError(absl::StrCat("cannot write ", vocab_path));
    }
  }
  LOG(INFO) << "REFIT wrote " << model_path << " and " << vocab_path;
  return absl::OkStatus();
}

}  // namespace sentencepiece::unigram_refit
