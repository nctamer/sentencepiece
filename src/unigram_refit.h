// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// FIXED-VOCABULARY UNIGRAM SCORE REFIT.
//
// This is a CALIBRATION operation, and it is deliberately not any of the three
// things it resembles:
//
//   - it is not fresh Unigram training: no seed vocabulary, no candidate
//     generation, no pruning, no vocab_size;
//   - it is not continuation expansion: nothing is added, so there is no
//     inherited/extension split, no lambda gauge and no required coverage
//     extension;
//   - it is not a one-pass count: the E-step is forward-backward posterior
//     occupancy over the fixed lattice, never Viterbi token counts.
//
// Given an existing UNIGRAM ModelProto and a weighted corpus, the piece
// inventory, IDs, strings and types are held EXACTLY fixed and only the
// trainable NORMAL scores are re-estimated.
//
// Why it exists: a base vocabulary is chosen from a corpus where every
// distinct normalized record has weight 1, which is right for support
// selection and wrong as probability geometry. Continuation freezes whatever
// score geometry it inherits (up to one global gauge), so those artificial
// weights would become permanent. Refit replaces them with the real weighted
// corpus's geometry over the identical support, and the result is a drop-in
// prior for continuation.
//
// PROBABILITY FAMILY, read off the model rather than assumed. In
// unigram::Model::PopulateNodes the ONLY stored scores that reach the lattice
// are NORMAL ones:
//   - USER_DEFINED nodes are scored GetUserDefinedScore(length) = 0.1*(len-1),
//     a synthetic bonus; their stored score is never read;
//   - UNKNOWN scores the fallback node at min_score() - kUnkPenalty, derived
//     from the NORMAL scores, never from the stored <unk> score;
//   - CONTROL and BYTE live in reserved_id_map_ and are not in the trie at
//     all, so they never appear as lattice nodes;
//   - UNUSED is in the trie but explicitly skipped.
// So the simplex is over NORMAL pieces, exactly as in fresh training (where
// the trainer model holds only NORMAL candidates and meta pieces are appended
// afterwards). Refit therefore normalizes over NORMAL pieces and leaves every
// non-NORMAL stored score bit-identical.

#ifndef UNIGRAM_REFIT_H_
#define UNIGRAM_REFIT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_trainer.h"

namespace sentencepiece::unigram_refit {

struct RefitOptions {
  // Deterministic fixed-support EM. There is no adaptive shrinking and no
  // membership-based stopping criterion: support is fixed by definition.
  int32_t num_iterations = 8;
  int32_t num_threads = 1;
  // Numerical device only. A NORMAL piece with zero posterior occupancy keeps
  // its slot and gets a finite score; it is never dropped. The floor enters
  // BOTH the numerator and the denominator (see RunMStep).
  double count_floor = 1e-30;
  // Optional deterministic early stop on objective improvement. <= 0 disables.
  double objective_tolerance = 0.0;
};

// OBJECTIVE SEMANTICS. `objective` is the weighted mean negative
// log-likelihood of the corpus. It is non-increasing FROM ONE EM ITERATION TO
// THE NEXT. `initial_objective` is measured with the INPUT model's stored
// scores and is not part of that chain: an input model is not required to be
// normalized, and an exp-sum over its NORMAL pieces above 1 inflates every
// path likelihood, so the first M-step -- which projects onto the simplex --
// may legitimately raise it once. From a normalized start the guarantee holds
// from the first step.
struct RefitIterationStats {
  double objective = 0.0;       // weighted mean negative log-likelihood
  int64_t floored_normal = 0;   // NORMAL pieces whose raw count hit the floor
};

struct RefitStats {
  int64_t piece_count = 0;
  int64_t normal_piece_count = 0;
  int64_t special_piece_count = 0;
  int64_t distinct_record_count = 0;
  int64_t weighted_record_count = 0;
  int32_t em_iterations = 0;
  double initial_objective = 0.0;
  double final_objective = 0.0;
  int64_t changed_normal_scores = 0;
  std::vector<RefitIterationStats> per_iteration;
};

// The production guard, not merely a test helper. Called before the output
// model is written; a violation aborts the run rather than emitting a model.
absl::Status VerifyFixedSupportContract(const ModelProto& input,
                                        const ModelProto& output);

// `corpus_spec` supplies input files/format and max_sentence_length only. The
// INPUT MODEL is authoritative for normalization: its normalizer_spec and
// denormalizer_spec are used and copied through, and a caller normalizer is
// never substituted.
absl::Status RefitFixedVocabulary(const ModelProto& input,
                                  const TrainerSpec& corpus_spec,
                                  const TrainerComponents& components,
                                  const RefitOptions& options,
                                  ModelProto* output, RefitStats* stats);

// Convenience wrapper: read the model, refit, write <model_prefix>.model and
// <model_prefix>.vocab.
absl::Status RefitFixedVocabularyToFiles(absl::string_view input_model_path,
                                         const TrainerSpec& corpus_spec,
                                         const RefitOptions& options,
                                         absl::string_view model_prefix);

}  // namespace sentencepiece::unigram_refit

#endif  // UNIGRAM_REFIT_H_
