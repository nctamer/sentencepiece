// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef UNIGRAM_CONTINUATION_TRAINER_H_
#define UNIGRAM_CONTINUATION_TRAINER_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "continuation_io.h"
#include "sentencepiece_model.pb.h"
#include "trainer_interface.h"

namespace sentencepiece::unigram {

// Both of these are the continuation contract's two hardest guards, and both
// are stated over values rather than over trainer state so a test can reach
// them. A guard nothing can trigger is a guard nobody has checked.

// Deterministic bracketed bisection for a monotone objective. Fails - rather
// than returning a bracket endpoint - when the root is not bracketed, when the
// objective leaves the finite range, or when the interval never closes.
absl::Status BisectMonotoneRoot(absl::string_view what,
                                const std::function<double(double)>& f,
                                bool increasing, double seed_lo, double seed_hi,
                                double* root);

// The prior-prefix invariant: every inherited ID keeps its index, its bytes
// and its type; inherited non-NORMAL scores are bit-identical; and every
// inherited NORMAL score moved by exactly `lambda * length` within one float32
// rounding. A property of the two protos and the gauge, nothing else.
absl::Status VerifyPriorPrefixInvariant(const ModelProto& prior,
                                        const ModelProto& output,
                                        double lambda);

// True continuation of a native Unigram ModelProto. Inherited NORMAL scores
// are constrained to s_i + lambda * length(i); only extension probabilities
// and their shared probability mass are learned from the continuation corpus.
class ContinuationTrainer : public TrainerInterface {
 public:
  ContinuationTrainer(const TrainerSpec& trainer_spec,
                      const NormalizerSpec& normalizer_spec,
                      const NormalizerSpec& denormalizer_spec)
      : TrainerInterface(trainer_spec, normalizer_spec, denormalizer_spec) {}

  absl::Status Train() override;

 private:
  struct InheritedNormal {
    int external_id = -1;
    std::string piece;
    double prior_score = 0.0;
    int additive_length = 0;
  };

  struct ExtensionCandidate {
    std::string piece;
    double inherited_best_score = 0.0;
    double score = 0.0;
  };

  absl::Status LoadAndValidatePrior();
  // Decides between inheriting the prior's normalization, accepting an
  // identical caller request, and refusing a conflicting one.
  absl::Status ReconcileNormalization();
  absl::Status VerifyCorpusCoverage() const;
  absl::Status MakeWeightedExtensionCandidates();
  absl::Status InitializeContinuationScores();
  absl::Status RunContinuationEM();
  absl::Status RunEStep(const std::vector<ExtensionCandidate>& extensions,
                        double lambda, std::vector<float>* expected,
                        double* objective) const;
  absl::Status RunConstrainedMStep(const std::vector<float>& expected,
                                   std::vector<ExtensionCandidate>* extensions,
                                   double* lambda) const;
  // Both lambdas are roots of monotone objectives. They report failure
  // instead of returning a bracket endpoint that merely ran out of iterations.
  absl::Status SolveInitialLambda(
      const std::vector<ExtensionCandidate>& extensions, double* lambda) const;
  absl::Status SolveMStepLambda(const std::vector<float>& expected,
                                size_t extension_count, double* lambda) const;
  double LogBaseMass(double lambda) const;
  double BaseMass(double lambda, double* derivative) const;
  ModelProto BuildWorkingModel(
      const std::vector<ExtensionCandidate>& extensions,
      double lambda) const;
  // Applies the free function above to this run's prior and gauge.
  absl::Status VerifyPriorPrefix(const ModelProto& output) const;
  absl::Status FinalizeArtifacts();

  ModelProto prior_model_;
  std::string prior_model_bytes_;
  continuation::PreparedCorpus corpus_;
  std::vector<InheritedNormal> inherited_normal_;
  std::vector<ExtensionCandidate> extension_candidates_;
  int prior_unk_id_ = -1;
  int extension_target_ = 0;
  double lambda_ = 0.0;
};

}  // namespace sentencepiece::unigram

#endif  // UNIGRAM_CONTINUATION_TRAINER_H_
