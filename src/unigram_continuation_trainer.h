// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef UNIGRAM_CONTINUATION_TRAINER_H_
#define UNIGRAM_CONTINUATION_TRAINER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include "absl/container/flat_hash_set.h"
#include <vector>

#include "absl/status/status.h"
#include "continuation_io.h"
#include "sentencepiece_model.pb.h"
#include "unigram_model.h"
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

// CANONICAL BOUNDARY POLICY for Unigram continuation, serialized into
// ExpansionResult.boundary_policy.
//
// --continuation_fence_strings is NOT an execution knob like
// --continuation_spill_dir or --continuation_spill_entries: it changes the
// candidate universe and therefore the tokenizer that can be learned. Without
// this, a fenced and an unfenced run could ship the same serialized
// TrainerSpec while having trained under different boundary policies, which is
// not acceptable provenance for a modelling parameter.
//
// What is recorded is the EFFECTIVE NORMALIZED UNIQUE surface set -- what
// candidate generation actually matched -- not the caller's logical spelling.
// The normalizer that produced those surfaces is identified by
// prior_model_sha256, so the pair is self-describing.
//
// Encoding, versioned and unambiguous for arbitrary bytes:
//
//   unigram_explicit_fences_v1:[]              no explicit curriculum fence
//   unigram_explicit_fences_v1:[<e1>,<e2>]     e_i sorted, percent-escaped
//
// Every byte outside [A-Za-z0-9._-] is escaped as %XX, so a surface containing
// a comma or a bracket cannot be confused with the separator. Sorting is by
// surface bytes, so the policy is independent of the caller's list order and
// of duplicates -- exactly like the matcher it describes.
std::string EncodeUnigramBoundaryPolicy(
    const std::set<std::string>& normalized_fence_surfaces);

// True continuation of a native Unigram ModelProto. Inherited NORMAL scores
// are constrained to s_i + lambda * length(i); only extension probabilities
// and their shared probability mass are learned from the continuation corpus.
//
// WHAT THE SCALAR GAUGE ACTUALLY PROVES, stated narrowly on purpose.
//
// For two segmentations of the SAME normalized surface that use inherited
// NORMAL pieces only, the total length is the same, so the lambda*length terms
// cancel and their score DIFFERENCE is unchanged. That is the theorem:
//
//     inherited-submodel segmentation geometry is preserved.
//
// It is NOT the claim that the expanded tokenizer emits the old token
// sequence. Once extensions are appended, an extension is deliberately allowed
// to beat the inherited segmentation -- that is the entire point of expanding
// the vocabulary -- so the surface may retokenize. Any wording along the lines
// of "old tokenization is preserved" is too strong and should not be used.
//
// The theorem is also about the inherited NORMAL probabilistic submodel only.
// USER_DEFINED nodes are scored by the lattice's maximal-matching bonus
// (GetUserDefinedScore) rather than by their stored score, and the UNKNOWN
// fallback is scored from min_score() - kUnkPenalty, so neither participates in
// the cancellation argument. Their stored scores are held bit-identical, which
// is a separate guarantee.
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
    uint64_t freq = 0;
    // h_i: additive length contributed by INHERITED pieces only in this
    // candidate's covered-basis decomposition. Required coverage-extension atoms carry
    // a FREE score, not a gauge-shifted one, so they contribute no lambda
    // exposure. For a candidate containing m required atoms, h = L - m; for a
    // candidate with none, h = L and the old equation is recovered exactly.
    int basis_inherited_len = 0;
    bool is_required = false;
  };

  absl::Status LoadAndValidatePrior();
  // Decides between inheriting the prior's normalization, accepting an
  // identical caller request, and refusing a conflicting one.
  absl::Status ReconcileNormalization();
  // Collects corpus characters the prior cannot spell; they become
  // extension candidates. Non-const: it fills required_coverage_extensions_ (the required coverage extensions).
  absl::Status VerifyCorpusCoverage();
  absl::Status MakeWeightedExtensionCandidates();
  absl::Status InitializeContinuationScores();
  // Iteration-zero diagnostic: Viterbi frequency vs posterior mass.
  // Staged, non-interpreting candidate tracker (see .cc).
  absl::Status ReportTracked(absl::string_view stage,
                             const std::vector<float>* expected_in,
                             const std::vector<double>* loss_in) const;
  absl::Status RunContinuationEM();
  // PERSISTENT WORKING MODEL.
  //
  // Support (piece strings/ids/types/count) changes only when pruning changes
  // the extension set. Scores change on every M-step. Rebuilding a million
  // protobuf pieces, the piece maps and a Darts trie for a score change is
  // pure waste, and it is what the continuation E-step used to do on EVERY
  // sub-iteration.
  //
  // DECLARATION ORDER IS LOAD-BEARING: `model` holds a pointer to `proto`, and
  // members are destroyed in reverse declaration order, so declaring `proto`
  // first guarantees the Model dies before its backing proto. Never mutate
  // proto.pieces() structurally while `model` is alive.
  struct WorkingModelState {
    ModelProto proto;
    std::unique_ptr<Model> model;
    uint64_t support_generation = 0;
    uint64_t support_builds = 0;
    uint64_t score_refreshes = 0;
  };

  // Full reconstruction. Call ONLY for a support change.
  absl::Status RebuildWorkingModel(
      const std::vector<ExtensionCandidate>& extensions, double lambda,
      WorkingModelState* state) const;
  // Score-only, in place. Must not alter count/strings/ids/types.
  absl::Status UpdateWorkingScores(
      const std::vector<ExtensionCandidate>& extensions, double lambda,
      WorkingModelState* state) const;

  absl::Status RunEStep(const Model& model, size_t support_size,
                        std::vector<float>* expected, double* objective) const;
  absl::Status RunConstrainedMStep(const std::vector<float>& expected,
                                   std::vector<ExtensionCandidate>* extensions,
                                   double* lambda) const;
  // EXPERIMENTAL, CURRENTLY UNUSED. An attempt at a per-extension deletion
  // loss. Its claim to mirror Trainer::PruneSentencePieces has been WITHDRAWN:
  // that estimator re-estimates surviving probabilities freely, which a frozen
  // gauge-constrained prior forbids. See the definition in the .cc.
  //
  // This is NOT the expected count. Ranking extensions by expected count -- as
  // this trainer did until 2026-09-10 -- keeps whatever is most FREQUENT,
  // which systematically prefers pieces that merely fill a representational
  // gap over pieces that compress a frequent phrase, because a gap-filler's
  // count is undiluted while a phrase's count is split with the parent pieces
  // that can already spell it. Deletion loss instead weights a piece's
  // frequency by how expensive its fallback is, so a moderately frequent
  // piece with a costly alternative can and should outrank a very frequent
  // piece whose alternative is nearly as good.
  //
  // `loss[i]` is +inf for a candidate with no alternative segmentation (it is
  // the only way to spell itself, so it must be kept) and -inf for one whose
  // own Viterbi path is already split (it is unreachable and free to drop).
  // Removed 2026-09-10: ComputeExtensionDeletionLoss() and
  // SolveInitialLambda() were the experimental deletion-loss pruner and its
  // pre-covered-basis initializer. Neither had a production or test call site
  // under the covered-basis initializer, and the deletion-loss statistics they
  // fed were being read out of never-filled vectors in RunContinuationEM().
  // Optional extension candidates are ranked by forward-backward posterior
  // expected occupancy; deletion loss is not active.

  // Covered-basis normalization:
  //   Z_B(lambda) + sum_{k in R} e^{r_k} + sum_{i in O} e^{beta_i + lambda*h_i} = 1
  // The SAME function is used by the root solver and by score assignment, so
  // the initialized working model normalizes by construction rather than by
  // coincidence.
  absl::Status SolveCoveredBasisLambda(
      const std::vector<ExtensionCandidate>& extensions, double required_mass,
      double* lambda) const;
  absl::Status SolveMStepLambda(const std::vector<float>& expected,
                                size_t extension_count, double* lambda) const;
  double LogBaseMass(double lambda) const;
  double BaseMass(double lambda, double* derivative) const;
  ModelProto BuildWorkingModel(
      const std::vector<ExtensionCandidate>& extensions,
      double lambda) const;
  // Applies the free function above to this run's prior and gauge.
  absl::Status VerifyPriorPrefix(const ModelProto& output) const;
  // Refuses to emit a model that cannot spell its own training corpus.
  absl::Status VerifyFinalCoverage() const;
  absl::Status FinalizeArtifacts();

  // Characters present in the corpus but absent from the prior. Admitted
  // as EXTENSION candidates, never as inherited state.
  std::vector<std::string> required_coverage_extensions_;
  // Extensions that must appear in the final model because the corpus is
  // otherwise unrepresentable. They consume extension budget but never enter
  // the deletion-loss ranking.
  absl::flat_hash_set<std::string> required_extensions_;
  // Effective normalized fence surfaces, kept for provenance.
  std::set<std::string> explicit_fence_surfaces_;
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
