// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef BPE_CONTINUATION_TRAINER_H_
#define BPE_CONTINUATION_TRAINER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <utility>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "continuation_io.h"
#include "normalizer.h"
#include "sentencepiece_model.pb.h"
#include "trainer_interface.h"

namespace sentencepiece::bpe {

// BPE continuation is intentionally a separate trainer from ordinary BPE.
// Its starting state is an inherited merge program, not a fresh character
// corpus with strings protected from pruning.
std::string EncodeBpeBoundaryPolicy(
    const std::set<std::string>& normalized_fence_surfaces);

class ContinuationTrainer : public TrainerInterface {
 public:
  ContinuationTrainer(const TrainerSpec& trainer_spec,
                      const NormalizerSpec& normalizer_spec,
                      const NormalizerSpec& denormalizer_spec)
      : TrainerInterface(trainer_spec, normalizer_spec, denormalizer_spec) {}

  absl::Status Train() override;

 private:
  // Canonical token symbol. Learned symbols have unique construction: a later
  // rank never aliases/reuses an existing child surface.
  struct Symbol {
    const Symbol* left = nullptr;
    const Symbol* right = nullptr;
    string_util::UnicodeText chars;
    uint64_t fp = 0;
    bool is_unk = false;
    // A USER_DEFINED occurrence. Never a merge participant on either side.
    bool frozen = false;

    [[nodiscard]] bool IsBigram() const {
      return left != nullptr && right != nullptr;
    }
    [[nodiscard]] std::string ToString() const;
  };

  // Training candidate identity is the ordinary flat BPE pair (left,right).
  // When a hierarchy sidecar is present it contributes occurrence-local
  // SUPPORT to this one global candidate; it never creates scoped aliases.
  // positions contains every current flat occurrence of the pair.
  struct Candidate {
    Symbol* left = nullptr;
    Symbol* right = nullptr;
    Symbol* result = nullptr;
    std::string left_text;
    std::string right_text;
    uint64_t freq = 0;
    bool active = true;
    bool pending = false;
    bool needs_recomputation = true;
    absl::btree_set<uint64_t> positions;
  };

  struct Position {
    int sid;
    int left;
    int right;
  };

  // One grammar parent whose direct children partition [begin,end). `cuts`
  // contains begin, every direct-child boundary, and end. Crossing one of
  // its internal cuts is legal only when both current tokens start/end on cuts
  // of this same parent: each side is therefore a whole child or a consecutive
  // union of whole children.
  struct HierarchyGate {
    size_t begin = 0;
    size_t end = 0;
    int level = 0;
    bool enabled = true;
    std::vector<size_t> cuts;
  };

  struct HierarchyRecord {
    std::vector<HierarchyGate> gates;
    absl::flat_hash_map<size_t, int> gate_at_boundary;
  };

  struct QueueEntry {
    uint64_t freq;
    Candidate* candidate;
  };

  struct QueueEntryComparator {
    static bool ByteLess(const std::string& a, const std::string& b) {
      return std::lexicographical_compare(
          a.begin(), a.end(), b.begin(), b.end(),
          [](char x, char y) {
            return static_cast<unsigned char>(x) <
                   static_cast<unsigned char>(y);
          });
    }

    bool operator()(const QueueEntry& e1, const QueueEntry& e2) const {
      // Highest hierarchy-supported weighted count wins, then unsigned UTF-8
      // byte order of (left,right) is the total deterministic key. Do NOT
      // tie-break on concatenated child text: a+bc and ab+c both produce abc.
      if (e1.freq != e2.freq) return e1.freq < e2.freq;
      if (e1.candidate->left_text != e2.candidate->left_text) {
        return ByteLess(e2.candidate->left_text, e1.candidate->left_text);
      }
      return ByteLess(e2.candidate->right_text, e1.candidate->right_text);
    }
  };

  static uint64_t EncodePos(int sid, int l, int r);
  static Position DecodePos(uint64_t n);

  Symbol* GetAtomicSymbol(absl::string_view atom);
  Symbol* GetFrozenSymbol(absl::string_view piece);
  Symbol* GetPairSymbol(const Symbol* left, const Symbol* right);
  Candidate* GetCandidate(Symbol* left, Symbol* right);
  Candidate* FindCandidate(const Symbol* left, const Symbol* right) const;
  void ComputeFreq(Candidate* candidate) const;
  int GetNextIndex(int sid, int index) const;
  int GetPrevIndex(int sid, int index) const;
  void AddNewPair(int sid, int left, int right);
  void ResetFreq(int sid, int left, int right, const Candidate* best);
  // Apply the selected ordinary pair globally to the current flat state.
  // candidate->freq is hierarchy support, not the total application weight.
  absl::Status AcceptCandidate(Candidate* candidate);
  void DrainPendingQueue();
  // After inherited/base replay, rebuild the candidate index over the CURRENT
  // flat segmentation. The hierarchy affects support/counting only; inherited
  // and learned state transitions are ordinary context-free BPE.
  absl::Status RebuildHierarchySupportIndex();

  // Training-only hierarchy. Empty bpe_hierarchy_file means ordinary
  // continuation semantics. The sidecar is keyed by the already-normalized
  // corpus surface, so it remains aligned after PreparedCorpus aggregation.
  absl::Status LoadHierarchy();
  // True when merging these two CURRENT flat tokens produces a source span
  // compatible with the immutable laminar hierarchy. This is a selection
  // witness only; a selected pair is still applied globally.
  bool HierarchySupportsMerge(int sid, int left, int right) const;

  // Segments `text` into the declared reversible atomic alphabet. Exactly one
  // segmentation is required: zero parses means the adapter omitted an atom,
  // and multiple parses mean the alphabet representation is ambiguous.
  absl::Status SegmentAtoms(absl::string_view text,
                            std::vector<std::string>* atoms) const;
  // Segments one record into corpus symbols: USER_DEFINED occurrences first
  // (longest prefix match, the native rule), each one frozen; every run of
  // text between them through SegmentAtoms.
  absl::Status SegmentRecord(
      absl::string_view text, std::vector<Symbol*>* symbols,
      std::vector<int>* fence_groups,
      std::vector<std::pair<size_t, size_t>>* byte_spans);
  // --continuation_fence_strings for BPE: the same G3 semantics as Unigram
  // continuation (logical strings normalized with the effective normalizer,
  // occurrences unioned, no piece may overlap a fenced character).
  absl::Status LoadExplicitFences();

  absl::Status LoadAndValidateSpec();
  // The inherited text pipeline and special-token ABI are authoritative, the
  // same rule Unigram continuation has always applied to its prior model.
  absl::Status ReconcileContinuationContract();
  void FillEffectiveContract(ContinuationContract* out) const;
  // SHA-256 over the inherited ordered piece table ("<id>\t<piece>\t<type>").
  std::string BaseIdMapSha256() const;
  absl::Status ValidateMergeProgram(
      const std::vector<ExpansionMerge>& merges,
      bool require_all_declared_pieces) const;
  absl::Status InitializeCorpusSymbols();
  absl::Status ReplayMerges(const std::vector<ExpansionMerge>& merges,
                            absl::string_view label);
  absl::Status LearnExpansion();
  std::vector<ExpansionMerge> EffectiveMergeTable() const;
  // SHA-256 over canonical PreparedCorpus order and the exact final ordered
  // token strings. This proves trainer state, not merely token cardinality.
  std::string FinalSegmentationSha256() const;
  // Rank of each declared (left, right) pair. Built once per finalization:
  // a real inherited tokenizer carries ~10^5 merges, so rebuilding this per
  // candidate piece would make reachability verification quadratic in the
  // size of the base tokenizer.
  using PairRanks = absl::flat_hash_map<std::pair<std::string, std::string>,
                                        int>;
  static PairRanks BuildPairRanks(const std::vector<ExpansionMerge>& merges);
  bool IsReachable(absl::string_view piece, const PairRanks& pair_rank) const;
  absl::Status FinalizeArtifacts();

  // Native SentencePiece BPE inference does not read a merge table. It merges
  // whichever adjacent pair has the best-scoring CONCATENATION in the
  // vocabulary. That is a different rule from an explicit (left, right)
  // program, and the two agree only under the conditions checked here. Returns
  // OK when a native ModelProto can reproduce `merges` exactly; otherwise the
  // status explains which condition failed and names an offending piece.
  absl::Status VerifyNativeMergeEquivalence(
      const std::vector<ExpansionPiece>& pieces,
      const std::vector<ExpansionMerge>& merges) const;
  absl::Status BuildNativeModel(const std::vector<ExpansionMerge>& merges,
                                ModelProto* model) const;

  ExpansionSpec expansion_spec_;
  continuation::PreparedCorpus corpus_;
  std::vector<ExpansionPiece> base_pieces_;
  std::vector<ExpansionPiece> bootstrap_pieces_;
  std::vector<ExpansionPiece> learned_pieces_;
  std::vector<ExpansionMerge> base_merges_;
  std::vector<ExpansionMerge> bootstrap_merges_;
  std::vector<ExpansionMerge> learned_merges_;
  // Populated only when TrainerSpec.bpe_reference_trace_file is set.
  std::vector<std::string> reference_trace_lines_;

  absl::flat_hash_set<std::string> existing_piece_strings_;
  absl::flat_hash_set<std::string> atomic_piece_strings_;
  absl::flat_hash_map<std::string, int> piece_external_id_by_string_;
  // Pair -> child ID for already-emitted constructions. A learned pair is
  // never selected twice. Every newly learned child must also be a fresh piece
  // string, which prevents a later rank from manufacturing an operand of an
  // earlier rank (the monotonicity invariant required by flat ranked replay).
  std::map<std::pair<std::string, std::string>, int> pair_external_id_;
  // USER_DEFINED base piece strings and their longest-prefix matcher. The
  // matcher borrows the strings, so the set must outlive it.
  std::set<std::string> user_defined_piece_strings_;
  std::unique_ptr<normalizer::PrefixMatcher> user_defined_matcher_;
  std::set<std::string> fence_surfaces_;
  std::unique_ptr<normalizer::PrefixMatcher> fence_matcher_;
  // Per position: 0 = open text, k > 0 = inside fence occurrence k. A pair
  // whose two positions carry different groups is never formed, so no merge
  // starts inside, ends inside, contains or spans a fence occurrence.
  std::vector<std::vector<int>> fence_group_;

  // Optional completion-gated grammar state, one entry per PreparedCorpus row.
  // span_begin_/span_end_ are indexed like symbols_; a merge keeps the left
  // slot and extends its end to the consumed right token's end.
  std::vector<HierarchyRecord> hierarchy_;
  std::vector<std::vector<size_t>> span_begin_;
  std::vector<std::vector<size_t>> span_end_;
  std::string hierarchy_sha256_;
  // False while replaying the inherited/base program; true only while
  // selecting appended continuation merges. The hierarchy changes candidate
  // SUPPORT only and never vetoes application of an inherited or learned rule.
  bool hierarchy_support_enabled_ = false;

  std::vector<std::string> atomic_pieces_ordered_;
  absl::flat_hash_map<std::string, Symbol*> live_by_string_;

  int first_new_external_id_ = -1;
  int next_external_id_ = -1;
  int target_new_pieces_ = 0;

  absl::flat_hash_map<uint64_t, Symbol*> symbols_cache_;
  std::map<std::pair<std::string, std::string>, Candidate*>
      candidate_cache_;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryComparator>
      pq_;
  std::vector<Candidate*> pending_queue_;
  std::vector<std::unique_ptr<Candidate>> allocated_candidates_;
  std::vector<std::unique_ptr<Symbol>> allocated_;
  std::vector<std::vector<Symbol*>> symbols_;
  // Intrusive live-neighbor links over the fixed occurrence slots. Merges
  // tombstone the right slot but update only these two links, so neighbour
  // discovery is O(1) instead of scanning across an ever-growing run of nulls.
  std::vector<std::vector<int>> prev_live_;
  std::vector<std::vector<int>> next_live_;
};

}  // namespace sentencepiece::bpe

#endif  // BPE_CONTINUATION_TRAINER_H_
