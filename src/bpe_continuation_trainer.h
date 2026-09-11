// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef BPE_CONTINUATION_TRAINER_H_
#define BPE_CONTINUATION_TRAINER_H_

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
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
class ContinuationTrainer : public TrainerInterface {
 public:
  ContinuationTrainer(const TrainerSpec& trainer_spec,
                      const NormalizerSpec& normalizer_spec,
                      const NormalizerSpec& denormalizer_spec)
      : TrainerInterface(trainer_spec, normalizer_spec, denormalizer_spec) {}

  absl::Status Train() override;

 private:
  struct Symbol {
    const Symbol* left = nullptr;
    const Symbol* right = nullptr;
    string_util::UnicodeText chars;
    uint64_t fp = 0;
    uint64_t freq = 0;
    bool is_unk = false;
    // A USER_DEFINED occurrence. Never a merge participant on either side,
    // which is exactly what native training achieves by replacing the
    // occurrence with a pretokenization boundary (trainer_interface.cc).
    bool frozen = false;
    bool active = true;
    bool pending = false;
    bool needs_recomputation = true;
    absl::btree_set<uint64_t> positions;

    [[nodiscard]] bool IsBigram() const {
      return left != nullptr && right != nullptr;
    }
    [[nodiscard]] std::string ToString() const;
  };

  struct Position {
    int sid;
    int left;
    int right;
  };

  struct QueueEntry {
    uint64_t freq;
    Symbol* symbol;
  };

  struct QueueEntryComparator {
    bool operator()(const QueueEntry& e1, const QueueEntry& e2) const {
      if (e1.freq != e2.freq) return e1.freq < e2.freq;
      if (e1.symbol->chars.size() != e2.symbol->chars.size()) {
        return e1.symbol->chars.size() > e2.symbol->chars.size();
      }
      return e1.symbol->chars > e2.symbol->chars;
    }
  };

  static uint64_t EncodePos(int sid, int l, int r);
  static Position DecodePos(uint64_t n);

  Symbol* GetAtomicSymbol(absl::string_view atom);
  Symbol* GetFrozenSymbol(absl::string_view piece);
  Symbol* GetPairSymbol(const Symbol* left, const Symbol* right);
  void ComputeFreq(Symbol* symbol) const;
  int GetNextIndex(int sid, int index) const;
  int GetPrevIndex(int sid, int index) const;
  void AddNewPair(int sid, int left, int right);
  void ResetFreq(int sid, int left, int right, const Symbol* best);
  absl::Status AcceptSymbol(Symbol* symbol);
  void DrainPendingQueue();

  // Segments `text` into the declared reversible atomic alphabet. Exactly one
  // segmentation is required: zero parses means the adapter omitted an atom,
  // and multiple parses mean the alphabet representation is ambiguous.
  absl::Status SegmentAtoms(absl::string_view text,
                            std::vector<std::string>* atoms) const;
  // Segments one record into corpus symbols: USER_DEFINED occurrences first
  // (longest prefix match, the native rule), each one frozen; every run of
  // text between them through SegmentAtoms.
  absl::Status SegmentRecord(absl::string_view text,
                             std::vector<Symbol*>* symbols);

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

  absl::flat_hash_set<std::string> existing_piece_strings_;
  absl::flat_hash_set<std::string> atomic_piece_strings_;
  // USER_DEFINED base piece strings and their longest-prefix matcher. The
  // matcher borrows the strings, so the set must outlive it.
  std::set<std::string> user_defined_piece_strings_;
  std::unique_ptr<normalizer::PrefixMatcher> user_defined_matcher_;
  std::vector<std::string> atomic_pieces_ordered_;
  absl::flat_hash_map<std::string, Symbol*> live_by_string_;

  int first_new_external_id_ = -1;
  int next_external_id_ = -1;
  int target_new_pieces_ = 0;

  absl::flat_hash_map<uint64_t, Symbol*> symbols_cache_;
  std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryComparator>
      pq_;
  std::vector<Symbol*> pending_queue_;
  std::vector<std::unique_ptr<Symbol>> allocated_;
  std::vector<std::vector<Symbol*>> symbols_;
};

}  // namespace sentencepiece::bpe

#endif  // BPE_CONTINUATION_TRAINER_H_
