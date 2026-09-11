// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// AUTHORITATIVE EXPLICIT-PROGRAM BPE INFERENCE.
//
// Continuation learns an explicit ranked (left,right) merge program. Native
// SentencePiece BPE does not execute such a table -- it merges whichever
// adjacent pair forms the best-scoring vocabulary piece -- so when a piece
// admits several vocabulary decompositions the two disagree and
// BuildNativeModel() deliberately withholds a .model.
//
// That left the merge program as the authoritative tokenizer with no runtime.
// This class is that runtime, and it lives here rather than in each consumer
// so there is exactly ONE implementation of the merge semantics the
// continuation tests already pin.
//
// Semantics, in order:
//   1. normalize with the artifact's own authoritative normalizer;
//   2. USER_DEFINED pieces are recognized FIRST by longest prefix match,
//      exactly as native SentencePiece does (bpe_model.cc, PrefixMatcher):
//      each occurrence is one frozen unit that no merge may enter or cross;
//   3. everything else starts from single characters;
//   4. among adjacent non-frozen pairs present in the program, LOWEST
//      effective rank wins, leftmost occurrence first;
//   5. external IDs are preserved exactly.

#ifndef EXPANSION_PROCESSOR_H_
#define EXPANSION_PROCESSOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "normalizer.h"
#include "sentencepiece_model.pb.h"

namespace sentencepiece::expansion {

struct TokenSpan {
  int id = 0;
  std::string piece;
  int begin = 0;  // UTF-8 byte offset into the NORMALIZED text, half-open
  int end = 0;
};

struct CompletionGate {
  int level = 0;  // audit/debug label only
  std::vector<int> cuts;  // normalized UTF-8 child boundaries incl. begin/end
};

class ExpansionProcessor {
 public:
  ExpansionProcessor() = default;

  // Load from a serialized ExpansionResult (the authoritative artifact).
  absl::Status Load(const ExpansionResult& result);
  absl::Status LoadFromFile(absl::string_view path);

  const absl::Status& status() const { return status_; }

  // Flat encode fails closed for hierarchical artifacts.
  absl::Status Encode(absl::string_view text, std::vector<TokenSpan>* out) const;
  absl::Status EncodeIds(absl::string_view text, std::vector<int>* ids) const;
  // Base/bootstrap merges are unconditional; learned continuation merges are
  // occurrence-locally gated by the supplied grammar completion hierarchy.
  absl::Status EncodeWithHierarchy(absl::string_view text,
                                   const std::vector<CompletionGate>& gates,
                                   std::vector<TokenSpan>* out) const;
  absl::Status EncodeIdsWithHierarchy(
      absl::string_view text, const std::vector<CompletionGate>& gates,
      std::vector<int>* ids) const;
  bool RequiresHierarchy() const { return requires_hierarchy_; }
  absl::Status Decode(const std::vector<int>& ids, std::string* out) const;

  int GetPieceSize() const { return static_cast<int>(id_to_piece_.size()); }
  int unk_id() const { return unk_id_; }
  const std::string& IdToPiece(int id) const;
  int PieceToId(absl::string_view piece) const;
  // ModelProto::SentencePiece::Type of `id` (NORMAL=1, UNKNOWN=2, CONTROL=3,
  // USER_DEFINED=4 ...); -1 when out of range.
  int IdToType(int id) const;
  bool IsUserDefined(int id) const;

  // "<id>\t<piece>\t<type>" per line, SHA-256. Two tokenizers of the same size
  // with different meanings differ here; this is the warm-start identity.
  std::string IdMapSha256() const;
  std::string Normalize(absl::string_view text) const;

 private:
  absl::Status status_;
  std::vector<std::string> id_to_piece_;
  std::vector<int> id_to_type_;
  struct MergeRule { int rank = 0; bool hierarchy_gated = false; };
  absl::flat_hash_map<std::string, int> piece_to_id_;
  absl::flat_hash_map<std::string, MergeRule> merge_rule_;
  std::unique_ptr<normalizer::Normalizer> normalizer_;
  // Longest-prefix matcher over the USER_DEFINED piece strings. Owns nothing;
  // the strings live in id_to_piece_.
  std::unique_ptr<normalizer::PrefixMatcher> user_defined_matcher_;
  NormalizerSpec normalizer_spec_;
  int unk_id_ = 0;
  std::string unk_piece_ = "<unk>";
  bool requires_hierarchy_ = false;
  bool has_inherited_merge_program_ = false;

  absl::Status EncodeImpl(absl::string_view text,
                          const std::vector<CompletionGate>* gates,
                          std::vector<TokenSpan>* out) const;
};

}  // namespace sentencepiece::expansion

#endif  // EXPANSION_PROCESSOR_H_
