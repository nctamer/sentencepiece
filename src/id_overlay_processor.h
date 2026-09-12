// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Integer-ID BPE overlay runtime for Melisma.
//
// This deliberately does NOT tokenize text. Its input is the already-produced
// base tokenizer ID stream. Ranked binary rules may fire only strictly between
// the configured fence IDs. Fence/control IDs are frozen and remain visible.

#ifndef ID_OVERLAY_PROCESSOR_H_
#define ID_OVERLAY_PROCESSOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "sentencepiece_model.pb.h"

namespace sentencepiece::overlay {

struct OverlayStats {
  uint64_t candidates_pushed = 0;
  uint64_t candidates_popped = 0;
  uint64_t stale_pops = 0;
  uint64_t dropout_skips = 0;
  uint64_t merges = 0;
};

class IdOverlayProcessor {
 public:
  IdOverlayProcessor() = default;

  absl::Status Load(const IdOverlayProgram& program);
  absl::Status LoadFromSerialized(absl::string_view serialized);
  absl::Status LoadFromFile(absl::string_view path);

  // Applies the ranked overlay to base/model IDs. Protocol errors are rejected.
  // An unclosed final block is accepted only when allow_unclosed=true.
  absl::Status EncodeIds(const std::vector<int>& ids, double dropout,
                         uint64_t seed, bool allow_unclosed,
                         std::vector<int>* out,
                         OverlayStats* stats = nullptr) const;

  // Expands every overlay ID to its complete base-ID sequence, while validating
  // the same fence state machine. Expansion happens before caller-side special
  // token filtering/decoding.
  absl::Status ExpandIds(const std::vector<int>& ids, bool allow_unclosed,
                         std::vector<int>* out) const;

  int base_vocab_size() const { return base_vocab_size_; }
  int model_vocab_size() const { return model_vocab_size_; }
  int first_overlay_id() const { return first_overlay_id_; }
  int open_fence_id() const { return open_fence_id_; }
  int close_fence_id() const { return close_fence_id_; }
  int rule_count() const { return static_cast<int>(rules_.size()); }
  const std::vector<int>& overlay_ids() const { return overlay_ids_; }
  int max_input_ids() const { return max_input_ids_; }
  int max_expansion_ids() const { return max_expansion_ids_; }
  const std::string& base_identity_sha256() const {
    return base_identity_sha256_;
  }

 private:
  struct Rule {
    int left = -1;
    int right = -1;
    int child = -1;
    uint32_t rank = 0;
  };

  static uint64_t PairKey(int left, int right);
  bool IsProtected(int id) const;
  bool IsOverlay(int id) const;
  absl::Status ApplyBlock(const std::vector<int>& block, double dropout,
                          uint64_t seed, std::vector<int>* out,
                          OverlayStats* stats) const;

  absl::flat_hash_map<uint64_t, Rule> rules_;
  absl::flat_hash_set<int> protected_ids_;
  std::vector<std::vector<int>> expansion_by_id_;
  std::vector<int> overlay_ids_;

  int base_vocab_size_ = 0;
  int model_vocab_size_ = 0;
  int first_overlay_id_ = 0;
  int open_fence_id_ = -1;
  int close_fence_id_ = -1;
  int max_input_ids_ = 0;
  int max_expansion_ids_ = 0;
  std::string base_identity_sha256_;
};

}  // namespace sentencepiece::overlay

#endif  // ID_OVERLAY_PROCESSOR_H_
