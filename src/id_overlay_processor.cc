// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "id_overlay_processor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>

#include "absl/strings/str_cat.h"
#include "filesystem.h"

namespace sentencepiece::overlay {
namespace {

uint64_t SplitMix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

bool DropOccurrence(double p, uint64_t seed, uint32_t rank, int position,
                    uint32_t left_version, uint32_t right_version) {
  if (p <= 0.0) return false;
  if (p >= 1.0) return true;
  uint64_t h = seed;
  h = SplitMix64(h ^ static_cast<uint64_t>(rank));
  h = SplitMix64(h ^ static_cast<uint64_t>(
                         static_cast<uint32_t>(position)));
  h = SplitMix64(h ^ (static_cast<uint64_t>(left_version) << 32) ^
                 static_cast<uint64_t>(right_version));
  // Exact 53-bit uniform variate, independent of libstdc++ RNG details.
  const double u = static_cast<double>(h >> 11) *
                   (1.0 / 9007199254740992.0);
  return u < p;
}

}  // namespace

uint64_t IdOverlayProcessor::PairKey(int left, int right) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
         static_cast<uint32_t>(right);
}

bool IdOverlayProcessor::IsProtected(int id) const {
  return protected_ids_.contains(id);
}

bool IdOverlayProcessor::IsOverlay(int id) const {
  return id >= first_overlay_id_ && id < model_vocab_size_ &&
         !expansion_by_id_[id].empty();
}

absl::Status IdOverlayProcessor::Load(const IdOverlayProgram& program) {
  // Publish a complete validated program, never partially installed rules.
  IdOverlayProcessor candidate;
  const auto status = candidate.LoadValidated(program);
  if (!status.ok()) return status;
  candidate.loaded_ = true;
  *this = std::move(candidate);
  return absl::OkStatus();
}

absl::Status IdOverlayProcessor::LoadValidated(const IdOverlayProgram& program) {
  rules_.clear();
  protected_ids_.clear();
  expansion_by_id_.clear();
  overlay_ids_.clear();
  base_identity_sha256_.clear();

  if (!program.IsInitialized()) {
    return absl::InvalidArgumentError("overlay program lacks required fields");
  }
  if (program.schema_version() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "unsupported IdOverlayProgram schema_version ",
        program.schema_version(), " (expected 1)"));
  }
  base_vocab_size_ = program.base_vocab_size();
  model_vocab_size_ = program.model_vocab_size();
  first_overlay_id_ = program.first_overlay_id();
  open_fence_id_ = program.open_fence_id();
  close_fence_id_ = program.close_fence_id();
  max_input_ids_ = program.max_input_ids();
  max_expansion_ids_ = program.max_expansion_ids();
  base_identity_sha256_ = program.base_identity_sha256();

  if (base_identity_sha256_.empty()) {
    return absl::InvalidArgumentError("base_identity_sha256 is empty");
  }
  if (base_vocab_size_ <= 0 || model_vocab_size_ < base_vocab_size_ ||
      first_overlay_id_ < base_vocab_size_ ||
      first_overlay_id_ > model_vocab_size_) {
    return absl::InvalidArgumentError(
        "invalid base/model/first-overlay vocabulary bounds");
  }
  if (open_fence_id_ < 0 || open_fence_id_ >= base_vocab_size_ ||
      close_fence_id_ < 0 || close_fence_id_ >= base_vocab_size_ ||
      open_fence_id_ == close_fence_id_) {
    return absl::InvalidArgumentError("invalid or colliding fence IDs");
  }
  if (max_input_ids_ <= 0 || max_expansion_ids_ <= 0) {
    return absl::InvalidArgumentError("overlay limits must be positive");
  }

  expansion_by_id_.resize(model_vocab_size_);
  for (int id = 0; id < base_vocab_size_; ++id) {
    expansion_by_id_[id] = {id};
  }

  for (int id : program.protected_base_ids()) {
    if (id < 0 || id >= base_vocab_size_) {
      return absl::InvalidArgumentError(
          absl::StrCat("protected base id out of range: ", id));
    }
    protected_ids_.insert(id);
  }
  // Fence tokens are unconditionally frozen even if an exporter forgot to
  // duplicate them in protected_base_ids.
  protected_ids_.insert(open_fence_id_);
  protected_ids_.insert(close_fence_id_);

  absl::flat_hash_set<uint32_t> ranks;
  absl::flat_hash_set<int> children;
  absl::flat_hash_set<std::vector<int>> expansions;
  std::vector<const IdOverlayRule*> ordered;
  ordered.reserve(program.rules_size());
  for (const auto& r : program.rules()) ordered.push_back(&r);
  std::sort(ordered.begin(), ordered.end(),
            [](const IdOverlayRule* a, const IdOverlayRule* b) {
              return a->rank() < b->rank();
            });

  for (const IdOverlayRule* rp : ordered) {
    const auto& r = *rp;
    if (!ranks.insert(r.rank()).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate overlay rank ", r.rank()));
    }
    if (r.left_id() < 0 || r.left_id() >= model_vocab_size_ ||
        r.right_id() < 0 || r.right_id() >= model_vocab_size_ ||
        r.child_id() < first_overlay_id_ ||
        r.child_id() >= model_vocab_size_) {
      return absl::InvalidArgumentError(
          absl::StrCat("overlay rule ", r.rank(), " has an invalid ID"));
    }
    if (IsProtected(r.left_id()) || IsProtected(r.right_id())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "overlay rule ", r.rank(), " consumes a protected/fence ID"));
    }
    if (r.left_id() >= base_vocab_size_ &&
        expansion_by_id_[r.left_id()].empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "overlay rule ", r.rank(), " left operand does not already exist"));
    }
    if (r.right_id() >= base_vocab_size_ &&
        expansion_by_id_[r.right_id()].empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "overlay rule ", r.rank(), " right operand does not already exist"));
    }
    if (!children.insert(r.child_id()).second ||
        !expansion_by_id_[r.child_id()].empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "overlay child ID is not fresh: ", r.child_id()));
    }

    const uint64_t key = PairKey(r.left_id(), r.right_id());
    if (rules_.contains(key)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "duplicate overlay pair at rank ", r.rank()));
    }

    const size_t expansion_size = expansion_by_id_[r.left_id()].size() +
                                  expansion_by_id_[r.right_id()].size();
    if (expansion_size > static_cast<size_t>(max_expansion_ids_)) {
      return absl::ResourceExhaustedError("overlay macro expansion exceeds limit");
    }
    std::vector<int> expected = expansion_by_id_[r.left_id()];
    expected.insert(expected.end(), expansion_by_id_[r.right_id()].begin(),
                    expansion_by_id_[r.right_id()].end());
    if (expected.size() > static_cast<size_t>(max_expansion_ids_)) {
      return absl::ResourceExhaustedError(absl::StrCat(
          "overlay expansion exceeds max_expansion_ids at rank ", r.rank()));
    }
    if (r.base_expansion_size() != static_cast<int>(expected.size()) ||
        !std::equal(expected.begin(), expected.end(),
                    r.base_expansion().begin())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "overlay rule ", r.rank(),
          " carries a base expansion inconsistent with its operands"));
    }
    // A protected ID cannot be hidden inside a macro because no rule may use
    // it as an operand. Keep this explicit as an artifact-integrity witness.
    for (int id : expected) {
      if (IsProtected(id)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "overlay rule ", r.rank(),
            " expansion contains a protected/fence ID"));
      }
    }

    if (!expansions.insert(expected).second) {
      return absl::InvalidArgumentError("duplicate expanded overlay atom sequence");
    }
    expansion_by_id_[r.child_id()] = std::move(expected);
    overlay_ids_.push_back(r.child_id());
    rules_.emplace(key, Rule{r.left_id(), r.right_id(), r.child_id(),
                             r.rank()});
  }
  std::sort(overlay_ids_.begin(), overlay_ids_.end());
  return absl::OkStatus();
}

absl::Status IdOverlayProcessor::LoadFromSerialized(
    absl::string_view serialized) {
  if (serialized.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::ResourceExhaustedError("serialized overlay exceeds parser limit");
  }
  IdOverlayProgram program;
  if (!program.ParseFromArray(serialized.data(),
                              static_cast<int>(serialized.size()))) {
    return absl::InvalidArgumentError(
        "bytes are not a serialized IdOverlayProgram");
  }
  return Load(program);
}

absl::Status IdOverlayProcessor::LoadFromFile(absl::string_view path) {
  auto input = filesystem::NewReadableFile(path, true);
  const absl::Status input_status = input->status();
  if (!input_status.ok()) return input_status;
  std::string blob;
  if (!input->ReadAll(&blob)) {
    return absl::InternalError(absl::StrCat("cannot read ", path));
  }
  return LoadFromSerialized(blob);
}

absl::Status IdOverlayProcessor::ApplyBlock(
    const std::vector<int>& block, double dropout, uint64_t seed,
    std::vector<int>* out, OverlayStats* stats) const {
  if (dropout < 0.0 || dropout > 1.0 || !std::isfinite(dropout)) {
    return absl::InvalidArgumentError("overlay dropout must be in [0,1]");
  }
  if (block.empty()) return absl::OkStatus();

  struct Sym {
    int id;
    int prev;
    int next;
    int position;
    uint32_t version = 0;
    bool alive = true;
  };
  struct Candidate {
    uint32_t rank;
    int position;
    int left;
    int right;
    uint32_t left_version;
    uint32_t right_version;
  };
  struct Worse {
    bool operator()(const Candidate& a, const Candidate& b) const {
      if (a.rank != b.rank) return a.rank > b.rank;
      return a.position > b.position;
    }
  };

  std::vector<Sym> syms;
  syms.reserve(block.size());
  for (size_t i = 0; i < block.size(); ++i) {
    const int id = block[i];
    if (id < 0 || id >= model_vocab_size_) {
      return absl::OutOfRangeError(
          absl::StrCat("input id out of model vocabulary: ", id));
    }
    if (id == open_fence_id_ || id == close_fence_id_) {
      return absl::InvalidArgumentError(
          "fence ID reached ApplyBlock; protocol scanner is inconsistent");
    }
    if (id >= base_vocab_size_ && !IsOverlay(id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("unknown extension ID inside overlay block: ", id));
    }
    syms.push_back(
        {id, static_cast<int>(i) - 1,
         i + 1 < block.size() ? static_cast<int>(i) + 1 : -1,
         static_cast<int>(i), 0, true});
  }

  std::priority_queue<Candidate, std::vector<Candidate>, Worse> heap;
  auto push_pair = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    const Sym& l = syms[left];
    const Sym& r = syms[right];
    if (!l.alive || !r.alive || l.next != right || r.prev != left ||
        IsProtected(l.id) || IsProtected(r.id)) {
      return;
    }
    const auto it = rules_.find(PairKey(l.id, r.id));
    if (it == rules_.end()) return;
    heap.push({it->second.rank, l.position, left, right, l.version, r.version});
    if (stats != nullptr) ++stats->candidates_pushed;
  };
  for (size_t i = 0; i + 1 < syms.size(); ++i) {
    push_pair(static_cast<int>(i), static_cast<int>(i + 1));
  }

  while (!heap.empty()) {
    const Candidate c = heap.top();
    heap.pop();
    if (stats != nullptr) ++stats->candidates_popped;
    Sym& l = syms[c.left];
    Sym& r = syms[c.right];
    if (!l.alive || !r.alive || l.version != c.left_version ||
        r.version != c.right_version || l.next != c.right ||
        r.prev != c.left) {
      if (stats != nullptr) ++stats->stale_pops;
      continue;
    }
    const auto it = rules_.find(PairKey(l.id, r.id));
    if (it == rules_.end() || it->second.rank != c.rank) {
      if (stats != nullptr) ++stats->stale_pops;
      continue;
    }

    // The decision belongs to this live occurrence/version. A skipped
    // occurrence is not reinserted. If a neighboring merge later changes it,
    // push_pair creates a new version and a new decision.
    if (DropOccurrence(dropout, seed, c.rank, c.position, l.version,
                       r.version)) {
      if (stats != nullptr) ++stats->dropout_skips;
      continue;
    }

    const int prev = l.prev;
    const int next = r.next;
    l.id = it->second.child;
    l.next = next;
    ++l.version;
    if (next >= 0) syms[next].prev = c.left;
    r.alive = false;
    ++r.version;
    if (stats != nullptr) ++stats->merges;
    push_pair(prev, c.left);
    push_pair(c.left, next);
  }

  for (int cur = 0; cur >= 0;) {
    if (syms[cur].alive) out->push_back(syms[cur].id);
    cur = syms[cur].next;
  }
  return absl::OkStatus();
}

absl::Status IdOverlayProcessor::EncodeIds(
    const std::vector<int>& ids, double dropout, uint64_t seed,
    bool allow_unclosed, std::vector<int>* out, OverlayStats* stats) const {
  if (!loaded_) return absl::FailedPreconditionError("overlay is not loaded");
  if (out == nullptr) {
    return absl::InvalidArgumentError("EncodeIds output is null");
  }
  out->clear();
  if (stats != nullptr) *stats = OverlayStats{};
  if (ids.size() > static_cast<size_t>(max_input_ids_)) {
    return absl::ResourceExhaustedError("input exceeds max_input_ids");
  }
  if (dropout < 0.0 || dropout > 1.0 || !std::isfinite(dropout)) {
    return absl::InvalidArgumentError("overlay dropout must be in [0,1]");
  }
  out->reserve(ids.size());

  bool inside = false;
  std::vector<int> block;
  for (size_t pos = 0; pos < ids.size(); ++pos) {
    const int id = ids[pos];
    if (id < 0 || id >= model_vocab_size_) {
      return absl::OutOfRangeError(
          absl::StrCat("input id out of model vocabulary at ", pos, ": ", id));
    }
    if (!inside) {
      if (id == close_fence_id_) {
        return absl::InvalidArgumentError(
            absl::StrCat("stray close fence at token ", pos));
      }
      if (id == open_fence_id_) {
        inside = true;
        out->push_back(id);
        block.clear();
        continue;
      }
      if (id >= base_vocab_size_) {
        return absl::InvalidArgumentError(absl::StrCat(
            "overlay/extension id outside fence at token ", pos, ": ", id));
      }
      out->push_back(id);
      continue;
    }

    if (id == open_fence_id_) {
      return absl::InvalidArgumentError(
          absl::StrCat("nested open fence at token ", pos));
    }
    if (id == close_fence_id_) {
      const absl::Status block_status =
          ApplyBlock(block, dropout, SplitMix64(seed ^ pos), out, stats);
      if (!block_status.ok()) return block_status;
      block.clear();
      out->push_back(id);
      inside = false;
      continue;
    }
    block.push_back(id);
  }
  if (inside) {
    if (!allow_unclosed) {
      return absl::InvalidArgumentError("unclosed overlay block");
    }
    const absl::Status block_status =
        ApplyBlock(block, dropout, SplitMix64(seed ^ ids.size()), out, stats);
    if (!block_status.ok()) return block_status;
  }
  return absl::OkStatus();
}

absl::Status IdOverlayProcessor::ExpandIds(const std::vector<int>& ids,
                                           bool allow_unclosed,
                                           std::vector<int>* out) const {
  if (!loaded_) return absl::FailedPreconditionError("overlay is not loaded");
  if (out == nullptr) {
    return absl::InvalidArgumentError("ExpandIds output is null");
  }
  out->clear();
  if (ids.size() > static_cast<size_t>(max_input_ids_)) {
    return absl::ResourceExhaustedError("input exceeds max_input_ids");
  }
  bool inside = false;
  for (size_t pos = 0; pos < ids.size(); ++pos) {
    const int id = ids[pos];
    if (id < 0 || id >= model_vocab_size_) {
      return absl::OutOfRangeError(
          absl::StrCat("input id out of model vocabulary at ", pos, ": ", id));
    }
    const size_t width = IsOverlay(id) ? expansion_by_id_[id].size() : 1;
    if (width > static_cast<size_t>(max_input_ids_) - out->size()) {
      return absl::ResourceExhaustedError("expanded sequence exceeds max_input_ids");
    }
    if (!inside) {
      if (id == close_fence_id_) {
        return absl::InvalidArgumentError(
            absl::StrCat("stray close fence at token ", pos));
      }
      if (id == open_fence_id_) {
        inside = true;
        out->push_back(id);
        continue;
      }
      if (id >= base_vocab_size_) {
        return absl::InvalidArgumentError(absl::StrCat(
            "overlay/extension id outside fence at token ", pos, ": ", id));
      }
      out->push_back(id);
      continue;
    }

    if (id == open_fence_id_) {
      return absl::InvalidArgumentError(
          absl::StrCat("nested open fence at token ", pos));
    }
    if (id == close_fence_id_) {
      out->push_back(id);
      inside = false;
      continue;
    }
    if (id < base_vocab_size_) {
      out->push_back(id);
    } else if (IsOverlay(id)) {
      const auto& expansion = expansion_by_id_[id];
      out->insert(out->end(), expansion.begin(), expansion.end());
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("unknown extension ID inside fence: ", id));
    }
  }
  if (inside && !allow_unclosed) {
    return absl::InvalidArgumentError("unclosed overlay block");
  }
  return absl::OkStatus();
}

}  // namespace sentencepiece::overlay
