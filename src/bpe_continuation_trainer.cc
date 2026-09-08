// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "bpe_continuation_trainer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ret_check.h"
#include "util.h"

namespace sentencepiece::bpe {

std::string ContinuationTrainer::Symbol::ToString() const {
  return string_util::UnicodeTextToUTF8(chars);
}

uint64_t ContinuationTrainer::EncodePos(int sid, int l, int r) {
  CHECK_GE(l, 0);
  CHECK_GE(r, 0);
  CHECK_LE(l, std::numeric_limits<uint16_t>::max());
  CHECK_LE(r, std::numeric_limits<uint16_t>::max());
  return (static_cast<uint64_t>(sid) << 32) |
         (static_cast<uint64_t>(l) << 16) | static_cast<uint64_t>(r);
}

ContinuationTrainer::Position ContinuationTrainer::DecodePos(uint64_t n) {
  return Position{static_cast<int>(n >> 32),
                  static_cast<int>((n >> 16) & 0xffff),
                  static_cast<int>(n & 0xffff)};
}

ContinuationTrainer::Symbol* ContinuationTrainer::GetAtomicSymbol(
    absl::string_view atom) {
  const uint64_t fp = absl::HashOf(uint64_t{0}, atom);
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    CHECK_EQ(it->second->ToString(), atom)
        << "hash collision in BPE continuation atomic alphabet";
    return it->second;
  }

  auto s = std::make_unique<Symbol>();
  s->fp = fp;
  s->chars = string_util::UTF8ToUnicodeText(atom);
  s->is_unk = (s->chars.size() == 1 && s->chars.front() == kUNKChar);
  s->freq = 1;
  Symbol* out = s.get();
  symbols_cache_.emplace(fp, out);
  allocated_.push_back(std::move(s));
  return out;
}

ContinuationTrainer::Symbol* ContinuationTrainer::GetPairSymbol(
    const Symbol* left, const Symbol* right) {
  if (left == nullptr || right == nullptr || left->is_unk || right->is_unk) {
    return nullptr;
  }
  const uint64_t fp = absl::HashOf(uint64_t{1}, left->fp, right->fp);
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    CHECK_EQ(it->second->left, left)
        << "hash collision in BPE continuation pair cache";
    CHECK_EQ(it->second->right, right)
        << "hash collision in BPE continuation pair cache";
    return it->second;
  }

  string_util::UnicodeText chars = left->chars;
  chars.insert(chars.end(), right->chars.begin(), right->chars.end());
  if (!IsValidSentencePiece(chars)) return nullptr;

  auto s = std::make_unique<Symbol>();
  s->fp = fp;
  s->left = left;
  s->right = right;
  s->chars = std::move(chars);
  Symbol* out = s.get();
  symbols_cache_.emplace(fp, out);
  allocated_.push_back(std::move(s));
  return out;
}

void ContinuationTrainer::ComputeFreq(Symbol* symbol) const {
  if (!symbol->needs_recomputation) return;
  symbol->freq = 0;
  for (auto it = symbol->positions.begin(); it != symbol->positions.end();) {
    const Position pos = DecodePos(*it);
    if (symbol->left != symbols_[pos.sid][pos.left] ||
        symbol->right != symbols_[pos.sid][pos.right]) {
      it = symbol->positions.erase(it);
    } else {
      symbol->freq += static_cast<uint64_t>(sentences_[pos.sid].second);
      ++it;
    }
  }
  symbol->needs_recomputation = false;
}

int ContinuationTrainer::GetNextIndex(int sid, int index) const {
  for (size_t i = static_cast<size_t>(index + 1); i < symbols_[sid].size();
       ++i) {
    if (symbols_[sid][i] != nullptr) return static_cast<int>(i);
  }
  return -1;
}

int ContinuationTrainer::GetPrevIndex(int sid, int index) const {
  for (int i = index - 1; i >= 0; --i) {
    if (symbols_[sid][i] != nullptr) return i;
  }
  return -1;
}

void ContinuationTrainer::AddNewPair(int sid, int left, int right) {
  if (left == -1 || right == -1) return;
  Symbol* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol == nullptr) return;
  symbol->positions.insert(EncodePos(sid, left, right));
  if (!symbol->pending) {
    symbol->pending = true;
    pending_queue_.push_back(symbol);
  }
}

void ContinuationTrainer::ResetFreq(int sid, int left, int right,
                                    const Symbol* best) {
  if (left == -1 || right == -1) return;
  Symbol* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr && symbol != best) symbol->needs_recomputation = true;
}

absl::Status ContinuationTrainer::AcceptSymbol(Symbol* symbol) {
  for (const uint64_t encoded_pos : symbol->positions) {
    const Position pos = DecodePos(encoded_pos);
    if (symbols_[pos.sid][pos.left] == nullptr) continue;
    RET_CHECK(symbols_[pos.sid][pos.right] != nullptr);

    const int next = GetNextIndex(pos.sid, pos.right);
    const int prev = GetPrevIndex(pos.sid, pos.left);
    ResetFreq(pos.sid, prev, pos.left, symbol);
    ResetFreq(pos.sid, pos.right, next, symbol);

    symbols_[pos.sid][pos.left] = symbol;
    symbols_[pos.sid][pos.right] = nullptr;
    AddNewPair(pos.sid, prev, pos.left);
    AddNewPair(pos.sid, pos.left, next);
  }

  symbols_cache_.erase(symbol->fp);
  symbol->active = false;
  return absl::OkStatus();
}

void ContinuationTrainer::DrainPendingQueue() {
  for (Symbol* symbol : pending_queue_) {
    symbol->pending = false;
    if (!symbol->active) continue;
    ComputeFreq(symbol);
    pq_.push({symbol->freq, symbol});
  }
  pending_queue_.clear();
}

absl::Status ContinuationTrainer::SegmentAtoms(
    absl::string_view text, std::vector<std::string>* atoms) const {
  RET_CHECK(atoms != nullptr);
  atoms->clear();
  if (text.empty()) return absl::OkStatus();
  if (atomic_pieces_ordered_.empty()) {
    return absl::InvalidArgumentError(
        "BPE continuation has an empty reversible atomic alphabet");
  }

  // ways[pos] is capped at two: 0 = impossible, 1 = unique, 2 = ambiguous.
  // Positions are byte offsets. Because every transition consumes a complete
  // structurally-valid atomic token, every reachable position is a valid UTF-8
  // boundary even when an atom spans multiple Unicode scalars.
  const size_t n = text.size();
  std::vector<unsigned char> ways(n + 1, 0);
  std::vector<int> choice(n + 1, -1);
  ways[n] = 1;

  for (size_t reverse = 0; reverse < n; ++reverse) {
    const size_t pos = n - reverse - 1;
    int total = 0;
    int unique_choice = -1;
    for (size_t i = 0; i < atomic_pieces_ordered_.size(); ++i) {
      const std::string& atom = atomic_pieces_ordered_[i];
      if (atom.size() > n - pos || ways[pos + atom.size()] == 0) continue;
      if (text.substr(pos, atom.size()) != atom) continue;
      if (total == 0 && ways[pos + atom.size()] == 1) {
        unique_choice = static_cast<int>(i);
      } else {
        unique_choice = -1;
      }
      total = std::min(2, total + static_cast<int>(ways[pos + atom.size()]));
      if (total == 2) unique_choice = -1;
    }
    ways[pos] = static_cast<unsigned char>(total);
    if (total == 1) choice[pos] = unique_choice;
  }

  if (ways[0] == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "text cannot be segmented by the inherited reversible atomic "
        "alphabet: ", text));
  }
  if (ways[0] != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "declared BPE atomic alphabet is ambiguous for text: ", text));
  }

  size_t pos = 0;
  while (pos < n) {
    const int index = choice[pos];
    if (index < 0 || index >= static_cast<int>(atomic_pieces_ordered_.size())) {
      return absl::InternalError(
          "unique atomic segmentation lost its reconstruction choice");
    }
    const std::string& atom = atomic_pieces_ordered_[index];
    atoms->push_back(atom);
    pos += atom.size();
  }
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::ValidateMergeProgram(
    const std::vector<ExpansionMerge>& merges,
    bool require_all_declared_pieces) const {
  std::set<std::string> constructible(atomic_pieces_ordered_.begin(),
                                      atomic_pieces_ordered_.end());
  std::set<std::pair<std::string, std::string>> seen_pairs;
  std::set<std::string> constructed_children;
  std::map<std::string, std::pair<int, bool>> pieces;

  auto add_piece = [&](const ExpansionPiece& piece) -> absl::Status {
    const auto inserted = pieces.emplace(
        piece.piece(), std::make_pair(piece.external_id(), piece.mergeable()));
    if (!inserted.second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate declared piece string: ", piece.piece()));
    }
    return absl::OkStatus();
  };
  for (const auto& piece : base_pieces_) ABSL_RETURN_IF_ERROR(add_piece(piece));
  for (const auto& piece : bootstrap_pieces_) {
    ABSL_RETURN_IF_ERROR(add_piece(piece));
  }
  for (const auto& piece : learned_pieces_) {
    ABSL_RETURN_IF_ERROR(add_piece(piece));
  }

  for (size_t i = 0; i < merges.size(); ++i) {
    const ExpansionMerge& merge = merges[i];
    if (merge.rank() != static_cast<int>(i)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "effective merge ranks must be contiguous from zero; expected ", i,
          " got ", merge.rank()));
    }
    if (merge.left().empty() || merge.right().empty()) {
      return absl::InvalidArgumentError("merge parents must be nonempty");
    }
    if (!seen_pairs.insert({merge.left(), merge.right()}).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "duplicate merge pair in effective program: ", merge.left(), " + ",
          merge.right()));
    }
    if (!constructible.count(merge.left()) ||
        !constructible.count(merge.right())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "merge parents are not constructible at effective rank ",
          merge.rank(), ": ", merge.left(), " + ", merge.right()));
    }

    const std::string child = merge.left() + merge.right();
    const auto piece_it = pieces.find(child);
    if (piece_it == pieces.end() || !piece_it->second.second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "merge child is missing or nonmergeable at effective rank ",
          merge.rank(), ": ", child));
    }
    if (merge.external_id() != piece_it->second.first) {
      return absl::InvalidArgumentError(absl::StrCat(
          "merge child external_id mismatch for ", child, ": merge says ",
          merge.external_id(), " but piece says ", piece_it->second.first));
    }
    if (!constructed_children.insert(child).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "piece has multiple merge constructions in effective program: ",
          child));
    }
    constructible.insert(child);
  }

  if (require_all_declared_pieces) {
    auto require_piece = [&](const ExpansionPiece& piece) -> absl::Status {
      if (piece.mergeable() && !piece.atomic() &&
          !constructible.count(piece.piece())) {
        return absl::InvalidArgumentError(absl::StrCat(
            "mergeable declared piece has no construction in effective rank "
            "program: ",
            piece.piece()));
      }
      return absl::OkStatus();
    };
    for (const auto& piece : base_pieces_) {
      ABSL_RETURN_IF_ERROR(require_piece(piece));
    }
    for (const auto& piece : bootstrap_pieces_) {
      ABSL_RETURN_IF_ERROR(require_piece(piece));
    }
    for (const auto& piece : learned_pieces_) {
      ABSL_RETURN_IF_ERROR(require_piece(piece));
    }
  }

  return absl::OkStatus();
}

absl::Status ContinuationTrainer::LoadAndValidateSpec() {
  if (trainer_spec_.expansion_spec().empty()) {
    return absl::InvalidArgumentError(
        "BPE continuation requires --expansion_spec");
  }
  if (!trainer_spec_.unigram_prior_model().empty()) {
    return absl::InvalidArgumentError(
        "expansion_spec and unigram_prior_model are mutually exclusive");
  }
  if (trainer_spec_.input_sentence_size() != 0) {
    return absl::InvalidArgumentError(
        "BPE continuation forbids input sampling");
  }
  if (trainer_spec_.treat_whitespace_as_suffix()) {
    return absl::InvalidArgumentError(
        "BPE continuation v1 does not support treat_whitespace_as_suffix");
  }

  ABSL_RETURN_IF_ERROR(continuation::ReadExpansionSpec(
      trainer_spec_.expansion_spec(), &expansion_spec_));
  if (expansion_spec_.schema_version() != 1) {
    return absl::InvalidArgumentError("unsupported ExpansionSpec schema_version");
  }
  if (expansion_spec_.model_type() != EXPANSION_BPE) {
    return absl::InvalidArgumentError(
        "BPE continuation requires ExpansionSpec.model_type=EXPANSION_BPE");
  }
  if (!expansion_spec_.preserve_base_ids()) {
    return absl::InvalidArgumentError(
        "continuation requires preserve_base_ids=true");
  }
  if (expansion_spec_.base_pieces().empty()) {
    return absl::InvalidArgumentError("ExpansionSpec has no base pieces");
  }

  base_pieces_.assign(expansion_spec_.base_pieces().begin(),
                      expansion_spec_.base_pieces().end());
  std::sort(base_pieces_.begin(), base_pieces_.end(),
            [](const ExpansionPiece& a, const ExpansionPiece& b) {
              return std::make_tuple(a.external_id(), a.piece()) <
                     std::make_tuple(b.external_id(), b.piece());
            });

  std::set<int> occupied_ids;
  std::map<std::string, ExpansionPiece> base_by_string;
  int max_id = -1;
  for (auto& piece : base_pieces_) {
    if (piece.external_id() < 0 || piece.piece().empty()) {
      return absl::InvalidArgumentError(
          "every base piece needs a nonnegative external_id and nonempty string");
    }
    if (!occupied_ids.insert(piece.external_id()).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate base external_id: ", piece.external_id()));
    }
    if (!string_util::IsStructurallyValid(piece.piece())) {
      return absl::InvalidArgumentError(
          absl::StrCat("base piece is not structurally valid UTF-8: ",
                       piece.piece()));
    }
    if (!base_by_string.emplace(piece.piece(), piece).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate base piece string: ", piece.piece()));
    }
    if (piece.atomic() && !piece.mergeable()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "atomic BPE pieces must be mergeable: ", piece.piece()));
    }
    max_id = std::max(max_id, piece.external_id());
    existing_piece_strings_.insert(piece.piece());
    if (piece.atomic() ||
        (piece.mergeable() && string_util::UTF8Len(piece.piece()) == 1)) {
      piece.set_atomic(true);
      atomic_piece_strings_.insert(piece.piece());
    }
  }
  atomic_pieces_ordered_.assign(atomic_piece_strings_.begin(),
                                atomic_piece_strings_.end());
  std::sort(atomic_pieces_ordered_.begin(), atomic_pieces_ordered_.end());
  if (atomic_pieces_ordered_.empty()) {
    return absl::InvalidArgumentError(
        "ExpansionSpec has no mergeable reversible atomic BPE pieces");
  }

  first_new_external_id_ = expansion_spec_.first_new_external_id() >= 0
                               ? expansion_spec_.first_new_external_id()
                               : max_id + 1;
  if (first_new_external_id_ <= max_id) {
    return absl::InvalidArgumentError(absl::StrCat(
        "first_new_external_id overlaps occupied base IDs: ",
        first_new_external_id_, " <= ", max_id));
  }
  next_external_id_ = first_new_external_id_;

  bootstrap_pieces_.assign(expansion_spec_.bootstrap_pieces().begin(),
                           expansion_spec_.bootstrap_pieces().end());
  std::map<std::string, int> bootstrap_ids;
  for (auto& piece : bootstrap_pieces_) {
    if (piece.piece().empty() || !piece.mergeable() || piece.atomic() ||
        piece.type() != ModelProto::SentencePiece::NORMAL) {
      return absl::InvalidArgumentError(
          "bootstrap BPE pieces must be nonempty, constructed, mergeable "
          "NORMAL pieces");
    }
    if (!string_util::IsStructurallyValid(piece.piece())) {
      return absl::InvalidArgumentError(
          absl::StrCat("bootstrap piece is not structurally valid UTF-8: ",
                       piece.piece()));
    }
    if (!existing_piece_strings_.insert(piece.piece()).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("bootstrap/base piece collision: ", piece.piece()));
    }
    if (piece.external_id() >= 0 && piece.external_id() != next_external_id_) {
      return absl::InvalidArgumentError(absl::StrCat(
          "bootstrap IDs must append contiguously; expected ", next_external_id_,
          " got ", piece.external_id()));
    }
    piece.set_external_id(next_external_id_++);
    if (!occupied_ids.insert(piece.external_id()).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("bootstrap external ID collision: ", piece.external_id()));
    }
    bootstrap_ids[piece.piece()] = piece.external_id();
  }

  target_new_pieces_ = expansion_spec_.requested_new_pieces();
  if (target_new_pieces_ == 0) {
    target_new_pieces_ = trainer_spec_.vocab_size() - next_external_id_;
  }
  if (target_new_pieces_ < 0) {
    return absl::InvalidArgumentError(
        "requested/final vocabulary is smaller than the occupied external ID "
        "range after bootstrap allocation");
  }

  base_merges_.assign(expansion_spec_.base_merges().begin(),
                      expansion_spec_.base_merges().end());
  bootstrap_merges_.assign(expansion_spec_.bootstrap_merges().begin(),
                           expansion_spec_.bootstrap_merges().end());
  auto rank_sort = [](const ExpansionMerge& a, const ExpansionMerge& b) {
    return a.rank() < b.rank();
  };
  std::sort(base_merges_.begin(), base_merges_.end(), rank_sort);
  std::sort(bootstrap_merges_.begin(), bootstrap_merges_.end(), rank_sort);

  auto validate_local_ranks = [](const std::vector<ExpansionMerge>& merges,
                                 absl::string_view label) -> absl::Status {
    for (size_t i = 0; i < merges.size(); ++i) {
      if (merges[i].rank() != static_cast<int>(i)) {
        return absl::InvalidArgumentError(absl::StrCat(
            label, " merge ranks must be unique and contiguous from zero; "
            "expected ",
            i, " got ", merges[i].rank()));
      }
      if (merges[i].left().empty() || merges[i].right().empty()) {
        return absl::InvalidArgumentError(
            absl::StrCat(label, " merge parents must be nonempty"));
      }
    }
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(validate_local_ranks(base_merges_, "base"));
  ABSL_RETURN_IF_ERROR(validate_local_ranks(bootstrap_merges_, "bootstrap"));

  // First prove the inherited base tokenizer is a valid program on its own.
  // Rank-prepending bootstrap state is not allowed to retroactively make a
  // malformed inherited tokenizer constructible.
  std::set<std::string> base_constructible(atomic_pieces_ordered_.begin(),
                                           atomic_pieces_ordered_.end());
  std::set<std::pair<std::string, std::string>> all_pairs;
  std::set<std::string> base_children;
  for (auto& merge : base_merges_) {
    if (!all_pairs.insert({merge.left(), merge.right()}).second) {
      return absl::InvalidArgumentError("duplicate inherited BPE merge pair");
    }
    if (!base_constructible.count(merge.left()) ||
        !base_constructible.count(merge.right())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "malformed inherited merge order at rank ", merge.rank(), ": ",
          merge.left(), " + ", merge.right()));
    }
    const std::string child = merge.left() + merge.right();
    const auto child_it = base_by_string.find(child);
    if (child_it == base_by_string.end() || !child_it->second.mergeable()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "inherited merge child is missing or nonmergeable: ", child));
    }
    if (child_it->second.atomic()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "piece cannot be both an atomic alphabet symbol and an inherited "
          "merge child: ",
          child));
    }
    if (!base_children.insert(child).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "inherited piece has multiple merge constructions: ", child));
    }
    merge.set_external_id(child_it->second.external_id());
    base_constructible.insert(child);
  }
  for (const auto& piece : base_pieces_) {
    if (piece.mergeable() && !piece.atomic() &&
        !base_constructible.count(piece.piece())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "mergeable inherited piece is neither atomic nor constructed by the "
          "inherited merge program: ",
          piece.piece()));
    }
  }

  std::set<std::string> bootstrap_children;
  for (auto& merge : bootstrap_merges_) {
    if (!all_pairs.insert({merge.left(), merge.right()}).second) {
      return absl::InvalidArgumentError(
          "bootstrap merge duplicates an inherited/bootstrap pair");
    }
    const std::string child = merge.left() + merge.right();
    const auto child_it = bootstrap_ids.find(child);
    if (child_it == bootstrap_ids.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "bootstrap merge child is not declared as a bootstrap piece: ",
          child));
    }
    if (!bootstrap_children.insert(child).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "bootstrap piece has multiple merge constructions: ", child));
    }
    merge.set_external_id(child_it->second);
  }

  // This is the single authority for constructibility and rank semantics. With
  // allow_rank_prepend=true it validates bootstrap -> base; otherwise it
  // validates base -> bootstrap. Training will replay this exact same order and
  // serialization will emit this exact same order.
  const std::vector<ExpansionMerge> effective_prefix = EffectiveMergeTable();
  ABSL_RETURN_IF_ERROR(
      ValidateMergeProgram(effective_prefix, /*require_all_declared_pieces=*/true));

  return absl::OkStatus();
}

absl::Status ContinuationTrainer::InitializeCorpusSymbols() {
  symbols_.clear();
  allocated_.clear();
  symbols_cache_.clear();
  live_by_string_.clear();
  pq_ = decltype(pq_)();
  pending_queue_.clear();

  sentences_ = corpus_.sentences;
  symbols_.resize(sentences_.size());
  for (size_t sid = 0; sid < sentences_.size(); ++sid) {
    std::vector<std::string> atoms;
    ABSL_RETURN_IF_ERROR(SegmentAtoms(sentences_[sid].first, &atoms));
    for (const std::string& atom : atoms) {
      Symbol* symbol = GetAtomicSymbol(atom);
      symbols_[sid].push_back(symbol);
      live_by_string_[atom] = symbol;
    }
  }

  for (size_t sid = 0; sid < symbols_.size(); ++sid) {
    for (size_t i = 1; i < symbols_[sid].size(); ++i) {
      AddNewPair(static_cast<int>(sid), static_cast<int>(i - 1),
                 static_cast<int>(i));
    }
  }
  DrainPendingQueue();
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::ReplayMerges(
    const std::vector<ExpansionMerge>& merges, absl::string_view label) {
  int applied = 0;
  int absent = 0;
  for (const auto& merge : merges) {
    const auto left = live_by_string_.find(merge.left());
    const auto right = live_by_string_.find(merge.right());
    if (left == live_by_string_.end() || right == live_by_string_.end()) {
      ++absent;
      continue;
    }
    Symbol* symbol = GetPairSymbol(left->second, right->second);
    if (symbol == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          label, " merge is incompatible with current trainer piece-shape "
          "constraints at effective rank ", merge.rank(), ": ", merge.left(),
          " + ", merge.right()));
    }
    symbol->needs_recomputation = true;
    ComputeFreq(symbol);
    if (symbol->freq == 0) {
      ++absent;
      continue;
    }
    live_by_string_[merge.left() + merge.right()] = symbol;
    ABSL_RETURN_IF_ERROR(AcceptSymbol(symbol));
    DrainPendingQueue();
    ++applied;
  }
  LOG(INFO) << "BPE continuation replayed " << applied << " " << label
            << " merges; " << absent << " were absent from this corpus";
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::LearnExpansion() {
  while (static_cast<int>(learned_pieces_.size()) < target_new_pieces_) {
    Symbol* best = nullptr;
    while (!pq_.empty()) {
      QueueEntry entry = pq_.top();
      pq_.pop();
      Symbol* symbol = entry.symbol;
      if (!symbol->active || entry.freq != symbol->freq) continue;
      if (symbol->needs_recomputation) {
        ComputeFreq(symbol);
        pq_.push({symbol->freq, symbol});
        continue;
      }
      if (symbol->freq == 0) continue;
      best = symbol;
      break;
    }

    if (best == nullptr) break;
    if (!best->IsBigram()) {
      return absl::InternalError("BPE continuation selected a non-bigram");
    }
    const std::string child = best->ToString();
    if (existing_piece_strings_.contains(child)) {
      // A rediscovered inherited/bootstrap string is not a new token and does
      // not consume expansion budget. Applying an alternative ancestry here
      // would mutate the corpus with a merge that is absent from the exported
      // rank program, so the candidate is retired instead.
      symbols_cache_.erase(best->fp);
      best->active = false;
      continue;
    }

    ExpansionPiece piece;
    piece.set_external_id(next_external_id_++);
    piece.set_piece(child);
    piece.set_type(ModelProto::SentencePiece::NORMAL);
    piece.set_mergeable(true);
    piece.set_atomic(false);
    piece.set_score(-static_cast<float>(base_merges_.size() +
                                        bootstrap_merges_.size() +
                                        learned_pieces_.size()));

    ExpansionMerge merge;
    merge.set_left(best->left->ToString());
    merge.set_right(best->right->ToString());
    merge.set_external_id(piece.external_id());

    // Exact parent provenance is captured at the acceptance point, before the
    // corpus is mutated. No exporter is ever asked to infer a split later.
    learned_pieces_.push_back(piece);
    learned_merges_.push_back(merge);
    existing_piece_strings_.insert(child);

    ABSL_RETURN_IF_ERROR(AcceptSymbol(best));
    live_by_string_[child] = best;
    DrainPendingQueue();
  }

  if (static_cast<int>(learned_pieces_.size()) != target_new_pieces_ &&
      trainer_spec_.hard_vocab_limit()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "BPE continuation could only learn ", learned_pieces_.size(), " of ",
        target_new_pieces_, " requested expansion pieces"));
  }
  return absl::OkStatus();
}

std::vector<ExpansionMerge> ContinuationTrainer::EffectiveMergeTable() const {
  std::vector<ExpansionMerge> out;
  out.reserve(base_merges_.size() + bootstrap_merges_.size() +
              learned_merges_.size());

  int rank = 0;
  if (expansion_spec_.allow_rank_prepend()) {
    for (const auto& source : bootstrap_merges_) {
      ExpansionMerge merge = source;
      merge.set_rank(rank++);
      out.push_back(std::move(merge));
    }
  }
  for (const auto& source : base_merges_) {
    ExpansionMerge merge = source;
    merge.set_rank(rank++);
    out.push_back(std::move(merge));
  }
  if (!expansion_spec_.allow_rank_prepend()) {
    for (const auto& source : bootstrap_merges_) {
      ExpansionMerge merge = source;
      merge.set_rank(rank++);
      out.push_back(std::move(merge));
    }
  }
  for (const auto& source : learned_merges_) {
    ExpansionMerge merge = source;
    merge.set_rank(rank++);
    out.push_back(std::move(merge));
  }
  return out;
}

bool ContinuationTrainer::IsReachable(
    absl::string_view piece, const std::vector<ExpansionMerge>& merges) const {
  std::map<std::pair<std::string, std::string>, int> pair_rank;
  for (const auto& merge : merges) {
    pair_rank[{merge.left(), merge.right()}] = merge.rank();
  }

  std::vector<std::string> symbols;
  if (!SegmentAtoms(piece, &symbols).ok()) return false;

  size_t guard = 0;
  while (symbols.size() > 1 && guard++ <= merges.size() + symbols.size() + 1) {
    int best_rank = std::numeric_limits<int>::max();
    std::pair<std::string, std::string> best_pair;
    bool found = false;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto key = std::make_pair(symbols[i], symbols[i + 1]);
      const auto it = pair_rank.find(key);
      if (it != pair_rank.end() && it->second < best_rank) {
        best_rank = it->second;
        best_pair = key;
        found = true;
      }
    }
    if (!found) break;

    std::vector<std::string> next;
    next.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == best_pair.first &&
          symbols[i + 1] == best_pair.second) {
        next.push_back(symbols[i] + symbols[i + 1]);
        i += 2;
      } else {
        next.push_back(symbols[i]);
        ++i;
      }
    }
    symbols.swap(next);
  }
  return symbols.size() == 1 && symbols.front() == piece;
}

absl::Status ContinuationTrainer::BuildNativeModel(ModelProto* model) const {
  std::vector<ExpansionPiece> pieces = base_pieces_;
  pieces.insert(pieces.end(), bootstrap_pieces_.begin(), bootstrap_pieces_.end());
  pieces.insert(pieces.end(), learned_pieces_.begin(), learned_pieces_.end());
  std::sort(pieces.begin(), pieces.end(), [](const ExpansionPiece& a,
                                             const ExpansionPiece& b) {
    return a.external_id() < b.external_id();
  });
  int unknown_count = 0;
  for (size_t i = 0; i < pieces.size(); ++i) {
    if (pieces[i].external_id() != static_cast<int>(i)) {
      return absl::FailedPreconditionError(
          "external IDs are not contiguous from zero; ExpansionResult is "
          "authoritative and no native SentencePiece ModelProto is emitted");
    }
    if (pieces[i].type() == ModelProto::SentencePiece::UNKNOWN) ++unknown_count;
  }
  if (unknown_count != 1) {
    return absl::FailedPreconditionError(
        "external tokenizer does not expose exactly one SentencePiece UNKNOWN; "
        "ExpansionResult is authoritative and no native ModelProto is emitted");
  }

  model->Clear();
  for (const auto& piece : pieces) {
    auto* out = model->add_pieces();
    out->set_piece(piece.piece());
    out->set_score(piece.score());
    out->set_type(piece.type());
  }
  *model->mutable_trainer_spec() = trainer_spec_;
  model->mutable_trainer_spec()->set_vocab_size(model->pieces_size());
  *model->mutable_normalizer_spec() = normalizer_spec_;
  if (!denormalizer_spec_.normalization_rule_tsv().empty()) {
    *model->mutable_denormalizer_spec() = denormalizer_spec_;
  }
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::FinalizeArtifacts() {
  const std::vector<ExpansionMerge> effective = EffectiveMergeTable();

  // Stronger than per-piece reachability: first prove that the complete table
  // we are about to serialize is a valid rank program from the declared atomic
  // alphabet, including inherited, bootstrap and learned constructions.
  ABSL_RETURN_IF_ERROR(
      ValidateMergeProgram(effective, /*require_all_declared_pieces=*/true));

  int unreachable = 0;
  for (const auto& piece : learned_pieces_) {
    if (!IsReachable(piece.piece(), effective)) {
      ++unreachable;
      LOG(ERROR) << "unreachable learned BPE piece id=" << piece.external_id()
                 << " piece=" << piece.piece();
    }
  }
  if (unreachable != 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "BPE expansion failed reachability verification: ", unreachable,
        " learned pieces are unreachable in the serialized final merge table"));
  }

  ExpansionResult result;
  result.set_schema_version(1);
  result.set_model_type(EXPANSION_BPE);
  for (const auto& piece : base_pieces_) *result.add_base_pieces() = piece;
  for (const auto& piece : bootstrap_pieces_) {
    *result.add_bootstrap_pieces() = piece;
  }
  for (const auto& piece : learned_pieces_) *result.add_learned_pieces() = piece;

  const int bootstrap_count = static_cast<int>(bootstrap_merges_.size());
  const int base_count = static_cast<int>(base_merges_.size());
  for (const auto& merge : effective) {
    if (expansion_spec_.allow_rank_prepend()) {
      if (merge.rank() < bootstrap_count) {
        *result.add_bootstrap_merges() = merge;
      } else if (merge.rank() < bootstrap_count + base_count) {
        *result.add_base_merges() = merge;
      } else {
        *result.add_learned_merges() = merge;
      }
    } else {
      if (merge.rank() < base_count) {
        *result.add_base_merges() = merge;
      } else if (merge.rank() < base_count + bootstrap_count) {
        *result.add_bootstrap_merges() = merge;
      } else {
        *result.add_learned_merges() = merge;
      }
    }
  }

  result.set_first_new_external_id(first_new_external_id_);
  result.set_requested_new_pieces(target_new_pieces_);
  result.set_actual_new_pieces(static_cast<int>(learned_pieces_.size()));
  result.set_unreachable_pieces(0);
  result.set_rank_prepend(expansion_spec_.allow_rank_prepend());
  result.set_vocab_sha256(expansion_spec_.vocab_sha256());
  result.set_merges_sha256(expansion_spec_.merges_sha256());
  result.set_tokenizer_sha256(expansion_spec_.tokenizer_sha256());
  result.set_pretokenizer_sha256(expansion_spec_.pretokenizer_sha256());
  result.set_boundary_policy(expansion_spec_.boundary_policy());

  std::vector<ExpansionPiece> all_pieces = base_pieces_;
  all_pieces.insert(all_pieces.end(), bootstrap_pieces_.begin(),
                    bootstrap_pieces_.end());
  all_pieces.insert(all_pieces.end(), learned_pieces_.begin(),
                    learned_pieces_.end());

  if (!trainer_spec_.model_prefix().empty()) {
    ABSL_RETURN_IF_ERROR(continuation::WriteMergeTable(
        trainer_spec_.model_prefix() + ".merges", effective));
    ABSL_RETURN_IF_ERROR(continuation::WriteExpansionVocab(
        trainer_spec_.model_prefix() + ".vocab", all_pieces));
  }

  const std::string result_path = !trainer_spec_.expansion_result().empty()
                                      ? trainer_spec_.expansion_result()
                                      : trainer_spec_.model_prefix() + ".expansion";
  if (!result_path.empty()) {
    // The success sidecar is deliberately written only after full-program and
    // learned-piece reachability validation has passed.
    ABSL_RETURN_IF_ERROR(
        continuation::WriteExpansionResult(result_path, result));
  }

  ModelProto native;
  absl::Status native_status = BuildNativeModel(&native);
  if (native_status.ok()) {
    *native.mutable_expansion_result() = result;
    if (output_model_proto_ != nullptr) {
      *output_model_proto_ = native;
    } else if (!trainer_spec_.model_prefix().empty()) {
      ABSL_RETURN_IF_ERROR(continuation::WriteModelProto(
          trainer_spec_.model_prefix() + ".model", native));
    }
  } else if (output_model_proto_ != nullptr) {
    return native_status;
  } else {
    LOG(INFO) << native_status.message();
  }

  return absl::OkStatus();
}

absl::Status ContinuationTrainer::Train() {
  ABSL_RETURN_IF_ERROR(status());
  RET_CHECK_EQ(TrainerSpec::BPE, trainer_spec_.model_type());

  existing_piece_strings_.clear();
  atomic_piece_strings_.clear();
  atomic_pieces_ordered_.clear();
  live_by_string_.clear();
  base_pieces_.clear();
  bootstrap_pieces_.clear();
  learned_pieces_.clear();
  base_merges_.clear();
  bootstrap_merges_.clear();
  learned_merges_.clear();
  sentences_.clear();
  final_pieces_.clear();
  allocated_.clear();
  symbols_cache_.clear();
  symbols_.clear();
  pq_ = decltype(pq_)();
  pending_queue_.clear();

  ABSL_RETURN_IF_ERROR(LoadAndValidateSpec());
  ABSL_RETURN_IF_ERROR(continuation::LoadPreparedCorpus(
      trainer_spec_, normalizer_spec_, components_, &corpus_));
  ABSL_RETURN_IF_ERROR(InitializeCorpusSymbols());

  // Replay the same inherited/bootstrap rank program that will be serialized.
  // This is the critical continuation invariant: training and exported BPE
  // tokenization start from one identical state, including rank prepending.
  const std::vector<ExpansionMerge> effective_prefix = EffectiveMergeTable();
  ABSL_RETURN_IF_ERROR(
      ReplayMerges(effective_prefix, "effective inherited/bootstrap"));
  ABSL_RETURN_IF_ERROR(LearnExpansion());
  ABSL_RETURN_IF_ERROR(FinalizeArtifacts());

  allocated_.clear();
  symbols_cache_.clear();
  live_by_string_.clear();
  symbols_.clear();
  pq_ = decltype(pq_)();
  pending_queue_.clear();
  return absl::OkStatus();
}

}  // namespace sentencepiece::bpe
