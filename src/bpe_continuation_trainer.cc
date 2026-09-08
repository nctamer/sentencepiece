// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "bpe_continuation_trainer.h"

#include <algorithm>
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

ContinuationTrainer::Symbol* ContinuationTrainer::GetCharSymbol(char32_t c) {
  const uint64_t fp = absl::HashOf(uint64_t{0}, static_cast<uint32_t>(c));
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) return it->second;

  auto s = std::make_unique<Symbol>();
  s->is_unk = (kUNKChar == c);
  s->fp = fp;
  s->chars.push_back(c);
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
  if (it != symbols_cache_.end()) return it->second;

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
              return std::tie(a.external_id(), a.piece()) <
                     std::tie(b.external_id(), b.piece());
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
    if (!base_by_string.emplace(piece.piece(), piece).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate base piece string: ", piece.piece()));
    }
    max_id = std::max(max_id, piece.external_id());
    existing_piece_strings_.insert(piece.piece());
    if (piece.atomic() ||
        (piece.mergeable() && string_util::UTF8Len(piece.piece()) == 1)) {
      atomic_piece_strings_.insert(piece.piece());
      piece.set_atomic(true);
    }
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
  for (auto& piece : bootstrap_pieces_) {
    if (piece.piece().empty() || !piece.mergeable() ||
        piece.type() != ModelProto::SentencePiece::NORMAL) {
      return absl::InvalidArgumentError(
          "bootstrap BPE pieces must be nonempty mergeable NORMAL pieces");
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
    occupied_ids.insert(piece.external_id());
  }

  target_new_pieces_ = expansion_spec_.requested_new_pieces();
  if (target_new_pieces_ == 0) {
    target_new_pieces_ = trainer_spec_.vocab_size() -
                         static_cast<int>(base_pieces_.size()) -
                         static_cast<int>(bootstrap_pieces_.size());
  }
  if (target_new_pieces_ < 0) {
    return absl::InvalidArgumentError(
        "requested vocabulary is smaller than inherited + bootstrap state");
  }

  base_merges_.assign(expansion_spec_.base_merges().begin(),
                      expansion_spec_.base_merges().end());
  std::sort(base_merges_.begin(), base_merges_.end(),
            [](const ExpansionMerge& a, const ExpansionMerge& b) {
              return a.rank() < b.rank();
            });

  std::set<std::string> constructible = atomic_piece_strings_;
  std::set<std::pair<std::string, std::string>> merge_pairs;
  std::set<std::string> constructed_children;
  for (size_t i = 0; i < base_merges_.size(); ++i) {
    auto& merge = base_merges_[i];
    if (merge.rank() != static_cast<int>(i)) {
      return absl::InvalidArgumentError(
          "base merge ranks must be unique and contiguous from zero");
    }
    if (!merge_pairs.insert({merge.left(), merge.right()}).second) {
      return absl::InvalidArgumentError("duplicate base merge pair");
    }
    if (!constructible.count(merge.left()) ||
        !constructible.count(merge.right())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "malformed base merge order: parents are not yet constructible at rank ",
          merge.rank(), ": ", merge.left(), " + ", merge.right()));
    }
    const std::string child = merge.left() + merge.right();
    const auto child_it = base_by_string.find(child);
    if (child_it == base_by_string.end() || !child_it->second.mergeable()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "base merge child is missing or nonmergeable: ", child));
    }
    if (!constructed_children.insert(child).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("base piece has multiple merge constructions: ", child));
    }
    merge.set_external_id(child_it->second.external_id());
    constructible.insert(child);
  }

  for (const auto& piece : base_pieces_) {
    if (piece.mergeable() && piece.type() == ModelProto::SentencePiece::NORMAL &&
        !constructible.count(piece.piece())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "mergeable base piece has no atomic/inherited construction: ",
          piece.piece()));
    }
  }

  bootstrap_merges_.assign(expansion_spec_.bootstrap_merges().begin(),
                           expansion_spec_.bootstrap_merges().end());
  std::sort(bootstrap_merges_.begin(), bootstrap_merges_.end(),
            [](const ExpansionMerge& a, const ExpansionMerge& b) {
              return a.rank() < b.rank();
            });
  std::map<std::string, int> bootstrap_ids;
  for (const auto& piece : bootstrap_pieces_) {
    bootstrap_ids[piece.piece()] = piece.external_id();
  }
  for (size_t i = 0; i < bootstrap_merges_.size(); ++i) {
    auto& merge = bootstrap_merges_[i];
    if (merge.rank() != static_cast<int>(i)) {
      return absl::InvalidArgumentError(
          "bootstrap merge ranks must be unique and contiguous from zero");
    }
    if (!merge_pairs.insert({merge.left(), merge.right()}).second) {
      return absl::InvalidArgumentError(
          "bootstrap merge duplicates an inherited/bootstrap pair");
    }
    if (!constructible.count(merge.left()) ||
        !constructible.count(merge.right())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "bootstrap merge parents are not constructible: ", merge.left(),
          " + ", merge.right()));
    }
    const std::string child = merge.left() + merge.right();
    const auto child_it = bootstrap_ids.find(child);
    if (child_it == bootstrap_ids.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "bootstrap merge child is not declared as a bootstrap piece: ",
          child));
    }
    merge.set_external_id(child_it->second);
    constructible.insert(child);
  }
  for (const auto& piece : bootstrap_pieces_) {
    if (!constructible.count(piece.piece())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "bootstrap piece has no valid construction: ", piece.piece()));
    }
  }

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
    for (const char32_t c : string_util::UTF8ToUnicodeText(sentences_[sid].first)) {
      const std::string atom = string_util::UnicodeCharToUTF8(c);
      if (!atomic_piece_strings_.contains(atom)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "continuation corpus requires an atom absent from the inherited "
            "reversible alphabet: ", atom));
      }
      Symbol* symbol = GetCharSymbol(c);
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
          "constraints: ", merge.left(), " + ", merge.right()));
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
      // A rediscovered inherited/bootstrap string is not a new token and must
      // not consume budget. Do not apply an unrecorded alternative ancestry.
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

    // Provenance is captured at the acceptance point, before the corpus is
    // mutated. There is never a later split-guessing step.
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
    const auto key = std::make_pair(merge.left(), merge.right());
    const auto it = pair_rank.find(key);
    if (it == pair_rank.end() || merge.rank() < it->second) {
      pair_rank[key] = merge.rank();
    }
  }

  std::vector<std::string> symbols;
  for (const char32_t c : string_util::UTF8ToUnicodeText(piece)) {
    std::string atom = string_util::UnicodeCharToUTF8(c);
    if (!atomic_piece_strings_.contains(atom)) return false;
    symbols.push_back(std::move(atom));
  }

  size_t guard = 0;
  while (symbols.size() > 1 && guard++ <= merges.size() + piece.size() + 1) {
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
  int unreachable = 0;
  for (const auto& piece : learned_pieces_) {
    if (!IsReachable(piece.piece(), effective)) {
      ++unreachable;
      LOG(ERROR) << "unreachable learned BPE piece id=" << piece.external_id()
                 << " piece=" << piece.piece();
    }
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
  result.set_unreachable_pieces(unreachable);
  result.set_rank_prepend(expansion_spec_.allow_rank_prepend());
  result.set_vocab_sha256(expansion_spec_.vocab_sha256());
  result.set_merges_sha256(expansion_spec_.merges_sha256());
  result.set_tokenizer_sha256(expansion_spec_.tokenizer_sha256());
  result.set_pretokenizer_sha256(expansion_spec_.pretokenizer_sha256());
  result.set_boundary_policy(expansion_spec_.boundary_policy());

  const std::string result_path = !trainer_spec_.expansion_result().empty()
                                      ? trainer_spec_.expansion_result()
                                      : trainer_spec_.model_prefix() + ".expansion";
  if (!result_path.empty()) {
    ABSL_RETURN_IF_ERROR(
        continuation::WriteExpansionResult(result_path, result));
  }

  if (unreachable != 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "BPE expansion failed reachability verification: ", unreachable,
        " learned pieces are unreachable in the serialized final merge table"));
  }

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
  base_pieces_.clear();
  bootstrap_pieces_.clear();
  learned_pieces_.clear();
  base_merges_.clear();
  bootstrap_merges_.clear();
  learned_merges_.clear();
  sentences_.clear();
  final_pieces_.clear();

  ABSL_RETURN_IF_ERROR(LoadAndValidateSpec());
  ABSL_RETURN_IF_ERROR(continuation::LoadPreparedCorpus(
      trainer_spec_, normalizer_spec_, components_, &corpus_));
  ABSL_RETURN_IF_ERROR(InitializeCorpusSymbols());

  // Exact inherited state is established before ordinary learning. An
  // inherited merge absent from this domain corpus is retained in the final
  // merge program but naturally has no occurrence to apply here.
  ABSL_RETURN_IF_ERROR(ReplayMerges(base_merges_, "inherited"));
  ABSL_RETURN_IF_ERROR(ReplayMerges(bootstrap_merges_, "bootstrap"));
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
