// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "bpe_continuation_trainer.h"

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"

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
#include "filesystem.h"
#include "ret_check.h"
#include "util.h"

ABSL_DECLARE_FLAG(std::string, continuation_fence_strings);

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

ContinuationTrainer::Symbol* ContinuationTrainer::GetFrozenSymbol(
    absl::string_view piece) {
  const uint64_t fp = absl::HashOf(uint64_t{2}, piece);
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    CHECK_EQ(it->second->ToString(), piece)
        << "hash collision in BPE continuation USER_DEFINED symbols";
    return it->second;
  }
  auto s = std::make_unique<Symbol>();
  s->fp = fp;
  s->chars = string_util::UTF8ToUnicodeText(piece);
  s->frozen = true;
  s->freq = 1;
  Symbol* out = s.get();
  symbols_cache_.emplace(fp, out);
  allocated_.push_back(std::move(s));
  return out;
}

ContinuationTrainer::Symbol* ContinuationTrainer::GetPairSymbol(
    const Symbol* left, const Symbol* right) {
  if (left == nullptr || right == nullptr || left->is_unk || right->is_unk ||
      left->frozen || right->frozen) {
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
  symbol->hierarchy_blocked = false;
  for (auto it = symbol->positions.begin(); it != symbol->positions.end();) {
    const Position pos = DecodePos(*it);
    if (symbol->left != symbols_[pos.sid][pos.left] ||
        symbol->right != symbols_[pos.sid][pos.right]) {
      it = symbol->positions.erase(it);
    } else {
      if (!CanMerge(pos.sid, pos.left, pos.right)) {
        symbol->hierarchy_blocked = true;
      } else {
        symbol->freq += static_cast<uint64_t>(sentences_[pos.sid].second);
      }
      ++it;
    }
  }
  // A context-free BPE rank may only be learned when EVERY occurrence that
  // exists after all earlier ranks is legal.  A blocked occurrence can vanish
  // later when an earlier-rank child-completion merge consumes one operand, so
  // this state is intentionally recomputed rather than made permanent.
  if (symbol->hierarchy_blocked) symbol->freq = 0;
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
  if (fence_group_[sid][left] != fence_group_[sid][right]) return;
  Symbol* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol == nullptr || !symbol->active) return;

  // Keep both legal and currently-blocked occurrences.  A pair is eligible
  // only when ComputeFreq sees that ALL of its current occurrences are legal.
  // This is rank-local safety: an incomplete occurrence may disappear after a
  // lower-rank merge completes the child that contains it.
  symbol->positions.insert(EncodePos(sid, left, right));
  symbol->needs_recomputation = true;
  if (!symbol->pending) {
    symbol->pending = true;
    pending_queue_.push_back(symbol);
  }
}

void ContinuationTrainer::ResetFreq(int sid, int left, int right,
                                    const Symbol* best) {
  if (left == -1 || right == -1) return;
  Symbol* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol == nullptr || symbol == best || !symbol->active) return;
  symbol->needs_recomputation = true;
  // A formerly blocked global pair may become safe precisely because this
  // adjacency is about to disappear. Requeue it even if no new occurrence of
  // that pair is created by the accepted merge.
  if (!symbol->pending) {
    symbol->pending = true;
    pending_queue_.push_back(symbol);
  }
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
    if (!span_end_.empty()) {
      span_end_[pos.sid][pos.left] = span_end_[pos.sid][pos.right];
    }
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

absl::Status ContinuationTrainer::SegmentRecord(
    absl::string_view text, std::vector<Symbol*>* symbols,
    std::vector<int>* fence_groups,
    std::vector<std::pair<size_t, size_t>>* byte_spans) {
  RET_CHECK(symbols != nullptr);
  RET_CHECK(fence_groups != nullptr);
  RET_CHECK(byte_spans != nullptr);
  symbols->clear();
  fence_groups->clear();
  byte_spans->clear();

  // Fence groups per BYTE, from the explicit fence surfaces: every byte
  // boundary is probed and overlapping matches are unioned into one
  // occurrence, exactly as the Unigram fence mask is built.
  std::vector<int> byte_group(text.size(), 0);
  if (fence_matcher_ != nullptr) {
    int group = 0;
    size_t covered_to = 0;   // exclusive end of the occurrence being extended
    for (size_t i = 0; i < text.size();) {
      bool found = false;
      const int len = fence_matcher_->PrefixMatch(text.substr(i), &found);
      if (found) {
        const size_t end = i + static_cast<size_t>(len);
        if (i >= covered_to) ++group;          // a new occurrence
        for (size_t b = i; b < end; ++b) byte_group[b] = group;
        covered_to = std::max(covered_to, end);
        i += static_cast<size_t>(
            std::min<int>(string_util::OneCharLen(text.data() + i),
                          static_cast<int>(text.size() - i)));
      } else {
        i += static_cast<size_t>(len);
      }
    }
  }

  std::vector<std::string> atoms;
  size_t cursor = 0;   // byte offset of the next symbol
  auto push = [&](Symbol* symbol, size_t nbytes) -> absl::Status {
    const int g = byte_group[cursor];
    for (size_t b = cursor; b < cursor + nbytes; ++b) {
      if (byte_group[b] != g) {
        return absl::InvalidArgumentError(absl::StrCat(
            "a fence boundary falls inside the atomic symbol \"",
            symbol->ToString(), "\"; fence strings must align with the "
            "reversible atomic alphabet"));
      }
    }
    symbols->push_back(symbol);
    fence_groups->push_back(g);
    byte_spans->push_back({cursor, cursor + nbytes});
    cursor += nbytes;
    return absl::OkStatus();
  };
  auto flush_run = [&](absl::string_view run) -> absl::Status {
    if (run.empty()) return absl::OkStatus();
    ABSL_RETURN_IF_ERROR(SegmentAtoms(run, &atoms));
    for (const std::string& atom : atoms) {
      ABSL_RETURN_IF_ERROR(push(GetAtomicSymbol(atom), atom.size()));
    }
    return absl::OkStatus();
  };
  if (user_defined_matcher_ == nullptr) return flush_run(text);

  size_t run_begin = 0;
  for (size_t i = 0; i < text.size();) {
    bool found = false;
    const int len = user_defined_matcher_->PrefixMatch(text.substr(i), &found);
    if (!found) {
      i += static_cast<size_t>(len);
      continue;
    }
    ABSL_RETURN_IF_ERROR(flush_run(text.substr(run_begin, i - run_begin)));
    ABSL_RETURN_IF_ERROR(
        push(GetFrozenSymbol(text.substr(i, len)), static_cast<size_t>(len)));
    i += static_cast<size_t>(len);
    run_begin = i;
  }
  return flush_run(text.substr(run_begin));
}

bool ContinuationTrainer::CanMerge(int sid, int left, int right) const {
  if (hierarchy_.empty()) return true;
  if (sid < 0 || sid >= static_cast<int>(hierarchy_.size()) ||
      left < 0 || right < 0) {
    return false;
  }
  const size_t boundary = span_end_[sid][left];
  if (boundary != span_begin_[sid][right]) return false;
  const HierarchyRecord& record = hierarchy_[sid];
  const auto it = record.gate_at_boundary.find(boundary);
  if (it == record.gate_at_boundary.end()) return true;

  const HierarchyGate& gate = record.gates[it->second];
  const size_t begin = span_begin_[sid][left];
  const size_t end = span_end_[sid][right];
  if (begin < gate.begin || end > gate.end) return false;
  return std::binary_search(gate.cuts.begin(), gate.cuts.end(), begin) &&
         std::binary_search(gate.cuts.begin(), gate.cuts.end(), end);
}

int ContinuationTrainer::GrammarLevelForPair(int sid, int left,
                                             int right) const {
  if (hierarchy_.empty() || sid < 0 ||
      sid >= static_cast<int>(hierarchy_.size()) || left < 0 || right < 0) {
    return 0;
  }
  const size_t boundary = span_end_[sid][left];
  const auto it = hierarchy_[sid].gate_at_boundary.find(boundary);
  return it == hierarchy_[sid].gate_at_boundary.end()
             ? 0
             : hierarchy_[sid].gates[it->second].level;
}

int ContinuationTrainer::MaxGrammarLevel(const Symbol* symbol) const {
  if (hierarchy_.empty() || symbol == nullptr) return 0;
  int level = 0;
  for (const uint64_t encoded_pos : symbol->positions) {
    const Position pos = DecodePos(encoded_pos);
    if (symbol->left != symbols_[pos.sid][pos.left] ||
        symbol->right != symbols_[pos.sid][pos.right]) {
      continue;
    }
    level = std::max(level,
                     GrammarLevelForPair(pos.sid, pos.left, pos.right));
  }
  return level;
}

absl::Status ContinuationTrainer::LoadHierarchy() {
  hierarchy_.clear();
  hierarchy_sha256_.clear();
  const std::string& filename = trainer_spec_.bpe_hierarchy_file();
  if (filename.empty()) return absl::OkStatus();
  if (fence_matcher_ != nullptr || !fence_surfaces_.empty()) {
    return absl::InvalidArgumentError(
        "bpe_hierarchy_file and continuation_fence_strings are mutually "
        "exclusive boundary policies");
  }

  auto input = filesystem::NewReadableFile(filename);
  if (!input->status().ok()) return input->status();

  std::string canonical;
  std::string line;
  if (!input->ReadLine(&line) ||
      line != "# sentencepiece-bpe-hierarchy-v1") {
    return absl::InvalidArgumentError(
        "BPE hierarchy sidecar must start with "
        "'# sentencepiece-bpe-hierarchy-v1'");
  }
  absl::StrAppend(&canonical, line, "\n");

  absl::flat_hash_map<std::string, int> sid_of;
  sid_of.reserve(corpus_.sentences.size());
  for (size_t sid = 0; sid < corpus_.sentences.size(); ++sid) {
    sid_of[corpus_.sentences[sid].first] = static_cast<int>(sid);
  }

  hierarchy_.resize(corpus_.sentences.size());
  std::vector<bool> seen(corpus_.sentences.size(), false);
  while (input->ReadLine(&line)) {
    absl::StrAppend(&canonical, line, "\n");
    if (line.empty() || line[0] == '#') continue;
    const std::vector<std::string> fields = absl::StrSplit(line, '\t');
    if (fields.size() != 2) {
      return absl::InvalidArgumentError(
          "BPE hierarchy row must be <normalized-text><tab><gate-spec>");
    }
    const auto sid_it = sid_of.find(fields[0]);
    if (sid_it == sid_of.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "BPE hierarchy contains a row absent from the normalized corpus: ",
          fields[0]));
    }
    const int sid = sid_it->second;
    if (seen[sid]) {
      return absl::InvalidArgumentError(
          absl::StrCat("duplicate BPE hierarchy row: ", fields[0]));
    }
    seen[sid] = true;
    HierarchyRecord& record = hierarchy_[sid];

    if (!fields[1].empty()) {
      for (absl::string_view gate_text : absl::StrSplit(fields[1], ';')) {
        if (gate_text.empty()) continue;
        const size_t colon = gate_text.find(':');
        if (colon == absl::string_view::npos) {
          return absl::InvalidArgumentError(
              "hierarchy gate must be <level>:<cut0>,<cut1>,...,<cutN>");
        }
        int level = 0;
        if (!absl::SimpleAtoi(gate_text.substr(0, colon), &level) ||
            level <= 0) {
          return absl::InvalidArgumentError(
              "hierarchy gate level must be a positive integer");
        }
        HierarchyGate gate;
        gate.level = level;
        for (absl::string_view one :
             absl::StrSplit(gate_text.substr(colon + 1), ',')) {
          size_t cut = 0;
          if (!absl::SimpleAtoi(one, &cut)) {
            return absl::InvalidArgumentError(
                absl::StrCat("invalid hierarchy byte cut: ", one));
          }
          gate.cuts.push_back(cut);
        }
        if (gate.cuts.size() < 3) {
          return absl::InvalidArgumentError(
              "hierarchy parent needs at least two children");
        }
        if (!std::is_sorted(gate.cuts.begin(), gate.cuts.end()) ||
            std::adjacent_find(gate.cuts.begin(), gate.cuts.end()) !=
                gate.cuts.end()) {
          return absl::InvalidArgumentError(
              "hierarchy cuts must be strictly increasing");
        }
        gate.begin = gate.cuts.front();
        gate.end = gate.cuts.back();
        const std::string& text = corpus_.sentences[sid].first;
        if (gate.end > text.size()) {
          return absl::InvalidArgumentError(
              "hierarchy gate extends past normalized text");
        }
        for (size_t cut : gate.cuts) {
          if (cut != 0 && cut != text.size() &&
              (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) {
            return absl::InvalidArgumentError(
                "hierarchy cut falls inside a UTF-8 codepoint");
          }
        }

        const int gate_index = static_cast<int>(record.gates.size());
        for (size_t i = 1; i + 1 < gate.cuts.size(); ++i) {
          if (!record.gate_at_boundary.emplace(gate.cuts[i], gate_index)
                   .second) {
            return absl::InvalidArgumentError(absl::StrCat(
                "two hierarchy parents claim the same child boundary at byte ",
                gate.cuts[i]));
          }
        }
        record.gates.push_back(std::move(gate));
      }
    }

    for (size_t i = 0; i < record.gates.size(); ++i) {
      for (size_t j = i + 1; j < record.gates.size(); ++j) {
        const auto& a = record.gates[i];
        const auto& b = record.gates[j];
        const bool disjoint = a.end <= b.begin || b.end <= a.begin;
        const bool a_contains = a.begin <= b.begin && b.end <= a.end;
        const bool b_contains = b.begin <= a.begin && a.end <= b.end;
        if (!(disjoint || a_contains || b_contains)) {
          return absl::InvalidArgumentError(
              "BPE hierarchy spans are not laminar");
        }
      }
    }
  }
  if (!input->status().ok()) return input->status();
  for (size_t sid = 0; sid < seen.size(); ++sid) {
    if (!seen[sid]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "BPE hierarchy is missing normalized corpus row: ",
          corpus_.sentences[sid].first));
    }
  }
  hierarchy_sha256_ = continuation::Sha256Hex(canonical);
  LOG(INFO) << "Loaded completion-gated BPE hierarchy for "
            << hierarchy_.size() << " records, sha256=" << hierarchy_sha256_;
  return absl::OkStatus();
}

absl::Status ContinuationTrainer::LoadExplicitFences() {
  fence_surfaces_.clear();
  fence_matcher_.reset();
  const std::string spec = absl::GetFlag(FLAGS_continuation_fence_strings);
  if (spec.empty()) return absl::OkStatus();
  // Normalized with the EFFECTIVE (reconciled, inherited) normalizer -- the
  // regime that produced the corpus records -- so "PL:" becomes the surface
  // that actually occurs, e.g. U+2581 "PL:" under add_dummy_prefix.
  normalizer::Normalizer fence_normalizer(normalizer_spec_, trainer_spec_);
  ABSL_RETURN_IF_ERROR(fence_normalizer.status());
  for (const auto& logical : absl::StrSplit(spec, ',')) {
    const std::string one(logical);
    if (one.empty()) {
      return absl::InvalidArgumentError(
          "--continuation_fence_strings contains an empty entry");
    }
    const std::string surface = fence_normalizer.Normalize(one);
    if (surface.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "continuation fence string ", one,
          " normalizes to the empty string"));
    }
    const bool inserted = fence_surfaces_.insert(surface).second;
    LOG(INFO) << "FENCE logical=\"" << one << "\" normalized=\"" << surface
              << "\"" << (inserted ? "" : " (duplicate, deduplicated)");
  }
  std::set<absl::string_view> views;
  for (const auto& sfc : fence_surfaces_) views.insert(sfc);
  fence_matcher_ = std::make_unique<normalizer::PrefixMatcher>(views);
  LOG(INFO) << "FENCE normalized_unique=" << fence_surfaces_.size()
            << " matcher=active (BPE continuation)";
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
  if (trainer_spec_.GetExtension(::sentencepiece::split_by_interval) ||
      trainer_spec_.GetExtension(::sentencepiece::split_by_barline)) {
    return absl::InvalidArgumentError(
        "split_by_interval/split_by_barline are legacy fresh-training "
        "pretokenization shims; continuation carries its domain fence in "
        "ExpansionSpec.boundary_policy and one input record per training unit");
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
    if (piece.type() == ModelProto::SentencePiece::USER_DEFINED) {
      // Native semantics: prefix-matched before any merge, frozen on both
      // sides. A spec that declares one mergeable or atomic is asking for a
      // tokenizer native inference cannot be, so it is refused, not corrected.
      if (piece.mergeable() || piece.atomic()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "USER_DEFINED piece must be declared nonmergeable and nonatomic: ",
            piece.piece()));
      }
      user_defined_piece_strings_.insert(piece.piece());
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
  if (!user_defined_piece_strings_.empty()) {
    std::set<absl::string_view> views(user_defined_piece_strings_.begin(),
                                      user_defined_piece_strings_.end());
    user_defined_matcher_ = std::make_unique<normalizer::PrefixMatcher>(views);
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
    if (next_external_id_ == std::numeric_limits<int>::max()) {
      return absl::OutOfRangeError(
          "bootstrap allocation exhausted the external ID range");
    }
    piece.set_external_id(next_external_id_++);
    if (!occupied_ids.insert(piece.external_id()).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("bootstrap external ID collision: ", piece.external_id()));
    }
    bootstrap_ids[piece.piece()] = piece.external_id();
  }

  // Budget is the number of pieces the run may LEARN. Base and bootstrap
  // pieces are inherited state and never spend it. An explicit zero is a
  // replay-only run and must stay distinguishable from "unset", so the request
  // is read by field presence rather than by value.
  if (expansion_spec_.has_requested_new_pieces()) {
    target_new_pieces_ = expansion_spec_.requested_new_pieces();
    if (target_new_pieces_ < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "requested_new_pieces must be nonnegative, got ",
          target_new_pieces_));
    }
  } else {
    // Unset: fall back to filling the declared final vocabulary. vocab_size is
    // a total over the whole external ID space, so the already-occupied range
    // is subtracted rather than added to.
    target_new_pieces_ = trainer_spec_.vocab_size() - next_external_id_;
    if (target_new_pieces_ < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "vocab_size ", trainer_spec_.vocab_size(),
          " is smaller than the occupied external ID range after bootstrap "
          "allocation (", next_external_id_, "); set requested_new_pieces "
          "explicitly to make a replay-only run intentional"));
    }
  }

  // 5.5: every ID this run will hand out must exist before any of them is
  // handed out. next_external_id_ is a plain int and must never wrap.
  if (target_new_pieces_ >
      std::numeric_limits<int>::max() - next_external_id_) {
    return absl::OutOfRangeError(absl::StrCat(
        "continuation would allocate external IDs past the representable "
        "range: first learned ID ", next_external_id_, " plus ",
        target_new_pieces_, " learned pieces"));
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
    // The adapter may name the child it believes this merge builds. When it
    // does, disagreeing with the piece table is an adapter bug, and rewriting
    // the field would hide it: provenance that silently agrees with whatever
    // it is compared against proves nothing.
    if (merge.external_id() >= 0 &&
        merge.external_id() != child_it->second.external_id()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "inherited merge at rank ", merge.rank(), " claims to build ",
          child, " as external_id ", merge.external_id(),
          " but the piece table gives it ", child_it->second.external_id()));
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
    if (merge.external_id() >= 0 && merge.external_id() != child_it->second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "bootstrap merge at rank ", merge.rank(), " claims to build ", child,
          " as external_id ", merge.external_id(),
          " but the bootstrap piece was allocated ", child_it->second));
    }
    merge.set_external_id(child_it->second);
  }

  // Shape options (max_sentencepiece_length, split_by_whitespace,
  // split_by_unicode_script, split_by_number, split_digits) are fresh-training
  // heuristics about what a NEW piece may look like. They are applied to merge
  // candidates through IsValidSentencePiece, which means a contradictory
  // setting would make an inherited merge unbuildable. Inherited state is not
  // negotiable, so a configuration that cannot even express it is rejected
  // here, before the corpus is read and before any merge is replayed.
  {
    auto check_declared = [&](const ExpansionPiece& piece,
                              absl::string_view label) -> absl::Status {
      if (piece.type() != ModelProto::SentencePiece::NORMAL ||
          !piece.mergeable() || piece.atomic()) {
        // Only constructed pieces travel through the merge machinery.
        return absl::OkStatus();
      }
      const string_util::UnicodeText chars =
          string_util::UTF8ToUnicodeText(piece.piece());
      if (IsValidSentencePiece(chars)) return absl::OkStatus();
      return absl::FailedPreconditionError(absl::StrCat(
          "trainer shape constraints cannot express the ", label,
          " piece \"", piece.piece(), "\" (", chars.size(),
          " characters); continuation must not silently drop inherited state. "
          "Check max_sentencepiece_length=",
          trainer_spec_.max_sentencepiece_length(),
          ", split_by_whitespace=", trainer_spec_.split_by_whitespace(),
          ", split_by_unicode_script=",
          trainer_spec_.split_by_unicode_script(),
          ", split_by_number=", trainer_spec_.split_by_number(),
          ", split_digits=", trainer_spec_.split_digits()));
    };
    for (const auto& piece : base_pieces_) {
      ABSL_RETURN_IF_ERROR(check_declared(piece, "inherited"));
    }
    for (const auto& piece : bootstrap_pieces_) {
      ABSL_RETURN_IF_ERROR(check_declared(piece, "bootstrap"));
    }
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
  fence_group_.assign(sentences_.size(), {});
  span_begin_.assign(sentences_.size(), {});
  span_end_.assign(sentences_.size(), {});

  // EncodePos packs the two symbol indexes of a position into 16 bits each.
  // An over-long record is a legitimate input, not a programming error, so it
  // is refused here with a status instead of aborting the process inside
  // EncodePos's CHECK.
  constexpr size_t kMaxAtomsPerRecord = 1u << 16;

  // Every pair frequency is bounded by the total weighted position mass, so
  // proving that sum fits in uint64_t proves no accumulation can wrap. A
  // record contributes atoms*weight positions.
  uint64_t weighted_positions = 0;

  for (size_t sid = 0; sid < sentences_.size(); ++sid) {
    std::vector<Symbol*> record;
    std::vector<std::pair<size_t, size_t>> byte_spans;
    ABSL_RETURN_IF_ERROR(SegmentRecord(sentences_[sid].first, &record,
                                       &fence_group_[sid], &byte_spans));
    if (record.size() > kMaxAtomsPerRecord) {
      return absl::OutOfRangeError(absl::StrCat(
          "continuation training unit segments into ", record.size(),
          " atomic symbols, which exceeds the ", kMaxAtomsPerRecord,
          " this trainer can index; split the record or shrink "
          "max_sentence_length"));
    }
    const uint64_t weight = static_cast<uint64_t>(sentences_[sid].second);
    if (weight != 0 &&
        record.size() > (std::numeric_limits<uint64_t>::max() -
                         weighted_positions) / weight) {
      return absl::OutOfRangeError(
          "weighted continuation corpus exceeds the exact integer range of "
          "BPE pair frequencies; reduce the TSV counts");
    }
    weighted_positions += static_cast<uint64_t>(record.size()) * weight;

    absl::flat_hash_set<size_t> atomic_boundaries;
    atomic_boundaries.insert(0);
    atomic_boundaries.insert(sentences_[sid].first.size());
    for (const auto& span : byte_spans) {
      atomic_boundaries.insert(span.first);
      atomic_boundaries.insert(span.second);
      span_begin_[sid].push_back(span.first);
      span_end_[sid].push_back(span.second);
    }
    if (!hierarchy_.empty()) {
      for (const HierarchyGate& gate : hierarchy_[sid].gates) {
        for (size_t cut : gate.cuts) {
          if (!atomic_boundaries.contains(cut)) {
            return absl::InvalidArgumentError(absl::StrCat(
                "hierarchy cut at byte ", cut,
                " falls inside an atomic/USER_DEFINED symbol in record: ",
                sentences_[sid].first));
          }
        }
      }
    }

    for (Symbol* symbol : record) {
      symbols_[sid].push_back(symbol);
      // A frozen USER_DEFINED unit is not addressable by a merge, so it is
      // deliberately absent from live_by_string_.
      if (!symbol->frozen) live_by_string_[symbol->ToString()] = symbol;
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
    if (symbol->hierarchy_blocked) {
      return absl::FailedPreconditionError(absl::StrCat(
          label, " merge is context-dependent under the completion hierarchy "
          "at effective rank ", merge.rank(), ": ", merge.left(), " + ",
          merge.right()));
    }
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
    RET_CHECK(!best->hierarchy_blocked)
        << "blocked hierarchy pair reached the positive-frequency queue";
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
    merge.set_weighted_count(best->freq);
    merge.set_grammar_level(MaxGrammarLevel(best));

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

ContinuationTrainer::PairRanks ContinuationTrainer::BuildPairRanks(
    const std::vector<ExpansionMerge>& merges) {
  PairRanks pair_rank;
  pair_rank.reserve(merges.size());
  for (const auto& merge : merges) {
    pair_rank[{merge.left(), merge.right()}] = merge.rank();
  }
  return pair_rank;
}

bool ContinuationTrainer::IsReachable(absl::string_view piece,
                                      const PairRanks& pair_rank) const {
  std::vector<std::string> symbols;
  if (!SegmentAtoms(piece, &symbols).ok()) return false;

  size_t guard = 0;
  while (symbols.size() > 1 &&
         guard++ <= pair_rank.size() + symbols.size() + 1) {
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

absl::Status ContinuationTrainer::VerifyNativeMergeEquivalence(
    const std::vector<ExpansionPiece>& pieces,
    const std::vector<ExpansionMerge>& merges) const {
  // Native inference merges a pair when the CONCATENATION is in the
  // vocabulary. Only NORMAL pieces are eligible: CONTROL sits in the reserved
  // id map, USER_DEFINED is prefix-matched and frozen, and neither can be a
  // merge participant. A non-mergeable NORMAL piece has no such protection, so
  // it would silently join merges the program never declared.
  absl::flat_hash_set<std::string> normal;
  for (const auto& piece : pieces) {
    const bool is_normal = piece.type() == ModelProto::SentencePiece::NORMAL;
    if (is_normal && !piece.mergeable()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "piece is NORMAL but declared nonmergeable, which native BPE cannot "
          "express: ",
          piece.piece()));
    }
    if (is_normal) normal.insert(piece.piece());
  }

  // Rank is carried as a float32 score. Beyond 2^24 consecutive integers stop
  // being distinguishable, and two merges sharing a score is an ordering the
  // program did not ask for.
  constexpr int kMaxExactFloatRank = 1 << 24;
  if (static_cast<int64_t>(merges.size()) > kMaxExactFloatRank) {
    return absl::FailedPreconditionError(absl::StrCat(
        "merge program has ", merges.size(),
        " ranks, which float32 piece scores cannot order exactly (limit ",
        kMaxExactFloatRank, ")"));
  }

  absl::flat_hash_map<std::string, std::pair<std::string, std::string>> recorded;
  for (const auto& merge : merges) {
    recorded[merge.left() + merge.right()] = {merge.left(), merge.right()};
  }

  // Native inference starts from a longest-prefix match over USER_DEFINED
  // pieces and falls back to single characters. A multi-codepoint atom is
  // therefore never produced as a starting unit, and no merge may build it.
  for (const std::string& atom : atomic_pieces_ordered_) {
    if (string_util::UTF8Len(atom) != 1) {
      return absl::FailedPreconditionError(absl::StrCat(
          "atomic alphabet symbol spans multiple Unicode scalars, which native "
          "BPE cannot reconstruct: ",
          atom));
    }
  }

  // The decisive condition. An explicit program merges the pair (l, r); native
  // inference merges any adjacent pair whose concatenation is in the
  // vocabulary. They agree only when each piece admits a single such split.
  for (const auto& piece : pieces) {
    if (piece.type() != ModelProto::SentencePiece::NORMAL) continue;
    const std::string& child = piece.piece();
    const auto recorded_it = recorded.find(child);
    int splits = 0;
    std::string offending_left;
    for (size_t cut = 1; cut < child.size(); ++cut) {
      // Only cut on a UTF-8 character boundary: a piece split mid-sequence is
      // not structurally valid and can never be in the vocabulary.
      if ((static_cast<unsigned char>(child[cut]) & 0xC0) == 0x80) continue;
      const std::string left = child.substr(0, cut);
      const std::string right = child.substr(cut);
      if (!normal.contains(left) || !normal.contains(right)) continue;
      ++splits;
      if (recorded_it == recorded.end() ||
          recorded_it->second != std::make_pair(left, right)) {
        offending_left = left;
      }
    }
    if (recorded_it == recorded.end()) {
      // An atom, or any piece the program never constructs. Native inference
      // must have no way to build it either.
      if (splits != 0) {
        return absl::FailedPreconditionError(absl::StrCat(
            "piece has no merge in the program but native BPE could build it "
            "from vocabulary pieces (\"",
            offending_left, "\" + ...): ", child));
      }
      continue;
    }
    if (splits != 1) {
      return absl::FailedPreconditionError(absl::StrCat(
          "piece admits ", splits,
          " vocabulary splits, so native BPE may merge a pair the program does "
          "not declare (e.g. \"",
          offending_left, "\" + ...): ", child));
    }
  }

  return absl::OkStatus();
}

// RECONCILE THE INHERITED TEXT PIPELINE.
//
// Unigram continuation has always treated its prior model's normalizer as
// authoritative (unigram_continuation_trainer.cc, ReconcileNormalization).
// BPE continuation did not, and passed the CALLER's spec straight to the
// corpus loader. A run that omitted --normalization_rule_name therefore
// trained under the CLI default nmt_nfkc while its own base had been built
// under identity -- an internally inconsistent pipeline that no log line
// flagged. Measured on the music corpus the two agreed on 200k/200k records,
// so nothing shipped was wrong; that was luck, not a contract.
//
// Rule, identical to Unigram's:
//   caller omitted / left at default -> inherit the base normalizer
//   caller equals the base           -> accept
//   caller conflicts explicitly      -> fail BEFORE training
absl::Status ContinuationTrainer::ReconcileContinuationContract() {
  if (!expansion_spec_.has_contract()) {
    // Pre-contract artifact: nothing authoritative to inherit. Keep the
    // caller's pipeline and say so in the effective contract.
    return absl::OkStatus();
  }
  const ContinuationContract& base = expansion_spec_.contract();
  if (base.has_normalizer_spec()) {
    const NormalizerSpec& want = base.normalizer_spec();
    NormalizerSpec have = normalizer_spec_;
    const bool caller_is_default =
        have.name().empty() || have.name() == "nmt_nfkc";
    if (caller_is_default) {
      normalizer_spec_ = want;
    } else {
      NormalizerSpec a = want, b = have;
      a.clear_precompiled_charsmap();
      b.clear_precompiled_charsmap();
      if (a.SerializeAsString() != b.SerializeAsString()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "BPE continuation normalizer conflicts with the inherited "
            "tokenizer: base name=", want.name(),
            " add_dummy_prefix=", want.add_dummy_prefix(),
            " remove_extra_whitespaces=", want.remove_extra_whitespaces(),
            " escape_whitespaces=", want.escape_whitespaces(),
            " but caller asked for name=", have.name(),
            " add_dummy_prefix=", have.add_dummy_prefix(),
            " remove_extra_whitespaces=", have.remove_extra_whitespaces(),
            " escape_whitespaces=", have.escape_whitespaces(),
            ". The base tokenizer's text pipeline is authoritative; omit the "
            "normalizer flags to inherit it, or pass exactly the same ones."));
      }
      normalizer_spec_ = want;
    }
  }
  if (base.has_denormalizer_spec() &&
      denormalizer_spec_.normalization_rule_tsv().empty()) {
    denormalizer_spec_ = base.denormalizer_spec();
  }
  return absl::OkStatus();
}

// The EFFECTIVE contract: what this run actually trained under, after
// reconciliation. Written into ExpansionResult so the artifact answers
// "which normalizer / which split_* / which special tokens" without a log.
void ContinuationTrainer::FillEffectiveContract(ContinuationContract* out) const {
  if (expansion_spec_.has_contract()) *out = expansion_spec_.contract();
  *out->mutable_normalizer_spec() = normalizer_spec_;
  if (!denormalizer_spec_.normalization_rule_tsv().empty()) {
    *out->mutable_denormalizer_spec() = denormalizer_spec_;
  }
  out->set_max_sentencepiece_length(trainer_spec_.max_sentencepiece_length());
  out->set_split_by_whitespace(trainer_spec_.split_by_whitespace());
  out->set_split_by_unicode_script(trainer_spec_.split_by_unicode_script());
  out->set_split_by_number(trainer_spec_.split_by_number());
  out->set_split_digits(trainer_spec_.split_digits());
  out->set_treat_whitespace_as_suffix(trainer_spec_.treat_whitespace_as_suffix());
  out->set_allow_whitespace_only_pieces(
      trainer_spec_.allow_whitespace_only_pieces());
  out->set_hard_vocab_limit(trainer_spec_.hard_vocab_limit());
  out->set_input_format(trainer_spec_.input_format());
  out->set_input_sentence_size(
      static_cast<int32_t>(trainer_spec_.input_sentence_size()));
  out->set_character_coverage(trainer_spec_.character_coverage());
  out->set_corpus_records(static_cast<int64_t>(corpus_.sentences.size()));
  out->set_corpus_weight(corpus_.weighted_sentence_count);
  // Special-token ABI is INHERITED, never taken from this run's CLI. If the
  // base carried none, derive what the piece table actually shows.
  if (!expansion_spec_.has_contract() ||
      !expansion_spec_.contract().has_unk_id()) {
    int unk_id = -1;
    for (const auto& p : base_pieces_) {
      if (p.type() == ModelProto::SentencePiece::UNKNOWN) unk_id = p.external_id();
    }
    out->set_unk_id(unk_id);
    out->set_bos_id(-1);
    out->set_eos_id(-1);
    out->set_pad_id(-1);
  }
  out->set_base_id_map_sha256(BaseIdMapSha256());
  if (!hierarchy_sha256_.empty()) {
    out->set_bpe_hierarchy_sha256(hierarchy_sha256_);
  }
}

// Identity of the inherited ordered piece table: "<id>\t<piece>\t<type>" per
// line. Two tokenizers with the same size and different meanings differ here.
std::string ContinuationTrainer::BaseIdMapSha256() const {
  std::vector<ExpansionPiece> pieces = base_pieces_;
  pieces.insert(pieces.end(), bootstrap_pieces_.begin(), bootstrap_pieces_.end());
  std::sort(pieces.begin(), pieces.end(),
            [](const ExpansionPiece& a, const ExpansionPiece& b) {
              return a.external_id() < b.external_id();
            });
  std::string blob;
  for (const auto& p : pieces) {
    absl::StrAppend(&blob, p.external_id(), "\t", p.piece(), "\t",
                    static_cast<int>(p.type()), "\n");
  }
  return continuation::Sha256Hex(blob);
}

absl::Status ContinuationTrainer::BuildNativeModel(
    const std::vector<ExpansionMerge>& merges, ModelProto* model) const {
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

  ABSL_RETURN_IF_ERROR(VerifyNativeMergeEquivalence(pieces, merges));

  // Scores are the merge program, re-expressed in the only ordering native
  // inference reads. score = -rank makes "best score" and "lowest rank" the
  // same relation, and the adapter's own scores are deliberately discarded:
  // they order nothing here and keeping them would order the wrong thing.
  absl::flat_hash_map<std::string, int> rank_of;
  for (const auto& merge : merges) {
    rank_of[merge.left() + merge.right()] = merge.rank();
  }

  model->Clear();
  for (const auto& piece : pieces) {
    auto* out = model->add_pieces();
    out->set_piece(piece.piece());
    out->set_type(piece.type());
    if (piece.type() != ModelProto::SentencePiece::NORMAL) {
      out->set_score(0.0);
      continue;
    }
    const auto it = rank_of.find(piece.piece());
    // Atoms are never merge children and never compete for a merge, so they
    // sit above every constructed piece, as they do in a natively trained BPE
    // model.
    out->set_score(it == rank_of.end() ? 0.0F
                                       : -static_cast<float>(it->second));
  }
  // SPECIAL-TOKEN METADATA IS INHERITED, NOT COPIED FROM THIS RUN'S CLI.
  //
  // Copying trainer_spec_ wholesale wrote the CLI defaults bos_id=1/eos_id=2
  // into the emitted model. On a music tokenizer those IDs are ordinary
  // inherited pieces, so the .model would have declared two real pieces to be
  // BOS and EOS -- a semantically false artifact that loads without complaint.
  // Derive the ABI from the piece table and the inherited contract instead,
  // and refuse to emit rather than write something untrue.
  TrainerSpec spec = trainer_spec_;
  spec.set_vocab_size(model->pieces_size());
  int unk_id = -1;
  for (int i = 0; i < model->pieces_size(); ++i) {
    if (model->pieces(i).type() == ModelProto::SentencePiece::UNKNOWN) unk_id = i;
  }
  if (unk_id < 0) {
    return absl::FailedPreconditionError(
        "native model has no UNKNOWN piece; refusing to emit");
  }
  spec.set_unk_id(unk_id);

  auto declared = [&](int id, ModelProto::SentencePiece::Type want,
                      const char* what) -> absl::Status {
    if (id < 0) return absl::OkStatus();
    if (id >= model->pieces_size() || model->pieces(id).type() != want) {
      return absl::FailedPreconditionError(absl::StrCat(
          "inherited contract declares ", what, "=", id,
          " but that ID is not a CONTROL piece in the inherited table; "
          "refusing to emit a native model whose special-token metadata "
          "contradicts its own pieces"));
    }
    return absl::OkStatus();
  };
  int bos = -1, eos = -1, pad = -1;
  if (expansion_spec_.has_contract()) {
    const ContinuationContract& c = expansion_spec_.contract();
    bos = c.bos_id(); eos = c.eos_id(); pad = c.pad_id();
    if (c.has_unk_id() && c.unk_id() != unk_id) {
      return absl::FailedPreconditionError(absl::StrCat(
          "inherited contract declares unk_id=", c.unk_id(),
          " but the piece table's UNKNOWN is at ", unk_id));
    }
    if (c.has_unk_piece()) spec.set_unk_piece(c.unk_piece());
    if (c.has_unk_surface()) spec.set_unk_surface(c.unk_surface());
    if (c.has_byte_fallback()) spec.set_byte_fallback(c.byte_fallback());
  }
  ABSL_RETURN_IF_ERROR(declared(bos, ModelProto::SentencePiece::CONTROL, "bos_id"));
  ABSL_RETURN_IF_ERROR(declared(eos, ModelProto::SentencePiece::CONTROL, "eos_id"));
  ABSL_RETURN_IF_ERROR(declared(pad, ModelProto::SentencePiece::CONTROL, "pad_id"));
  spec.set_bos_id(bos);
  spec.set_eos_id(eos);
  spec.set_pad_id(pad);

  *model->mutable_trainer_spec() = spec;
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
  const PairRanks pair_rank = BuildPairRanks(effective);
  for (const auto& piece : learned_pieces_) {
    if (!IsReachable(piece.piece(), pair_rank)) {
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
  if (!hierarchy_sha256_.empty()) {
    result.set_boundary_policy(
        absl::StrCat("bpe_hierarchical_completion_v1:", hierarchy_sha256_));
  } else if (!fence_surfaces_.empty()) {
    result.set_boundary_policy(EncodeBpeBoundaryPolicy(fence_surfaces_));
  } else {
    result.set_boundary_policy(expansion_spec_.boundary_policy());
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

  const std::string result_path = !trainer_spec_.expansion_result().empty()
                                      ? trainer_spec_.expansion_result()
                                      : trainer_spec_.model_prefix() + ".expansion";
  if (!result_path.empty()) {
    // The success sidecar is deliberately written only after full-program and
    // learned-piece reachability validation has passed.
    ABSL_RETURN_IF_ERROR(
        continuation::WriteExpansionResult(result_path, result));
  }

  FillEffectiveContract(result.mutable_contract());

  ModelProto native;
  absl::Status native_status = BuildNativeModel(effective, &native);
  result.set_native_model_emitted(native_status.ok());
  if (!native_status.ok()) {
    result.set_native_model_refusal(std::string(native_status.message()));
  }
  if (!result_path.empty()) {
    // Rewrite the sidecar now that native status is known, so a reader never
    // sees an artifact that claims nothing about native representability.
    ABSL_RETURN_IF_ERROR(
        continuation::WriteExpansionResult(result_path, result));
  }
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

// bpe_explicit_fences_v1:[<e1>,<e2>]  e_i sorted by bytes, %XX-escaped outside
// [A-Za-z0-9._-]: the same encoding as unigram_explicit_fences_v1, under a
// prefix that names the trainer, so a reader knows which semantics applied.
std::string EncodeBpeBoundaryPolicy(
    const std::set<std::string>& normalized_fence_surfaces) {
  std::string out = "bpe_explicit_fences_v1:[";
  bool first = true;
  for (const auto& surface : normalized_fence_surfaces) {
    if (!first) out += ",";
    first = false;
    for (const unsigned char c : surface) {
      const bool safe = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                        c == '-';
      if (safe) {
        out += static_cast<char>(c);
      } else {
        static constexpr char kHex[] = "0123456789ABCDEF";
        out += '%';
        out += kHex[c >> 4];
        out += kHex[c & 0x0F];
      }
    }
  }
  out += "]";
  return out;
}

absl::Status ContinuationTrainer::Train() {
  ABSL_RETURN_IF_ERROR(status());
  RET_CHECK_EQ(TrainerSpec::BPE, trainer_spec_.model_type());

  existing_piece_strings_.clear();
  atomic_piece_strings_.clear();
  atomic_pieces_ordered_.clear();
  user_defined_matcher_.reset();
  user_defined_piece_strings_.clear();
  fence_surfaces_.clear();
  fence_matcher_.reset();
  fence_group_.clear();
  hierarchy_.clear();
  span_begin_.clear();
  span_end_.clear();
  hierarchy_sha256_.clear();
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
  ABSL_RETURN_IF_ERROR(ReconcileContinuationContract());
  ABSL_RETURN_IF_ERROR(LoadExplicitFences());
  ABSL_RETURN_IF_ERROR(continuation::LoadPreparedCorpus(
      trainer_spec_, normalizer_spec_, components_, &corpus_));
  ABSL_RETURN_IF_ERROR(LoadHierarchy());
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
