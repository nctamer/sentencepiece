// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "expansion_processor.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>

#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "continuation_io.h"
#include "filesystem.h"
#include "util.h"

namespace sentencepiece::expansion {
namespace {
constexpr char kSep = '\x01';   // never occurs in a piece; pairs join on it
std::string Key(absl::string_view l, absl::string_view r) {
  return absl::StrCat(l, std::string(1, kSep), r);
}
}  // namespace

absl::Status ExpansionProcessor::Load(const ExpansionResult& result) {
  id_to_piece_.clear(); id_to_type_.clear();
  piece_to_id_.clear(); merge_rules_.clear();
  user_defined_matcher_.reset();
  requires_hierarchy_ = false;
  has_inherited_merge_program_ = false;

  std::vector<ExpansionPiece> pieces;
  for (const auto& p : result.base_pieces()) pieces.push_back(p);
  for (const auto& p : result.bootstrap_pieces()) pieces.push_back(p);
  for (const auto& p : result.learned_pieces()) pieces.push_back(p);
  std::sort(pieces.begin(), pieces.end(),
            [](const ExpansionPiece& a, const ExpansionPiece& b) {
              return a.external_id() < b.external_id();
            });
  if (pieces.empty()) {
    status_ = absl::InvalidArgumentError("ExpansionResult has no pieces");
    return status_;
  }
  id_to_piece_.resize(pieces.size());
  id_to_type_.resize(pieces.size());
  int unknown_count = 0;
  for (size_t i = 0; i < pieces.size(); ++i) {
    if (pieces[i].external_id() != static_cast<int>(i)) {
      status_ = absl::FailedPreconditionError(absl::StrCat(
          "external IDs are not contiguous from zero at index ", i,
          "; this artifact cannot be used as a dense tokenizer"));
      return status_;
    }
    id_to_piece_[i] = pieces[i].piece();
    id_to_type_[i] = static_cast<int>(pieces[i].type());
    piece_to_id_[pieces[i].piece()] = static_cast<int>(i);
    if (pieces[i].type() == ModelProto::SentencePiece::UNKNOWN) {
      ++unknown_count;
      unk_id_ = static_cast<int>(i);
      unk_piece_ = pieces[i].piece();
    }
  }
  if (unknown_count != 1) {
    status_ = absl::FailedPreconditionError(absl::StrCat(
        "expected exactly one UNKNOWN piece, found ", unknown_count));
    return status_;
  }

  // USER_DEFINED pieces are matched before any merge runs, the same role they
  // have in native inference. They are declared as pieces, never as merge
  // children, and the trainer refuses a program that says otherwise; here we
  // only build the matcher.
  {
    std::set<absl::string_view> user_defined;
    for (size_t i = 0; i < pieces.size(); ++i) {
      if (pieces[i].type() != ModelProto::SentencePiece::USER_DEFINED) continue;
      if (id_to_piece_[i].empty()) {
        status_ = absl::FailedPreconditionError(
            absl::StrCat("USER_DEFINED piece ", i, " is empty"));
        return status_;
      }
      user_defined.insert(id_to_piece_[i]);
    }
    if (!user_defined.empty()) {
      user_defined_matcher_ =
          std::make_unique<normalizer::PrefixMatcher>(user_defined);
    }
  }

  const std::string hierarchy_prefix = "bpe_hierarchical_completion_v1:";
  requires_hierarchy_ =
      result.boundary_policy().compare(0, hierarchy_prefix.size(),
                                       hierarchy_prefix) == 0;

  struct RankedMerge {
    ExpansionMerge merge;
    int scope_level = -1;
  };
  std::vector<RankedMerge> merges;
  has_inherited_merge_program_ =
      result.base_merges_size() != 0 || result.bootstrap_merges_size() != 0;

  for (const auto& m : result.base_merges()) {
    if (m.has_scope_level() && m.scope_level() >= 0) {
      status_ = absl::FailedPreconditionError(
          "scoped inherited base merges are unsupported without the inherited "
          "hierarchy provider");
      return status_;
    }
    merges.push_back({m, -1});
  }
  for (const auto& m : result.bootstrap_merges()) {
    if (m.has_scope_level() && m.scope_level() >= 0) {
      status_ = absl::FailedPreconditionError(
          "scoped bootstrap merges are unsupported without their hierarchy "
          "provider");
      return status_;
    }
    merges.push_back({m, -1});
  }
  for (const auto& m : result.learned_merges()) {
    if (requires_hierarchy_) {
      if (!m.has_scope_level() || m.scope_level() < 0) {
        status_ = absl::FailedPreconditionError(
            "hierarchical ExpansionResult contains an unscoped learned merge; "
            "rebuild it with scoped operation semantics");
        return status_;
      }
      merges.push_back({m, m.scope_level()});
    } else {
      if (m.has_scope_level() && m.scope_level() >= 0) {
        status_ = absl::FailedPreconditionError(
            "scoped learned merge appears without a hierarchical "
            "boundary_policy");
        return status_;
      }
      merges.push_back({m, -1});
    }
  }
  std::sort(merges.begin(), merges.end(),
            [](const RankedMerge& a, const RankedMerge& b) {
              return a.merge.rank() < b.merge.rank();
            });
  for (size_t i = 0; i < merges.size(); ++i) {
    const ExpansionMerge& merge = merges[i].merge;
    for (const std::string& side : {merge.left(), merge.right()}) {
      const auto it = piece_to_id_.find(side);
      if (it != piece_to_id_.end() &&
          id_to_type_[it->second] ==
              static_cast<int>(ModelProto::SentencePiece::USER_DEFINED)) {
        status_ = absl::FailedPreconditionError(absl::StrCat(
            "merge at rank ", merge.rank(), " names the USER_DEFINED piece \"",
            side, "\" as a child; USER_DEFINED pieces are frozen units and "
            "never merge participants"));
        return status_;
      }
    }
    auto& rules = merge_rules_[Key(merge.left(), merge.right())];
    for (const MergeRule& existing : rules) {
      if (existing.scope_level == merges[i].scope_level) {
        status_ = absl::FailedPreconditionError(absl::StrCat(
            "duplicate merge operation for pair ", merge.left(), " + ",
            merge.right(), " at scope ", merges[i].scope_level));
        return status_;
      }
    }
    rules.push_back(
        MergeRule{static_cast<int>(i), merges[i].scope_level});
  }

  // THE ARTIFACT'S OWN NORMALIZER IS AUTHORITATIVE. Falling back to a default
  // here would reintroduce exactly the identity-vs-nmt_nfkc mismatch the
  // contract exists to prevent.
  if (result.has_contract() && result.contract().has_normalizer_spec()) {
    normalizer_spec_ = result.contract().normalizer_spec();
  } else {
    status_ = absl::FailedPreconditionError(
        "ExpansionResult carries no normalizer contract; refusing to guess a "
        "text pipeline. Re-seal this artifact with its authoritative "
        "normalizer before using it for inference.");
    return status_;
  }
  normalizer_ = std::make_unique<normalizer::Normalizer>(normalizer_spec_);
  ABSL_RETURN_IF_ERROR(normalizer_->status());
  status_ = absl::OkStatus();
  return status_;
}

absl::Status ExpansionProcessor::LoadFromFile(absl::string_view path) {
  auto input = filesystem::NewReadableFile(path, true);
  ABSL_RETURN_IF_ERROR(input->status());
  std::string blob;
  if (!input->ReadAll(&blob)) {
    return absl::InternalError(absl::StrCat("cannot read ", path));
  }
  ExpansionResult r;
  if (!r.ParseFromString(blob)) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, " is not a serialized ExpansionResult"));
  }
  return Load(r);
}

std::string ExpansionProcessor::Normalize(absl::string_view text) const {
  if (normalizer_ == nullptr) return std::string(text);
  return normalizer_->Normalize(text);
}

const std::string& ExpansionProcessor::IdToPiece(int id) const {
  static const std::string kEmpty;
  if (id < 0 || id >= static_cast<int>(id_to_piece_.size())) return kEmpty;
  return id_to_piece_[id];
}

int ExpansionProcessor::PieceToId(absl::string_view piece) const {
  const auto it = piece_to_id_.find(std::string(piece));
  return it == piece_to_id_.end() ? unk_id_ : it->second;
}

int ExpansionProcessor::IdToType(int id) const {
  if (id < 0 || id >= static_cast<int>(id_to_type_.size())) return -1;
  return id_to_type_[id];
}

bool ExpansionProcessor::IsUserDefined(int id) const {
  return IdToType(id) ==
         static_cast<int>(ModelProto::SentencePiece::USER_DEFINED);
}

absl::Status ExpansionProcessor::Encode(
    absl::string_view text, std::vector<TokenSpan>* out) const {
  if (requires_hierarchy_) {
    return absl::FailedPreconditionError(
        "hierarchical ExpansionResult requires per-input completion gates; "
        "use EncodeWithHierarchy instead of flat Encode");
  }
  return EncodeImpl(text, nullptr, out);
}

absl::Status ExpansionProcessor::EncodeIds(
    absl::string_view text, std::vector<int>* ids) const {
  std::vector<TokenSpan> spans;
  ABSL_RETURN_IF_ERROR(Encode(text, &spans));
  ids->clear();
  ids->reserve(spans.size());
  for (const auto& s : spans) ids->push_back(s.id);
  return absl::OkStatus();
}

absl::Status ExpansionProcessor::EncodeWithHierarchy(
    absl::string_view text, const std::vector<CompletionGate>& gates,
    std::vector<TokenSpan>* out) const {
  return EncodeImpl(text, &gates, out);
}

absl::Status ExpansionProcessor::EncodeIdsWithHierarchy(
    absl::string_view text, const std::vector<CompletionGate>& gates,
    std::vector<int>* ids) const {
  std::vector<TokenSpan> spans;
  ABSL_RETURN_IF_ERROR(EncodeWithHierarchy(text, gates, &spans));
  ids->clear();
  ids->reserve(spans.size());
  for (const auto& s : spans) ids->push_back(s.id);
  return absl::OkStatus();
}

absl::Status ExpansionProcessor::EncodeImpl(
    absl::string_view text, const std::vector<CompletionGate>* gates,
    std::vector<TokenSpan>* out) const {
  ABSL_RETURN_IF_ERROR(status_);
  if (out == nullptr) {
    return absl::InvalidArgumentError("Encode output must not be null");
  }
  out->clear();
  const std::string norm = Normalize(text);

  // Compile the shallow laminar hierarchy once for this input. The hot path
  // performs one boundary lookup plus two binary searches over a parent's cuts.
  absl::flat_hash_map<int, int> gate_at_boundary;
  if (gates != nullptr) {
    for (size_t gi = 0; gi < gates->size(); ++gi) {
      const CompletionGate& gate = (*gates)[gi];
      if (gate.level <= 0 || gate.cuts.size() < 3 ||
          !std::is_sorted(gate.cuts.begin(), gate.cuts.end()) ||
          std::adjacent_find(gate.cuts.begin(), gate.cuts.end()) !=
              gate.cuts.end()) {
        return absl::InvalidArgumentError(
            "completion gate cuts must be strictly increasing with >=2 children");
      }
      if (gate.cuts.front() < 0 ||
          gate.cuts.back() > static_cast<int>(norm.size())) {
        return absl::InvalidArgumentError(
            "completion gate lies outside normalized input");
      }
      for (int cut : gate.cuts) {
        if (cut != 0 && cut != static_cast<int>(norm.size()) &&
            (static_cast<unsigned char>(norm[cut]) & 0xC0) == 0x80) {
          return absl::InvalidArgumentError(
              "completion gate cut falls inside a UTF-8 codepoint");
        }
      }
      for (size_t ci = 1; ci + 1 < gate.cuts.size(); ++ci) {
        if (!gate_at_boundary
                 .emplace(gate.cuts[ci], static_cast<int>(gi))
                 .second) {
          return absl::InvalidArgumentError(
              "two completion parents claim the same child boundary");
        }
      }
    }
    for (size_t i = 0; i < gates->size(); ++i) {
      const auto& a = (*gates)[i].cuts;
      for (size_t j = i + 1; j < gates->size(); ++j) {
        const auto& b = (*gates)[j].cuts;
        const bool disjoint = a.back() <= b.front() || b.back() <= a.front();
        const bool a_contains = a.front() <= b.front() && b.back() <= a.back();
        const bool b_contains = b.front() <= a.front() && a.back() <= b.back();
        if (!(disjoint || a_contains || b_contains)) {
          return absl::InvalidArgumentError(
              "completion hierarchy spans are not laminar");
        }
      }
    }
  } else if (requires_hierarchy_) {
    return absl::FailedPreconditionError(
        "hierarchical ExpansionResult requires per-input completion gates");
  }

  struct Sym {
    std::string s;
    int begin = 0;
    int end = 0;
    int prev = -1;
    int next = -1;
    int version = 0;
    bool frozen = false;
    bool alive = true;
  };
  std::vector<Sym> syms;
  for (size_t i = 0; i < norm.size();) {
    const absl::string_view rest(norm.data() + i, norm.size() - i);
    bool found = false;
    size_t len = 0;
    if (user_defined_matcher_ != nullptr) {
      len = static_cast<size_t>(
          user_defined_matcher_->PrefixMatch(rest, &found));
    }
    if (!found) {
      len = std::min<size_t>(string_util::OneCharLen(norm.data() + i),
                             norm.size() - i);
    }
    const int idx = static_cast<int>(syms.size());
    syms.push_back(
        {norm.substr(i, len), static_cast<int>(i),
         static_cast<int>(i + len), idx - 1, -1, 0, found, true});
    if (idx > 0) syms[idx - 1].next = idx;
    i += len;
  }
  if (syms.empty()) return absl::OkStatus();

  std::vector<bool> gate_enabled(gates == nullptr ? 0 : gates->size(), true);

  // Exact current occurrence scope: 0 ordinary/internal, >0 a completed
  // child boundary, -2 hierarchy-ineligible.
  auto occurrence_scope = [&](const Sym& left, const Sym& right) -> int {
    if (gates == nullptr) return 0;
    const int boundary = left.end;
    if (boundary != right.begin) return -2;
    const auto it = gate_at_boundary.find(boundary);
    if (it == gate_at_boundary.end()) return 0;
    if (!gate_enabled[it->second]) return 0;
    const CompletionGate& gate = (*gates)[it->second];
    if (left.begin < gate.cuts.front() || right.end > gate.cuts.back()) {
      return -2;
    }
    if (!std::binary_search(gate.cuts.begin(), gate.cuts.end(), left.begin) ||
        !std::binary_search(gate.cuts.begin(), gate.cuts.end(), right.end)) {
      return -2;
    }
    return gate.level;
  };

  auto matching_rule = [&](const Sym& left, const Sym& right,
                           bool scoped_phase) -> const MergeRule* {
    const auto it = merge_rules_.find(Key(left.s, right.s));
    if (it == merge_rules_.end()) return nullptr;
    if (!scoped_phase) {
      const MergeRule* best = nullptr;
      for (const MergeRule& rule : it->second) {
        if (rule.scope_level != -1) continue;
        if (best == nullptr || rule.rank < best->rank) best = &rule;
      }
      return best;
    }
    const int scope = occurrence_scope(left, right);
    if (scope < 0) return nullptr;
    const MergeRule* best = nullptr;
    for (const MergeRule& rule : it->second) {
      if (rule.scope_level != scope) continue;
      if (best == nullptr || rule.rank < best->rank) best = &rule;
    }
    return best;
  };

  struct Candidate {
    int rank;
    int scope_level;
    int left;
    int right;
    int left_version;
    int right_version;
  };
  struct Worse {
    bool operator()(const Candidate& a, const Candidate& b) const {
      if (a.rank != b.rank) return a.rank > b.rank;
      return a.left > b.left;
    }
  };

  auto run_phase = [&](bool scoped_phase) {
    std::priority_queue<Candidate, std::vector<Candidate>, Worse> pq;

    auto push_pair = [&](int left, int right) {
      if (left < 0 || right < 0) return;
      const Sym& l = syms[left];
      const Sym& r = syms[right];
      if (!l.alive || !r.alive || l.next != right || r.prev != left ||
          l.frozen || r.frozen) {
        return;
      }
      const MergeRule* rule = matching_rule(l, r, scoped_phase);
      if (rule == nullptr) return;
      pq.push({rule->rank, rule->scope_level, left, right,
               l.version, r.version});
    };

    int cur = 0;
    while (cur >= 0 && cur < static_cast<int>(syms.size()) &&
           !syms[cur].alive) {
      cur = syms[cur].next;
    }
    while (cur >= 0) {
      const int next = syms[cur].next;
      if (next >= 0) push_pair(cur, next);
      cur = next;
    }

    while (!pq.empty()) {
      const Candidate c = pq.top();
      pq.pop();
      Sym& left = syms[c.left];
      Sym& right = syms[c.right];
      if (!left.alive || !right.alive || left.version != c.left_version ||
          right.version != c.right_version || left.next != c.right ||
          right.prev != c.left) {
        continue;
      }
      const MergeRule* rule = matching_rule(left, right, scoped_phase);
      if (rule == nullptr || rule->rank != c.rank ||
          rule->scope_level != c.scope_level) {
        continue;
      }

      const int prev = left.prev;
      const int next = right.next;
      left.s += right.s;
      left.end = right.end;
      ++left.version;
      left.next = next;
      if (next >= 0) syms[next].prev = c.left;
      right.alive = false;
      ++right.version;
      push_pair(prev, c.left);
      push_pair(c.left, next);
    }
  };

  // Inherited/base/bootstrap (and all rules in a flat artifact) are
  // unconditional. A newly introduced hierarchy may not veto them.
  run_phase(false);

  if (requires_hierarchy_ && gates != nullptr) {
    for (size_t gi = 0; gi < gates->size(); ++gi) {
      const CompletionGate& gate = (*gates)[gi];
      int cur = 0;
      while (cur >= 0 && cur < static_cast<int>(syms.size()) &&
             !syms[cur].alive) {
        cur = syms[cur].next;
      }
      while (cur >= 0) {
        const Sym& sym = syms[cur];
        auto cut = std::upper_bound(gate.cuts.begin(), gate.cuts.end(),
                                    sym.begin);
        const bool crosses_internal =
            cut != gate.cuts.end() && *cut < sym.end &&
            *cut < gate.cuts.back();
        if (crosses_internal) {
          const bool begin_is_cut =
              std::binary_search(gate.cuts.begin(), gate.cuts.end(),
                                 sym.begin);
          const bool end_is_cut =
              std::binary_search(gate.cuts.begin(), gate.cuts.end(), sym.end);
          if (!(begin_is_cut && end_is_cut)) {
            if (!has_inherited_merge_program_) {
              return absl::FailedPreconditionError(
                  "fresh hierarchical artifact has an initial token that "
                  "partially crosses a grammar child boundary; refusing to "
                  "disable a gate without an inherited tokenizer to preserve");
            }
            gate_enabled[gi] = false;
            break;
          }
        }
        cur = sym.next;
      }
    }
  }

  // Appended hierarchical rules are exact-scope operations.
  run_phase(true);

  int at = 0;
  while (at >= 0 && at < static_cast<int>(syms.size()) &&
         !syms[at].alive) {
    at = syms[at].next;
  }
  while (at >= 0) {
    const Sym& sym = syms[at];
    const auto it = piece_to_id_.find(sym.s);
    out->push_back({it == piece_to_id_.end() ? unk_id_ : it->second,
                    sym.s, sym.begin, sym.end});
    at = sym.next;
  }
  return absl::OkStatus();
}

absl::Status ExpansionProcessor::Decode(const std::vector<int>& ids,
                                        std::string* out) const {
  ABSL_RETURN_IF_ERROR(status_);
  std::string joined;
  for (const int id : ids) {
    if (id < 0 || id >= static_cast<int>(id_to_piece_.size())) {
      return absl::OutOfRangeError(absl::StrCat("id ", id, " out of range"));
    }
    // unk_surface is empty under the RNNT contract, so UNKNOWN contributes
    // no bytes. A non-empty surface would make decode non-invertible.
    if (id == unk_id_) continue;
    absl::StrAppend(&joined, id_to_piece_[id]);
  }
  // Invert the whitespace escaping the normalizer applied.
  std::string text = joined;
  if (normalizer_spec_.escape_whitespaces()) {
    const std::string mark = "\xe2\x96\x81";
    std::string tmp;
    tmp.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
      if (text.compare(i, mark.size(), mark) == 0) {
        tmp.push_back(' ');
        i += mark.size();
      } else {
        tmp.push_back(text[i++]);
      }
    }
    text.swap(tmp);
  }
  if (normalizer_spec_.add_dummy_prefix() && !text.empty() && text[0] == ' ') {
    text.erase(text.begin());
  }
  *out = text;
  return absl::OkStatus();
}

std::string ExpansionProcessor::IdMapSha256() const {
  std::string blob;
  for (size_t i = 0; i < id_to_piece_.size(); ++i) {
    absl::StrAppend(&blob, i, "\t", id_to_piece_[i], "\t", id_to_type_[i], "\n");
  }
  return continuation::Sha256Hex(blob);
}

}  // namespace sentencepiece::expansion
