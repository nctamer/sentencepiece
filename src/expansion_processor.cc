// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "expansion_processor.h"

#include <algorithm>
#include <limits>

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
  piece_to_id_.clear(); merge_rank_.clear();

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

  std::vector<ExpansionMerge> merges;
  for (const auto& m : result.base_merges()) merges.push_back(m);
  for (const auto& m : result.bootstrap_merges()) merges.push_back(m);
  for (const auto& m : result.learned_merges()) merges.push_back(m);
  std::sort(merges.begin(), merges.end(),
            [](const ExpansionMerge& a, const ExpansionMerge& b) {
              return a.rank() < b.rank();
            });
  for (size_t i = 0; i < merges.size(); ++i) {
    // Effective rank is position in the sorted program, so a table whose
    // declared ranks are sparse still applies in the intended order.
    merge_rank_.emplace(Key(merges[i].left(), merges[i].right()),
                        static_cast<int>(i));
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

absl::Status ExpansionProcessor::Encode(absl::string_view text,
                                        std::vector<TokenSpan>* out) const {
  ABSL_RETURN_IF_ERROR(status_);
  out->clear();
  const std::string norm = Normalize(text);

  // Atoms: one entry per UTF-8 character, carrying its byte span so the spans
  // survive every merge. Spans are byte offsets into the NORMALIZED text.
  struct Sym { std::string s; int begin; int end; };
  std::vector<Sym> syms;
  for (size_t i = 0; i < norm.size();) {
    const size_t len = std::min<size_t>(
        string_util::OneCharLen(norm.data() + i), norm.size() - i);
    syms.push_back({norm.substr(i, len), static_cast<int>(i),
                    static_cast<int>(i + len)});
    i += len;
  }
  if (syms.empty()) return absl::OkStatus();

  while (syms.size() > 1) {
    int best = std::numeric_limits<int>::max();
    size_t at = syms.size();
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      const auto it = merge_rank_.find(Key(syms[i].s, syms[i + 1].s));
      if (it != merge_rank_.end() && it->second < best) {
        best = it->second;
        at = i;   // leftmost occurrence of the best rank wins
      }
    }
    if (at == syms.size()) break;
    syms[at] = {syms[at].s + syms[at + 1].s, syms[at].begin, syms[at + 1].end};
    syms.erase(syms.begin() + at + 1);
  }

  out->reserve(syms.size());
  for (const auto& s : syms) {
    const auto it = piece_to_id_.find(s.s);
    out->push_back({it == piece_to_id_.end() ? unk_id_ : it->second, s.s,
                    s.begin, s.end});
  }
  return absl::OkStatus();
}

absl::Status ExpansionProcessor::EncodeIds(absl::string_view text,
                                           std::vector<int>* ids) const {
  std::vector<TokenSpan> spans;
  ABSL_RETURN_IF_ERROR(Encode(text, &spans));
  ids->clear();
  ids->reserve(spans.size());
  for (const auto& s : spans) ids->push_back(s.id);
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
