// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "continuation_io.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "filesystem.h"
#include "normalizer.h"
#include "trainer_interface.h"

namespace sentencepiece::continuation {
namespace {

absl::Status ReadAll(absl::string_view filename, std::string* bytes) {
  auto input = filesystem::NewReadableFile(filename, true);
  if (!input->status().ok()) return input->status();
  if (!input->ReadAll(bytes)) {
    return absl::DataLossError(absl::StrCat("failed to read: ", filename));
  }
  return absl::OkStatus();
}

absl::Status WriteAll(absl::string_view filename, absl::string_view bytes) {
  auto output = filesystem::NewWritableFile(filename, true);
  if (!output->status().ok()) return output->status();
  if (!output->Write(bytes)) {
    return absl::DataLossError(absl::StrCat("failed to write: ", filename));
  }
  return absl::OkStatus();
}

uint32_t RotR(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

absl::Status LoadPreparedCorpus(const TrainerSpec& trainer_spec,
                                const NormalizerSpec& normalizer_spec,
                                const TrainerComponents& components,
                                PreparedCorpus* corpus) {
  if (corpus == nullptr) {
    return absl::InvalidArgumentError("PreparedCorpus must not be null");
  }
  corpus->sentences.clear();
  corpus->weighted_sentence_count = 0;

  if (trainer_spec.input_sentence_size() != 0) {
    return absl::InvalidArgumentError(
        "continuation forbids input_sentence_size sampling; provide the fixed "
        "prepared corpus directly");
  }
  if (!trainer_spec.pretokenization_delimiter().empty() ||
      components.pretokenizer != nullptr) {
    return absl::InvalidArgumentError(
        "continuation v1 uses one input record as the hard training fence; "
        "pretokenization_delimiter/callback must not be combined with it");
  }
  if (!(trainer_spec.input_format().empty() ||
        trainer_spec.input_format() == "text" ||
        trainer_spec.input_format() == "tsv")) {
    return absl::InvalidArgumentError(
        "continuation input_format must be text or tsv");
  }

  const bool has_iterator = components.sentence_iterator != nullptr;
  const bool has_files = !trainer_spec.input().empty();
  if (has_iterator == has_files) {
    return absl::InvalidArgumentError(
        "SentenceIterator and trainer_spec.input() must be exclusive");
  }

  std::unique_ptr<SentenceIterator> owned_iterator;
  SentenceIterator* iterator = components.sentence_iterator;
  if (iterator == nullptr) {
    owned_iterator = std::make_unique<MultiFileSentenceIterator>(
        std::vector<std::string>(trainer_spec.input().begin(),
                                 trainer_spec.input().end()));
    iterator = owned_iterator.get();
  }

  normalizer::Normalizer normalizer(normalizer_spec, trainer_spec);
  if (!normalizer.status().ok()) return normalizer.status();

  const bool is_tsv = trainer_spec.input_format() == "tsv";
  // Ordered, so the canonical corpus order does not depend on a hash seed.
  std::map<std::string, int64_t> aggregated;
  for (; !iterator->done(); iterator->Next()) {
    std::string sentence = iterator->value();
    int64_t freq = 1;
    if (is_tsv) {
      const std::vector<std::string> fields = absl::StrSplit(sentence, '\t');
      if (fields.size() != 2) {
        return absl::InvalidArgumentError(absl::StrCat(
            "continuation TSV must be <unit><tab><count>: ", sentence));
      }
      sentence = fields[0];
      if (!absl::SimpleAtoi(fields[1], &freq) || freq < 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("invalid continuation TSV count: ", fields[1]));
      }
    }
    if (sentence.empty()) continue;
    if (static_cast<int64_t>(sentence.size()) >
        trainer_spec.max_sentence_length()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "continuation refuses to silently drop an overlong training unit: ",
          sentence.size(), " > ", trainer_spec.max_sentence_length()));
    }

    std::string normalized = normalizer.Normalize(sentence);
    if (normalized.empty()) continue;

    if (freq > std::numeric_limits<int64_t>::max() -
                   corpus->weighted_sentence_count) {
      return absl::OutOfRangeError("weighted continuation corpus count overflow");
    }
    corpus->weighted_sentence_count += freq;

    // Continuation reads its corpus as a MULTISET of weighted records, not as
    // a sequence of lines. Identical records are folded together and the
    // result is kept in one canonical order.
    //
    // That is what makes "a weighted TSV equals physical repetition" true by
    // construction rather than approximately. Otherwise the two spellings of
    // the same corpus take different summation paths - n*p once against p
    // added n times - and float addition is not associative, so they disagree
    // in the last bits. Those bits then decide which of two equally scored
    // extension pieces gets the lower external ID, and an ID is an ABI.
    //
    // It also makes a run independent of the order lines happen to sit in the
    // input file, which is the same reproducibility promise stated once more.
    auto inserted = aggregated.emplace(std::move(normalized), freq);
    if (!inserted.second) {
      int64_t& total = inserted.first->second;
      if (freq > std::numeric_limits<int64_t>::max() - total) {
        return absl::OutOfRangeError(
            "weighted continuation corpus count overflow for a single record");
      }
      total += freq;
    }
  }
  if (!iterator->status().ok()) return iterator->status();
  if (aggregated.empty()) {
    return absl::InvalidArgumentError("continuation corpus is empty");
  }
  corpus->sentences.reserve(aggregated.size());
  for (auto& entry : aggregated) {
    corpus->sentences.emplace_back(entry.first, entry.second);
  }
  return absl::OkStatus();
}

absl::Status ReadExpansionSpec(absl::string_view filename,
                               ExpansionSpec* spec) {
  if (filename.empty()) {
    return absl::InvalidArgumentError("expansion_spec path is empty");
  }
  std::string bytes;
  auto status = ReadAll(filename, &bytes);
  if (!status.ok()) return status;
  if (!spec->ParseFromString(bytes)) {
    return absl::InvalidArgumentError(
        absl::StrCat("malformed ExpansionSpec: ", filename));
  }
  return absl::OkStatus();
}

absl::Status ReadModelProto(absl::string_view filename, ModelProto* model,
                            std::string* raw_bytes) {
  if (filename.empty()) {
    return absl::InvalidArgumentError("prior model path is empty");
  }
  std::string bytes;
  auto status = ReadAll(filename, &bytes);
  if (!status.ok()) return status;
  if (!model->ParseFromString(bytes)) {
    return absl::InvalidArgumentError(
        absl::StrCat("malformed ModelProto: ", filename));
  }
  if (raw_bytes != nullptr) *raw_bytes = std::move(bytes);
  return absl::OkStatus();
}

absl::Status WriteExpansionResult(absl::string_view filename,
                                  const ExpansionResult& result) {
  return WriteAll(filename, result.SerializeAsString());
}

absl::Status WriteModelProto(absl::string_view filename,
                             const ModelProto& model) {
  return WriteAll(filename, model.SerializeAsString());
}

absl::Status WriteExpansionVocab(
    absl::string_view filename,
    const std::vector<ExpansionPiece>& externally_ordered_pieces) {
  std::vector<ExpansionPiece> pieces = externally_ordered_pieces;
  std::sort(pieces.begin(), pieces.end(), [](const ExpansionPiece& a,
                                             const ExpansionPiece& b) {
    if (a.external_id() != b.external_id()) {
      return a.external_id() < b.external_id();
    }
    return a.piece() < b.piece();
  });
  auto output = filesystem::NewWritableFile(filename);
  if (!output->status().ok()) return output->status();
  for (const auto& piece : pieces) {
    // %.9g round-trips float32 exactly. absl::StrCat's default float
    // formatting does not, so the .vocab used to be a lossy view of scores the
    // .model and .expansion carry exactly -- which made "the three artifacts
    // describe the same tokenizer" untestable at the text layer.
    if (!output->WriteLine(
            absl::StrCat(piece.piece(), "\t",
                         absl::StrFormat("%.9g", piece.score()), "\t",
                         piece.external_id()))) {
      return absl::DataLossError(
          absl::StrCat("failed to write vocab: ", filename));
    }
  }
  return absl::OkStatus();
}

absl::Status WriteMergeTable(
    absl::string_view filename,
    const std::vector<ExpansionMerge>& rank_ordered_merges) {
  std::vector<ExpansionMerge> merges = rank_ordered_merges;
  std::sort(merges.begin(), merges.end(), [](const ExpansionMerge& a,
                                             const ExpansionMerge& b) {
    return a.rank() < b.rank();
  });
  auto output = filesystem::NewWritableFile(filename);
  if (!output->status().ok()) return output->status();
  for (const auto& merge : merges) {
    if (!output->WriteLine(absl::StrCat(merge.left(), "\t", merge.right()))) {
      return absl::DataLossError(
          absl::StrCat("failed to write merge table: ", filename));
    }
  }
  return absl::OkStatus();
}

std::string Sha256Hex(absl::string_view bytes) {
  static constexpr std::array<uint32_t, 64> k = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

  std::vector<uint8_t> data(bytes.begin(), bytes.end());
  const uint64_t bit_len = static_cast<uint64_t>(data.size()) * 8;
  data.push_back(0x80);
  while ((data.size() % 64) != 56) data.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8) {
    data.push_back(static_cast<uint8_t>((bit_len >> shift) & 0xff));
  }

  std::array<uint32_t, 8> h = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

  for (size_t off = 0; off < data.size(); off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      const size_t p = off + 4 * i;
      w[i] = (static_cast<uint32_t>(data[p]) << 24) |
             (static_cast<uint32_t>(data[p + 1]) << 16) |
             (static_cast<uint32_t>(data[p + 2]) << 8) |
             static_cast<uint32_t>(data[p + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = RotR(w[i - 15], 7) ^ RotR(w[i - 15], 18) ^
                          (w[i - 15] >> 3);
      const uint32_t s1 = RotR(w[i - 2], 17) ^ RotR(w[i - 2], 19) ^
                          (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
      const uint32_t S0 = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = S0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }

  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (uint32_t x : h) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      out.push_back(hex[(x >> shift) & 0xf]);
    }
  }
  return out;
}

}  // namespace sentencepiece::continuation
