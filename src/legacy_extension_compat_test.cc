// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// The fork shipped four TrainerSpec extensions (200/201/204/205) before
// first-class continuation existed. Models and CLI scripts produced then must
// keep decoding to the same meanings, and 202/203 must stay retired. These are
// wire-ABI oracles: they decode fixed bytes rather than round-tripping the
// current schema against itself, so renumbering a field fails here even if the
// schema stays self-consistent.

#include <gtest/gtest.h>

#include <string>

#include "absl/status/status.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_trainer.h"

namespace sentencepiece {
namespace {

// Appends the base-128 varint encoding of `value`.
void AppendVarint(uint64_t value, std::string* out) {
  while (value >= 0x80) {
    out->push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  out->push_back(static_cast<char>(value));
}

void AppendTag(int field_number, int wire_type, std::string* out) {
  AppendVarint((static_cast<uint64_t>(field_number) << 3) | wire_type, out);
}

void AppendBoolField(int field_number, bool value, std::string* out) {
  AppendTag(field_number, 0, out);
  AppendVarint(value ? 1 : 0, out);
}

void AppendStringField(int field_number, absl::string_view value,
                       std::string* out) {
  AppendTag(field_number, 2, out);
  AppendVarint(value.size(), out);
  out->append(value.data(), value.size());
}

// A TrainerSpec exactly as a pre-continuation build would have written it.
std::string LegacyTrainerSpecBytes() {
  std::string bytes;
  AppendStringField(1, "corpus.txt", &bytes);   // input
  AppendStringField(2, "m", &bytes);            // model_prefix
  AppendBoolField(200, true, &bytes);           // split_by_interval
  AppendBoolField(201, true, &bytes);           // split_by_barline
  AppendStringField(204, "protected.txt", &bytes);  // protected_pieces_file
  AppendStringField(205, "seed.merges", &bytes);    // seed_merges_file
  return bytes;
}

TEST(LegacyExtensionCompatTest, FieldNumbersAreFrozen) {
  EXPECT_EQ(200, split_by_interval.number());
  EXPECT_EQ(201, split_by_barline.number());
  EXPECT_EQ(204, protected_pieces_file.number());
  EXPECT_EQ(205, seed_merges_file.number());
}

TEST(LegacyExtensionCompatTest, LegacyBytesDecodeToOriginalMeanings) {
  TrainerSpec spec;
  ASSERT_TRUE(spec.ParseFromString(LegacyTrainerSpecBytes()));

  ASSERT_EQ(1, spec.input_size());
  EXPECT_EQ("corpus.txt", spec.input(0));
  EXPECT_EQ("m", spec.model_prefix());

  EXPECT_TRUE(spec.GetExtension(split_by_interval));
  EXPECT_TRUE(spec.GetExtension(split_by_barline));
  EXPECT_EQ("protected.txt", spec.GetExtension(protected_pieces_file));
  EXPECT_EQ("seed.merges", spec.GetExtension(seed_merges_file));
}

TEST(LegacyExtensionCompatTest, CurrentWriterProducesLegacyBytes) {
  TrainerSpec spec;
  spec.add_input("corpus.txt");
  spec.set_model_prefix("m");
  spec.SetExtension(split_by_interval, true);
  spec.SetExtension(split_by_barline, true);
  spec.SetExtension(protected_pieces_file, "protected.txt");
  spec.SetExtension(seed_merges_file, "seed.merges");

  // Serialization is deterministic for this shape: fields ascend by number and
  // every one of them is present, so the bytes must match the legacy encoder.
  EXPECT_EQ(LegacyTrainerSpecBytes(), spec.SerializeAsString());
}

TEST(LegacyExtensionCompatTest, DefaultsMatchLegacyDefaults) {
  const TrainerSpec spec;
  EXPECT_FALSE(spec.GetExtension(split_by_interval));
  EXPECT_FALSE(spec.GetExtension(split_by_barline));
  EXPECT_EQ("", spec.GetExtension(protected_pieces_file));
  EXPECT_EQ("", spec.GetExtension(seed_merges_file));
}

// 202/203 were the progressive BPE phase budgets. They are outside the
// declared extension range now, so nothing can claim them; a stray varint on
// either number must survive as an unknown field rather than bind to anything.
TEST(LegacyExtensionCompatTest, RetiredNumbersBindToNothing) {
  std::string bytes;
  AppendStringField(2, "m", &bytes);
  AppendBoolField(202, true, &bytes);
  AppendBoolField(203, true, &bytes);

  TrainerSpec spec;
  ASSERT_TRUE(spec.ParseFromString(bytes));
  EXPECT_EQ("m", spec.model_prefix());
  EXPECT_FALSE(spec.GetExtension(split_by_interval));
  EXPECT_FALSE(spec.GetExtension(split_by_barline));
  EXPECT_EQ("", spec.GetExtension(protected_pieces_file));
  EXPECT_EQ("", spec.GetExtension(seed_merges_file));
}

TEST(LegacyExtensionCompatTest, CliNamesStillParse) {
  TrainerSpec spec;
  ASSERT_TRUE(SentencePieceTrainer::SetProtoField("split_by_interval", "true",
                                                  &spec)
                  .ok());
  ASSERT_TRUE(
      SentencePieceTrainer::SetProtoField("split_by_barline", "true", &spec)
          .ok());
  ASSERT_TRUE(SentencePieceTrainer::SetProtoField("protected_pieces_file",
                                                  "protected.txt", &spec)
                  .ok());
  ASSERT_TRUE(SentencePieceTrainer::SetProtoField("seed_merges_file",
                                                  "seed.merges", &spec)
                  .ok());
  EXPECT_TRUE(spec.GetExtension(split_by_interval));
  EXPECT_TRUE(spec.GetExtension(split_by_barline));
  EXPECT_EQ("protected.txt", spec.GetExtension(protected_pieces_file));
  EXPECT_EQ("seed.merges", spec.GetExtension(seed_merges_file));
}

// The legacy shims name fresh-training protected vocabulary. They never denote
// inherited state, so pairing them with a continuation request is ambiguous.
TEST(LegacyExtensionCompatTest, RejectsLegacyCombinedWithContinuation) {
  auto base_spec = []() {
    TrainerSpec spec;
    spec.add_input("corpus.txt");
    spec.set_model_prefix("m");
    spec.set_model_type(TrainerSpec::BPE);
    spec.set_vocab_size(64);
    return spec;
  };

  NormalizerSpec normalizer_spec;
  ASSERT_TRUE(SentencePieceTrainer::PopulateNormalizerSpec(&normalizer_spec)
                  .ok());

  {
    TrainerSpec spec = base_spec();
    spec.set_expansion_spec("spec.pb");
    spec.SetExtension(protected_pieces_file, "protected.txt");
    EXPECT_FALSE(
        SentencePieceTrainer::Train(spec, normalizer_spec, nullptr).ok());
  }
  {
    TrainerSpec spec = base_spec();
    spec.set_model_type(TrainerSpec::UNIGRAM);
    spec.set_unigram_prior_model("prior.model");
    spec.SetExtension(protected_pieces_file, "protected.txt");
    EXPECT_FALSE(
        SentencePieceTrainer::Train(spec, normalizer_spec, nullptr).ok());
  }
  {
    TrainerSpec spec = base_spec();
    spec.set_expansion_spec("spec.pb");
    spec.set_unigram_prior_model("prior.model");
    EXPECT_FALSE(
        SentencePieceTrainer::Train(spec, normalizer_spec, nullptr).ok());
  }
  {
    // seed_merges_file is BPE-only.
    TrainerSpec spec = base_spec();
    spec.set_model_type(TrainerSpec::UNIGRAM);
    spec.SetExtension(seed_merges_file, "seed.merges");
    EXPECT_FALSE(
        SentencePieceTrainer::Train(spec, normalizer_spec, nullptr).ok());
  }
}

}  // namespace
}  // namespace sentencepiece
