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
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "filesystem.h"
#include "model_interface.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "util.h"

// The whitespace marker U+2581, which is an ordinary piece character here.
#define WS "\xe2\x96\x81"

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


// ---------------------------------------------------------------------------
// Legacy behaviour, not just legacy wire format. These pin what the four
// extension fields DO, so a future upstream merge that quietly changes it
// fails here rather than in a training run.
// ---------------------------------------------------------------------------

std::string TempFile(absl::string_view leaf) {
  return filesystem::JoinPath(::testing::TempDir(), leaf);
}

void WriteLinesTo(const std::string& path,
                  const std::vector<std::string>& lines) {
  auto output = filesystem::NewWritableFile(path);
  ASSERT_TRUE(output->status().ok());
  for (const auto& line : lines) ASSERT_TRUE(output->WriteLine(line));
}

// split_by_interval breaks only before whitespace that begins an interval
// token: '/' or '1'-'9' or '|'. '0' is a grace duration and never one.
TEST(LegacyExtensionCompatTest, SplitIntoWordsByInterval) {
  {
    const auto v = SplitIntoWords(
        WS "|4/4k0" WS "PR:" WS "C5" WS "1/4" WS "PL:" WS "A-3", false, false,
        /*split_by_interval=*/true, /*split_by_barline=*/false);
    ASSERT_EQ(2, v.size());
    EXPECT_EQ(WS "|4/4k0" WS "PR:" WS "C5", v[0]);
    EXPECT_EQ(WS "1/4" WS "PL:" WS "A-3", v[1]);
  }
  {
    // Whitespace before a letter is interior, not a boundary.
    const auto v = SplitIntoWords(WS "PR:" WS "C5" WS "D4", false, false, true,
                                  false);
    ASSERT_EQ(1, v.size());
    EXPECT_EQ(WS "PR:" WS "C5" WS "D4", v[0]);
  }
  {
    // '0' is a grace duration, so it does not open an interval.
    const auto v =
        SplitIntoWords(WS "PR:" WS "0/4" WS "C5", false, false, true, false);
    ASSERT_EQ(1, v.size());
  }
}

// split_by_barline is the coarser fence: only '|' counts.
TEST(LegacyExtensionCompatTest, SplitIntoWordsByBarline) {
  {
    const auto v = SplitIntoWords(
        WS "|4/4k0" WS "PR:" WS "C5" WS "1/4" WS "PL:" WS "A-3" WS "|3/4k0" WS
           "D5",
        false, false, /*split_by_interval=*/false, /*split_by_barline=*/true);
    ASSERT_EQ(2, v.size());
    EXPECT_EQ(WS "|4/4k0" WS "PR:" WS "C5" WS "1/4" WS "PL:" WS "A-3", v[0]);
    EXPECT_EQ(WS "|3/4k0" WS "D5", v[1]);
  }
  {
    // A digit does NOT open a boundary in barline mode.
    const auto v =
        SplitIntoWords(WS "|4/4k0" WS "PR:" WS "C5" WS "1/4", false, false,
                       false, true);
    ASSERT_EQ(1, v.size());
  }
}

// The guarantee consumers need from <model_prefix>.merges: every recorded
// merge composes two pieces that are THEMSELVES in the vocabulary, so the list
// reconstructs without guessing a split back from the piece string.
TEST(LegacyExtensionCompatTest, MergeListIsReconstructible) {
  const std::string model_prefix = TempFile("legacy_merges_model");
  const std::string input_file = TempFile("legacy_merges_input");
  WriteLinesTo(input_file, {"abracadabra", "pineapple", "hellohe"});

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  absl::StrCat("--model_prefix=", model_prefix,
                               " --input=", input_file,
                               " --vocab_size=40 --model_type=bpe"
                               " --normalization_rule_name=identity"))
                  .ok());

  SentencePieceProcessor processor;
  ASSERT_TRUE(processor.Load(model_prefix + ".model").ok());
  std::unordered_set<std::string> vocab;
  for (const auto& piece : processor.model_proto().pieces()) {
    vocab.insert(piece.piece());
  }

  auto input = filesystem::NewReadableFile(model_prefix + ".merges");
  ASSERT_TRUE(input->status().ok());
  std::string line;
  int merges = 0;
  while (input->ReadLine(&line)) {
    const auto tab = line.find('\t');
    ASSERT_NE(std::string::npos, tab) << "merge line is not left<TAB>right";
    const std::string left = line.substr(0, tab);
    const std::string right = line.substr(tab + 1);
    EXPECT_TRUE(vocab.count(left + right)) << "merged piece missing: " << line;
    EXPECT_TRUE(vocab.count(left)) << "left half missing: " << left;
    EXPECT_TRUE(vocab.count(right)) << "right half missing: " << right;
    ++merges;
  }
  EXPECT_GT(merges, 0);
}

// Replaying the seed's merges changes how the corpus is segmented, so pieces
// learned afterwards compose seed pieces instead of contradicting them.
TEST(LegacyExtensionCompatTest, BpeProtectedPiecesAndSeedMergeReplay) {
  const std::string input_file = TempFile("legacy_seed_input");
  const std::string protected_file = TempFile("legacy_seed_protected");
  const std::string merges_file = TempFile("legacy_seed_merges");
  const std::string model_prefix = TempFile("legacy_seed_model");

  WriteLinesTo(input_file, std::vector<std::string>(200, "abcabcabc"));
  WriteLinesTo(protected_file, {"ab", "abc"});
  WriteLinesTo(merges_file, {"a\tb", "ab\tc"});

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  absl::StrCat("--model_prefix=", model_prefix,
                               " --input=", input_file,
                               " --vocab_size=32 --model_type=bpe"
                               " --normalization_rule_name=identity"
                               " --hard_vocab_limit=false"
                               " --protected_pieces_file=", protected_file,
                               " --seed_merges_file=", merges_file))
                  .ok());

  SentencePieceProcessor processor;
  ASSERT_TRUE(processor.Load(model_prefix + ".model").ok());
  // Protected pieces are in the vocabulary even though BPE never had to learn
  // them, because the replay consumed their pairs before the main loop.
  EXPECT_NE(processor.unk_id(), processor.PieceToId("ab"));
  EXPECT_NE(processor.unk_id(), processor.PieceToId("abc"));
}

// Protection has to survive every prune stage, including ones added upstream
// after the field existed.
TEST(LegacyExtensionCompatTest, UnigramProtectedPiecesSurvivePruning) {
  const std::string input_file = TempFile("legacy_uni_input");
  const std::string protected_file = TempFile("legacy_uni_protected");
  const std::string model_prefix = TempFile("legacy_uni_model");

  std::vector<std::string> lines;
  for (int i = 0; i < 200; ++i) {
    lines.push_back("PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5 1/4 PL: G3 B3 D4");
  }
  for (int i = 0; i < 200; ++i) {
    lines.push_back("PR: E4 1/8 PL: B-2 D3 F3 1/16 PR: g4 a4 1/4 PL: C3 E3");
  }
  // Rare enough that ordinary EM pruning would drop these shapes.
  for (int i = 0; i < 10; ++i) lines.push_back("PR: A--0 1/4 PL: B--1");
  WriteLinesTo(input_file, lines);

  const std::vector<std::string> protect = {WS "A-", "-0", WS "B-", "-1", "##"};
  WriteLinesTo(protected_file, protect);

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  absl::StrCat("--model_prefix=", model_prefix,
                               " --input=", input_file,
                               " --vocab_size=50 --model_type=unigram"
                               " --split_by_whitespace=true"
                               " --split_by_unicode_script=false"
                               " --split_by_number=false"
                               " --character_coverage=1.0"
                               " --max_sentence_length=500000"
                               " --hard_vocab_limit=false"
                               " --protected_pieces_file=", protected_file))
                  .ok());

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(model_prefix + ".model").ok());
  for (const auto& piece : protect) {
    // "##" never occurs in this corpus at all: zero Viterbi frequency is still
    // protection, which is the case an always_keep check alone gets wrong.
    EXPECT_NE(sp.unk_id(), sp.PieceToId(piece))
        << "protected piece dropped: " << piece;
  }
}

// auto_character_coverage prunes candidates by a global objective and can drop
// a protected piece, which is the one thing the field promises. The
// combination is refused rather than allowed to violate it silently.
TEST(LegacyExtensionCompatTest, RejectsProtectedPiecesWithAutoCoverage) {
  const std::string input_file = TempFile("legacy_autocov_input");
  const std::string protected_file = TempFile("legacy_autocov_protected");
  const std::string model_prefix = TempFile("legacy_autocov_model");
  WriteLinesTo(input_file, std::vector<std::string>(50, "abcabc"));
  WriteLinesTo(protected_file, {"ab"});

  EXPECT_FALSE(SentencePieceTrainer::Train(
                   absl::StrCat("--model_prefix=", model_prefix,
                                " --input=", input_file,
                                " --vocab_size=32 --model_type=bpe"
                                " --normalization_rule_name=identity"
                                " --auto_character_coverage=true"
                                " --protected_pieces_file=", protected_file))
                   .ok());
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
