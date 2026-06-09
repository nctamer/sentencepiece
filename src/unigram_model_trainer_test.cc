// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#include "unigram_model_trainer.h"

#include <string>
#include <vector>

#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "util.h"

namespace sentencepiece {
namespace unigram {

// Space symbol
#define WS "\xe2\x96\x81"

TEST(UnigramTrainerTest, TrainerModelTest) {
  TrainerSpec trainer_spec;
  NormalizerSpec normalizer_spec;
  const TrainerModel model(trainer_spec, normalizer_spec);
  EXPECT_EQ(EncodeResult(), model.Encode("test"));
}

struct TrainerResult {
  std::string sentence_pieces;
  std::vector<std::pair<std::string, float>> seed_pieces_and_probs;
};

TrainerResult RunTrainer(const std::vector<std::string>& input, int size,
                         const bool use_dp = false, const float dp_noise = 0.0,
                         const uint32_t dp_clip = 0) {
  const std::string input_file = util::JoinPath(::testing::TempDir(), "input");
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "model");
  {
    auto output = filesystem::NewWritableFile(input_file);
    for (const auto& line : input) {
      output->WriteLine(line);
    }
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_type(TrainerSpec::UNIGRAM);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(size - 3);  // remove <unk>, <s>, </s>
  trainer_spec.set_model_prefix(model_prefix);

  trainer_spec.set_enable_differential_privacy(use_dp);
  trainer_spec.set_differential_privacy_noise_level(dp_noise);
  trainer_spec.set_differential_privacy_clipping_threshold(dp_clip);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);

  NormalizerSpec denormalizer_spec;

  std::vector<std::pair<std::string, float>> seed_pieces;

  {
    Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
    EXPECT_OK(trainer.LoadSentences());
    TrainerModel::SentencePieces res = trainer.MakeSeedSentencePieces();

    for (const auto& piece : res) {
      seed_pieces.emplace_back(piece.first, piece.second);
    }
  }

  std::vector<std::string> pieces;

  {
    Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
    EXPECT_TRUE(trainer.Train().ok());

    SentencePieceProcessor processor;
    EXPECT_TRUE(processor.Load(model_prefix + ".model").ok());

    const auto& model = processor.model_proto();

    // remove <unk>, <s>, </s>
    for (int i = 3; i < model.pieces_size(); ++i) {
      pieces.emplace_back(model.pieces(i).piece());
    }
  }

  TrainerResult res;
  res.seed_pieces_and_probs = seed_pieces;
  std::sort(pieces.begin(), pieces.end());
  res.sentence_pieces = absl::StrJoin(pieces, " ");
  return res;
}

TEST(UnigramTrainerTest, BasicTest) {
  const auto& res = RunTrainer(
      {"magnanimity \t 5", "Pineapple \t 6", "i have an apple and a pen \t 1",
       "Overly \t 6", "Available \t 3"},
      30);

  // Check seed pieces.
  EXPECT_EQ(56, res.seed_pieces_and_probs.size());

  // Check final pieces.
  EXPECT_EQ(
      "A O Overly P Pineapple a b d e g h i l le m magnanimity n p r t v y ▁ "
      "▁an",
      res.sentence_pieces);
}

TEST(UnigramTrainerTest, BasicDPTest) {
  // no noise, clipping.
  {
    const auto& res = RunTrainer(
        {"magnanimity \t 5", "Pineapple \t 6", "i have an apple and a pen \t 1",
         "Overly \t 6", "Available \t 5"},
        22, true /*use_dp*/, 0 /*dp_noise*/, 4 /*dp_clipping*/);

    // Got 38 instead of 27 seeds.
    EXPECT_EQ(38, res.seed_pieces_and_probs.size());

    // And they are equiv to if the last sentence was not there.
    const auto& res_nodp = RunTrainer(
        {"magnanimity \t 5", "Pineapple \t 6", "Overly \t 6", "Available \t 5"},
        22);

    EXPECT_EQ(res.seed_pieces_and_probs, res_nodp.seed_pieces_and_probs);

    // Check final pieces.
    EXPECT_EQ(res.sentence_pieces, res_nodp.sentence_pieces);
  }
}

namespace {

static constexpr char kTestInputData[] = "wagahaiwa_nekodearu.txt";

TEST(UnigramTrainerTest, EndToEndTest) {
  const std::string input = util::JoinPath(::testing::SrcDir(), kTestInputData);

  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=",
                       util::JoinPath(::testing::TempDir(), "tmp_model"),
                       " --input=", input,
                       " --vocab_size=8000 --normalization_rule_name=identity",
                       " --model_type=unigram --user_defined_symbols=<user>",
                       " --control_symbols=<ctrl> --max_sentence_length=2048"))
          .ok());

  SentencePieceProcessor sp;
  EXPECT_TRUE(
      sp.Load(util::JoinPath(::testing::TempDir(), "tmp_model.model")).ok());
  EXPECT_EQ(8000, sp.GetPieceSize());

  const int cid = sp.PieceToId("<ctrl>");
  const int uid = sp.PieceToId("<user>");
  EXPECT_TRUE(sp.IsControl(cid));
  EXPECT_FALSE(sp.IsUnknown(uid));

  std::vector<std::string> tok;

  EXPECT_TRUE(sp.Encode("", &tok).ok());
  EXPECT_TRUE(tok.empty());

  EXPECT_TRUE(sp.Encode("吾輩《わがはい》は猫である。名前はまだ無い。"
                        "どこで生れたかとんと見当《けんとう》がつかぬ。"
                        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だ"
                        "けは記憶している"
                        "。",
                        &tok)
                  .ok());
  // TODO(taku): Temporally disable this test on Windows.
#ifndef OS_WIN
  LOG(INFO) << "[" << absl::StrJoin(tok, " ") << std::endl;
  EXPECT_EQ(
      WS
      " 吾輩 《 わが はい 》 は猫である 。 名前はまだ 無 い 。 どこ で 生 れた "
      "か とん と 見当 《 けん とう 》 が つか ぬ 。 何でも 薄 暗 い じめ じめ "
      "した 所で ニャーニャー 泣 い ていた 事 だけは 記憶 している 。",
      absl::StrJoin(tok, " "));
#endif
}

TEST(UnigramTrainerTest, ProtectedPiecesTest) {
  // Train with protected sub-event pieces. Verify they survive despite being
  // rare, and are used for encoding rare events.
  const std::string input_file =
      util::JoinPath(::testing::TempDir(), "intermo_unigram_input");
  const std::string protected_file =
      util::JoinPath(::testing::TempDir(), "protected_pieces");
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "unigram_protected");
  {
    auto output = filesystem::NewWritableFile(input_file);
    // Common events (high frequency)
    for (int i = 0; i < 200; ++i) {
      output->WriteLine("PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5 1/4 PL: G3 B3 D4");
    }
    for (int i = 0; i < 200; ++i) {
      output->WriteLine("PR: E4 1/8 PL: B-2 D3 F3 1/16 PR: g4 a4 1/4 PL: C3 E3");
    }
    // Rare events with double-flats (low frequency)
    for (int i = 0; i < 10; ++i) {
      output->WriteLine("PR: A--0 1/4 PL: B--1 1/8 PR: C##5");
    }
  }

  // Protected sub-event pieces that should survive despite low frequency
  {
    auto output = filesystem::NewWritableFile(protected_file);
    output->WriteLine(WS "A-");
    output->WriteLine("-0");
    output->WriteLine(WS "B-");
    output->WriteLine("-1");
    output->WriteLine("##");
  }

  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=", model_prefix,
                       " --input=", input_file,
                       " --vocab_size=50"
                       " --model_type=unigram"
                       " --split_by_whitespace=true"
                       " --split_by_unicode_script=false"
                       " --split_by_number=false"
                       " --character_coverage=1.0"
                       " --max_sentence_length=500000"
                       " --protected_pieces_file=", protected_file))
          .ok());

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(model_prefix + ".model").ok());

  // All protected pieces must exist in the final model
  std::vector<std::string> protected_pieces = {WS "A-", "-0", WS "B-", "-1", "##"};
  for (const auto &piece : protected_pieces) {
    EXPECT_NE(sp.PieceToId(piece), sp.unk_id())
        << "Protected piece " << piece << " not found in model";
  }

  // Encoding rare event should use the protected sub-event pieces
  std::vector<std::string> tokens;
  ASSERT_TRUE(sp.Encode("PR: A--0", &tokens).ok());
  // Should NOT be all single characters — protected pieces should be used
  EXPECT_LT(tokens.size(), 8u)
      << "A--0 should use sub-event pieces, not character fallback";
}

TEST(UnigramTrainerTest, ProtectedPiecesZeroFreqTest) {
  // Test that protected pieces survive even when they have zero Viterbi
  // frequency under the new boundary regime. This reproduces the bug where
  // PruneSentencePieces drops always_keep pieces when freq==0.
  const std::string input_file =
      util::JoinPath(::testing::TempDir(), "zero_freq_input");
  const std::string protected_file =
      util::JoinPath(::testing::TempDir(), "zero_freq_protected");
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "zero_freq_model");
  {
    auto output = filesystem::NewWritableFile(input_file);
    // Corpus has NO occurrences of "XY" or "ZW" as substrings.
    // The protected pieces will have zero frequency in Viterbi paths.
    for (int i = 0; i < 500; ++i) {
      output->WriteLine("PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5");
    }
  }

  // Protect pieces that exist as substrings but are never optimal
  // in Viterbi paths (longer alternatives always win).
  // "P1" can appear in "PL:" context but unigram prefers "PL:" as one piece.
  // "-3C" spans two events; with split_by_whitespace it can't be used.
  {
    auto output = filesystem::NewWritableFile(protected_file);
    output->WriteLine("-3");   // rare sub-event piece (A-3 → A + -3 vs A-3)
    output->WriteLine("4F");   // never useful: crosses character boundary oddly
    output->WriteLine(WS "PR:");  // this one IS commonly used
  }

  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=", model_prefix,
                       " --input=", input_file,
                       " --vocab_size=25"
                       " --model_type=unigram"
                       " --split_by_whitespace=true"
                       " --split_by_unicode_script=false"
                       " --split_by_number=false"
                       " --character_coverage=1.0"
                       " --max_sentence_length=500000"
                       " --protected_pieces_file=", protected_file))
          .ok());

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(model_prefix + ".model").ok());

  // ▁PR: should survive (high frequency + protected)
  EXPECT_NE(sp.PieceToId(WS "PR:"), sp.unk_id())
      << "Protected piece " WS "PR: not found";

  // -3 is a valid substring (in "A-3") but may not be optimal in Viterbi
  // (unigram prefers "▁A-3" as one piece). Protection must keep it anyway.
  EXPECT_NE(sp.PieceToId("-3"), sp.unk_id())
      << "Protected piece -3 (low freq) dropped - protection bug";

  // 4F spans "C4 F4" boundary chars — exists as substring but never in Viterbi
  EXPECT_NE(sp.PieceToId("4F"), sp.unk_id())
      << "Protected piece 4F (zero freq) dropped - protection bug";
}

}  // namespace
}  // namespace unigram
}  // namespace sentencepiece
