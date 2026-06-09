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

#include "bpe_model_trainer.h"

#include <string>
#include <vector>

#include "filesystem.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "testharness.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "util.h"

namespace sentencepiece {
namespace bpe {
namespace {

// Space symbol
#define WS "\xe2\x96\x81"

std::string RunTrainer(
    const std::vector<std::string> &input, int size,
    const std::vector<std::string> &user_defined_symbols = {}) {
  const std::string input_file = util::JoinPath(::testing::TempDir(), "input");
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "model");
  {
    auto output = filesystem::NewWritableFile(input_file);
    for (const auto &line : input) {
      output->WriteLine(line);
    }
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(size - 3);  // remove <unk>, <s>, </s>
  trainer_spec.set_model_prefix(model_prefix);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);

  NormalizerSpec denormalizer_spec;

  for (const auto &w : user_defined_symbols) {
    trainer_spec.add_user_defined_symbols(w);
  }

  Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
  EXPECT_TRUE(trainer.Train().ok());

  SentencePieceProcessor processor;
  EXPECT_TRUE(processor.Load(model_prefix + ".model").ok());

  const auto &model = processor.model_proto();
  std::vector<std::string> pieces;

  // remove <unk>, <s>, </s>
  for (int i = 3; i < model.pieces_size(); ++i) {
    pieces.emplace_back(model.pieces(i).piece());
  }

  return absl::StrJoin(pieces, " ");
}

TEST(BPETrainerTest, BasicTest) {
  EXPECT_EQ("ab ra abra ad cad abracad abracadabra ac br a b r c d",
            RunTrainer({"abracadabra"}, 20));
  EXPECT_EQ("ap le app apple en in ine pen p e a l n i",
            RunTrainer({"pen", "pineapple", "apple"}, 20));
  EXPECT_EQ("he ll llo hello hellohe el lo oh hel ohe e h l o",
            RunTrainer({"hellohe"}, 20));
  EXPECT_EQ("app le en in ine pen pine ne pe e l n p i",
            RunTrainer({"pen", "pineapple", "apple"}, 20, {"app"}));
}

static constexpr char kTestInputData[] = "wagahaiwa_nekodearu.txt";

TEST(BPETrainerTest, EndToEndTest) {
  const std::string input = util::JoinPath(::testing::SrcDir(), kTestInputData);

  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=",
                       util::JoinPath(::testing::TempDir(), "tmp_model"),
                       " --input=", input,
                       " --vocab_size=8000 --normalization_rule_name=identity"
                       " --model_type=bpe --control_symbols=<ctrl> "
                       "--max_sentence_length=2048"))
          .ok());

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(std::string(util::JoinPath(::testing::TempDir(),
                                                 "tmp_model.model")))
                  .ok());
  EXPECT_EQ(8000, sp.GetPieceSize());

  const int cid = sp.PieceToId("<ctrl>");
  EXPECT_TRUE(sp.IsControl(cid));

  std::vector<std::string> tok;
  ASSERT_TRUE(sp.Encode("", &tok).ok());
  ASSERT_TRUE(tok.empty());

  EXPECT_TRUE(sp.Encode("吾輩《わがはい》は猫である。名前はまだ無い。"
                        "どこで生れたかとんと見当《けんとう》がつかぬ。"
                        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だ"
                        "けは記憶している"
                        "。",
                        &tok)
                  .ok());
  EXPECT_EQ(WS
            " 吾輩 《 わが はい 》 は猫 である 。 名前 はまだ 無い 。 "
            "どこで 生 れた か とん と見 当 《 けんとう 》 が つかぬ 。 "
            "何でも 薄 暗 いじ め じ め した 所で ニャー ニャー 泣 いていた "
            "事 だけは 記憶 している 。",
            absl::StrJoin(tok, " "));
}

TEST(BPETrainerTest, ProgressiveConstraintTest) {
  // Train with split_by_barline + phase budgets on intermo-like data.
  // Phase 1 (first 10 merges): within-event only (no internal whitespace)
  // Phase 2 (next 10 merges): cross-event within-moment (no internal ws+digit/|)
  // Phase 3 (remaining): cross-moment within-bar (no internal ws+|)
  const std::string input_file =
      util::JoinPath(::testing::TempDir(), "intermo_input");
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "intermo_model");
  {
    auto output = filesystem::NewWritableFile(input_file);
    // Repeat to get sufficient frequency
    for (int i = 0; i < 100; ++i) {
      output->WriteLine(
          "|4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5 |3/4k0 PR: G4");
    }
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input_file);
  trainer_spec.set_vocab_size(80);
  trainer_spec.set_model_prefix(model_prefix);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_barline(true);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_character_coverage(1.0);
  trainer_spec.set_max_sentence_length(500000);
  trainer_spec.set_phase1_merge_budget(10);
  trainer_spec.set_phase2_merge_budget(10);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(true);

  NormalizerSpec denormalizer_spec;

  Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
  EXPECT_TRUE(trainer.Train().ok());

  SentencePieceProcessor processor;
  EXPECT_TRUE(processor.Load(model_prefix + ".model").ok());

  const auto &model = processor.model_proto();

  // Check phase 1 pieces (first 10 after meta pieces): no internal whitespace
  int phase1_violations = 0;
  int phase2_violations = 0;
  const int meta_size = 3;  // <unk>, <s>, </s>
  for (int i = meta_size; i < model.pieces_size(); ++i) {
    const std::string &piece = model.pieces(i).piece();
    const int merge_idx = i - meta_size;

    // Check for internal whitespace (▁ at pos > 0)
    bool has_internal_ws = false;
    bool has_interval_boundary = false;
    for (size_t p = 3; p < piece.size(); p++) {
      if (piece.substr(p, 3) == WS) {
        has_internal_ws = true;
        if (p + 3 < piece.size()) {
          char next = piece[p + 3];
          if ((next >= '0' && next <= '9') || next == '|') {
            has_interval_boundary = true;
          }
        }
      }
    }

    if (merge_idx < 10 && has_internal_ws) {
      phase1_violations++;
    }
    if (merge_idx < 20 && has_interval_boundary) {
      phase2_violations++;
    }
  }

  EXPECT_EQ(0, phase1_violations)
      << "Phase 1 pieces should not have internal whitespace";
  EXPECT_EQ(0, phase2_violations)
      << "Phase 2 pieces should not cross interval boundaries";
}

}  // namespace
}  // namespace bpe
}  // namespace sentencepiece
