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

TEST(BPETrainerTest, SavesTheMergeEachPieceCameFrom) {
  // The guarantee consumers need: every learned piece is left+right of a
  // recorded merge, and BOTH halves are themselves in the vocabulary. That is
  // what makes a merge list reconstructible WITHOUT guessing a split.
  const std::string model_prefix =
      util::JoinPath(::testing::TempDir(), "merges_model");
  const std::string input_file =
      util::JoinPath(::testing::TempDir(), "merges_input");
  {
    auto output = filesystem::NewWritableFile(input_file);
    output->WriteLine("abracadabra");
    output->WriteLine("pineapple");
    output->WriteLine("hellohe");
  }
  ASSERT_TRUE(SentencePieceTrainer::Train(
                  absl::StrCat("--model_prefix=", model_prefix,
                               " --input=", input_file,
                               " --vocab_size=40 --model_type=bpe"
                               " --normalization_rule_name=identity"))
                  .ok());

  SentencePieceProcessor processor;
  ASSERT_TRUE(processor.Load(model_prefix + ".model").ok());
  std::unordered_set<std::string> vocab;
  for (const auto &piece : processor.model_proto().pieces()) {
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

}  // namespace
}  // namespace bpe
}  // namespace sentencepiece
