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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "trainer_interface.h"
#include "util.h"

namespace sentencepiece {
namespace bpe {
namespace {

// Space symbol
#define WS "\xe2\x96\x81"

std::string RunTrainer(
    const std::vector<std::string>& input, int size,
    const std::vector<std::string>& user_defined_symbols = {}) {
  const std::string input_file =
      filesystem::JoinPath(::testing::TempDir(), "input");
  const std::string model_prefix =
      filesystem::JoinPath(::testing::TempDir(), "model");
  {
    auto output = filesystem::NewWritableFile(input_file);
    for (const auto& line : input) {
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

  for (const auto& w : user_defined_symbols) {
    trainer_spec.add_user_defined_symbols(w);
  }

  Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
  EXPECT_TRUE(trainer.Train().ok());

  SentencePieceProcessor processor;
  EXPECT_TRUE(processor.Load(model_prefix + ".model").ok());

  const auto& model = processor.model_proto();
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
  const std::string input =
      filesystem::JoinPath(::testing::SrcDir(), kTestInputData);

  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=",
                       filesystem::JoinPath(::testing::TempDir(), "tmp_model"),
                       " --input=", input,
                       " --vocab_size=8000 --normalization_rule_name=identity"
                       " --model_type=bpe --control_symbols=<ctrl> "
                       "--max_sentence_length=2048"))
          .ok());

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(std::string(filesystem::JoinPath(::testing::TempDir(),
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

TEST(BPETrainerTest, AutoCharacterCoverageTest) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_auto_character_coverage, true);

  const std::string input_file =
      filesystem::JoinPath(::testing::TempDir(), "input_auto_bpe");
  const std::string model_prefix =
      filesystem::JoinPath(::testing::TempDir(), "model_auto_bpe");
  {
    auto output = filesystem::NewWritableFile(input_file);
    // Repeat some high-frequency words with multibyte characters
    for (int i = 0; i < 20; ++i) {
      output->WriteLine("こんにちは世界");
      output->WriteLine("Hello world");
    }
    // Low frequency rare characters
    output->WriteLine("稀少文字：ゐゑ驫");
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input_file);
  trainer_spec.set_byte_fallback(true);
  trainer_spec.set_vocab_size(
      275);  // tight vocab budget to force rare chars to fallback
  trainer_spec.set_model_prefix(model_prefix);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);

  NormalizerSpec denormalizer_spec;

  Trainer trainer(trainer_spec, normalizer_spec, denormalizer_spec);
  EXPECT_TRUE(trainer.Train().ok());

  SentencePieceProcessor processor;
  EXPECT_TRUE(processor.Load(model_prefix + ".model").ok());
  EXPECT_EQ(275, processor.GetPieceSize());

  const auto& model = processor.model_proto();
  // 1. Verify that all non-byte pieces are structurally valid UTF-8.
  for (int i = 0; i < model.pieces_size(); ++i) {
    const auto& sp = model.pieces(i);
    if (sp.type() == ModelProto::SentencePiece::BYTE) {
      continue;
    }
    EXPECT_TRUE(string_util::IsStructurallyValid(sp.piece()))
        << "Piece " << sp.piece() << " is not valid UTF-8!";
  }

  // 2. High-frequency characters should be intact.
  std::vector<std::string> pieces;
  EXPECT_TRUE(processor.Encode("こんにちは", &pieces).ok());
  for (const auto& p : pieces) {
    EXPECT_TRUE(string_util::IsStructurallyValid(p));
  }

  // 3. Rare characters (e.g. "驫" only appeared once, pruned due to budget)
  // should be byte-fallback. "驫" is UTF-8: 0xE9 0xA9 0xAB.
  EXPECT_TRUE(processor.Encode("驫", &pieces).ok());
  EXPECT_EQ(3, pieces.size());
  EXPECT_EQ("<0xE9>", pieces[0]);
  EXPECT_EQ("<0xA9>", pieces[1]);
  EXPECT_EQ("<0xAB>", pieces[2]);

  // 4. Completely unseen characters should also be byte-fallback.
  // "鬱" is UTF-8: 0xE9 0xAC 0xB1.
  EXPECT_TRUE(processor.Encode("鬱", &pieces).ok());
  EXPECT_EQ(3, pieces.size());
  EXPECT_EQ("<0xE9>", pieces[0]);
  EXPECT_EQ("<0xAC>", pieces[1]);
  EXPECT_EQ("<0xB1>", pieces[2]);

  // 5. Decode should perfectly roundtrip.
  std::string decoded;
  EXPECT_TRUE(processor.Decode(pieces, &decoded).ok());
  EXPECT_EQ("鬱", decoded);
}

TEST(BPETrainerTest, EndToEndTestWithAutoCharacterCoverage) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_auto_character_coverage, true);

  const std::string input =
      filesystem::JoinPath(::testing::SrcDir(), kTestInputData);
  const std::string model_prefix =
      filesystem::JoinPath(::testing::TempDir(), "tmp_model_auto_bpe");

  ASSERT_TRUE(
      SentencePieceTrainer::Train(
          absl::StrCat("--model_prefix=", model_prefix, " --input=", input,
                       " --vocab_size=8000 --normalization_rule_name=identity"
                       " --model_type=bpe --byte_fallback=true "
                       "--max_sentence_length=2048"))
          .ok());

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(model_prefix + ".model").ok());
  EXPECT_EQ(8000, sp.GetPieceSize());

  // Verify all pieces are valid UTF-8 (except byte pieces)
  const auto& model = sp.model_proto();
  for (int i = 0; i < model.pieces_size(); ++i) {
    const auto& piece = model.pieces(i);
    if (piece.type() == ModelProto::SentencePiece::BYTE) continue;
    EXPECT_TRUE(string_util::IsStructurallyValid(piece.piece()))
        << "Invalid piece: " << piece.piece();
  }

  std::vector<std::string> tok;
  EXPECT_TRUE(sp.Encode("吾輩は猫である。未知の漢字驫。", &tok).ok());
  EXPECT_FALSE(tok.empty());

  std::string decoded;
  EXPECT_TRUE(sp.Decode(tok, &decoded).ok());
  EXPECT_EQ("吾輩は猫である。未知の漢字驫。", decoded);
}

TEST(BPETrainerTest, CompletionHierarchyUnlocksOnlyWholeChildren) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "hier_bpe_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "hier_bpe.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "hier_bpe.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "hier_bpe_model");
  const std::string result_path = prefix + ".expansion";

  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abcd\t5"));
  }
  {
    // One parent with children "ab" and "cd".  The byte boundary at 2 may be
    // crossed only when the current left token starts at 0 and the current
    // right token ends at 4.  Thus a+b and c+d must happen before ab+cd.
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    ASSERT_TRUE(out->WriteLine("abcd\t1:0,2,4"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(5);
  expansion.set_requested_new_pieces(3);

  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type, bool mergeable,
                 bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  add(2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  add(3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  add(4, "d", ModelProto::SentencePiece::NORMAL, true, true);
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input);
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_prefix(prefix);
  trainer_spec.set_vocab_size(8);
  trainer_spec.set_expansion_spec(spec_path);
  trainer_spec.set_expansion_result(result_path);
  trainer_spec.set_bpe_hierarchy_file(hierarchy);
  trainer_spec.set_input_sentence_size(0);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_split_digits(false);
  trainer_spec.set_bos_id(-1);
  trainer_spec.set_eos_id(-1);
  trainer_spec.set_pad_id(-1);
  trainer_spec.set_hard_vocab_limit(true);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  normalizer_spec.set_remove_extra_whitespaces(false);
  NormalizerSpec denormalizer_spec;

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  trainer_spec, normalizer_spec, denormalizer_spec)
                  .ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(3, result.learned_merges_size());

  EXPECT_EQ("a", result.learned_merges(0).left());
  EXPECT_EQ("b", result.learned_merges(0).right());
  EXPECT_EQ(0, result.learned_merges(0).grammar_level());
  EXPECT_EQ(5, result.learned_merges(0).weighted_count());

  EXPECT_EQ("c", result.learned_merges(1).left());
  EXPECT_EQ("d", result.learned_merges(1).right());
  EXPECT_EQ(0, result.learned_merges(1).grammar_level());
  EXPECT_EQ(5, result.learned_merges(1).weighted_count());

  EXPECT_EQ("ab", result.learned_merges(2).left());
  EXPECT_EQ("cd", result.learned_merges(2).right());
  EXPECT_EQ(1, result.learned_merges(2).grammar_level());
  EXPECT_EQ(5, result.learned_merges(2).weighted_count());
  EXPECT_TRUE(absl::StartsWith(
      result.boundary_policy(), "bpe_hierarchical_completion_v1:"));
  EXPECT_EQ(result.contract().bpe_hierarchy_sha256(),
            result.boundary_policy().substr(
                std::string("bpe_hierarchical_completion_v1:").size()));
}

TEST(BPETrainerTest, ContractNormalizerComparesByValueNotPresence) {
  // A spec authored from scratch carries no normalization_rule_tsv field;
  // the CLI always sets it (to ""). Same pipeline, different presence bits:
  // this must reconcile, not fail.
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "norm_presence_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "norm_presence.spec");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "norm_presence_model");
  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abab\t5"));
  }
  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(3);
  expansion.set_requested_new_pieces(1);
  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type, bool mergeable,
                 bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  add(2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  auto* ns = expansion.mutable_contract()->mutable_normalizer_spec();
  ns->set_name("identity");
  ns->set_add_dummy_prefix(false);
  ns->set_remove_extra_whitespaces(false);
  ns->set_escape_whitespaces(true);
  ASSERT_FALSE(ns->has_normalization_rule_tsv());
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input);
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_prefix(prefix);
  trainer_spec.set_vocab_size(4);
  trainer_spec.set_expansion_spec(spec_path);
  trainer_spec.set_expansion_result(prefix + ".expansion");
  trainer_spec.set_input_sentence_size(0);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_split_digits(false);
  trainer_spec.set_bos_id(-1);
  trainer_spec.set_eos_id(-1);
  trainer_spec.set_pad_id(-1);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  normalizer_spec.set_remove_extra_whitespaces(false);
  normalizer_spec.set_escape_whitespaces(true);
  normalizer_spec.set_normalization_rule_tsv("");
  ASSERT_TRUE(normalizer_spec.has_normalization_rule_tsv());
  NormalizerSpec denormalizer_spec;
  EXPECT_TRUE(SentencePieceTrainer::Train(
                  trainer_spec, normalizer_spec, denormalizer_spec)
                  .ok());

  // A real value conflict is still refused.
  normalizer_spec.set_add_dummy_prefix(true);
  EXPECT_FALSE(SentencePieceTrainer::Train(
                   trainer_spec, normalizer_spec, denormalizer_spec)
                   .ok());
}

TEST(BPETrainerTest, CompletionHierarchyRejectsContextDependentGlobalPair) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "hier_global_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "hier_global.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "hier_global.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "hier_global_model");
  const std::string result_path = prefix + ".expansion";

  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abcd\t5"));
    ASSERT_TRUE(out->WriteLine("xabcd\t7"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    // In the first row ab|cd are complete siblings. In the second row the
    // same eventual surface pair sits inside xab|cd: bare "ab" is only a
    // suffix of the left child, so ab+cd must NOT become a context-free merge.
    ASSERT_TRUE(out->WriteLine("abcd\t1:0,2,4"));
    ASSERT_TRUE(out->WriteLine("xabcd\t1:0,3,5"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(6);
  expansion.set_requested_new_pieces(3);
  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type, bool mergeable,
                 bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  add(2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  add(3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  add(4, "d", ModelProto::SentencePiece::NORMAL, true, true);
  add(5, "x", ModelProto::SentencePiece::NORMAL, true, true);
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input);
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_prefix(prefix);
  trainer_spec.set_vocab_size(9);
  trainer_spec.set_expansion_spec(spec_path);
  trainer_spec.set_expansion_result(result_path);
  trainer_spec.set_bpe_hierarchy_file(hierarchy);
  trainer_spec.set_input_sentence_size(0);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_split_digits(false);
  trainer_spec.set_bos_id(-1);
  trainer_spec.set_eos_id(-1);
  trainer_spec.set_pad_id(-1);
  trainer_spec.set_hard_vocab_limit(true);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  normalizer_spec.set_remove_extra_whitespaces(false);
  NormalizerSpec denormalizer_spec;
  ASSERT_TRUE(SentencePieceTrainer::Train(
                  trainer_spec, normalizer_spec, denormalizer_spec)
                  .ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(3, result.learned_merges_size());
  bool saw_ab_cd = false;
  for (const auto& merge : result.learned_merges()) {
    if (merge.left() == "ab" && merge.right() == "cd") saw_ab_cd = true;
  }
  EXPECT_FALSE(saw_ab_cd);
  // The third slot goes to the safe completion of the first child in xab|cd.
  EXPECT_EQ("x", result.learned_merges(2).left());
  EXPECT_EQ("ab", result.learned_merges(2).right());
}



}  // namespace
}  // namespace bpe
}  // namespace sentencepiece
