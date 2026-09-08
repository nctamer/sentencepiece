// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "trainer_factory.h"
#include "util.h"

namespace sentencepiece {
namespace {

std::string TempPath(absl::string_view leaf) {
  return filesystem::JoinPath(::testing::TempDir(), leaf);
}

template <typename Proto>
bool WriteProto(absl::string_view path, const Proto& proto) {
  auto output = filesystem::NewWritableFile(path, true);
  if (!output->status().ok()) return false;
  return output->Write(proto.SerializeAsString());
}

template <typename Proto>
bool ReadProto(absl::string_view path, Proto* proto) {
  auto input = filesystem::NewReadableFile(path, true);
  if (!input->status().ok()) return false;
  std::string bytes;
  if (!input->ReadAll(&bytes)) return false;
  return proto->ParseFromString(bytes);
}

bool WriteLines(absl::string_view path,
                const std::vector<std::string>& lines) {
  auto output = filesystem::NewWritableFile(path);
  if (!output->status().ok()) return false;
  for (const auto& line : lines) {
    if (!output->WriteLine(line)) return false;
  }
  return true;
}

NormalizerSpec IdentityNormalizer() {
  NormalizerSpec normalizer;
  normalizer.set_name("identity");
  normalizer.set_add_dummy_prefix(false);
  normalizer.set_remove_extra_whitespaces(false);
  normalizer.set_escape_whitespaces(true);
  return normalizer;
}

ExpansionPiece* AddPiece(ExpansionSpec* spec, int id, absl::string_view piece,
                         ModelProto::SentencePiece::Type type,
                         bool mergeable, bool atomic) {
  ExpansionPiece* out = spec->add_base_pieces();
  out->set_external_id(id);
  out->set_piece(std::string(piece));
  out->set_type(type);
  out->set_score(0.0f);
  out->set_mergeable(mergeable);
  out->set_atomic(atomic);
  return out;
}

ExpansionMerge* AddBaseMerge(ExpansionSpec* spec, int rank,
                             absl::string_view left,
                             absl::string_view right) {
  ExpansionMerge* merge = spec->add_base_merges();
  merge->set_rank(rank);
  merge->set_left(std::string(left));
  merge->set_right(std::string(right));
  return merge;
}

ExpansionPiece* AddBootstrapPiece(ExpansionSpec* spec,
                                  absl::string_view piece) {
  ExpansionPiece* out = spec->add_bootstrap_pieces();
  out->set_external_id(-1);
  out->set_piece(std::string(piece));
  out->set_type(ModelProto::SentencePiece::NORMAL);
  out->set_score(0.0f);
  out->set_mergeable(true);
  out->set_atomic(false);
  return out;
}

ExpansionMerge* AddBootstrapMerge(ExpansionSpec* spec, int rank,
                                  absl::string_view left,
                                  absl::string_view right) {
  ExpansionMerge* merge = spec->add_bootstrap_merges();
  merge->set_rank(rank);
  merge->set_left(std::string(left));
  merge->set_right(std::string(right));
  return merge;
}

TrainerSpec BpeContinuationSpec(absl::string_view input,
                                absl::string_view expansion_spec,
                                absl::string_view result,
                                absl::string_view prefix, int vocab_size) {
  TrainerSpec spec;
  spec.set_model_type(TrainerSpec::BPE);
  spec.add_input(std::string(input));
  spec.set_input_format("text");
  spec.set_model_prefix(std::string(prefix));
  spec.set_vocab_size(vocab_size);
  spec.set_expansion_spec(std::string(expansion_spec));
  spec.set_expansion_result(std::string(result));
  spec.set_hard_vocab_limit(true);
  spec.set_split_by_whitespace(false);
  spec.set_split_by_unicode_script(false);
  spec.set_split_by_number(false);
  spec.set_shuffle_input_sentence(false);
  return spec;
}

absl::Status RunTrainer(const TrainerSpec& spec,
                        const NormalizerSpec& normalizer) {
  NormalizerSpec denormalizer;
  std::unique_ptr<TrainerInterface> trainer =
      TrainerFactory::Create(spec, normalizer, denormalizer);
  return trainer->Train();
}

ExpansionSpec BasicAbcdSpec() {
  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_requested_new_pieces(1);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN,
           false, false);
  AddPiece(&expansion, 1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 4, "d", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 5, "ab", ModelProto::SentencePiece::NORMAL, true, false);
  AddPiece(&expansion, 6, "cd", ModelProto::SentencePiece::NORMAL, true, false);
  AddBaseMerge(&expansion, 0, "a", "b");
  AddBaseMerge(&expansion, 1, "c", "d");
  return expansion;
}

TEST(BPEContinuationContractTest, ReplaysInheritedStateBeforeLearning) {
  const std::string input = TempPath("continuation_bpe_abcd_input.txt");
  const std::string spec_path = TempPath("continuation_bpe_abcd.pb");
  const std::string result_path = TempPath("continuation_bpe_abcd.result");
  const std::string prefix = TempPath("continuation_bpe_abcd_model");
  ASSERT_TRUE(WriteLines(input, {"abcd", "abcd", "abcd"}));
  const ExpansionSpec expansion = BasicAbcdSpec();
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  const TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 8);
  ASSERT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  ASSERT_EQ(1, result.learned_pieces_size());
  ASSERT_EQ(1, result.learned_merges_size());
  EXPECT_EQ(7, result.learned_pieces(0).external_id());
  EXPECT_EQ("abcd", result.learned_pieces(0).piece());
  EXPECT_EQ("ab", result.learned_merges(0).left());
  EXPECT_EQ("cd", result.learned_merges(0).right());
  EXPECT_EQ(2, result.learned_merges(0).rank());
  EXPECT_EQ(0, result.unreachable_pieces());
}

ExpansionSpec RankCompetitionSpec(bool prepend) {
  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_allow_rank_prepend(prepend);
  expansion.set_requested_new_pieces(1);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN,
           false, false);
  AddPiece(&expansion, 1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 4, "ab", ModelProto::SentencePiece::NORMAL, true, false);
  AddBaseMerge(&expansion, 0, "a", "b");
  AddBootstrapPiece(&expansion, "bc");
  AddBootstrapMerge(&expansion, 0, "b", "c");
  return expansion;
}

TEST(BPEContinuationContractTest, RankPrependReplayMatchesSerializedProgram) {
  const std::string input = TempPath("continuation_bpe_prepend_input.txt");
  const std::string spec_path = TempPath("continuation_bpe_prepend.pb");
  const std::string result_path = TempPath("continuation_bpe_prepend.result");
  const std::string prefix = TempPath("continuation_bpe_prepend_model");
  ASSERT_TRUE(WriteLines(input, {"abc", "abc", "abc"}));
  const ExpansionSpec expansion = RankCompetitionSpec(true);
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  const TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 7);
  ASSERT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  ASSERT_EQ(1, result.bootstrap_merges_size());
  ASSERT_EQ(1, result.base_merges_size());
  ASSERT_EQ(1, result.learned_merges_size());
  EXPECT_EQ(0, result.bootstrap_merges(0).rank());
  EXPECT_EQ("b", result.bootstrap_merges(0).left());
  EXPECT_EQ("c", result.bootstrap_merges(0).right());
  EXPECT_EQ(1, result.base_merges(0).rank());
  EXPECT_EQ("a", result.base_merges(0).left());
  EXPECT_EQ("b", result.base_merges(0).right());
  EXPECT_EQ(2, result.learned_merges(0).rank());
  EXPECT_EQ("a", result.learned_merges(0).left());
  EXPECT_EQ("bc", result.learned_merges(0).right());
  EXPECT_EQ("abc", result.learned_pieces(0).piece());
}

TEST(BPEContinuationContractTest, RankAppendReplayMatchesSerializedProgram) {
  const std::string input = TempPath("continuation_bpe_append_input.txt");
  const std::string spec_path = TempPath("continuation_bpe_append.pb");
  const std::string result_path = TempPath("continuation_bpe_append.result");
  const std::string prefix = TempPath("continuation_bpe_append_model");
  ASSERT_TRUE(WriteLines(input, {"abc", "abc", "abc"}));
  const ExpansionSpec expansion = RankCompetitionSpec(false);
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  const TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 7);
  ASSERT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  ASSERT_EQ(1, result.learned_merges_size());
  EXPECT_EQ("ab", result.learned_merges(0).left());
  EXPECT_EQ("c", result.learned_merges(0).right());
  EXPECT_EQ("abc", result.learned_pieces(0).piece());
}

TEST(BPEContinuationContractTest, RejectsPrependDependencyOnLaterBaseMerge) {
  const std::string spec_path = TempPath("continuation_bpe_bad_prepend.pb");
  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_allow_rank_prepend(true);
  expansion.set_requested_new_pieces(0);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN,
           false, false);
  AddPiece(&expansion, 1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 4, "ab", ModelProto::SentencePiece::NORMAL, true, false);
  AddBaseMerge(&expansion, 0, "a", "b");
  AddBootstrapPiece(&expansion, "abc");
  AddBootstrapMerge(&expansion, 0, "ab", "c");
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  TrainerSpec trainer = BpeContinuationSpec(
      TempPath("deliberately_missing_corpus.txt"), spec_path,
      TempPath("continuation_bpe_bad_prepend.result"),
      TempPath("continuation_bpe_bad_prepend_model"), 6);
  const absl::Status status = RunTrainer(trainer, IdentityNormalizer());
  EXPECT_FALSE(status.ok());
  EXPECT_NE(std::string::npos,
            std::string(status.message()).find("not constructible"));
}

TEST(BPEContinuationContractTest, SupportsMultiCodepointAtomicAlphabet) {
  const std::string input = TempPath("continuation_bpe_atomic_input.txt");
  const std::string spec_path = TempPath("continuation_bpe_atomic.pb");
  const std::string result_path = TempPath("continuation_bpe_atomic.result");
  const std::string prefix = TempPath("continuation_bpe_atomic_model");
  ASSERT_TRUE(WriteLines(input, {"abc", "abc"}));

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_requested_new_pieces(1);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN,
           false, false);
  AddPiece(&expansion, 1, "ab", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 2, "c", ModelProto::SentencePiece::NORMAL, true, true);
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  const TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 4);
  ASSERT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  ASSERT_EQ(1, result.learned_merges_size());
  EXPECT_EQ("ab", result.learned_merges(0).left());
  EXPECT_EQ("c", result.learned_merges(0).right());
  EXPECT_EQ("abc", result.learned_pieces(0).piece());
}

TEST(BPEContinuationContractTest, AllocatesAfterMaximumOccupiedExternalId) {
  const std::string input = TempPath("continuation_bpe_id_input.txt");
  const std::string spec_path = TempPath("continuation_bpe_id.pb");
  const std::string result_path = TempPath("continuation_bpe_id.result");
  ASSERT_TRUE(WriteLines(input, {"ab", "ab"}));

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_requested_new_pieces(1);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN,
           false, false);
  AddPiece(&expansion, 1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 20, "<reserved>", ModelProto::SentencePiece::CONTROL,
           false, false);
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  const TrainerSpec trainer = BpeContinuationSpec(
      input, spec_path, result_path, TempPath("continuation_bpe_id_model"), 22);
  ASSERT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  EXPECT_EQ(21, result.first_new_external_id());
  ASSERT_EQ(1, result.learned_pieces_size());
  EXPECT_EQ(21, result.learned_pieces(0).external_id());
}

ModelProto MakeTinyUnigramPrior() {
  ModelProto prior;
  TrainerSpec* trainer = prior.mutable_trainer_spec();
  trainer->set_model_type(TrainerSpec::UNIGRAM);
  trainer->set_vocab_size(3);
  trainer->set_unk_id(0);
  trainer->set_bos_id(-1);
  trainer->set_eos_id(-1);
  trainer->set_pad_id(-1);
  trainer->set_split_by_whitespace(false);
  trainer->set_split_by_unicode_script(false);
  trainer->set_split_by_number(false);

  *prior.mutable_normalizer_spec() = IdentityNormalizer();

  auto* unk = prior.add_pieces();
  unk->set_piece("<unk>");
  unk->set_score(0.0f);
  unk->set_type(ModelProto::SentencePiece::UNKNOWN);

  auto* a = prior.add_pieces();
  a->set_piece("a");
  a->set_score(-0.4f);
  a->set_type(ModelProto::SentencePiece::NORMAL);

  auto* b = prior.add_pieces();
  b->set_piece("b");
  b->set_score(-1.2f);
  b->set_type(ModelProto::SentencePiece::NORMAL);
  return prior;
}

TrainerSpec UnigramContinuationSpec(absl::string_view input,
                                    absl::string_view input_format,
                                    absl::string_view prior,
                                    absl::string_view result,
                                    absl::string_view prefix) {
  TrainerSpec spec;
  spec.set_model_type(TrainerSpec::UNIGRAM);
  spec.add_input(std::string(input));
  spec.set_input_format(std::string(input_format));
  spec.set_model_prefix(std::string(prefix));
  spec.set_vocab_size(4);
  spec.set_unigram_prior_model(std::string(prior));
  spec.set_expansion_result(std::string(result));
  spec.set_hard_vocab_limit(true);
  spec.set_num_threads(1);
  spec.set_num_sub_iterations(2);
  spec.set_shrinking_factor(0.75f);
  spec.set_unk_id(0);
  spec.set_bos_id(-1);
  spec.set_eos_id(-1);
  spec.set_pad_id(-1);
  spec.set_split_by_whitespace(false);
  spec.set_split_by_unicode_script(false);
  spec.set_split_by_number(false);
  spec.set_shuffle_input_sentence(false);
  return spec;
}

TEST(UnigramContinuationContractTest, PreservesPriorIdsAndScoreGeometry) {
  const std::string prior_path = TempPath("continuation_unigram_prior.model");
  const std::string input = TempPath("continuation_unigram_input.txt");
  const std::string result_path = TempPath("continuation_unigram.result");
  const std::string prefix = TempPath("continuation_unigram_model");
  const ModelProto prior = MakeTinyUnigramPrior();
  ASSERT_TRUE(WriteProto(prior_path, prior));
  ASSERT_TRUE(WriteLines(input, {"ab", "ab"}));

  const TrainerSpec trainer = UnigramContinuationSpec(
      input, "text", prior_path, result_path, prefix);
  // Deliberately pass a caller normalizer that differs from the prior. The
  // prior ModelProto is authoritative for continuation normalization.
  NormalizerSpec caller_normalizer;
  ASSERT_TRUE(RunTrainer(trainer, caller_normalizer).ok());

  ModelProto output;
  ASSERT_TRUE(ReadProto(prefix + ".model", &output));
  ASSERT_EQ(4, output.pieces_size());
  for (int id = 0; id < prior.pieces_size(); ++id) {
    EXPECT_EQ(prior.pieces(id).piece(), output.pieces(id).piece());
    EXPECT_EQ(prior.pieces(id).type(), output.pieces(id).type());
  }
  EXPECT_EQ("ab", output.pieces(3).piece());
  EXPECT_EQ(ModelProto::SentencePiece::NORMAL, output.pieces(3).type());
  EXPECT_EQ(prior.normalizer_spec().SerializeAsString(),
            output.normalizer_spec().SerializeAsString());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  EXPECT_EQ(3, result.first_new_external_id());
  EXPECT_EQ(3, result.prior_piece_count());
  ASSERT_EQ(1, result.learned_pieces_size());
  EXPECT_EQ(3, result.learned_pieces(0).external_id());
  EXPECT_LT(result.unigram_score_gauge_max_error(), 1e-5);

  const double shift_a = output.pieces(1).score() - prior.pieces(1).score();
  const double shift_b = output.pieces(2).score() - prior.pieces(2).score();
  EXPECT_NEAR(shift_a, result.unigram_lambda(), 1e-5);
  EXPECT_NEAR(shift_b, result.unigram_lambda(), 1e-5);
  EXPECT_NEAR(output.pieces(1).score() - output.pieces(2).score(),
              prior.pieces(1).score() - prior.pieces(2).score(), 1e-6);
}

TEST(UnigramContinuationContractTest, SpecializesHeadAndFallsBackOnTail) {
  const std::string prior_path = TempPath("continuation_unigram_head_prior.model");
  const std::string input = TempPath("continuation_unigram_head_input.txt");
  const std::string prefix = TempPath("continuation_unigram_head_model");
  const ModelProto prior = MakeTinyUnigramPrior();
  ASSERT_TRUE(WriteProto(prior_path, prior));
  ASSERT_TRUE(WriteLines(input, {"ab", "ab"}));

  const TrainerSpec trainer = UnigramContinuationSpec(
      input, "text", prior_path, TempPath("continuation_unigram_head.result"),
      prefix);
  ASSERT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  SentencePieceProcessor processor;
  ASSERT_TRUE(processor.Load(prefix + ".model").ok());
  std::vector<int> ids;
  ASSERT_TRUE(processor.Encode("ab", &ids).ok());
  ASSERT_EQ(1, ids.size());
  EXPECT_EQ(3, ids[0]);

  ASSERT_TRUE(processor.Encode("ba", &ids).ok());
  ASSERT_EQ(2, ids.size());
  EXPECT_EQ(2, ids[0]);
  EXPECT_EQ(1, ids[1]);
}

TEST(UnigramContinuationContractTest, WeightedTsvMatchesPhysicalRepetition) {
  const std::string prior_path = TempPath("continuation_unigram_weight_prior.model");
  const ModelProto prior = MakeTinyUnigramPrior();
  ASSERT_TRUE(WriteProto(prior_path, prior));

  const std::string raw_input = TempPath("continuation_unigram_weight_raw.txt");
  const std::string tsv_input = TempPath("continuation_unigram_weight.tsv");
  ASSERT_TRUE(WriteLines(raw_input, {"ab", "ab"}));
  ASSERT_TRUE(WriteLines(tsv_input, {"ab\t2"}));

  const std::string raw_result = TempPath("continuation_unigram_weight_raw.result");
  const std::string tsv_result = TempPath("continuation_unigram_weight_tsv.result");
  const TrainerSpec raw = UnigramContinuationSpec(
      raw_input, "text", prior_path, raw_result,
      TempPath("continuation_unigram_weight_raw_model"));
  const TrainerSpec tsv = UnigramContinuationSpec(
      tsv_input, "tsv", prior_path, tsv_result,
      TempPath("continuation_unigram_weight_tsv_model"));
  ASSERT_TRUE(RunTrainer(raw, NormalizerSpec()).ok());
  ASSERT_TRUE(RunTrainer(tsv, NormalizerSpec()).ok());

  ExpansionResult raw_expansion;
  ExpansionResult tsv_expansion;
  ASSERT_TRUE(ReadProto(raw_result, &raw_expansion));
  ASSERT_TRUE(ReadProto(tsv_result, &tsv_expansion));
  ASSERT_EQ(1, raw_expansion.learned_pieces_size());
  ASSERT_EQ(1, tsv_expansion.learned_pieces_size());
  EXPECT_EQ(raw_expansion.learned_pieces(0).piece(),
            tsv_expansion.learned_pieces(0).piece());
  EXPECT_NEAR(raw_expansion.learned_pieces(0).score(),
              tsv_expansion.learned_pieces(0).score(), 1e-7);
  EXPECT_NEAR(raw_expansion.unigram_lambda(),
              tsv_expansion.unigram_lambda(), 1e-10);
}

TEST(ContinuationContractTest, DeterministicBpeResultBytes) {
  const ExpansionSpec expansion = BasicAbcdSpec();
  std::string first_bytes;
  for (int run = 0; run < 2; ++run) {
    const std::string suffix = run == 0 ? "one" : "two";
    const std::string input = TempPath("continuation_bpe_deterministic_" + suffix + ".txt");
    const std::string spec_path = TempPath("continuation_bpe_deterministic_" + suffix + ".pb");
    const std::string result_path = TempPath("continuation_bpe_deterministic_" + suffix + ".result");
    ASSERT_TRUE(WriteLines(input, {"abcd", "abcd"}));
    ASSERT_TRUE(WriteProto(spec_path, expansion));
    const TrainerSpec trainer = BpeContinuationSpec(
        input, spec_path, result_path,
        TempPath("continuation_bpe_deterministic_" + suffix + "_model"), 8);
    ASSERT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());
    auto reader = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(reader->status().ok());
    std::string bytes;
    ASSERT_TRUE(reader->ReadAll(&bytes));
    if (run == 0) {
      first_bytes = std::move(bytes);
    } else {
      EXPECT_EQ(first_bytes, bytes);
    }
  }
}

}  // namespace
}  // namespace sentencepiece
