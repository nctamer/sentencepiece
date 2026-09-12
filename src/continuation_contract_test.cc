// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <cerrno>

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"
#include "filesystem.h"
#include "bpe_continuation_trainer.h"
#include "expansion_processor.h"
#include "normalizer.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "trainer_factory.h"
#include "unigram_continuation_trainer.h"
#include "util.h"

ABSL_DECLARE_FLAG(int32_t, continuation_spill_entries);
ABSL_DECLARE_FLAG(float, min_freq_alpha);
ABSL_DECLARE_FLAG(std::string, continuation_fence_strings);

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


// ---------------------------------------------------------------------------
// Native-model equivalence oracles.
//
// A continuation run is authoritative in its ExpansionResult and .merges. A
// native .model is a second, weaker artifact: SentencePiece BPE inference does
// not read a merge table at all, it merges whichever adjacent pair has the
// best-scoring concatenation in the vocabulary. These tests pin both halves of
// that contract - when a native model is emitted it must agree with the merge
// program exactly, and when it cannot it must not be emitted at all.
// ---------------------------------------------------------------------------

bool FileExists(absl::string_view path) {
  auto input = filesystem::NewReadableFile(path, true);
  return input->status().ok();
}

void RemoveIfPresent(const std::string& path) { std::remove(path.c_str()); }

// Every merge of the result, in effective rank order.
std::vector<ExpansionMerge> EffectiveMerges(const ExpansionResult& result) {
  std::vector<ExpansionMerge> merges;
  for (const auto& m : result.base_merges()) merges.push_back(m);
  for (const auto& m : result.bootstrap_merges()) merges.push_back(m);
  for (const auto& m : result.learned_merges()) merges.push_back(m);
  std::sort(merges.begin(), merges.end(),
            [](const ExpansionMerge& a, const ExpansionMerge& b) {
              return a.rank() < b.rank();
            });
  return merges;
}

// The authoritative reading of a merge program: repeatedly apply the declared
// pair of lowest rank, leftmost occurrence first. This is what an external
// rank-ordered tokenizer does, and what a native model has to reproduce.
std::vector<std::string> SimulateMergeProgram(
    const std::string& text, const std::vector<ExpansionMerge>& merges) {
  std::map<std::pair<std::string, std::string>, int> rank;
  for (const auto& merge : merges) {
    rank[{merge.left(), merge.right()}] = merge.rank();
  }

  std::vector<std::string> symbols;
  for (size_t i = 0; i < text.size();) {
    const int len = std::min<int>(string_util::OneCharLen(text.data() + i),
                                  text.size() - i);
    symbols.push_back(text.substr(i, len));
    i += len;
  }

  while (symbols.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_at = symbols.size();
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto it = rank.find({symbols[i], symbols[i + 1]});
      if (it != rank.end() && it->second < best_rank) {
        best_rank = it->second;
        best_at = i;
      }
    }
    if (best_at == symbols.size()) break;
    symbols[best_at] += symbols[best_at + 1];
    symbols.erase(symbols.begin() + best_at + 1);
  }
  return symbols;
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


TEST(BPEContinuationContractTest, NativeModelAgreesWithMergeProgram) {
  const std::string input = TempPath("continuation_native_input.txt");
  const std::string spec_path = TempPath("continuation_native.pb");
  const std::string result_path = TempPath("continuation_native.result");
  const std::string prefix = TempPath("continuation_native_model");
  RemoveIfPresent(prefix + ".model");
  ASSERT_TRUE(WriteLines(input, {"abcd", "abcd", "abcd"}));
  ASSERT_TRUE(WriteProto(spec_path, BasicAbcdSpec()));

  ASSERT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                             prefix, 8),
                         IdentityNormalizer())
                  .ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  const std::vector<ExpansionMerge> merges = EffectiveMerges(result);

  // Every piece of the program is exactly one vocabulary split here, so a
  // faithful native model is possible and must have been written.
  ASSERT_TRUE(FileExists(prefix + ".model"))
      << "expected an exactly-representable native model";

  SentencePieceProcessor processor;
  ASSERT_TRUE(processor.Load(prefix + ".model").ok());

  for (const std::string& text : {"abcd", "abc", "abab", "dcba", "abcdabcd"}) {
    std::vector<std::string> native;
    ASSERT_TRUE(processor.Encode(text, &native).ok()) << text;
    EXPECT_EQ(SimulateMergeProgram(text, merges), native)
        << "native inference disagreed with the merge program on: " << text;
  }
}

TEST(BPEContinuationContractTest, NativeScoresAreTheEffectiveRankProgram) {
  const std::string input = TempPath("continuation_scores_input.txt");
  const std::string spec_path = TempPath("continuation_scores.pb");
  const std::string result_path = TempPath("continuation_scores.result");
  const std::string prefix = TempPath("continuation_scores_model");
  ASSERT_TRUE(WriteLines(input, {"abcd", "abcd", "abcd"}));
  ASSERT_TRUE(WriteProto(spec_path, BasicAbcdSpec()));

  ASSERT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                             prefix, 8),
                         IdentityNormalizer())
                  .ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  ModelProto model;
  ASSERT_TRUE(ReadProto(prefix + ".model", &model));

  std::map<std::string, float> score;
  for (const auto& piece : model.pieces()) score[piece.piece()] = piece.score();

  // A merge child scores -rank, so "best score" and "lowest rank" are the same
  // relation. The adapter's own scores order nothing and are discarded.
  for (const auto& merge : EffectiveMerges(result)) {
    const std::string child = merge.left() + merge.right();
    ASSERT_TRUE(score.count(child)) << child;
    EXPECT_FLOAT_EQ(-static_cast<float>(merge.rank()), score[child]) << child;
  }
  // Atoms are never merge children and never compete for a merge.
  for (const std::string& atom : {"a", "b", "c", "d"}) {
    ASSERT_TRUE(score.count(atom)) << atom;
    EXPECT_FLOAT_EQ(0.0f, score[atom]) << atom;
  }
  // Learned ranks land after the whole inherited prefix.
  ASSERT_EQ(1, result.learned_merges_size());
  EXPECT_EQ(result.base_merges_size() + result.bootstrap_merges_size(),
            result.learned_merges(0).rank());
}

TEST(BPEContinuationContractTest, RefusesNativeModelWhenSplitIsAmbiguous) {
  // "abc" is reachable as both "a"+"bc" and "ab"+"c", and both halves of each
  // are in the vocabulary. Native inference chooses by concatenation score, so
  // it can take a pair this program never declares. The result and merge table
  // stay authoritative; the misleading .model is not written.
  for (const bool prepend : {false, true}) {
    const std::string tag = prepend ? "prepend" : "append";
    const std::string input = TempPath("continuation_ambig_" + tag + ".txt");
    const std::string spec_path = TempPath("continuation_ambig_" + tag + ".pb");
    const std::string result_path =
        TempPath("continuation_ambig_" + tag + ".result");
    const std::string prefix = TempPath("continuation_ambig_" + tag + "_model");
    RemoveIfPresent(prefix + ".model");
    ASSERT_TRUE(WriteLines(input, {"abc", "abc", "abc"}));
    ASSERT_TRUE(WriteProto(spec_path, RankCompetitionSpec(prepend)));

    ASSERT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                               prefix, 7),
                           IdentityNormalizer())
                    .ok());

    EXPECT_FALSE(FileExists(prefix + ".model"))
        << "emitted a native model that cannot represent the program (" << tag
        << ")";
    // The authoritative artifacts are still produced.
    EXPECT_TRUE(FileExists(result_path));
    EXPECT_TRUE(FileExists(prefix + ".merges"));
  }
}

TEST(BPEContinuationContractTest, RefusesNativeModelForMultiCodepointAtom) {
  // Native inference starts from single characters (or frozen USER_DEFINED
  // prefixes) and no merge may build an atom, so a multi-scalar atom can never
  // be reconstructed natively.
  const std::string input = TempPath("continuation_multiatom_input.txt");
  const std::string spec_path = TempPath("continuation_multiatom.pb");
  const std::string result_path = TempPath("continuation_multiatom.result");
  const std::string prefix = TempPath("continuation_multiatom_model");
  RemoveIfPresent(prefix + ".model");

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_requested_new_pieces(1);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false,
           false);
  AddPiece(&expansion, 1, "xy", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 2, "z", ModelProto::SentencePiece::NORMAL, true, true);
  ASSERT_TRUE(WriteProto(spec_path, expansion));
  ASSERT_TRUE(WriteLines(input, {"xyz", "xyz", "xyz"}));

  ASSERT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                             prefix, 4),
                         IdentityNormalizer())
                  .ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  ASSERT_EQ(1, result.learned_pieces_size());
  EXPECT_EQ("xyz", result.learned_pieces(0).piece());
  EXPECT_FALSE(FileExists(prefix + ".model"));
}


// ---------------------------------------------------------------------------
// Hardening: refuse rather than abort, and never spend budget on inherited
// state.
// ---------------------------------------------------------------------------

// A minimal spec whose only atom is "a".
ExpansionSpec SingleAtomSpec(int requested_new_pieces) {
  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_requested_new_pieces(requested_new_pieces);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false,
           false);
  AddPiece(&expansion, 1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  return expansion;
}

// EncodePos packs two 16-bit symbol indexes. An over-long record used to trip
// a CHECK and take the process down; valid input must produce a status.
TEST(BPEContinuationContractTest, RejectsRecordLongerThanPositionIndex) {
  const std::string input = TempPath("continuation_longrec_input.txt");
  const std::string spec_path = TempPath("continuation_longrec.pb");
  const std::string result_path = TempPath("continuation_longrec.result");
  const std::string prefix = TempPath("continuation_longrec_model");
  ASSERT_TRUE(WriteLines(input, {std::string((1 << 16) + 1, 'a')}));
  ASSERT_TRUE(WriteProto(spec_path, SingleAtomSpec(0)));

  TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 16);
  trainer.set_max_sentence_length(1 << 18);
  const absl::Status status = RunTrainer(trainer, IdentityNormalizer());
  EXPECT_EQ(absl::StatusCode::kOutOfRange, status.code()) << status;
}

// One atom shorter still fits, so the limit is a real boundary rather than a
// blanket refusal of long records.
TEST(BPEContinuationContractTest, AcceptsRecordAtThePositionIndexLimit) {
  const std::string input = TempPath("continuation_atlimit_input.txt");
  const std::string spec_path = TempPath("continuation_atlimit.pb");
  const std::string result_path = TempPath("continuation_atlimit.result");
  const std::string prefix = TempPath("continuation_atlimit_model");
  ASSERT_TRUE(WriteLines(input, {std::string(1 << 16, 'a')}));
  ASSERT_TRUE(WriteProto(spec_path, SingleAtomSpec(0)));

  TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 16);
  trainer.set_max_sentence_length(1 << 18);
  EXPECT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());
}

// Pair frequencies accumulate weight per position. The weighted position mass
// must be proven to fit before any of it is summed.
TEST(BPEContinuationContractTest, RejectsWeightedFrequencyOverflow) {
  const std::string input = TempPath("continuation_ovf_input.tsv");
  const std::string spec_path = TempPath("continuation_ovf.pb");
  const std::string result_path = TempPath("continuation_ovf.result");
  const std::string prefix = TempPath("continuation_ovf_model");
  ASSERT_TRUE(WriteLines(input, {"aaaa\t9000000000000000000"}));
  ASSERT_TRUE(WriteProto(spec_path, SingleAtomSpec(0)));

  TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 16);
  trainer.set_input_format("tsv");
  const absl::Status status = RunTrainer(trainer, IdentityNormalizer());
  EXPECT_EQ(absl::StatusCode::kOutOfRange, status.code()) << status;
}

// An explicit zero is a replay-only run, and must not be read as "unset".
TEST(BPEContinuationContractTest, ExplicitZeroBudgetIsReplayOnly) {
  const std::string input = TempPath("continuation_zero_input.txt");
  const std::string spec_path = TempPath("continuation_zero.pb");
  const std::string result_path = TempPath("continuation_zero.result");
  const std::string prefix = TempPath("continuation_zero_model");
  ASSERT_TRUE(WriteLines(input, {"abcd", "abcd", "abcd"}));
  ExpansionSpec expansion = BasicAbcdSpec();
  expansion.set_requested_new_pieces(0);
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  // vocab_size is deliberately larger than the occupied range: were the budget
  // derived from it, this run would learn a piece.
  ASSERT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                             prefix, 32),
                         IdentityNormalizer())
                  .ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  EXPECT_EQ(0, result.learned_pieces_size());
  EXPECT_EQ(0, result.learned_merges_size());
  EXPECT_EQ(0, result.requested_new_pieces());
  EXPECT_EQ(0, result.actual_new_pieces());
  // The inherited tokenizer is returned intact.
  EXPECT_EQ(7, result.base_pieces_size());
  EXPECT_EQ(2, result.base_merges_size());
  EXPECT_EQ(7, result.first_new_external_id());
}

// An inherited merge the continuation corpus never exercises stays in the
// tokenizer and costs no budget.
TEST(BPEContinuationContractTest, AbsentInheritedMergeIsKeptAndCostsNothing) {
  const std::string input = TempPath("continuation_absent_input.txt");
  const std::string spec_path = TempPath("continuation_absent.pb");
  const std::string result_path = TempPath("continuation_absent.result");
  const std::string prefix = TempPath("continuation_absent_model");
  // "cd" never occurs, so the (c, d) merge cannot fire.
  ASSERT_TRUE(WriteLines(input, {"abab", "abab", "abab"}));
  ExpansionSpec expansion = BasicAbcdSpec();
  expansion.set_requested_new_pieces(1);
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  ASSERT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                             prefix, 8),
                         IdentityNormalizer())
                  .ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  // Both inherited merges are still exported, in their inherited order.
  ASSERT_EQ(2, result.base_merges_size());
  EXPECT_EQ("a", result.base_merges(0).left());
  EXPECT_EQ("b", result.base_merges(0).right());
  EXPECT_EQ("c", result.base_merges(1).left());
  EXPECT_EQ("d", result.base_merges(1).right());
  // The full budget was still available to learning.
  ASSERT_EQ(1, result.learned_pieces_size());
  EXPECT_EQ("abab", result.learned_pieces(0).piece());
  EXPECT_EQ("ab", result.learned_merges(0).left());
  EXPECT_EQ("ab", result.learned_merges(0).right());
}

// A candidate whose string already exists is not a new token: no ID, no
// budget, and no alternative ancestry smuggled into the corpus.
TEST(BPEContinuationContractTest, DuplicateCandidateNeitherAllocatesNorSpends) {
  const std::string input = TempPath("continuation_dup_input.txt");
  const std::string spec_path = TempPath("continuation_dup.pb");
  const std::string result_path = TempPath("continuation_dup.result");
  const std::string prefix = TempPath("continuation_dup_model");
  // "ba" is more frequent than "cd", so it is the first candidate offered.
  ASSERT_TRUE(WriteLines(input, {"ba", "ba", "ba", "ba", "ba", "cd", "cd",
                                 "cd"}));

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_requested_new_pieces(1);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false,
           false);
  AddPiece(&expansion, 1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 4, "d", ModelProto::SentencePiece::NORMAL, true, true);
  // Present in the ID space but built by no merge the program declares.
  AddPiece(&expansion, 5, "ba", ModelProto::SentencePiece::NORMAL, false,
           false);
  ASSERT_TRUE(WriteProto(spec_path, expansion));

  ASSERT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                             prefix, 7),
                         IdentityNormalizer())
                  .ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  // The duplicate was retired, so the budget bought the next real candidate.
  ASSERT_EQ(1, result.learned_pieces_size());
  EXPECT_EQ("cd", result.learned_pieces(0).piece());
  EXPECT_EQ(6, result.learned_pieces(0).external_id());
  // No merge was invented for the pre-existing string.
  for (const auto& merge : result.learned_merges()) {
    EXPECT_NE("ba", merge.left() + merge.right());
  }
}

// Shape options describe what a NEW piece may look like. A setting that cannot
// even express the inherited tokenizer is a configuration error, and must be
// caught before the corpus is opened - this input path does not exist.
TEST(BPEContinuationContractTest, RejectsShapeThatCannotExpressInheritedPiece) {
  const std::string spec_path = TempPath("continuation_shape.pb");
  const std::string result_path = TempPath("continuation_shape.result");
  const std::string prefix = TempPath("continuation_shape_model");
  ASSERT_TRUE(WriteProto(spec_path, BasicAbcdSpec()));

  TrainerSpec trainer = BpeContinuationSpec(
      TempPath("continuation_shape_does_not_exist.txt"), spec_path,
      result_path, prefix, 8);
  trainer.set_max_sentencepiece_length(1);
  const absl::Status status = RunTrainer(trainer, IdentityNormalizer());
  EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;
  EXPECT_NE(std::string::npos, std::string(status.message()).find("ab"));
}

// The legacy interval/barline pretokenizer is fresh-training policy; a
// continuation run carries its fence in the spec instead.
TEST(BPEContinuationContractTest, RejectsLegacyBoundaryFlags) {
  const std::string spec_path = TempPath("continuation_legacyflag.pb");
  const std::string result_path = TempPath("continuation_legacyflag.result");
  const std::string prefix = TempPath("continuation_legacyflag_model");
  ASSERT_TRUE(WriteProto(spec_path, BasicAbcdSpec()));

  TrainerSpec trainer = BpeContinuationSpec(
      TempPath("continuation_legacyflag_input.txt"), spec_path, result_path,
      prefix, 8);
  trainer.SetExtension(split_by_interval, true);
  EXPECT_FALSE(RunTrainer(trainer, IdentityNormalizer()).ok());
}

// Training twice through one trainer object must not carry state across.
TEST(BPEContinuationContractTest, TrainerReuseDoesNotLeakState) {
  const std::string input = TempPath("continuation_reuse_input.txt");
  const std::string spec_path = TempPath("continuation_reuse.pb");
  const std::string result_path = TempPath("continuation_reuse.result");
  const std::string prefix = TempPath("continuation_reuse_model");
  ASSERT_TRUE(WriteLines(input, {"abcd", "abcd", "abcd"}));
  ASSERT_TRUE(WriteProto(spec_path, BasicAbcdSpec()));

  const TrainerSpec trainer_spec =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 8);
  NormalizerSpec denormalizer;
  std::unique_ptr<TrainerInterface> trainer = TrainerFactory::Create(
      trainer_spec, IdentityNormalizer(), denormalizer);

  ASSERT_TRUE(trainer->Train().ok());
  ExpansionResult first;
  ASSERT_TRUE(ReadProto(result_path, &first));

  ASSERT_TRUE(trainer->Train().ok());
  ExpansionResult second;
  ASSERT_TRUE(ReadProto(result_path, &second));

  EXPECT_EQ(first.SerializeAsString(), second.SerializeAsString());
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


// ---------------------------------------------------------------------------
// Unigram continuation: the prior is authoritative, and that has to be
// enforced rather than assumed.
// ---------------------------------------------------------------------------

// A prior with genuinely competing inherited paths: "ab" is a piece, and so
// are "a" and "b", so the same string has two inherited-only segmentations
// whose scores can be compared before and after continuation.
ModelProto MakeMultiPathUnigramPrior() {
  ModelProto prior;
  TrainerSpec* trainer = prior.mutable_trainer_spec();
  trainer->set_model_type(TrainerSpec::UNIGRAM);
  trainer->set_vocab_size(4);
  trainer->set_unk_id(0);
  trainer->set_bos_id(-1);
  trainer->set_eos_id(-1);
  trainer->set_pad_id(-1);
  trainer->set_split_by_whitespace(false);
  trainer->set_split_by_unicode_script(false);
  trainer->set_split_by_number(false);
  *prior.mutable_normalizer_spec() = IdentityNormalizer();

  struct Entry {
    const char* piece;
    float score;
    ModelProto::SentencePiece::Type type;
  };
  // score(ab) - (score(a) + score(b)) = -2.5 - (-3.0) = +0.5, so "ab" wins its
  // own Viterbi while "a" + "b" stays a real alternative.
  for (const Entry& e :
       {Entry{"<unk>", 0.0f, ModelProto::SentencePiece::UNKNOWN},
        Entry{"a", -1.0f, ModelProto::SentencePiece::NORMAL},
        Entry{"b", -2.0f, ModelProto::SentencePiece::NORMAL},
        Entry{"ab", -2.5f, ModelProto::SentencePiece::NORMAL}}) {
    auto* piece = prior.add_pieces();
    piece->set_piece(e.piece);
    piece->set_score(e.score);
    piece->set_type(e.type);
  }
  return prior;
}

double ScoreOf(const ModelProto& model, absl::string_view piece) {
  for (const auto& p : model.pieces()) {
    if (p.piece() == piece) return p.score();
  }
  ADD_FAILURE() << "piece missing from model: " << piece;
  return std::numeric_limits<double>::quiet_NaN();
}

// The gauge exists so inherited segmentation decisions survive continuation.
// The observable form of that promise is a score DIFFERENCE between two
// inherited paths over the same string, which must not move at all.
TEST(UnigramContinuationContractTest, InheritedPathGeometryIsUnchanged) {
  const std::string input = TempPath("continuation_uni_geom_input.txt");
  const std::string prior_path = TempPath("continuation_uni_geom.prior");
  const std::string result_path = TempPath("continuation_uni_geom.result");
  const std::string prefix = TempPath("continuation_uni_geom_model");

  const ModelProto prior = MakeMultiPathUnigramPrior();
  ASSERT_TRUE(WriteProto(prior_path, prior));
  // "abab" gives the extension something worth learning.
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(40, "abab")));

  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(6);  // four inherited + two extensions
  ASSERT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  ModelProto output;
  ASSERT_TRUE(ReadProto(prefix + ".model", &output));

  const double delta_before =
      ScoreOf(prior, "ab") - (ScoreOf(prior, "a") + ScoreOf(prior, "b"));
  const double delta_after =
      ScoreOf(output, "ab") - (ScoreOf(output, "a") + ScoreOf(output, "b"));
  EXPECT_NEAR(delta_before, delta_after, 1e-5)
      << "continuation moved inherited paths relative to each other";

  // And the decision that difference encodes is still the decision taken.
  SentencePieceProcessor processor;
  ASSERT_TRUE(processor.Load(prefix + ".model").ok());
  std::vector<std::string> pieces;
  ASSERT_TRUE(processor.Encode("ab", &pieces).ok());
  EXPECT_EQ(std::vector<std::string>({"ab"}), pieces);

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  // Every inherited ID, string and type is untouched, and extensions append.
  ASSERT_GE(output.pieces_size(), prior.pieces_size());
  for (int id = 0; id < prior.pieces_size(); ++id) {
    EXPECT_EQ(prior.pieces(id).piece(), output.pieces(id).piece()) << id;
    EXPECT_EQ(prior.pieces(id).type(), output.pieces(id).type()) << id;
  }
  EXPECT_EQ(prior.pieces_size(), result.prior_piece_count());
}

// A run that is asked for no extension must hand the prior back unchanged -
// no recalibration, no invented pieces, not even a nonzero gauge.
TEST(UnigramContinuationContractTest, ZeroExtensionReturnsThePriorUnchanged) {
  const std::string input = TempPath("continuation_uni_noop_input.txt");
  const std::string prior_path = TempPath("continuation_uni_noop.prior");
  const std::string result_path = TempPath("continuation_uni_noop.result");
  const std::string prefix = TempPath("continuation_uni_noop_model");

  const ModelProto prior = MakeMultiPathUnigramPrior();
  ASSERT_TRUE(WriteProto(prior_path, prior));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(20, "abab")));

  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(prior.pieces_size());  // nothing left to learn
  ASSERT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  ModelProto output;
  ASSERT_TRUE(ReadProto(prefix + ".model", &output));
  ASSERT_EQ(prior.pieces_size(), output.pieces_size());
  for (int id = 0; id < prior.pieces_size(); ++id) {
    EXPECT_EQ(prior.pieces(id).piece(), output.pieces(id).piece()) << id;
    EXPECT_EQ(prior.pieces(id).type(), output.pieces(id).type()) << id;
    EXPECT_FLOAT_EQ(prior.pieces(id).score(), output.pieces(id).score()) << id;
  }

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  EXPECT_EQ(0, result.learned_pieces_size());
  EXPECT_EQ(0, result.actual_new_pieces());
  EXPECT_DOUBLE_EQ(0.0, result.unigram_lambda());
}

TEST(UnigramContinuationContractTest, DeterministicAcrossRunsAndThreadCounts) {
  const std::string input = TempPath("continuation_uni_det_input.txt");
  const std::string prior_path = TempPath("continuation_uni_det.prior");
  ASSERT_TRUE(WriteProto(prior_path, MakeMultiPathUnigramPrior()));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(60, "abab")));

  auto run = [&](int num_threads, const std::string& tag) {
    const std::string result_path =
        TempPath("continuation_uni_det_" + tag + ".result");
    const std::string prefix =
        TempPath("continuation_uni_det_" + tag + "_model");
    TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                  result_path, prefix);
    trainer.set_vocab_size(6);
    trainer.set_num_threads(num_threads);
    EXPECT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());
    ModelProto output;
    EXPECT_TRUE(ReadProto(prefix + ".model", &output));
    return output;
  };

  // The piece table is the model. trainer_spec carries the output path, which
  // necessarily differs between two runs writing to two places.
  auto piece_table = [](const ModelProto& model) {
    ModelProto pieces_only;
    *pieces_only.mutable_pieces() = model.pieces();
    return pieces_only.SerializeAsString();
  };

  const ModelProto first = run(1, "a");
  const ModelProto again = run(1, "b");
  EXPECT_EQ(piece_table(first), piece_table(again))
      << "identical inputs produced different models";
  EXPECT_DOUBLE_EQ(first.expansion_result().unigram_lambda(),
                   again.expansion_result().unigram_lambda());

  // Thread count is a scheduling detail, not a modelling one.
  const ModelProto threaded = run(4, "c");
  ASSERT_EQ(first.pieces_size(), threaded.pieces_size());
  for (int id = 0; id < first.pieces_size(); ++id) {
    EXPECT_EQ(first.pieces(id).piece(), threaded.pieces(id).piece()) << id;
    EXPECT_EQ(first.pieces(id).type(), threaded.pieces(id).type()) << id;
    EXPECT_NEAR(first.pieces(id).score(), threaded.pieces(id).score(), 1e-4)
        << id;
  }
}

// Normalization is inherited, but a caller who asked for something else is
// told, not overruled in silence.
TEST(UnigramContinuationContractTest, NormalizationConflictIsRejected) {
  const std::string input = TempPath("continuation_uni_norm_input.txt");
  const std::string prior_path = TempPath("continuation_uni_norm.prior");
  const std::string result_path = TempPath("continuation_uni_norm.result");
  const std::string prefix = TempPath("continuation_uni_norm_model");
  ASSERT_TRUE(WriteProto(prior_path, MakeMultiPathUnigramPrior()));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(20, "abab")));

  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(6);

  // Unspecified: inherit the prior.
  EXPECT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  // Exactly the prior's: accepted.
  EXPECT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());

  // Populated and different: refused before training.
  NormalizerSpec conflicting = IdentityNormalizer();
  conflicting.set_add_dummy_prefix(true);
  const absl::Status status = RunTrainer(trainer, conflicting);
  EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;

  NormalizerSpec renamed = IdentityNormalizer();
  renamed.set_name("nmt_nfkc");
  EXPECT_FALSE(RunTrainer(trainer, renamed).ok());
}

// A prior carrying a non-default normalizer must still be honoured verbatim.
TEST(UnigramContinuationContractTest, CustomPriorNormalizerIsCarriedThrough) {
  const std::string input = TempPath("continuation_uni_cnorm_input.txt");
  const std::string prior_path = TempPath("continuation_uni_cnorm.prior");
  const std::string result_path = TempPath("continuation_uni_cnorm.result");
  const std::string prefix = TempPath("continuation_uni_cnorm_model");

  ModelProto prior = MakeMultiPathUnigramPrior();
  // Not the identity spec the other priors use, and not the CLI default.
  prior.mutable_normalizer_spec()->set_name("intermo_custom");
  prior.mutable_normalizer_spec()->set_remove_extra_whitespaces(true);
  ASSERT_TRUE(WriteProto(prior_path, prior));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(20, "abab")));

  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(6);
  ASSERT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  ModelProto output;
  ASSERT_TRUE(ReadProto(prefix + ".model", &output));
  EXPECT_EQ("intermo_custom", output.normalizer_spec().name());
  EXPECT_TRUE(output.normalizer_spec().remove_extra_whitespaces());
}


// ---------------------------------------------------------------------------
// Malformed input is refused, and refused for the stated reason. Each case
// mutates one field of an otherwise-valid spec, so what is under test is the
// mutation and not the scaffolding.
// ---------------------------------------------------------------------------

absl::Status RunWithSpec(const ExpansionSpec& expansion, absl::string_view tag,
                         const std::vector<std::string>& corpus = {"abcd",
                                                                   "abcd",
                                                                   "abcd"},
                         int vocab_size = 8) {
  const std::string input = TempPath(absl::StrCat("cont_bad_", tag, ".txt"));
  const std::string spec_path = TempPath(absl::StrCat("cont_bad_", tag, ".pb"));
  const std::string result_path =
      TempPath(absl::StrCat("cont_bad_", tag, ".result"));
  const std::string prefix = TempPath(absl::StrCat("cont_bad_", tag, "_model"));
  if (!WriteLines(input, corpus)) return absl::InternalError("write corpus");
  if (!WriteProto(spec_path, expansion)) {
    return absl::InternalError("write spec");
  }
  return RunTrainer(
      BpeContinuationSpec(input, spec_path, result_path, prefix, vocab_size),
      IdentityNormalizer());
}

TEST(BPEContinuationContractTest, RejectsDuplicateBaseExternalId) {
  ExpansionSpec expansion = BasicAbcdSpec();
  expansion.mutable_base_pieces(3)->set_external_id(2);  // collides with "b"
  EXPECT_FALSE(RunWithSpec(expansion, "dupid").ok());
}

TEST(BPEContinuationContractTest, RejectsDuplicateBasePieceString) {
  ExpansionSpec expansion = BasicAbcdSpec();
  expansion.mutable_base_pieces(3)->set_piece("b");  // "c" becomes a second "b"
  EXPECT_FALSE(RunWithSpec(expansion, "dupstr").ok());
}

TEST(BPEContinuationContractTest, RejectsNoncontiguousMergeRanks) {
  ExpansionSpec expansion = BasicAbcdSpec();
  expansion.mutable_base_merges(1)->set_rank(7);
  EXPECT_FALSE(RunWithSpec(expansion, "ranks").ok());
}

TEST(BPEContinuationContractTest, RejectsMergeGraphWithUnconstructibleParent) {
  ExpansionSpec expansion = BasicAbcdSpec();
  // "ab" is built at rank 0, so a merge consuming "abc" at rank 1 has a parent
  // nothing ever constructs.
  expansion.mutable_base_merges(1)->set_left("abc");
  expansion.mutable_base_merges(1)->set_right("d");
  EXPECT_FALSE(RunWithSpec(expansion, "graph").ok());
}

TEST(BPEContinuationContractTest, RejectsMergeChildIdMismatch) {
  ExpansionSpec expansion = BasicAbcdSpec();
  // "cd" is declared as ID 6; claim the merge builds something else.
  expansion.mutable_base_merges(1)->set_external_id(3);
  EXPECT_FALSE(RunWithSpec(expansion, "childid").ok());
}

TEST(BPEContinuationContractTest, RejectsBootstrapCollidingWithBase) {
  ExpansionSpec expansion = BasicAbcdSpec();
  AddBootstrapPiece(&expansion, "ab");  // already an inherited piece
  AddBootstrapMerge(&expansion, 0, "a", "b");
  EXPECT_FALSE(RunWithSpec(expansion, "bootcollide").ok());
}

TEST(BPEContinuationContractTest, RejectsFirstNewIdOverlappingOccupiedIds) {
  ExpansionSpec expansion = BasicAbcdSpec();
  expansion.set_first_new_external_id(4);  // "d" already holds 4
  EXPECT_FALSE(RunWithSpec(expansion, "firstid").ok());
}

TEST(BPEContinuationContractTest, RejectsAllocationPastTheIdRange) {
  ExpansionSpec expansion = BasicAbcdSpec();
  expansion.set_first_new_external_id(std::numeric_limits<int>::max() - 1);
  expansion.set_requested_new_pieces(8);
  const absl::Status status = RunWithSpec(expansion, "idcap");
  EXPECT_EQ(absl::StatusCode::kOutOfRange, status.code()) << status;
}

TEST(BPEContinuationContractTest, RejectsAmbiguousAtomicSegmentation) {
  // "ab" is an atom AND "a"/"b" are atoms, so "ab" has two parses.
  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_requested_new_pieces(0);
  AddPiece(&expansion, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false,
           false);
  AddPiece(&expansion, 1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  AddPiece(&expansion, 3, "ab", ModelProto::SentencePiece::NORMAL, true, true);
  EXPECT_FALSE(RunWithSpec(expansion, "ambig", {"ab", "ab"}, 16).ok());
}

TEST(BPEContinuationContractTest, RejectsUncoveredAtomicSegmentation) {
  // The corpus contains "z", which the declared alphabet cannot spell.
  EXPECT_FALSE(RunWithSpec(BasicAbcdSpec(), "uncovered", {"abcz", "abcz"}).ok());
}

// A weighted record must be exactly equivalent to physical repetition.
TEST(BPEContinuationContractTest, WeightedTsvMatchesPhysicalRepetition) {
  auto learn = [&](bool weighted) {
    const std::string tag = weighted ? "w" : "r";
    const std::string input = TempPath("cont_bpe_weight_" + tag +
                                       (weighted ? ".tsv" : ".txt"));
    const std::string spec_path = TempPath("cont_bpe_weight_" + tag + ".pb");
    const std::string result_path =
        TempPath("cont_bpe_weight_" + tag + ".result");
    const std::string prefix = TempPath("cont_bpe_weight_" + tag + "_model");

    std::vector<std::string> lines;
    if (weighted) {
      lines = {"abcd\t5", "abab\t2"};
    } else {
      for (int i = 0; i < 5; ++i) lines.push_back("abcd");
      for (int i = 0; i < 2; ++i) lines.push_back("abab");
    }
    EXPECT_TRUE(WriteLines(input, lines));
    ExpansionSpec expansion = BasicAbcdSpec();
    expansion.set_requested_new_pieces(2);
    EXPECT_TRUE(WriteProto(spec_path, expansion));

    TrainerSpec trainer =
        BpeContinuationSpec(input, spec_path, result_path, prefix, 9);
    if (weighted) trainer.set_input_format("tsv");
    trainer.set_hard_vocab_limit(false);
    EXPECT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());

    ExpansionResult result;
    EXPECT_TRUE(ReadProto(result_path, &result));
    std::vector<std::string> learned;
    for (const auto& piece : result.learned_pieces()) {
      learned.push_back(piece.piece());
    }
    return learned;
  };

  EXPECT_EQ(learn(false), learn(true));
}

// ---------------------------------------------------------------------------
// Unigram prior validation.
// ---------------------------------------------------------------------------

absl::Status RunWithPrior(const ModelProto& prior, absl::string_view tag,
                          int vocab_size = 6) {
  const std::string input = TempPath(absl::StrCat("cont_uni_bad_", tag, ".txt"));
  const std::string prior_path =
      TempPath(absl::StrCat("cont_uni_bad_", tag, ".prior"));
  const std::string result_path =
      TempPath(absl::StrCat("cont_uni_bad_", tag, ".result"));
  const std::string prefix =
      TempPath(absl::StrCat("cont_uni_bad_", tag, "_model"));
  if (!WriteLines(input, std::vector<std::string>(20, "abab"))) {
    return absl::InternalError("write corpus");
  }
  if (!WriteProto(prior_path, prior)) return absl::InternalError("write prior");
  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(vocab_size);
  return RunTrainer(trainer, NormalizerSpec());
}

TEST(UnigramContinuationContractTest, RejectsPriorOfTheWrongModelType) {
  ModelProto prior = MakeMultiPathUnigramPrior();
  prior.mutable_trainer_spec()->set_model_type(TrainerSpec::BPE);
  EXPECT_FALSE(RunWithPrior(prior, "type").ok());
}

TEST(UnigramContinuationContractTest, RejectsDuplicateOrInvalidPriorState) {
  {  // two pieces with the same string
    ModelProto prior = MakeMultiPathUnigramPrior();
    prior.mutable_pieces(2)->set_piece("a");
    EXPECT_FALSE(RunWithPrior(prior, "dup").ok());
  }
  {  // no UNKNOWN
    ModelProto prior = MakeMultiPathUnigramPrior();
    prior.mutable_pieces(0)->set_type(ModelProto::SentencePiece::NORMAL);
    EXPECT_FALSE(RunWithPrior(prior, "nounk").ok());
  }
  {  // two UNKNOWNs
    ModelProto prior = MakeMultiPathUnigramPrior();
    prior.mutable_pieces(1)->set_type(ModelProto::SentencePiece::UNKNOWN);
    EXPECT_FALSE(RunWithPrior(prior, "twounk").ok());
  }
  {  // a non-finite inherited score has no gauge to sit on
    ModelProto prior = MakeMultiPathUnigramPrior();
    prior.mutable_pieces(1)->set_score(
        std::numeric_limits<float>::infinity());
    EXPECT_FALSE(RunWithPrior(prior, "inf").ok());
  }
}

TEST(UnigramContinuationContractTest, AdmitsUnsupportedAlphabetAsRequiredCoverage) {
  // CONTRACT CHANGE, DELIBERATE. This used to assert that a corpus containing
  // a character the prior cannot spell is REJECTED. That behaviour was the
  // coverage defect: continuation admitted the missing character as an
  // extension candidate and then discarded it (and everything containing it)
  // at initialization, which made the whole affected family unreachable --
  // measured on the real corpus as 341,334 V-bearing candidates admitted and
  // 0 surviving. Missing characters are now admitted as REQUIRED atomic
  // COVERAGE EXTENSIONS: they consume extension budget, are protected from
  // pruning, remain free parameters in every M-step, and the prior is never
  // modified. So this must now SUCCEED.
  const std::string input = TempPath("cont_uni_alpha.txt");
  const std::string prior_path = TempPath("cont_uni_alpha.prior");
  const std::string result_path = TempPath("cont_uni_alpha.result");
  const std::string prefix = TempPath("cont_uni_alpha_model");
  ASSERT_TRUE(WriteProto(prior_path, MakeMultiPathUnigramPrior()));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(20, "zzzz")));
  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(6);
  EXPECT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());
}

TEST(UnigramContinuationContractTest, RejectsCapacityBelowThePrior) {
  // Unchanged contract: asking for a vocabulary smaller than the prior cannot
  // be honoured, because the prior is the floor and inherited pieces are never
  // pruned.
  EXPECT_FALSE(RunWithPrior(MakeMultiPathUnigramPrior(), "capacity",
                            /*vocab_size=*/2)
                   .ok());
}

TEST(UnigramContinuationContractTest, RejectsSeedSentencepiecesConflict) {
  const std::string input = TempPath("cont_uni_seed.txt");
  const std::string prior_path = TempPath("cont_uni_seed.prior");
  const std::string result_path = TempPath("cont_uni_seed.result");
  const std::string prefix = TempPath("cont_uni_seed_model");
  const std::string seed_path = TempPath("cont_uni_seed_pieces.txt");
  ASSERT_TRUE(WriteProto(prior_path, MakeMultiPathUnigramPrior()));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(20, "abab")));
  ASSERT_TRUE(WriteLines(seed_path, {"ab\t-1.0"}));

  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(6);
  trainer.set_seed_sentencepieces_file(seed_path);
  // The prior already fixes the starting vocabulary; a second seed source is
  // ambiguous rather than additive.
  EXPECT_FALSE(RunTrainer(trainer, NormalizerSpec()).ok());
}


// ---------------------------------------------------------------------------
// The guards themselves. Everything above proves the happy path keeps the
// contract; these prove that when the contract is broken the run stops. A
// guard nothing can trigger is a guard nobody has checked.
// ---------------------------------------------------------------------------

TEST(UnigramContinuationContractTest, GaugeInvariantRejectsAPerturbedScore) {
  const ModelProto prior = MakeMultiPathUnigramPrior();
  constexpr double kLambda = -0.25;

  // A faithful continuation: every inherited NORMAL score moved by exactly
  // lambda * length, and an extension appended.
  ModelProto output = prior;
  for (int id = 0; id < output.pieces_size(); ++id) {
    if (output.pieces(id).type() != ModelProto::SentencePiece::NORMAL) continue;
    const int length = static_cast<int>(
        string_util::UTF8Len(output.pieces(id).piece()));
    output.mutable_pieces(id)->set_score(static_cast<float>(
        prior.pieces(id).score() + kLambda * length));
  }
  auto* added = output.add_pieces();
  added->set_piece("abab");
  added->set_score(-4.0f);
  added->set_type(ModelProto::SentencePiece::NORMAL);

  ASSERT_TRUE(unigram::VerifyPriorPrefixInvariant(prior, output, kLambda).ok())
      << "a correctly gauged model must pass";

  {  // One inherited score nudged off the shared gauge.
    ModelProto bad = output;
    for (int id = 0; id < bad.pieces_size(); ++id) {
      if (bad.pieces(id).type() == ModelProto::SentencePiece::NORMAL) {
        bad.mutable_pieces(id)->set_score(bad.pieces(id).score() - 0.01f);
        break;
      }
    }
    const absl::Status status =
        unigram::VerifyPriorPrefixInvariant(prior, bad, kLambda);
    EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;
    EXPECT_NE(std::string::npos,
              std::string(status.message()).find("additive-length gauge"));
  }
  {  // An inherited piece restrung: same ID, different bytes.
    ModelProto bad = output;
    bad.mutable_pieces(1)->set_piece("zzz");
    EXPECT_FALSE(unigram::VerifyPriorPrefixInvariant(prior, bad, kLambda).ok());
  }
  {  // An inherited piece retyped.
    ModelProto bad = output;
    bad.mutable_pieces(1)->set_type(ModelProto::SentencePiece::USER_DEFINED);
    EXPECT_FALSE(unigram::VerifyPriorPrefixInvariant(prior, bad, kLambda).ok());
  }
  {  // A non-NORMAL inherited score is not allowed to drift at all.
    ModelProto bad = output;
    bad.mutable_pieces(0)->set_score(prior.pieces(0).score() - 0.5f);
    EXPECT_FALSE(unigram::VerifyPriorPrefixInvariant(prior, bad, kLambda).ok());
  }
  {  // Truncating the prior prefix.
    ModelProto bad = output;
    bad.mutable_pieces()->RemoveLast();
    bad.mutable_pieces()->RemoveLast();
    EXPECT_FALSE(unigram::VerifyPriorPrefixInvariant(prior, bad, kLambda).ok());
  }
}

TEST(UnigramContinuationContractTest, SolverReportsItsFailuresInsteadOfGuessing) {
  double root = 0.0;

  // A genuine root is found, so the failures below are about the objective and
  // not about the search being broken.
  ASSERT_TRUE(unigram::BisectMonotoneRoot(
                  "linear", [](double x) { return x - 3.0; },
                  /*increasing=*/true, -1.0, 1.0, &root)
                  .ok());
  EXPECT_NEAR(3.0, root, 1e-6);

  {  // Never crosses zero: unbracketed, and no lambda may be reported.
    const absl::Status status = unigram::BisectMonotoneRoot(
        "always positive", [](double) { return 1.0; }, true, -1.0, 1.0, &root);
    EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;
    EXPECT_NE(std::string::npos,
              std::string(status.message()).find("not bracketed"));
  }
  {  // Leaves the finite range while bracketing.
    const absl::Status status = unigram::BisectMonotoneRoot(
        "non-finite", [](double) {
          return std::numeric_limits<double>::quiet_NaN();
        }, true, -1.0, 1.0, &root);
    EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;
    EXPECT_NE(std::string::npos,
              std::string(status.message()).find("finite range"));
  }
  {  // Finite at the ends, non-finite in the interior.
    const absl::Status status = unigram::BisectMonotoneRoot(
        "non-finite interior", [](double x) {
          if (x <= -1.0) return -1.0;
          if (x >= 1.0) return 1.0;
          return std::numeric_limits<double>::infinity();
        }, true, -1.0, 1.0, &root);
    EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;
  }
  {  // Decreasing objectives are handled by the same search.
    ASSERT_TRUE(unigram::BisectMonotoneRoot(
                    "decreasing", [](double x) { return 5.0 - x; },
                    /*increasing=*/false, -1.0, 1.0, &root)
                    .ok());
    EXPECT_NEAR(5.0, root, 1e-6);
  }
}

// With a soft vocabulary limit, too few candidates is not a failure: the run
// returns the smaller extension it could actually justify, and returns the
// same one every time.
TEST(UnigramContinuationContractTest, SoftCapacityYieldsADeterministicSmallerExtension) {
  const std::string input = TempPath("cont_uni_soft.txt");
  const std::string prior_path = TempPath("cont_uni_soft.prior");
  ASSERT_TRUE(WriteProto(prior_path, MakeMultiPathUnigramPrior()));
  // One short repeated record offers very few distinct extension candidates.
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(30, "abab")));

  auto run = [&](const std::string& tag, bool hard) {
    const std::string result_path = TempPath("cont_uni_soft_" + tag + ".result");
    const std::string prefix = TempPath("cont_uni_soft_" + tag + "_model");
    TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                  result_path, prefix);
    trainer.set_vocab_size(400);  // far more than the corpus can supply
    trainer.set_hard_vocab_limit(hard);
    return std::make_pair(RunTrainer(trainer, NormalizerSpec()), prefix);
  };

  // Hard limit: the shortfall is an error.
  EXPECT_FALSE(run("hard", true).first.ok());

  // Soft limit: succeed with fewer, and identically on a rerun.
  auto first = run("a", false);
  auto second = run("b", false);
  ASSERT_TRUE(first.first.ok()) << first.first;
  ASSERT_TRUE(second.first.ok()) << second.first;

  ModelProto ma, mb;
  ASSERT_TRUE(ReadProto(first.second + ".model", &ma));
  ASSERT_TRUE(ReadProto(second.second + ".model", &mb));
  EXPECT_LT(ma.pieces_size(), 400) << "soft capacity should return fewer";
  EXPECT_GT(ma.pieces_size(), 0);
  ASSERT_EQ(ma.pieces_size(), mb.pieces_size());
  for (int i = 0; i < ma.pieces_size(); ++i) {
    EXPECT_EQ(ma.pieces(i).piece(), mb.pieces(i).piece()) << i;
    EXPECT_EQ(ma.pieces(i).score(), mb.pieces(i).score()) << i;
  }
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


// ---------------------------------------------------------------------------
// Required coverage extensions, spill exactness, and the inherited-meta fence.
//
// These four properties were each established by a measurement on the real
// corpus during the repair and are pinned here so a future edit cannot undo
// them silently.
// ---------------------------------------------------------------------------

// A prior whose alphabet is {a, b} plus a USER_DEFINED meta symbol. "z" is
// deliberately absent, so any corpus containing it needs a coverage repair.
ModelProto MakeMetaAlphabetUnigramPrior() {
  ModelProto prior = MakeMultiPathUnigramPrior();
  prior.clear_pieces();
  struct Entry {
    const char* piece;
    float score;
    ModelProto::SentencePiece::Type type;
  };
  for (const Entry& e :
       {Entry{"<unk>", 0.0f, ModelProto::SentencePiece::UNKNOWN},
        Entry{"a", -1.0f, ModelProto::SentencePiece::NORMAL},
        Entry{"b", -1.2f, ModelProto::SentencePiece::NORMAL},
        Entry{"<X>", -9.0f, ModelProto::SentencePiece::USER_DEFINED}}) {
    auto* piece = prior.add_pieces();
    piece->set_piece(e.piece);
    piece->set_score(e.score);
    piece->set_type(e.type);
  }
  prior.mutable_trainer_spec()->set_vocab_size(prior.pieces_size());
  return prior;
}

bool HasPiece(const ModelProto& model, absl::string_view piece) {
  for (const auto& p : model.pieces()) {
    if (p.piece() == piece) return true;
  }
  return false;
}

// PROPERTY: a required coverage extension is not free. It occupies one of the
// requested new slots, exactly like an ordinary extension, and the prior is
// never enlarged to make room for it. With a single slot available, the
// coverage repair takes it and nothing else is learned.
TEST(UnigramContinuationContractTest, RequiredCoverageConsumesExtensionCapacity) {
  const std::string prior_path = TempPath("cont_uni_cap_prior.model");
  const std::string input = TempPath("cont_uni_cap_input.txt");
  const std::string result_path = TempPath("cont_uni_cap.result");
  const std::string prefix = TempPath("cont_uni_cap_model");
  const ModelProto prior = MakeMultiPathUnigramPrior();  // <unk>, a, b, ab
  ASSERT_TRUE(WriteProto(prior_path, prior));
  // "z" is outside the prior alphabet; "abab" is richly attested so an
  // ordinary extension would certainly be learned if a slot were free.
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(40, "abababz")));

  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(prior.pieces_size() + 1);  // exactly ONE new slot
  ASSERT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  ModelProto output;
  ASSERT_TRUE(ReadProto(prefix + ".model", &output));
  ASSERT_EQ(prior.pieces_size() + 1, output.pieces_size());
  // The one slot went to the coverage repair, not to a frequent ordinary
  // candidate: capacity is shared, not extended.
  EXPECT_EQ("z", output.pieces(prior.pieces_size()).piece());
  EXPECT_EQ(ModelProto::SentencePiece::NORMAL,
            output.pieces(prior.pieces_size()).type());
  for (int id = 0; id < prior.pieces_size(); ++id) {
    EXPECT_EQ(prior.pieces(id).piece(), output.pieces(id).piece());
    EXPECT_EQ(prior.pieces(id).type(), output.pieces(id).type());
  }
}

// PROPERTY: `is_required` governs PRUNING ELIGIBILITY ONLY. The coverage
// extension's probability is a free parameter re-estimated from posterior
// expected counts in every constrained M-step, so making it more frequent in
// the corpus must move its score. A fabricated or pinned score would be
// identical across these two runs.
TEST(UnigramContinuationContractTest, RequiredCoverageIsFreelyReestimated) {
  const std::string prior_path = TempPath("cont_uni_reest_prior.model");
  const ModelProto prior = MakeMultiPathUnigramPrior();
  ASSERT_TRUE(WriteProto(prior_path, prior));

  auto run = [&](const std::string& tag,
                 const std::vector<std::string>& lines) {
    const std::string input = TempPath("cont_uni_reest_" + tag + ".txt");
    const std::string prefix = TempPath("cont_uni_reest_" + tag + "_model");
    EXPECT_TRUE(WriteLines(input, lines));
    TrainerSpec trainer = UnigramContinuationSpec(
        input, "text", prior_path, TempPath("cont_uni_reest_" + tag + ".result"),
        prefix);
    trainer.set_vocab_size(prior.pieces_size() + 1);
    EXPECT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());
    ModelProto out;
    EXPECT_TRUE(ReadProto(prefix + ".model", &out));
    return out;
  };

  const ModelProto rare = run("rare", std::vector<std::string>(40, "ababababz"));
  const ModelProto often = run("often", std::vector<std::string>(40, "zzzzzzzab"));
  ASSERT_TRUE(HasPiece(rare, "z"));
  ASSERT_TRUE(HasPiece(often, "z"));

  const double s_rare = ScoreOf(rare, "z");
  const double s_often = ScoreOf(often, "z");
  EXPECT_TRUE(std::isfinite(s_rare));
  EXPECT_TRUE(std::isfinite(s_often));
  // More posterior mass on "z" means a higher (less negative) log score.
  EXPECT_GT(s_often, s_rare + 1e-3)
      << "z score did not respond to its corpus frequency: " << s_rare
      << " vs " << s_often;
}

// PROPERTY: candidate extraction spills to disk and merges externally, and the
// spill threshold is a MEMORY knob, not a modelling one. A tiny threshold
// forces many runs and a k-way merge; a large one keeps everything in memory.
// Both must produce the identical artifact, because the aggregation is exact.
TEST(UnigramContinuationContractTest, SpillThresholdDoesNotChangeTheResult) {
  const std::string prior_path = TempPath("cont_uni_spill_prior.model");
  const std::string input = TempPath("cont_uni_spill_input.txt");
  const ModelProto prior = MakeMultiPathUnigramPrior();
  ASSERT_TRUE(WriteProto(prior_path, prior));
  std::vector<std::string> lines;
  for (int i = 0; i < 60; ++i) {
    lines.push_back("abababab");
    lines.push_back("babababa");
    lines.push_back("aabbaabb");
  }
  ASSERT_TRUE(WriteLines(input, lines));

  auto run = [&](const std::string& tag, int32_t spill) {
    const int32_t saved = absl::GetFlag(FLAGS_continuation_spill_entries);
    absl::SetFlag(&FLAGS_continuation_spill_entries, spill);
    const std::string prefix = TempPath("cont_uni_spill_" + tag + "_model");
    TrainerSpec trainer = UnigramContinuationSpec(
        input, "text", prior_path, TempPath("cont_uni_spill_" + tag + ".result"),
        prefix);
    trainer.set_vocab_size(prior.pieces_size() + 2);
    const absl::Status status = RunTrainer(trainer, NormalizerSpec());
    absl::SetFlag(&FLAGS_continuation_spill_entries, saved);
    EXPECT_TRUE(status.ok()) << tag << ": " << status;
    ModelProto out;
    EXPECT_TRUE(ReadProto(prefix + ".model", &out));
    return out;
  };

  const ModelProto tiny = run("tiny", 16);          // forces many spill runs
  const ModelProto huge = run("huge", 1 << 24);     // never spills
  ASSERT_EQ(tiny.pieces_size(), huge.pieces_size());
  for (int i = 0; i < tiny.pieces_size(); ++i) {
    EXPECT_EQ(tiny.pieces(i).piece(), huge.pieces(i).piece()) << "at id " << i;
    EXPECT_EQ(tiny.pieces(i).score(), huge.pieces(i).score()) << "at id " << i;
  }
}

// PROPERTY: an inherited meta symbol (USER_DEFINED / CONTROL / UNKNOWN / BYTE)
// is an atomic boundary. No extension candidate may contain one or straddle
// one, so an extension can only be built inside a fenced span.
TEST(UnigramContinuationContractTest, InheritedMetaSymbolsFenceCandidates) {
  const std::string prior_path = TempPath("cont_uni_fence_prior.model");
  const std::string input = TempPath("cont_uni_fence_input.txt");
  const std::string prefix = TempPath("cont_uni_fence_model");
  const ModelProto prior = MakeMetaAlphabetUnigramPrior();  // <unk>, a, b, <X>
  ASSERT_TRUE(WriteProto(prior_path, prior));
  // "aa" and "bb" are legal candidates. "a<X>", "<X>b" and "aa<X>bb" are not,
  // and "<X>" itself is inherited, not learnable.
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(60, "aa<X>bb")));

  TrainerSpec trainer = UnigramContinuationSpec(
      input, "text", prior_path, TempPath("cont_uni_fence.result"), prefix);
  trainer.set_vocab_size(prior.pieces_size() + 2);
  ASSERT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  ModelProto output;
  ASSERT_TRUE(ReadProto(prefix + ".model", &output));
  ASSERT_EQ(prior.pieces_size() + 2, output.pieces_size());
  for (int id = prior.pieces_size(); id < output.pieces_size(); ++id) {
    const std::string& piece = output.pieces(id).piece();
    EXPECT_EQ(std::string::npos, piece.find("<X>"))
        << "extension straddles or contains the inherited meta symbol: "
        << piece;
    EXPECT_EQ(std::string::npos, piece.find('<')) << piece;
    EXPECT_EQ(std::string::npos, piece.find('>')) << piece;
  }
  // The prior itself is untouched by the fence.
  for (int id = 0; id < prior.pieces_size(); ++id) {
    EXPECT_EQ(prior.pieces(id).piece(), output.pieces(id).piece());
    EXPECT_EQ(prior.pieces(id).type(), output.pieces(id).type());
  }
}


// ---------------------------------------------------------------------------
// Cleanup regressions: SPM_TRACK, spill scratch lifetime, exact required
// coverage frequencies, artifact score agreement, Unicode length, and the
// conditional K-independence claim.
// ---------------------------------------------------------------------------

// Counts entries under a directory tree, or -1 when it does not exist.
int CountDirEntries(const std::string& dir) {
  DIR* d = ::opendir(dir.c_str());
  if (d == nullptr) return -1;
  int n = 0;
  while (struct dirent* e = ::readdir(d)) {
    const std::string name = e->d_name;
    if (name != "." && name != "..") ++n;
  }
  ::closedir(d);
  return n;
}

// Any spm_continuation_spill_* directory still present under `base`.
std::vector<std::string> LeftoverSpillDirs(const std::string& base) {
  std::vector<std::string> found;
  DIR* d = ::opendir(base.c_str());
  if (d == nullptr) return found;
  while (struct dirent* e = ::readdir(d)) {
    const std::string name = e->d_name;
    if (name.rfind("spm_continuation_spill_", 0) == 0) {
      found.push_back(filesystem::JoinPath(base, name));
    }
  }
  ::closedir(d);
  return found;
}

class ScopedEnv {
 public:
  ScopedEnv(const char* key, const std::string& value) : key_(key) {
    const char* old = ::getenv(key);
    had_ = old != nullptr;
    if (had_) old_ = old;
    ::setenv(key, value.c_str(), 1);
  }
  ~ScopedEnv() {
    if (had_) {
      ::setenv(key_.c_str(), old_.c_str(), 1);
    } else {
      ::unsetenv(key_.c_str());
    }
  }
 private:
  std::string key_, old_;
  bool had_ = false;
};

// A prior over {a, b} whose corpus will also contain characters it cannot
// spell, so required coverage extensions are exercised.
ModelProto MakeAbUnigramPrior() {
  ModelProto prior = MakeTinyUnigramPrior();
  prior.mutable_trainer_spec()->set_vocab_size(prior.pieces_size());
  return prior;
}

// ITEM 1. SPM_TRACK used to index a never-filled vector -- undefined behaviour
// on every tracked run. This just has to COMPLETE.
TEST(UnigramContinuationContractTest, TrackingPathRunsWithoutInvalidAccess) {
  const std::string prior_path = TempPath("cont_track_prior.model");
  const std::string input = TempPath("cont_track_input.txt");
  const std::string prefix = TempPath("cont_track_model");
  ASSERT_TRUE(WriteProto(prior_path, MakeAbUnigramPrior()));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(60, "abababab")));

  ScopedEnv track("SPM_TRACK", "1");
  TrainerSpec trainer = UnigramContinuationSpec(
      input, "text", prior_path, TempPath("cont_track.result"), prefix);
  trainer.set_vocab_size(5);
  EXPECT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  ModelProto out;
  ASSERT_TRUE(ReadProto(prefix + ".model", &out));
  EXPECT_EQ(5, out.pieces_size());
}

// ITEM 3. Spill scratch is private and removed on EVERY exit path.
TEST(UnigramContinuationContractTest, SpillScratchIsRemovedOnSuccess) {
  const std::string base = TempPath("cont_spillbase_ok");
  ASSERT_EQ(0, ::mkdir(base.c_str(), 0700) == 0 || errno == EEXIST ? 0 : -1);
  ScopedEnv tmp("TMPDIR", base);

  const std::string prior_path = TempPath("cont_spillok_prior.model");
  const std::string input = TempPath("cont_spillok_input.txt");
  const std::string prefix = TempPath("cont_spillok_model");
  ASSERT_TRUE(WriteProto(prior_path, MakeAbUnigramPrior()));
  std::vector<std::string> lines;
  for (int i = 0; i < 80; ++i) {
    lines.push_back("abababab");
    lines.push_back("babababa");
  }
  ASSERT_TRUE(WriteLines(input, lines));

  const int32_t saved = absl::GetFlag(FLAGS_continuation_spill_entries);
  absl::SetFlag(&FLAGS_continuation_spill_entries, 16);  // force many runs
  TrainerSpec trainer = UnigramContinuationSpec(
      input, "text", prior_path, TempPath("cont_spillok.result"), prefix);
  trainer.set_vocab_size(5);
  const absl::Status status = RunTrainer(trainer, NormalizerSpec());
  absl::SetFlag(&FLAGS_continuation_spill_entries, saved);
  EXPECT_TRUE(status.ok()) << status;
  EXPECT_TRUE(LeftoverSpillDirs(base).empty())
      << "a successful run left spill files behind";
}

TEST(UnigramContinuationContractTest, SpillScratchIsRemovedOnFailure) {
  const std::string base = TempPath("cont_spillbase_fail");
  ::mkdir(base.c_str(), 0700);
  ScopedEnv tmp("TMPDIR", base);
  // Fail only AFTER at least one sorted run exists, so cleanup has something
  // real to remove.
  ScopedEnv fail("SPM_SPILL_FAIL_AFTER_RUNS", "1");

  const std::string prior_path = TempPath("cont_spillfail_prior.model");
  const std::string input = TempPath("cont_spillfail_input.txt");
  const std::string prefix = TempPath("cont_spillfail_model");
  ASSERT_TRUE(WriteProto(prior_path, MakeAbUnigramPrior()));
  std::vector<std::string> lines(80, "abababab");
  ASSERT_TRUE(WriteLines(input, lines));

  const int32_t saved = absl::GetFlag(FLAGS_continuation_spill_entries);
  absl::SetFlag(&FLAGS_continuation_spill_entries, 16);
  TrainerSpec trainer = UnigramContinuationSpec(
      input, "text", prior_path, TempPath("cont_spillfail.result"), prefix);
  trainer.set_vocab_size(5);
  const absl::Status status = RunTrainer(trainer, NormalizerSpec());
  absl::SetFlag(&FLAGS_continuation_spill_entries, saved);
  EXPECT_FALSE(status.ok()) << "the fault injection did not fire";
  for (const auto& dir : LeftoverSpillDirs(base)) {
    ADD_FAILURE() << "a failed run left " << CountDirEntries(dir)
                  << " spill file(s) in " << dir;
  }
}

// ITEM 4. Two missing characters with deliberately unequal weighted counts.
// Their INITIAL required probabilities must follow the exact frequency ratio,
// so this reads the initialization log rather than the post-EM model: final
// re-estimation would otherwise hide an initialization bug.
TEST(UnigramContinuationContractTest, RequiredCoverageFrequenciesAreExact) {
  const std::string prior_path = TempPath("cont_reqfreq_prior.model");
  const std::string input = TempPath("cont_reqfreq_input.tsv");
  const std::string prefix = TempPath("cont_reqfreq_model");
  const std::string dump = TempPath("cont_reqfreq_init.tsv");
  ASSERT_TRUE(WriteProto(prior_path, MakeAbUnigramPrior()));
  // 'Y' and the two-byte 'ü' are both absent from the prior. Weighted
  // occurrences: Y = 3*10 = 30, ue = 1*10 + 1*2 = 12. Deliberately unequal,
  // and deliberately not 1 -- the previous lookup returned exactly 1 for every
  // required character, which silently imposed an EQUAL split of the required
  // mass while the code claimed proportionality.
  ASSERT_TRUE(WriteLines(input, {"aYbYaYb\t10", "aüb\t10", "büa\t2"}));

  ScopedEnv dump_env("SPM_DUMP_REQUIRED_INIT", dump);
  TrainerSpec trainer = UnigramContinuationSpec(
      input, "tsv", prior_path, TempPath("cont_reqfreq.result"), prefix);
  trainer.set_vocab_size(5);   // 3 inherited + exactly the 2 required
  ASSERT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  // INITIALIZATION is what is inspected. Final EM re-estimation would hide an
  // initialization bug entirely: both pieces would still be present with
  // plausible scores.
  std::map<std::string, std::pair<uint64_t, double>> init;
  {
    auto in = filesystem::NewReadableFile(dump);
    ASSERT_TRUE(in->status().ok());
    std::string line;
    while (in->ReadLine(&line)) {
      if (line.empty()) continue;
      const std::vector<std::string> f = absl::StrSplit(line, '\t');
      ASSERT_EQ(3u, f.size()) << line;
      uint64_t freq = 0;
      double q = 0.0;
      ASSERT_TRUE(absl::SimpleAtoi(f[1], &freq));
      ASSERT_TRUE(absl::SimpleAtod(f[2], &q));
      init[f[0]] = {freq, q};
    }
  }
  ASSERT_EQ(2u, init.size());
  ASSERT_TRUE(init.count("Y"));
  ASSERT_TRUE(init.count("ü"));

  // Exact weighted counts, not synthetic 1s.
  EXPECT_EQ(30u, init["Y"].first);
  EXPECT_EQ(12u, init["ü"].first);

  // Initial required probabilities follow that exact ratio ...
  EXPECT_NEAR(30.0 / 12.0, init["Y"].second / init["ü"].second, 1e-9)
      << "required mass was not split in proportion to weighted frequency";
  // ... and together they take the whole residual, so the covered basis is
  // normalized rather than merely close.
  ModelProto prior;
  ASSERT_TRUE(ReadProto(prior_path, &prior));
  double base_mass = 0.0;
  for (const auto& p : prior.pieces()) {
    if (p.type() == ModelProto::SentencePiece::NORMAL) {
      base_mass += std::exp(static_cast<double>(p.score()));
    }
  }
  EXPECT_NEAR(1.0 - base_mass, init["Y"].second + init["ü"].second, 1e-12);

  ModelProto out;
  ASSERT_TRUE(ReadProto(prefix + ".model", &out));
  ASSERT_EQ(5, out.pieces_size());
  bool has_y = false, has_u = false;
  for (const auto& p : out.pieces()) {
    if (p.piece() == "Y") has_y = true;
    if (p.piece() == "ü") has_u = true;
  }
  EXPECT_TRUE(has_y);
  EXPECT_TRUE(has_u);
}

// ITEM 6. .model, .expansion and .vocab must describe ONE tokenizer.
TEST(UnigramContinuationContractTest, ArtifactsAgreeOnFinalScores) {
  const std::string prior_path = TempPath("cont_artifacts_prior.model");
  const std::string input = TempPath("cont_artifacts_input.txt");
  const std::string prefix = TempPath("cont_artifacts_model");
  const std::string result_path = TempPath("cont_artifacts.result");
  const ModelProto prior = MakeAbUnigramPrior();
  ASSERT_TRUE(WriteProto(prior_path, prior));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(60, "abababab")));

  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(5);
  ASSERT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());

  ModelProto model;
  ExpansionResult result;
  ASSERT_TRUE(ReadProto(prefix + ".model", &model));
  ASSERT_TRUE(ReadProto(result_path, &result));

  std::map<int, const ExpansionPiece*> by_id;
  for (const auto& p : result.base_pieces()) by_id[p.external_id()] = &p;
  for (const auto& p : result.learned_pieces()) by_id[p.external_id()] = &p;
  ASSERT_EQ(model.pieces_size(), static_cast<int>(by_id.size()));

  // The gauge really did move the inherited scores, or this proves nothing.
  bool some_inherited_moved = false;
  for (int id = 0; id < prior.pieces_size(); ++id) {
    if (prior.pieces(id).type() == ModelProto::SentencePiece::NORMAL &&
        prior.pieces(id).score() != model.pieces(id).score()) {
      some_inherited_moved = true;
    }
  }
  EXPECT_TRUE(some_inherited_moved)
      << "lambda was 0, so this test cannot distinguish pre- from post-gauge";

  for (int id = 0; id < model.pieces_size(); ++id) {
    const auto* sidecar = by_id[id];
    ASSERT_NE(nullptr, sidecar) << "id " << id << " missing from ExpansionResult";
    EXPECT_EQ(model.pieces(id).piece(), sidecar->piece()) << "id " << id;
    EXPECT_EQ(model.pieces(id).type(), sidecar->type()) << "id " << id;
    // Bit-identical: both are the same float32.
    EXPECT_EQ(model.pieces(id).score(), sidecar->score())
        << "id " << id << " (" << model.pieces(id).piece()
        << ") .model and .expansion disagree";
  }

  // ... and the textual .vocab round-trips those same float32 values.
  auto vocab = filesystem::NewReadableFile(prefix + ".vocab");
  ASSERT_TRUE(vocab->status().ok());
  std::string line;
  int rows = 0;
  while (vocab->ReadLine(&line)) {
    if (line.empty()) continue;
    const std::vector<std::string> f = absl::StrSplit(line, '\t');
    ASSERT_EQ(3u, f.size()) << line;
    int id = 0;
    ASSERT_TRUE(absl::SimpleAtoi(f[2], &id));
    float score = 0.0f;
    ASSERT_TRUE(absl::SimpleAtof(f[1], &score)) << line;
    ASSERT_LT(id, model.pieces_size());
    EXPECT_EQ(model.pieces(id).score(), score)
        << "id " << id << " .vocab does not round-trip the model score";
    ++rows;
  }
  EXPECT_EQ(model.pieces_size(), rows);
}

// ITEM 7. The short-candidate exemption from the dynamic frequency floor must
// be measured in CHARACTERS. Under byte length, "abc" was exempt while the
// same-length "aüc" (4 bytes) was not.
TEST(UnigramContinuationContractTest, ShortCandidateExemptionUsesUnicodeLength) {
  const std::string prior_path = TempPath("cont_ulen_prior.model");
  const std::string input = TempPath("cont_ulen_input.tsv");
  ModelProto prior = MakeAbUnigramPrior();
  for (const char* ch : {"c", "ü", "x"}) {
    auto* p = prior.add_pieces();
    p->set_piece(ch);
    p->set_score(-3.0f);
    p->set_type(ModelProto::SentencePiece::NORMAL);
  }
  prior.mutable_trainer_spec()->set_vocab_size(prior.pieces_size());
  ASSERT_TRUE(WriteProto(prior_path, prior));
  // "abc" and "aüc" are both 3 CHARACTERS and appear equally often; only
  // their byte lengths differ (3 vs 4).
  ASSERT_TRUE(WriteLines(input, {"abcx\t40", "aücx\t40"}));

  auto run = [&](const std::string& tag) {
    const std::string prefix = TempPath("cont_ulen_" + tag);
    TrainerSpec trainer = UnigramContinuationSpec(
        input, "tsv", prior_path, TempPath("cont_ulen_" + tag + ".result"),
        prefix);
    trainer.set_vocab_size(prior.pieces_size() + 6);
    trainer.set_hard_vocab_limit(false);
    EXPECT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());
    ModelProto m;
    EXPECT_TRUE(ReadProto(prefix + ".model", &m));
    return m;
  };

  // The filter is only active for a nonzero min_freq_alpha; at the validated
  // default the exemption never binds. Exercise it where it can bind.
  const float saved = absl::GetFlag(FLAGS_min_freq_alpha);
  absl::SetFlag(&FLAGS_min_freq_alpha, 1.0f);
  const ModelProto m = run("active");
  absl::SetFlag(&FLAGS_min_freq_alpha, saved);

  bool has_ascii = false, has_multibyte = false;
  for (const auto& p : m.pieces()) {
    if (p.piece() == "abc") has_ascii = true;
    if (p.piece() == "aüc") has_multibyte = true;
  }
  // Whatever the filter decides, it must decide the same for both: they are
  // the same length in characters and equally frequent.
  EXPECT_EQ(has_ascii, has_multibyte)
      << "the short-candidate exemption still depends on UTF-8 byte length";
}

// ITEM 8. Pool-size independence from the extension target, pinned at the
// validated default min_freq_alpha = 0 -- and only there.
TEST(UnigramContinuationContractTest, CandidatePoolIsKIndependentAtAlphaZero) {
  const std::string prior_path = TempPath("cont_kindep_prior.model");
  const std::string input = TempPath("cont_kindep_input.tsv");
  ASSERT_TRUE(WriteProto(prior_path, MakeAbUnigramPrior()));
  ASSERT_TRUE(WriteLines(input, {"abababab\t50", "babababa\t30",
                                 "aabbaabb\t20", "abbaabba\t14",
                                 "aaabbbab\t9"}));

  const float saved = absl::GetFlag(FLAGS_min_freq_alpha);
  absl::SetFlag(&FLAGS_min_freq_alpha, 0.0f);
  auto run = [&](int vocab_size) {
    const std::string prefix =
        TempPath(absl::StrCat("cont_kindep_", vocab_size));
    TrainerSpec trainer = UnigramContinuationSpec(
        input, "tsv", prior_path,
        TempPath(absl::StrCat("cont_kindep_", vocab_size, ".result")), prefix);
    trainer.set_vocab_size(vocab_size);
    trainer.set_seed_sentencepiece_size(64);   // pool size, set explicitly
    EXPECT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());
    ModelProto m;
    EXPECT_TRUE(ReadProto(prefix + ".model", &m));
    return m;
  };
  const ModelProto small = run(5);
  const ModelProto large = run(7);
  absl::SetFlag(&FLAGS_min_freq_alpha, saved);

  ASSERT_EQ(5, small.pieces_size());
  ASSERT_EQ(7, large.pieces_size());
  // A smaller K must be a PREFIX of the larger run's selection order: same
  // candidate pool, same ranking, fewer slots. If the pool moved with K this
  // would not hold.
  for (int id = 0; id < small.pieces_size(); ++id) {
    EXPECT_EQ(small.pieces(id).piece(), large.pieces(id).piece())
        << "at id " << id << ": the candidate pool moved with K";
  }
}


// ---------------------------------------------------------------------------
// G3: explicit string-keyed candidate fences.
//
// Every test here is an EXTRACTION-level test. None of them asks EM to select
// or reject a piece: whether a candidate survives pruning is an optimizer
// outcome, and it is not what the fence controls. The fence controls which
// candidates are PROPOSED.
// ---------------------------------------------------------------------------

using PieceFreq = std::pair<std::string, uint64_t>;

// Runs continuation with SPM_DUMP_CANDIDATES and returns the extracted pool.
std::vector<PieceFreq> ExtractCandidates(const ModelProto& prior,
                                         const std::vector<std::string>& lines,
                                         const std::string& fence_spec,
                                         const std::string& tag,
                                         int vocab_size = 0,
                                         std::string* dump_path_out = nullptr,
                                         absl::Status* status_out = nullptr) {
  const std::string prior_path = TempPath("g3_" + tag + "_prior.model");
  const std::string input = TempPath("g3_" + tag + "_input.txt");
  const std::string prefix = TempPath("g3_" + tag + "_model");
  const std::string dump = TempPath("g3_" + tag + "_cands.tsv");
  EXPECT_TRUE(WriteProto(prior_path, prior));
  EXPECT_TRUE(WriteLines(input, lines));
  ::remove(dump.c_str());

  const std::string saved_fence =
      absl::GetFlag(FLAGS_continuation_fence_strings);
  absl::SetFlag(&FLAGS_continuation_fence_strings, fence_spec);
  ScopedEnv dump_env("SPM_DUMP_CANDIDATES", dump);

  TrainerSpec trainer = UnigramContinuationSpec(
      input, "text", prior_path, TempPath("g3_" + tag + ".result"), prefix);
  trainer.set_vocab_size(vocab_size > 0 ? vocab_size : prior.pieces_size() + 2);
  trainer.set_max_sentencepiece_length(8);
  const absl::Status status = RunTrainer(trainer, NormalizerSpec());
  absl::SetFlag(&FLAGS_continuation_fence_strings, saved_fence);
  if (status_out != nullptr) *status_out = status;
  if (dump_path_out != nullptr) *dump_path_out = dump;

  std::vector<PieceFreq> out;
  auto in = filesystem::NewReadableFile(dump);
  if (!in->status().ok()) return out;
  std::string line;
  while (in->ReadLine(&line)) {
    if (line.empty()) continue;
    const std::vector<std::string> f = absl::StrSplit(line, '\t');
    if (f.size() != 3) continue;
    uint64_t freq = 0;
    if (!absl::SimpleAtoi(f[2], &freq)) continue;
    out.emplace_back(f[1], freq);
  }
  return out;
}

bool HasCandidate(const std::vector<PieceFreq>& pool, const std::string& p) {
  for (const auto& e : pool) {
    if (e.first == p) return true;
  }
  return false;
}

std::string ReadWhole(const std::string& path) {
  auto in = filesystem::NewReadableFile(path);
  if (!in->status().ok()) return "<unreadable>";
  std::string all;
  if (!in->ReadAll(&all)) return "<unreadable>";
  return all;
}

// The prior for the G3 tests: alphabet plus a USER_DEFINED meta symbol, so the
// union of both fence sources can be exercised. Context switches are ORDINARY
// NORMAL text and are deliberately NOT pieces of the prior.
ModelProto MakeG3Prior() {
  ModelProto prior;
  TrainerSpec* t = prior.mutable_trainer_spec();
  t->set_model_type(TrainerSpec::UNIGRAM);
  t->set_unk_id(0);
  t->set_bos_id(-1);
  t->set_eos_id(-1);
  t->set_pad_id(-1);
  t->set_split_by_whitespace(false);
  t->set_split_by_unicode_script(false);
  t->set_split_by_number(false);
  // The REAL corpus normalizer: identity, dummy prefix on, whitespace escaped.
  // This is what turns a logical "Vn:" into its boundary-bearing surface.
  NormalizerSpec* n = prior.mutable_normalizer_spec();
  n->set_name("identity");
  n->set_add_dummy_prefix(true);
  n->set_remove_extra_whitespaces(false);
  n->set_escape_whitespaces(true);

  auto add = [&prior](absl::string_view piece, float score,
                      ModelProto::SentencePiece::Type type) {
    auto* p = prior.add_pieces();
    p->set_piece(std::string(piece));
    p->set_score(score);
    p->set_type(type);
  };
  add("<unk>", 0.0f, ModelProto::SentencePiece::UNKNOWN);
  add("<X>", -9.0f, ModelProto::SentencePiece::USER_DEFINED);
  float score = -2.0f;
  for (const char* ch : {"\xe2\x96\x81", "a", "b", "c", "d", "e", "f", "n",
                         "L", "P", "R", "V", ":"}) {
    add(ch, score, ModelProto::SentencePiece::NORMAL);
    score -= 0.05f;
  }
  t->set_vocab_size(prior.pieces_size());
  return prior;
}

// A variant whose normalizer adds no dummy prefix, so a logical fence string
// IS its own surface. Used only where the property under test is about the
// MATCHER (overlapping spans), not about this corpus's normalization.
ModelProto MakeG3PriorNoDummyPrefix() {
  ModelProto prior = MakeG3Prior();
  prior.mutable_normalizer_spec()->set_add_dummy_prefix(false);
  return prior;
}

// Normalizes a logical string exactly the way the trainer does, so assertions
// never contain a guessed U+2581 spelling.
std::string NormalizeLikeTrainer(const ModelProto& prior,
                                 absl::string_view logical) {
  normalizer::Normalizer norm(prior.normalizer_spec(), prior.trainer_spec());
  EXPECT_TRUE(norm.status().ok());
  return norm.Normalize(logical);
}

// THE ASSUMPTION THE WHOLE FLAG RESTS ON, tested rather than assumed: a
// logical "Vn:" supplied by the caller must normalize to the surface that
// actually occurs in normalized training text. With add_dummy_prefix=true and
// escape_whitespaces=true that is the boundary-bearing form, which is also
// what stops it matching inside unrelated material like "fooVn:bar".
TEST(UnigramContinuationContractTest, LogicalFenceStringNormalizesToCorpusSurface) {
  const ModelProto prior = MakeG3Prior();
  const std::string surface = NormalizeLikeTrainer(prior, "Vn:");
  EXPECT_EQ("\xe2\x96\x81"
            "Vn:", surface)
      << "logical Vn: did not normalize to the boundary-bearing surface";

  // And the same normalizer applied to a training record really does contain
  // that surface, at a word boundary.
  const std::string record = NormalizeLikeTrainer(prior, "a Vn: b");
  EXPECT_NE(std::string::npos, record.find(surface))
      << "normalized record " << record << " does not contain " << surface;

  // A raw substring occurrence must NOT be the same string, or the fence would
  // match inside unrelated material.
  const std::string glued = NormalizeLikeTrainer(prior, "fooVn:bar");
  EXPECT_EQ(std::string::npos, glued.find(surface))
      << "the fence surface appears inside glued material " << glued;
}

TEST(UnigramContinuationContractTest, FenceRejectsMalformedConfiguration) {
  const ModelProto prior = MakeG3Prior();
  absl::Status status;
  ExtractCandidates(prior, {"a PL: b"}, "PL:,,Vn:", "malformed", 0, nullptr,
                    &status);
  EXPECT_FALSE(status.ok()) << "an empty list entry must be rejected";
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, status.code()) << status;
}

// OVERLAPPING MATCHES MUST BE UNIONED. With fences {abc, bcd} over "abcd" the
// old longest-match-then-skip loop marked [0,3) and never reconsidered
// position 1, leaving the final character admissible.
TEST(UnigramContinuationContractTest, OverlappingFenceMatchesAreUnioned) {
  // {abc, bcd} over "abcd" match at [2,5) and [3,6) of "eeabcdff". The old
  // longest-match-then-skip loop marked the first and then jumped past
  // position 3, so the final 'd' stayed admissible. The union must be [2,6).
  //
  // add_dummy_prefix is off here on purpose: the property under test is the
  // matcher's span union, and a dummy prefix would make "bcd" simply not occur
  // in the normalized text, so the test would pass without testing anything.
  const ModelProto prior = MakeG3PriorNoDummyPrefix();
  EXPECT_EQ("abc", NormalizeLikeTrainer(prior, "abc"));
  EXPECT_EQ("bcd", NormalizeLikeTrainer(prior, "bcd"));
  const std::vector<std::string> lines(40, "eeabcdff");
  EXPECT_EQ("eeabcdff", NormalizeLikeTrainer(prior, "eeabcdff"))
      << "the record does not normalize to itself; the offsets below are wrong";

  const auto off = ExtractCandidates(prior, lines, "", "overlap_off");
  const auto on = ExtractCandidates(prior, lines, "abc,bcd", "overlap_on");

  // Only ONE fence would leave part of "abcd" admissible; the union leaves
  // none of it.
  const auto single = ExtractCandidates(prior, lines, "abc", "overlap_single");
  bool single_leaves_d = false;
  for (const auto& e : single) {
    if (e.first.find('d') != std::string::npos) single_leaves_d = true;
  }
  EXPECT_TRUE(single_leaves_d)
      << "fencing only \"abc\" already removed every 'd'; the union test "
         "cannot distinguish anything";

  for (const auto& e : on) {
    EXPECT_EQ(std::string::npos, e.first.find_first_of("abcd"))
        << "candidate " << e.first << " overlaps the unioned fence";
  }
  bool off_has_some = false;
  for (const auto& e : off) {
    if (e.first.find_first_of("abcd") != std::string::npos) off_has_some = true;
  }
  EXPECT_TRUE(off_has_some) << "fence-off produced no candidate to remove";
  EXPECT_TRUE(HasCandidate(on, "ee") || HasCandidate(on, "ff"))
      << "legal spans were fenced too";
}

// Multi-byte fence strings must mark the correct UNICODE interval.
TEST(UnigramContinuationContractTest, MultiByteFenceMarksCorrectUnicodeSpan) {
  ModelProto prior = MakeG3Prior();
  for (const char* ch : {"\xc3\xbc", "\xe2\x82\xac"}) {   // 'ü' (2B), '€' (3B)
    auto* p = prior.add_pieces();
    p->set_piece(ch);
    p->set_score(-3.0f);
    p->set_type(ModelProto::SentencePiece::NORMAL);
  }
  prior.mutable_trainer_spec()->set_vocab_size(prior.pieces_size());
  const std::vector<std::string> lines(40, "aa \xc3\xbc\xe2\x82\xac bb");

  const auto off = ExtractCandidates(prior, lines, "", "mb_off");
  const auto on = ExtractCandidates(prior, lines, "\xc3\xbc\xe2\x82\xac",
                                    "mb_on");
  EXPECT_TRUE(HasCandidate(off, "\xc3\xbc\xe2\x82\xac"))
      << "the multi-byte candidate was not proposed even with the fence off";
  for (const auto& e : on) {
    EXPECT_EQ(std::string::npos, e.first.find("\xc3\xbc"))
        << "candidate " << e.first << " overlaps a fenced multi-byte character";
    EXPECT_EQ(std::string::npos, e.first.find("\xe2\x82\xac"))
        << "candidate " << e.first << " overlaps a fenced multi-byte character";
  }
  EXPECT_TRUE(HasCandidate(on, "aa") || HasCandidate(on, "bb"));
}

// THE CENTRAL G3 TEST. Fence OFF proposes context-containing and
// context-crossing candidates; fence ON proposes none of them, while
// candidates wholly inside the legal spans survive.
TEST(UnigramContinuationContractTest, ContextFencesRemoveExactlyContextCandidates) {
  const ModelProto prior = MakeG3Prior();
  std::vector<std::string> lines;
  for (int i = 0; i < 40; ++i) {
    lines.push_back("aa PL: bb");
    lines.push_back("cc PR: dd");
    lines.push_back("ee Vn: ff");
  }
  const auto off = ExtractCandidates(prior, lines, "", "ctx_off");
  const auto on = ExtractCandidates(prior, lines, "PL:,PR:,Vn:", "ctx_on");

  // Assertions are built THROUGH the normalizer, never by guessing a U+2581
  // spelling.
  const std::string pl = NormalizeLikeTrainer(prior, "PL:");   // the switch
  const std::string vn = NormalizeLikeTrainer(prior, "Vn:");
  // Crossing candidates are cut out of the NORMALIZED RECORD, not assembled
  // from separately normalized fragments: "a PL:" normalizes with its own
  // leading boundary and is not the substring that actually occurs here.
  const std::string record = NormalizeLikeTrainer(prior, "aa PL: bb");
  const size_t at = record.find(pl);
  ASSERT_NE(std::string::npos, at) << "record " << record;
  ASSERT_GT(at, 0u);
  // The whitespace marker is DERIVED from the normalized fence rather than
  // spelled out: pl is marker + "PL:", so everything below stays on character
  // boundaries even though the marker is three bytes.
  const std::string marker = pl.substr(0, pl.size() - std::string("PL:").size());
  ASSERT_FALSE(marker.empty());
  const std::string a_pl = record.substr(at - 1, 1 + pl.size());
  const std::string pl_b = record.substr(at, pl.size() + marker.size() + 1);
  const std::string a_pl_b =
      record.substr(at - 1, 1 + pl.size() + marker.size() + 1);

  // fence OFF: representative context candidates are proposed.
  EXPECT_TRUE(HasCandidate(off, pl)) << "missing " << pl;
  EXPECT_TRUE(HasCandidate(off, vn)) << "missing " << vn;
  EXPECT_TRUE(HasCandidate(off, a_pl)) << "missing left-crossing " << a_pl;
  EXPECT_TRUE(HasCandidate(off, pl_b)) << "missing right-crossing " << pl_b;
  EXPECT_TRUE(HasCandidate(off, a_pl_b)) << "missing spanning " << a_pl_b;

  // fence ON: those exact candidates are gone, and so is anything else that
  // touches a fenced character -- including the switch itself.
  for (const std::string& gone : {pl, vn, a_pl, pl_b, a_pl_b}) {
    EXPECT_FALSE(HasCandidate(on, gone)) << "still proposed: " << gone;
  }
  for (const auto& e : on) {
    for (const std::string& fenced :
         {NormalizeLikeTrainer(prior, "PL:"), NormalizeLikeTrainer(prior, "PR:"),
          NormalizeLikeTrainer(prior, "Vn:")}) {
      EXPECT_EQ(std::string::npos, e.first.find(fenced))
          << "candidate " << e.first << " contains fenced " << fenced;
    }
    // No candidate may contain any character of a fenced span, so the bare
    // marker characters cannot appear either.
    EXPECT_EQ(std::string::npos, e.first.find(':'))
        << "candidate " << e.first << " overlaps a fenced character";
  }

  // Candidates wholly inside the legal left/right spans remain.
  EXPECT_TRUE(HasCandidate(on, "aa")) << "legal left span was fenced";
  EXPECT_TRUE(HasCandidate(on, "bb")) << "legal right span was fenced";
  EXPECT_TRUE(HasCandidate(on, "ff")) << "legal right span was fenced";
}

// The fence governs candidate PROPOSAL, never the inherited model.
TEST(UnigramContinuationContractTest, FenceDoesNotTouchInheritedPieces) {
  ModelProto prior = MakeG3Prior();
  // An inherited NORMAL piece that CONTAINS the fenced context switch.
  const std::string inherited = NormalizeLikeTrainer(prior, "Vn: d");
  auto* p = prior.add_pieces();
  p->set_piece(inherited);
  p->set_score(-4.5f);
  p->set_type(ModelProto::SentencePiece::NORMAL);
  const int inherited_id = prior.pieces_size() - 1;
  prior.mutable_trainer_spec()->set_vocab_size(prior.pieces_size());

  const std::string prior_path = TempPath("g3_inherit_prior.model");
  const std::string input = TempPath("g3_inherit_input.txt");
  const std::string prefix = TempPath("g3_inherit_model");
  const std::string dump = TempPath("g3_inherit_cands.tsv");
  ASSERT_TRUE(WriteProto(prior_path, prior));
  std::vector<std::string> lines;
  for (int i = 0; i < 40; ++i) {
    lines.push_back("aa Vn: d bb");
    lines.push_back("cc Vn: e dd");
  }
  ASSERT_TRUE(WriteLines(input, lines));

  const std::string saved = absl::GetFlag(FLAGS_continuation_fence_strings);
  absl::SetFlag(&FLAGS_continuation_fence_strings, "Vn:");
  ScopedEnv dump_env("SPM_DUMP_CANDIDATES", dump);
  TrainerSpec trainer = UnigramContinuationSpec(
      input, "text", prior_path, TempPath("g3_inherit.result"), prefix);
  trainer.set_vocab_size(prior.pieces_size() + 2);
  trainer.set_max_sentencepiece_length(8);
  const absl::Status status = RunTrainer(trainer, NormalizerSpec());
  absl::SetFlag(&FLAGS_continuation_fence_strings, saved);
  ASSERT_TRUE(status.ok()) << status;

  ModelProto out;
  ASSERT_TRUE(ReadProto(prefix + ".model", &out));
  // Same ID, same string, same type.
  ASSERT_LT(inherited_id, out.pieces_size());
  EXPECT_EQ(inherited, out.pieces(inherited_id).piece());
  EXPECT_EQ(ModelProto::SentencePiece::NORMAL, out.pieces(inherited_id).type());
  // And its score obeys the ordinary gauge contract, fence or no fence.
  ExpansionResult result;
  ASSERT_TRUE(ReadProto(TempPath("g3_inherit.result"), &result));
  EXPECT_TRUE(unigram::VerifyPriorPrefixInvariant(prior, out,
                                                 result.unigram_lambda()).ok());

  // The E-step could still use it: it is in the model the trainer built, and
  // the gauge shifted it like any other inherited NORMAL piece.
  EXPECT_NE(prior.pieces(inherited_id).score(), out.pieces(inherited_id).score());

  // But no NEW candidate contains or crosses the fence.
  const std::string vn = NormalizeLikeTrainer(prior, "Vn:");
  auto in = filesystem::NewReadableFile(dump);
  ASSERT_TRUE(in->status().ok());
  std::string line;
  int rows = 0;
  while (in->ReadLine(&line)) {
    if (line.empty()) continue;
    const std::vector<std::string> f = absl::StrSplit(line, '\t');
    ASSERT_EQ(3u, f.size());
    EXPECT_EQ(std::string::npos, f[1].find(vn))
        << "new candidate " << f[1] << " contains the fenced switch";
    ++rows;
  }
  EXPECT_GT(rows, 0) << "no candidates at all; the test proves nothing";
}

// Typed-meta fencing (defect D) and explicit-string fencing must UNION, and
// the counters must keep them apart.
TEST(UnigramContinuationContractTest, MetaAndExplicitFencesUnion) {
  const ModelProto prior = MakeG3Prior();       // contains USER_DEFINED "<X>"
  std::vector<std::string> lines(60, "aa <X> bb Vn: cc");
  const auto on = ExtractCandidates(prior, lines, "Vn:", "union_on");
  ASSERT_FALSE(on.empty());

  const std::string vn = NormalizeLikeTrainer(prior, "Vn:");
  for (const auto& e : on) {
    EXPECT_EQ(std::string::npos, e.first.find("<X>"))
        << e.first << " crosses the inherited meta symbol";
    EXPECT_EQ(std::string::npos, e.first.find('<')) << e.first;
    EXPECT_EQ(std::string::npos, e.first.find(vn))
        << e.first << " overlaps the explicit fence";
    EXPECT_EQ(std::string::npos, e.first.find(':')) << e.first;
  }
  // Legal regions still produce candidates.
  EXPECT_TRUE(HasCandidate(on, "aa"));
  EXPECT_TRUE(HasCandidate(on, "bb"));
  EXPECT_TRUE(HasCandidate(on, "cc"));
}

// An empty/absent option must be a STRICT no-op, because G3 is a curriculum
// option and not a new default modelling rule.
TEST(UnigramContinuationContractTest, EmptyFenceOptionIsAStrictNoOp) {
  const ModelProto prior = MakeG3Prior();
  std::vector<std::string> lines;
  for (int i = 0; i < 40; ++i) {
    lines.push_back("aa PL: bb");
    lines.push_back("cc Vn: dd");
  }
  std::string dump_absent, dump_empty;
  const auto absent =
      ExtractCandidates(prior, lines, "", "noop_absent", 0, &dump_absent);
  const auto empty =
      ExtractCandidates(prior, lines, "", "noop_empty", 0, &dump_empty);

  // Candidate dumps byte-identical.
  EXPECT_EQ(ReadWhole(dump_absent), ReadWhole(dump_empty));
  ASSERT_EQ(absent.size(), empty.size());
  for (size_t i = 0; i < absent.size(); ++i) {
    EXPECT_EQ(absent[i].first, empty[i].first) << "at rank " << i;
    EXPECT_EQ(absent[i].second, empty[i].second) << "at rank " << i;
  }

  // And the final models agree bit for bit.
  ModelProto m1, m2;
  ASSERT_TRUE(ReadProto(TempPath("g3_noop_absent_model.model"), &m1));
  ASSERT_TRUE(ReadProto(TempPath("g3_noop_empty_model.model"), &m2));
  ASSERT_EQ(m1.pieces_size(), m2.pieces_size());
  for (int i = 0; i < m1.pieces_size(); ++i) {
    EXPECT_EQ(m1.pieces(i).piece(), m2.pieces(i).piece());
    EXPECT_EQ(m1.pieces(i).type(), m2.pieces(i).type());
    EXPECT_EQ(m1.pieces(i).score(), m2.pieces(i).score());
  }
}

// Fence ORDER must not be a modelling parameter, and duplicates must collapse.
TEST(UnigramContinuationContractTest, FenceSetIsOrderIndependentAndDeduplicated) {
  const ModelProto prior = MakeG3Prior();
  std::vector<std::string> lines;
  for (int i = 0; i < 40; ++i) {
    lines.push_back("aa PL: bb");
    lines.push_back("cc PR: dd");
    lines.push_back("ee Vn: ff");
  }
  std::string d1, d2, d3;
  ExtractCandidates(prior, lines, "PL:,PR:,Vn:", "order_a", 0, &d1);
  ExtractCandidates(prior, lines, "Vn:,PL:,PR:", "order_b", 0, &d2);
  ExtractCandidates(prior, lines, "Vn:,PL:,Vn:,PR:,PR:", "order_c", 0, &d3);
  const std::string a = ReadWhole(d1);
  EXPECT_EQ(a, ReadWhole(d2)) << "fence list order changed extraction";
  EXPECT_EQ(a, ReadWhole(d3)) << "duplicate fence entries changed extraction";
}


// ---------------------------------------------------------------------------
// G3 provenance: the fence set is a MODELLING parameter and must be recorded.
//
// --continuation_fence_strings changes the candidate universe, so unlike
// --continuation_spill_dir/-entries it can change the learned tokenizer. A
// fenced and an unfenced run would otherwise ship the same serialized
// TrainerSpec while having trained under different boundary policies.
// ---------------------------------------------------------------------------

std::string PolicyOf(const ModelProto& prior,
                     const std::vector<std::string>& lines,
                     const std::string& fence_spec, const std::string& tag,
                     std::string* embedded_out = nullptr) {
  const std::string prior_path = TempPath("g3p_" + tag + "_prior.model");
  const std::string input = TempPath("g3p_" + tag + "_input.txt");
  const std::string prefix = TempPath("g3p_" + tag + "_model");
  const std::string result_path = TempPath("g3p_" + tag + ".result");
  EXPECT_TRUE(WriteProto(prior_path, prior));
  EXPECT_TRUE(WriteLines(input, lines));

  const std::string saved = absl::GetFlag(FLAGS_continuation_fence_strings);
  absl::SetFlag(&FLAGS_continuation_fence_strings, fence_spec);
  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(prior.pieces_size() + 2);
  trainer.set_max_sentencepiece_length(8);
  const absl::Status status = RunTrainer(trainer, NormalizerSpec());
  absl::SetFlag(&FLAGS_continuation_fence_strings, saved);
  EXPECT_TRUE(status.ok()) << status;

  ExpansionResult standalone;
  EXPECT_TRUE(ReadProto(result_path, &standalone));
  if (embedded_out != nullptr) {
    ModelProto m;
    EXPECT_TRUE(ReadProto(prefix + ".model", &m));
    *embedded_out = m.expansion_result().boundary_policy();
  }
  return standalone.boundary_policy();
}

TEST(UnigramContinuationContractTest, BoundaryPolicyEncodingIsCanonical) {
  // Pure function, so the awkward cases are cheap to pin here rather than
  // through a training run.
  EXPECT_EQ("unigram_explicit_fences_v1:[]",
            unigram::EncodeUnigramBoundaryPolicy({}));
  EXPECT_EQ("unigram_explicit_fences_v1:[abc]",
            unigram::EncodeUnigramBoundaryPolicy({"abc"}));
  // Order- and duplicate-independent: a std::set, sorted by bytes.
  EXPECT_EQ(unigram::EncodeUnigramBoundaryPolicy({"a", "b"}),
            unigram::EncodeUnigramBoundaryPolicy({"b", "a", "b"}));
  // A surface containing the separator must not be confusable with two
  // surfaces. This is why the encoding escapes rather than trusting the data.
  EXPECT_NE(unigram::EncodeUnigramBoundaryPolicy({"a,b"}),
            unigram::EncodeUnigramBoundaryPolicy({"a", "b"}));
  EXPECT_EQ("unigram_explicit_fences_v1:[a%2Cb]",
            unigram::EncodeUnigramBoundaryPolicy({"a,b"}));
  // Brackets too, so the envelope cannot be forged.
  EXPECT_EQ("unigram_explicit_fences_v1:[%5D]",
            unigram::EncodeUnigramBoundaryPolicy({"]"}));
  // Multi-byte surfaces are escaped byte by byte and stay distinguishable.
  EXPECT_EQ("unigram_explicit_fences_v1:[%E2%96%81Vn%3A]",
            unigram::EncodeUnigramBoundaryPolicy({"\xe2\x96\x81Vn:"}));
}

TEST(UnigramContinuationContractTest, BoundaryPolicyRecordsTheEffectiveFenceSet) {
  const ModelProto prior = MakeG3Prior();
  std::vector<std::string> lines;
  for (int i = 0; i < 40; ++i) {
    lines.push_back("aa PL: bb");
    lines.push_back("cc PR: dd");
    lines.push_back("ee Vn: ff");
  }

  // 1. absent flag and explicitly empty flag agree, and are the canonical
  //    "no explicit curriculum fence" policy.
  std::string embedded_absent;
  const std::string absent =
      PolicyOf(prior, lines, "", "absent", &embedded_absent);
  EXPECT_EQ("unigram_explicit_fences_v1:[]", absent);

  // 2. reordered and duplicated fence lists are the same policy.
  std::string embedded_a;
  const std::string a =
      PolicyOf(prior, lines, "PL:,PR:,Vn:", "seta", &embedded_a);
  const std::string b = PolicyOf(prior, lines, "Vn:,PL:,PR:", "setb");
  const std::string c = PolicyOf(prior, lines, "Vn:,PL:,Vn:,PR:,PR:", "setc");
  EXPECT_EQ(a, b) << "fence list order changed the recorded policy";
  EXPECT_EQ(a, c) << "duplicate entries changed the recorded policy";
  EXPECT_NE(absent, a);

  // The recorded surfaces are the NORMALIZED ones actually matched, not the
  // caller's logical spelling.
  EXPECT_EQ(unigram::EncodeUnigramBoundaryPolicy(
                {NormalizeLikeTrainer(prior, "PL:"),
                 NormalizeLikeTrainer(prior, "PR:"),
                 NormalizeLikeTrainer(prior, "Vn:")}),
            a);

  // 3. a genuinely different fence set is a different policy.
  const std::string only_vn = PolicyOf(prior, lines, "Vn:", "onlyvn");
  EXPECT_NE(a, only_vn);
  EXPECT_NE(absent, only_vn);

  // 4. embedded ModelProto.expansion_result agrees with the standalone
  //    .expansion in both the empty and non-empty cases.
  EXPECT_EQ(absent, embedded_absent);
  EXPECT_EQ(a, embedded_a);

  // 5. the policy survives serialization/deserialization unchanged. Round-trip
  //    the REAL artifact rather than a hand-built message: ExpansionResult has
  //    required fields, so a partially populated one does not serialize at all
  //    and the test would pass or fail for the wrong reason.
  ExpansionResult on_disk;
  ASSERT_TRUE(ReadProto(TempPath("g3p_seta.result"), &on_disk));
  ASSERT_EQ(a, on_disk.boundary_policy());
  ExpansionResult back;
  ASSERT_TRUE(back.ParseFromString(on_disk.SerializeAsString()));
  EXPECT_EQ(a, back.boundary_policy());
}


// The continuation expected-count dump must be diagnostics ONLY: enabling it
// may not move a score, a lambda or the piece table. It reads a vector already
// in hand at EM exit.
TEST(UnigramContinuationContractTest, ContinuationExpectedDumpDoesNotChangeTheResult) {
  const ModelProto prior = MakeAbUnigramPrior();
  const std::string prior_path = TempPath("cont_expdump_prior.model");
  const std::string input = TempPath("cont_expdump_input.txt");
  ASSERT_TRUE(WriteProto(prior_path, prior));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(60, "abababab")));
  const std::string dump = TempPath("cont_expdump.tsv");
  ::remove(dump.c_str());

  auto run = [&](const std::string& tag, bool with_dump) {
    const std::string prefix = TempPath("cont_expdump_" + tag);
    std::unique_ptr<ScopedEnv> env;
    if (with_dump) {
      env = std::make_unique<ScopedEnv>("SPM_DUMP_CONTINUATION_EXPECTED", dump);
    }
    TrainerSpec trainer = UnigramContinuationSpec(
        input, "text", prior_path, TempPath("cont_expdump_" + tag + ".result"),
        prefix);
    trainer.set_vocab_size(prior.pieces_size() + 3);
    EXPECT_TRUE(RunTrainer(trainer, NormalizerSpec()).ok());
    ModelProto m;
    EXPECT_TRUE(ReadProto(prefix + ".model", &m));
    return m;
  };
  const ModelProto plain = run("plain", false);
  const ModelProto dumped = run("dumped", true);

  ASSERT_EQ(plain.pieces_size(), dumped.pieces_size());
  for (int i = 0; i < plain.pieces_size(); ++i) {
    EXPECT_EQ(plain.pieces(i).piece(), dumped.pieces(i).piece());
    EXPECT_EQ(plain.pieces(i).type(), dumped.pieces(i).type());
    EXPECT_EQ(plain.pieces(i).score(), dumped.pieces(i).score())
        << "enabling the dump moved the score at id " << i;
  }
  EXPECT_EQ(plain.expansion_result().unigram_lambda(),
            dumped.expansion_result().unigram_lambda());

  auto in = filesystem::NewReadableFile(dump);
  ASSERT_TRUE(in->status().ok());
  std::string line;
  int rows = 0, inherited = 0, extension = 0;
  while (in->ReadLine(&line)) {
    if (line.empty()) continue;
    const std::vector<std::string> f = absl::StrSplit(line, '\t');
    ASSERT_EQ(4u, f.size()) << line;
    if (f[2] == "inherited") ++inherited;
    if (f[2] == "extension") ++extension;
    ++rows;
  }
  EXPECT_EQ(plain.pieces_size(), rows);
  EXPECT_EQ(prior.pieces_size(), inherited);
  EXPECT_EQ(plain.pieces_size() - prior.pieces_size(), extension);
}


// ---------------------------------------------------------------------------
// Explicit-program BPE runtime, and the BPE continuation contract.
//
// The merge program is authoritative whenever native equivalence fails, so it
// needs a real tokenizer, not a research-only .merges file. These tests pin
// that runtime against the same semantics the continuation trainer validates.
// ---------------------------------------------------------------------------

NormalizerSpec MusicNormalizer() {
  NormalizerSpec n;
  n.set_name("identity");
  n.set_add_dummy_prefix(true);
  n.set_remove_extra_whitespaces(false);
  n.set_escape_whitespaces(true);
  return n;
}

// Builds an ExpansionResult by hand: the runtime must work on an artifact,
// independent of how it was produced.
ExpansionResult MakeProgram(const std::vector<std::pair<std::string, int>>& pieces,
                            const std::vector<std::pair<std::string, std::string>>& merges,
                            bool with_contract = true) {
  ExpansionResult r;
  r.set_schema_version(1);
  r.set_model_type(EXPANSION_BPE);
  int id = 0;
  for (const auto& [piece, type] : pieces) {
    auto* p = r.add_base_pieces();
    p->set_external_id(id++);
    p->set_piece(piece);
    p->set_type(static_cast<ModelProto::SentencePiece::Type>(type));
  }
  int rank = 0;
  for (const auto& [l, rr] : merges) {
    auto* m = r.add_base_merges();
    m->set_rank(rank++);
    m->set_left(l);
    m->set_right(rr);
  }
  r.set_first_new_external_id(id);
  if (with_contract) {
    *r.mutable_contract()->mutable_normalizer_spec() = MusicNormalizer();
    r.mutable_contract()->set_unk_id(0);
    r.mutable_contract()->set_bos_id(-1);
    r.mutable_contract()->set_eos_id(-1);
    r.mutable_contract()->set_pad_id(-1);
  }
  return r;
}

const int kNORMAL = ModelProto::SentencePiece::NORMAL;
const int kUNK = ModelProto::SentencePiece::UNKNOWN;

TEST(ExpansionProcessorTest, RefusesAnArtifactWithNoNormalizerContract) {
  // Guessing a text pipeline is how the identity-vs-nmt_nfkc mismatch happened.
  const ExpansionResult r = MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"a", kNORMAL}}, {},
      /*with_contract=*/false);
  expansion::ExpansionProcessor p;
  const absl::Status st = p.Load(r);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(std::string::npos,
            std::string(st.message()).find("normalizer contract"));
}

TEST(ExpansionProcessorTest, AppliesTheMergeProgramByRank) {
  // a+b -> ab at rank 0, ab+c -> abc at rank 1. Lowest rank first.
  const ExpansionResult r = MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"a", kNORMAL},
       {"b", kNORMAL}, {"c", kNORMAL}, {"ab", kNORMAL}, {"abc", kNORMAL}},
      {{"a", "b"}, {"ab", "c"}});
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());
  std::vector<expansion::TokenSpan> spans;
  ASSERT_TRUE(p.Encode("abc", &spans).ok());
  ASSERT_EQ(2u, spans.size());          // U+2581 then "abc"
  EXPECT_EQ("\xe2\x96\x81", spans[0].piece);
  EXPECT_EQ("abc", spans[1].piece);
  EXPECT_EQ(p.PieceToId("abc"), spans[1].id);

  // Exact UTF-8 byte spans over the NORMALIZED text: U+2581 is three bytes.
  EXPECT_EQ(0, spans[0].begin);
  EXPECT_EQ(3, spans[0].end);
  EXPECT_EQ(3, spans[1].begin);
  EXPECT_EQ(6, spans[1].end);

  std::string back;
  ASSERT_TRUE(p.Decode({spans[0].id, spans[1].id}, &back).ok());
  EXPECT_EQ("abc", back);
}


TEST(ExpansionProcessorTest, HierarchyEligibilityIsOccurrenceLocalAtRuntime) {
  ExpansionResult r;
  r.set_schema_version(1);
  r.set_model_type(EXPANSION_BPE);
  const std::vector<std::pair<std::string, int>> base = {
      {"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"/", kNORMAL},
      {"1", kNORMAL}, {"2", kNORMAL}, {"8", kNORMAL}};
  int id = 0;
  for (const auto& [piece, type] : base) {
    auto* p = r.add_base_pieces();
    p->set_external_id(id++);
    p->set_piece(piece);
    p->set_type(static_cast<ModelProto::SentencePiece::Type>(type));
  }
  for (const std::string& piece : {"12", "/12"}) {
    auto* p = r.add_learned_pieces();
    p->set_external_id(id++);
    p->set_piece(piece);
    p->set_type(ModelProto::SentencePiece::NORMAL);
  }
  auto* m0 = r.add_learned_merges();
  m0->set_rank(0); m0->set_left("1"); m0->set_right("2");
  m0->set_scope_level(0);
  auto* m1 = r.add_learned_merges();
  m1->set_rank(1); m1->set_left("/"); m1->set_right("12");
  m1->set_scope_level(1);
  *r.mutable_contract()->mutable_normalizer_spec() = MusicNormalizer();
  r.set_boundary_policy("bpe_hierarchical_completion_v1:test");

  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());
  EXPECT_TRUE(p.RequiresHierarchy());

  std::vector<expansion::TokenSpan> flat;
  EXPECT_FALSE(p.Encode("/12", &flat).ok())
      << "hierarchical artifacts must fail closed without occurrence gates";

  expansion::CompletionGate short_den;
  short_den.level = 1;
  short_den.cuts = {3, 4, 6};  // normalized "▁/12": "/" | "12"
  std::vector<expansion::TokenSpan> short_out;
  ASSERT_TRUE(p.EncodeWithHierarchy("/12", {short_den}, &short_out).ok());
  ASSERT_EQ(2u, short_out.size());
  EXPECT_EQ("\xe2\x96\x81", short_out[0].piece);
  EXPECT_EQ("/12", short_out[1].piece)
      << "/+12 is legal when 12 is the complete denominator child";

  expansion::CompletionGate long_den;
  long_den.level = 1;
  long_den.cuts = {3, 4, 7};  // normalized "▁/128": "/" | "128"
  std::vector<expansion::TokenSpan> long_out;
  ASSERT_TRUE(p.EncodeWithHierarchy("/128", {long_den}, &long_out).ok());
  std::vector<std::string> got;
  for (const auto& x : long_out) got.push_back(x.piece);
  EXPECT_EQ(std::vector<std::string>(
                {"\xe2\x96\x81", "/", "12", "8"}),
            got)
      << "the SAME global /+12 rule must be blocked only in the /128 "
         "occurrence; /128 must not poison /12 globally";
}


TEST(ExpansionProcessorTest, RejectsNestedGateCrossingOuterDirectChild) {
  const ExpansionResult r = MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL},
       {"a", kNORMAL}, {"b", kNORMAL}, {"c", kNORMAL}, {"d", kNORMAL}},
      {});
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());

  // Normalized surface is "▁abcd": byte 3 starts 'a'. The outer parent is
  // [3,7) with children "ab" | "cd" (cut 5). The nested parent [3,6)
  // crosses that child boundary but ends inside "cd", so it cannot belong to
  // one consistent constituent tree even though the intervals are laminar.
  expansion::CompletionGate outer;
  outer.level = 2;
  outer.cuts = {3, 5, 7};
  expansion::CompletionGate inner;
  inner.level = 1;
  inner.cuts = {3, 4, 6};

  std::vector<expansion::TokenSpan> out;
  const absl::Status st =
      p.EncodeWithHierarchy("abcd", {outer, inner}, &out);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(std::string::npos,
            std::string(st.message()).find("direct-child"));
}

TEST(ExpansionProcessorTest, SamePairCanHaveDistinctExactScopeRules) {
  ExpansionResult r;
  r.set_schema_version(1);
  r.set_model_type(EXPANSION_BPE);
  int id = 0;
  for (const auto& [piece, type] :
       std::vector<std::pair<std::string, int>>{
           {"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL},
           {"a", kNORMAL}, {"b", kNORMAL}}) {
    auto* p = r.add_base_pieces();
    p->set_external_id(id++);
    p->set_piece(piece);
    p->set_type(static_cast<ModelProto::SentencePiece::Type>(type));
  }
  auto* ab = r.add_learned_pieces();
  ab->set_external_id(id);
  ab->set_piece("ab");
  ab->set_type(ModelProto::SentencePiece::NORMAL);

  auto* ordinary = r.add_learned_merges();
  ordinary->set_rank(0);
  ordinary->set_left("a");
  ordinary->set_right("b");
  ordinary->set_external_id(id);
  ordinary->set_scope_level(0);

  auto* level1 = r.add_learned_merges();
  level1->set_rank(1);
  level1->set_left("a");
  level1->set_right("b");
  level1->set_external_id(id);
  level1->set_scope_level(1);

  *r.mutable_contract()->mutable_normalizer_spec() = MusicNormalizer();
  r.set_boundary_policy("bpe_hierarchical_completion_v1:test");

  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());

  // No completion boundary at a|b -> exact scope 0.
  std::vector<expansion::TokenSpan> ordinary_out;
  ASSERT_TRUE(p.EncodeWithHierarchy("ab", {}, &ordinary_out).ok());
  ASSERT_EQ(2u, ordinary_out.size());
  EXPECT_EQ("ab", ordinary_out[1].piece);

  // The same surface pair at a level-1 completion boundary must select the
  // scope-1 operation, not the earlier scope-0 operation.
  expansion::CompletionGate gate;
  gate.level = 1;
  gate.cuts = {3, 4, 5};  // normalized "▁ab": "a" | "b"
  std::vector<expansion::TokenSpan> scoped_out;
  ASSERT_TRUE(p.EncodeWithHierarchy("ab", {gate}, &scoped_out).ok());
  ASSERT_EQ(2u, scoped_out.size());
  EXPECT_EQ("ab", scoped_out[1].piece);
}

TEST(ExpansionProcessorTest, InheritedMergesAreNeverVetoedByNewHierarchy) {
  ExpansionResult r;
  r.set_schema_version(1);
  r.set_model_type(EXPANSION_BPE);
  const std::vector<std::pair<std::string, int>> pieces = {
      {"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"a", kNORMAL},
      {"b", kNORMAL}, {"c", kNORMAL}, {"ab", kNORMAL}};
  int id = 0;
  for (const auto& [piece, type] : pieces) {
    auto* p = r.add_base_pieces();
    p->set_external_id(id++);
    p->set_piece(piece);
    p->set_type(static_cast<ModelProto::SentencePiece::Type>(type));
  }
  auto* merge = r.add_base_merges();
  merge->set_rank(0); merge->set_left("a"); merge->set_right("b");
  *r.mutable_contract()->mutable_normalizer_spec() = MusicNormalizer();
  r.set_boundary_policy("bpe_hierarchical_completion_v1:test");

  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());
  // New grammar says a | bc. The inherited a+b merge crosses that child
  // boundary before bc is complete, but continuation is forbidden to change
  // inherited/base tokenization.
  expansion::CompletionGate gate;
  gate.level = 1;
  gate.cuts = {3, 4, 6};
  std::vector<expansion::TokenSpan> out;
  ASSERT_TRUE(p.EncodeWithHierarchy("abc", {gate}, &out).ok());
  std::vector<std::string> got;
  for (const auto& x : out) got.push_back(x.piece);
  EXPECT_EQ(std::vector<std::string>({"\xe2\x96\x81", "ab", "c"}), got);
}

TEST(ExpansionProcessorTest, LeftmostOccurrenceOfTheBestRankWins) {
  const ExpansionResult r = MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"a", kNORMAL},
       {"aa", kNORMAL}},
      {{"a", "a"}});
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());
  std::vector<std::string> got;
  std::vector<expansion::TokenSpan> spans;
  ASSERT_TRUE(p.Encode("aaa", &spans).ok());
  for (const auto& s : spans) got.push_back(s.piece);
  // leftmost "aa" merges first, leaving a trailing atom
  EXPECT_EQ(std::vector<std::string>({"\xe2\x96\x81", "aa", "a"}), got);
}

TEST(ExpansionProcessorTest, WhitespaceDummyPrefixAndRepeatedSpacesRoundTrip) {
  std::vector<std::pair<std::string, int>> pieces = {
      {"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}};
  for (const char* c : {"a", "b"}) pieces.push_back({c, kNORMAL});
  const ExpansionResult r = MakeProgram(pieces, {});
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());
  // remove_extra_whitespaces=false, so a doubled space survives as two marks.
  std::vector<expansion::TokenSpan> spans;
  ASSERT_TRUE(p.Encode("a  b", &spans).ok());
  int marks = 0;
  for (const auto& s : spans) {
    if (s.piece == "\xe2\x96\x81") ++marks;
  }
  EXPECT_EQ(3, marks) << "dummy prefix + two literal spaces";
  std::vector<int> ids;
  for (const auto& s : spans) ids.push_back(s.id);
  std::string back;
  ASSERT_TRUE(p.Decode(ids, &back).ok());
  EXPECT_EQ("a  b", back);
}

TEST(ExpansionProcessorTest, NonAsciiSurvivesIdentityNormalization) {
  const std::string mb = "\xc3\xbc";   // 'ü'
  const ExpansionResult r = MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {mb, kNORMAL}}, {});
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());
  std::vector<expansion::TokenSpan> spans;
  ASSERT_TRUE(p.Encode(mb, &spans).ok());
  ASSERT_EQ(2u, spans.size());
  EXPECT_EQ(mb, spans[1].piece);
  EXPECT_EQ(3, spans[1].begin);
  EXPECT_EQ(5, spans[1].end) << "two-byte character, exact span";
  std::string back;
  ASSERT_TRUE(p.Decode({spans[0].id, spans[1].id}, &back).ok());
  EXPECT_EQ(mb, back);
}

TEST(ExpansionProcessorTest, UnknownCharacterBecomesTheUnkIdAndDecodesEmpty) {
  const ExpansionResult r = MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"a", kNORMAL}}, {});
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());
  std::vector<expansion::TokenSpan> spans;
  ASSERT_TRUE(p.Encode("aZ", &spans).ok());
  ASSERT_EQ(3u, spans.size());
  EXPECT_EQ(p.unk_id(), spans[2].id);
  // unk_surface is empty under the RNNT contract.
  std::string back;
  ASSERT_TRUE(p.Decode({spans[2].id}, &back).ok());
  EXPECT_EQ("", back);
}

TEST(ExpansionProcessorTest, UnknownDoesNotRecoverNonAtomicSurfaceByLookup) {
  ExpansionResult r;
  r.set_schema_version(1);
  r.set_model_type(EXPANSION_BPE);
  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type,
                 bool mergeable, bool atomic) {
    auto* p = r.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "\xe2\x96\x81", ModelProto::SentencePiece::NORMAL, true, true);
  add(2, "a", ModelProto::SentencePiece::NORMAL, true, true);
  // Same surface as the OOV scalar, but deliberately not in the reversible
  // atomic alphabet. It must not capture runtime UNKNOWN by string lookup.
  add(3, "Z", ModelProto::SentencePiece::NORMAL, false, false);
  *r.mutable_contract()->mutable_normalizer_spec() = MusicNormalizer();

  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());
  std::vector<expansion::TokenSpan> spans;
  ASSERT_TRUE(p.Encode("aZ", &spans).ok());
  ASSERT_EQ(3u, spans.size());
  EXPECT_EQ(2, spans[1].id);
  EXPECT_EQ(p.unk_id(), spans[2].id);
  EXPECT_EQ("Z", spans[2].piece);
}

TEST(ExpansionProcessorTest, IdMapShaDistinguishesSameSizeDifferentMeaning) {
  const ExpansionResult a = MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"a", kNORMAL}}, {});
  const ExpansionResult b = MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"b", kNORMAL}}, {});
  expansion::ExpansionProcessor pa, pb;
  ASSERT_TRUE(pa.Load(a).ok());
  ASSERT_TRUE(pb.Load(b).ok());
  EXPECT_EQ(pa.GetPieceSize(), pb.GetPieceSize());
  EXPECT_NE(pa.IdMapSha256(), pb.IdMapSha256())
      << "same size, different ID map must not share a warm-start identity";
}


// ---------------------------------------------------------------------------
// BPE continuation: the inherited tokenizer semantics are authoritative.
// ---------------------------------------------------------------------------

ExpansionSpec AbcdSpecWithContract(const NormalizerSpec& norm) {
  ExpansionSpec spec = BasicAbcdSpec();
  *spec.mutable_contract()->mutable_normalizer_spec() = norm;
  spec.mutable_contract()->set_unk_id(0);
  spec.mutable_contract()->set_bos_id(-1);
  spec.mutable_contract()->set_eos_id(-1);
  spec.mutable_contract()->set_pad_id(-1);
  return spec;
}

absl::Status RunBpeWithNormalizer(const ExpansionSpec& spec,
                                  const NormalizerSpec& caller,
                                  const std::string& tag,
                                  ExpansionResult* out) {
  const std::string spec_path = TempPath("bc_" + tag + ".spec");
  const std::string input = TempPath("bc_" + tag + "_input.txt");
  const std::string result_path = TempPath("bc_" + tag + ".result");
  EXPECT_TRUE(WriteProto(spec_path, spec));
  EXPECT_TRUE(WriteLines(input, {"abcd", "abcd", "abcd", "dcba"}));
  TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path,
                          TempPath("bc_" + tag + "_model"), 8);
  const absl::Status st = RunTrainer(trainer, caller);
  if (st.ok()) EXPECT_TRUE(ReadProto(result_path, out));
  return st;
}

TEST(BPEContinuationContractTest, OmittedNormalizerInheritsTheBase) {
  // The caller passes the CLI default; the base says identity. Unigram
  // continuation has always inherited here. BPE used to silently train under
  // nmt_nfkc while its own base was identity.
  const ExpansionSpec spec = AbcdSpecWithContract(IdentityNormalizer());
  NormalizerSpec caller;              // default-constructed == "not specified"
  caller.set_name("nmt_nfkc");
  ExpansionResult out;
  ASSERT_TRUE(RunBpeWithNormalizer(spec, caller, "inherit", &out).ok());
  ASSERT_TRUE(out.has_contract());
  EXPECT_EQ("identity", out.contract().normalizer_spec().name())
      << "the base tokenizer's text pipeline is authoritative";
}

TEST(BPEContinuationContractTest, ConflictingNormalizerIsRefused) {
  ExpansionSpec spec = AbcdSpecWithContract(IdentityNormalizer());
  NormalizerSpec caller = IdentityNormalizer();
  caller.set_remove_extra_whitespaces(true);   // explicit and different
  ExpansionResult out;
  const absl::Status st =
      RunBpeWithNormalizer(spec, caller, "conflict", &out);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(std::string::npos,
            std::string(st.message()).find("conflicts with the inherited"));
}

TEST(BPEContinuationContractTest, ResultRecordsTheEffectiveTrainingPolicy) {
  const ExpansionSpec spec = AbcdSpecWithContract(IdentityNormalizer());
  ExpansionResult out;
  ASSERT_TRUE(RunBpeWithNormalizer(spec, IdentityNormalizer(), "policy", &out).ok());
  ASSERT_TRUE(out.has_contract());
  const ContinuationContract& c = out.contract();
  // Every question the README claims the artifact answers without a log.
  EXPECT_EQ("identity", c.normalizer_spec().name());
  EXPECT_TRUE(c.normalizer_spec().add_dummy_prefix() ||
              !c.normalizer_spec().add_dummy_prefix());   // present, not absent
  EXPECT_TRUE(c.has_split_by_whitespace());
  EXPECT_TRUE(c.has_split_by_unicode_script());
  EXPECT_TRUE(c.has_split_by_number());
  EXPECT_TRUE(c.has_split_digits());
  EXPECT_TRUE(c.has_max_sentencepiece_length());
  EXPECT_TRUE(c.has_input_format());
  EXPECT_TRUE(c.has_hard_vocab_limit());
  EXPECT_EQ(0, c.unk_id());
  EXPECT_EQ(-1, c.bos_id());
  EXPECT_EQ(-1, c.eos_id());
  EXPECT_EQ(-1, c.pad_id());
  EXPECT_FALSE(c.base_id_map_sha256().empty());
  EXPECT_GT(c.corpus_records(), 0);
  // And whether a native model exists is recorded, not left to the log.
  EXPECT_TRUE(out.has_native_model_emitted());
}

TEST(BPEContinuationContractTest, NativeModelNeverInventsBosEos) {
  // The CLI defaults are bos_id=1/eos_id=2. On this spec IDs 1 and 2 are
  // ordinary inherited NORMAL pieces, so copying the caller's TrainerSpec
  // would have declared two real pieces to be BOS and EOS.
  const ExpansionSpec spec = AbcdSpecWithContract(IdentityNormalizer());
  const std::string prefix = TempPath("bc_native_model");
  ExpansionResult out;
  ASSERT_TRUE(RunBpeWithNormalizer(spec, IdentityNormalizer(), "native", &out).ok());
  ModelProto m;
  if (!ReadProto(TempPath("bc_native_model") + ".model", &m)) {
    GTEST_SKIP() << "native model legitimately withheld for this program";
  }
  EXPECT_EQ(0, m.trainer_spec().unk_id());
  EXPECT_EQ(-1, m.trainer_spec().bos_id());
  EXPECT_EQ(-1, m.trainer_spec().eos_id());
  EXPECT_EQ(-1, m.trainer_spec().pad_id());
  EXPECT_EQ("identity", m.normalizer_spec().name());
}

TEST(BPEContinuationContractTest, ExplicitProcessorAgreesWithNativeWhenEquivalent) {
  // Where a native model IS emitted, the two runtimes must agree exactly --
  // that is the whole basis for ever emitting one.
  const ExpansionSpec spec = AbcdSpecWithContract(IdentityNormalizer());
  const std::string prefix = TempPath("bc_parity_model");
  const std::string spec_path = TempPath("bc_parity.spec");
  const std::string input = TempPath("bc_parity_input.txt");
  const std::string result_path = TempPath("bc_parity.result");
  ASSERT_TRUE(WriteProto(spec_path, spec));
  ASSERT_TRUE(WriteLines(input, {"abcd", "abcd", "abcd", "dcba"}));
  TrainerSpec trainer =
      BpeContinuationSpec(input, spec_path, result_path, prefix, 8);
  ASSERT_TRUE(RunTrainer(trainer, IdentityNormalizer()).ok());

  ModelProto m;
  if (!ReadProto(prefix + ".model", &m)) {
    GTEST_SKIP() << "native model withheld; parity is not claimable";
  }
  ExpansionResult r;
  ASSERT_TRUE(ReadProto(result_path, &r));
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(r).ok());

  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(prefix + ".model").ok());
  for (const std::string& probe : {std::string("abcd"), std::string("dcba"),
                                   std::string("abcdabcd"), std::string("ab")}) {
    std::vector<int> native;
    ASSERT_TRUE(sp.Encode(probe, &native).ok());
    std::vector<int> explicit_ids;
    ASSERT_TRUE(p.EncodeIds(probe, &explicit_ids).ok());
    EXPECT_EQ(native, explicit_ids) << "disagreement on " << probe;
  }
}


// ---------------------------------------------------------------------------
// USER_DEFINED pieces.
//
// Native SentencePiece gives a USER_DEFINED piece one role at inference: it is
// recognized by longest prefix match BEFORE any merge runs and is frozen on
// both sides (bpe_model.cc). Native training reaches the same statistics by
// replacing every occurrence with a pretokenization boundary
// (trainer_interface.cc). The explicit runtime and the continuation trainer
// implement that role, not an approximation of it.
// ---------------------------------------------------------------------------

const int kUSER = ModelProto::SentencePiece::USER_DEFINED;

// <unk>, U+2581, the atoms of "<|v|>", "a", "b", the USER_DEFINED "<|v|>",
// and a merge program that WOULD build "<|v" and "a<" from those atoms.
ExpansionResult UserDefinedProgram() {
  return MakeProgram(
      {{"<unk>", kUNK}, {"\xe2\x96\x81", kNORMAL}, {"<", kNORMAL},
       {"|", kNORMAL}, {"v", kNORMAL}, {">", kNORMAL}, {"a", kNORMAL},
       {"b", kNORMAL}, {"<|", kNORMAL}, {"<|v", kNORMAL}, {"a<", kNORMAL},
       {"<|v|>", kUSER}},
      {{"<", "|"}, {"<|", "v"}, {"a", "<"}});
}

TEST(ExpansionProcessorTest, UserDefinedIsRecognizedBeforeAnyMerge) {
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(UserDefinedProgram()).ok());
  const int ud = p.PieceToId("<|v|>");
  ASSERT_TRUE(p.IsUserDefined(ud));
  EXPECT_EQ(kUSER, p.IdToType(ud));

  // Inside: the program's "<|" and "<|v" merges must not fire within it.
  // Across: "a<" is a declared merge and must not consume the "<" of the
  // USER_DEFINED occurrence.
  std::vector<expansion::TokenSpan> spans;
  ASSERT_TRUE(p.Encode("a<|v|>b", &spans).ok());
  std::vector<std::string> got;
  for (const auto& sp : spans) got.push_back(sp.piece);
  EXPECT_EQ(std::vector<std::string>({"\xe2\x96\x81", "a", "<|v|>", "b"}), got);
  EXPECT_EQ(ud, spans[2].id);
  // Exact byte spans over the NORMALIZED text: U+2581 is 3 bytes.
  EXPECT_EQ(3, spans[1].begin); EXPECT_EQ(4, spans[1].end);
  EXPECT_EQ(4, spans[2].begin); EXPECT_EQ(9, spans[2].end);
  EXPECT_EQ(9, spans[3].begin); EXPECT_EQ(10, spans[3].end);

  // The same characters OUTSIDE a complete USER_DEFINED string still merge:
  // the freeze is the occurrence, not the alphabet.
  ASSERT_TRUE(p.Encode("<|v", &spans).ok());
  got.clear();
  for (const auto& sp : spans) got.push_back(sp.piece);
  EXPECT_EQ(std::vector<std::string>({"\xe2\x96\x81", "<|v"}), got);

  // Adjacent occurrences are two frozen units, never one merged one.
  std::vector<int> ids;
  ASSERT_TRUE(p.EncodeIds("<|v|><|v|>", &ids).ok());
  EXPECT_EQ(std::vector<int>({p.PieceToId("\xe2\x96\x81"), ud, ud}), ids);

  std::string back;
  ASSERT_TRUE(p.Decode({p.PieceToId("\xe2\x96\x81"), p.PieceToId("a"), ud,
                        p.PieceToId("b")}, &back).ok());
  EXPECT_EQ("a<|v|>b", back);
}

TEST(ExpansionProcessorTest, RejectsAMergeNamingAUserDefinedChild) {
  ExpansionResult r = UserDefinedProgram();
  auto* m = r.add_base_merges();
  m->set_rank(3); m->set_left("a"); m->set_right("<|v|>");
  expansion::ExpansionProcessor p;
  const absl::Status st = p.Load(r);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(std::string::npos,
            std::string(st.message()).find("USER_DEFINED"));
}

ExpansionSpec UserDefinedSpec(bool declare_mergeable) {
  ExpansionSpec e;
  e.set_schema_version(1);
  e.set_model_type(EXPANSION_BPE);
  e.set_preserve_base_ids(true);
  e.set_requested_new_pieces(2);
  AddPiece(&e, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  AddPiece(&e, 1, "<|v|>", ModelProto::SentencePiece::USER_DEFINED,
           declare_mergeable, false);
  int id = 2;
  for (const char* c : {"a", "b", "c", "<", "|", "v", ">"}) {
    AddPiece(&e, id++, c, ModelProto::SentencePiece::NORMAL, true, true);
  }
  *e.mutable_contract()->mutable_normalizer_spec() = IdentityNormalizer();
  return e;
}

TEST(BPEContinuationContractTest, RejectsAMergeableUserDefinedPiece) {
  const std::string input = TempPath("ud_mergeable_input.txt");
  const std::string spec_path = TempPath("ud_mergeable.pb");
  ASSERT_TRUE(WriteLines(input, {"a<|v|>b"}));
  ASSERT_TRUE(WriteProto(spec_path, UserDefinedSpec(/*declare_mergeable=*/true)));
  const absl::Status st = RunTrainer(
      BpeContinuationSpec(input, spec_path, TempPath("ud_mergeable.result"),
                          TempPath("ud_mergeable_model"), 11),
      IdentityNormalizer());
  EXPECT_FALSE(st.ok());
  EXPECT_NE(std::string::npos,
            std::string(st.message()).find("USER_DEFINED"));
}

TEST(BPEContinuationContractTest, UserDefinedIsInheritedFrozenAndNeverMerged) {
  const std::string input = TempPath("ud_frozen_input.txt");
  const std::string spec_path = TempPath("ud_frozen.pb");
  const std::string result_path = TempPath("ud_frozen.result");
  const std::string prefix = TempPath("ud_frozen_model");
  RemoveIfPresent(prefix + ".model");
  // The ONLY "<", "|", "v", ">" in the corpus sit inside the USER_DEFINED
  // occurrence, and "a<|v|>b" is by far the most frequent adjacency. Without
  // the freeze, the first learned merge would be a+"<" or "<"+"|".
  std::vector<std::string> lines;
  for (int i = 0; i < 20; ++i) lines.push_back("a<|v|>b");
  for (int i = 0; i < 3; ++i) lines.push_back("abc");
  ASSERT_TRUE(WriteLines(input, lines));
  ASSERT_TRUE(WriteProto(spec_path, UserDefinedSpec(false)));
  ASSERT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                             prefix, 11),
                         IdentityNormalizer())
                  .ok());

  ExpansionResult result;
  ASSERT_TRUE(ReadProto(result_path, &result));
  // Exact ID and type inheritance.
  ASSERT_EQ(9, result.base_pieces_size());
  EXPECT_EQ(1, result.base_pieces(1).external_id());
  EXPECT_EQ("<|v|>", result.base_pieces(1).piece());
  EXPECT_EQ(ModelProto::SentencePiece::USER_DEFINED, result.base_pieces(1).type());
  EXPECT_FALSE(result.base_pieces(1).mergeable());
  EXPECT_FALSE(result.base_pieces(1).atomic());

  // No learned piece contains a character that occurs only inside the
  // USER_DEFINED occurrence, and no learned merge names it.
  ASSERT_EQ(2, result.learned_pieces_size());
  for (const auto& piece : result.learned_pieces()) {
    EXPECT_EQ(std::string::npos, piece.piece().find_first_of("<|v>"))
        << "a merge entered or crossed a USER_DEFINED occurrence: "
        << piece.piece();
  }
  for (const auto& merge : result.learned_merges()) {
    EXPECT_NE("<|v|>", merge.left());
    EXPECT_NE("<|v|>", merge.right());
  }
  // The learnable statistics are exactly those of "abc": every learned
  // piece is drawn from {a, b, c}.
  for (const auto& piece : result.learned_pieces()) {
    EXPECT_EQ(std::string::npos, piece.piece().find_first_not_of("abc"))
        << piece.piece();
  }

  // Explicit runtime on the artifact: the freeze survives serialization.
  // (IdentityNormalizer() adds no dummy prefix, so there is no U+2581.)
  expansion::ExpansionProcessor p;
  ASSERT_TRUE(p.Load(result).ok());
  std::vector<int> ids;
  ASSERT_TRUE(p.EncodeIds("a<|v|>b", &ids).ok());
  EXPECT_EQ(std::vector<int>({p.PieceToId("a"), 1, p.PieceToId("b")}), ids);

  // Native parity, where a native model was emitted: SentencePiece's own
  // matcher must agree with the explicit runtime on every probe, including
  // the USER_DEFINED occurrences.
  ModelProto m;
  if (!ReadProto(prefix + ".model", &m)) {
    GTEST_SKIP() << "native model withheld; explicit runtime already checked";
  }
  EXPECT_EQ(ModelProto::SentencePiece::USER_DEFINED, m.pieces(1).type());
  SentencePieceProcessor sp;
  ASSERT_TRUE(sp.Load(prefix + ".model").ok());
  for (const std::string& probe :
       {std::string("a<|v|>b"), std::string("<|v|><|v|>"), std::string("ab<|v|>"),
        std::string("abc"), std::string("<|v")}) {
    std::vector<int> native, explicit_ids;
    ASSERT_TRUE(sp.Encode(probe, &native).ok());
    ASSERT_TRUE(p.EncodeIds(probe, &explicit_ids).ok());
    EXPECT_EQ(native, explicit_ids) << "disagreement on " << probe;
  }
}


// ---------------------------------------------------------------------------
// BPE continuation fences (--continuation_fence_strings), the G3 rule:
// no learned merge overlaps a fenced character. The fence unit itself is
// inherited whole and stays whole; the text on either side is never joined
// to it. Same flag, same normalization, same provenance encoding as Unigram.
// ---------------------------------------------------------------------------

ExpansionSpec FenceSpec() {
  ExpansionSpec e;
  e.set_schema_version(1);
  e.set_model_type(EXPANSION_BPE);
  e.set_preserve_base_ids(true);
  e.set_requested_new_pieces(1);
  AddPiece(&e, 0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  int id = 1;
  for (const char* c : {"x", "y", "P", ":"}) {
    AddPiece(&e, id++, c, ModelProto::SentencePiece::NORMAL, true, true);
  }
  AddPiece(&e, id++, "P:", ModelProto::SentencePiece::NORMAL, true, false);
  AddBaseMerge(&e, 0, "P", ":");
  *e.mutable_contract()->mutable_normalizer_spec() = IdentityNormalizer();
  return e;
}

TEST(BPEContinuationContractTest, ExplicitFenceBlocksMergesAcrossTheLabel) {
  const std::string input = TempPath("bpe_fence_input.txt");
  const std::string spec_path = TempPath("bpe_fence.pb");
  std::vector<std::string> lines;
  // "xP:y" is by far the most frequent adjacency; unfenced, x+P: wins.
  for (int i = 0; i < 20; ++i) lines.push_back("xP:y");
  for (int i = 0; i < 3; ++i) lines.push_back("xy");
  ASSERT_TRUE(WriteLines(input, lines));
  ASSERT_TRUE(WriteProto(spec_path, FenceSpec()));

  auto run = [&](const std::string& fences, const std::string& tag) {
    absl::SetFlag(&FLAGS_continuation_fence_strings, fences);
    const std::string result_path = TempPath("bpe_fence_" + tag + ".result");
    EXPECT_TRUE(RunTrainer(BpeContinuationSpec(input, spec_path, result_path,
                                               TempPath("bpe_fence_" + tag), 7),
                           IdentityNormalizer())
                    .ok());
    absl::SetFlag(&FLAGS_continuation_fence_strings, "");
    ExpansionResult r;
    EXPECT_TRUE(ReadProto(result_path, &r));
    return r;
  };

  const ExpansionResult unfenced = run("", "off");
  ASSERT_EQ(1, unfenced.learned_pieces_size());
  // x+P: and P:+y tie at 20; either one crosses the label.
  EXPECT_NE(std::string::npos, unfenced.learned_pieces(0).piece().find("P:"));
  EXPECT_EQ("", unfenced.boundary_policy());

  const ExpansionResult fenced = run("P:", "on");
  ASSERT_EQ(1, fenced.learned_pieces_size());
  EXPECT_EQ("xy", fenced.learned_pieces(0).piece())
      << "the only pair that does not overlap the fence";
  EXPECT_EQ("bpe_explicit_fences_v1:[P%3A]", fenced.boundary_policy());
  // the inherited fence unit was still replayed whole
  bool has_label = false;
  for (const auto& p : fenced.base_pieces()) has_label |= p.piece() == "P:";
  EXPECT_TRUE(has_label);
}

TEST(BPEContinuationContractTest, BpeBoundaryPolicyEncodingIsCanonical) {
  EXPECT_EQ("bpe_explicit_fences_v1:[]", bpe::EncodeBpeBoundaryPolicy({}));
  EXPECT_EQ("bpe_explicit_fences_v1:[%E2%96%81PL%3A,%E2%96%81Vn%3A]",
            bpe::EncodeBpeBoundaryPolicy({"\xe2\x96\x81Vn:", "\xe2\x96\x81PL:"}));
}

}  // namespace
}  // namespace sentencepiece
