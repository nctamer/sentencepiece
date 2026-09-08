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

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
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

TEST(UnigramContinuationContractTest, RejectsUnsupportedAlphabetAndCapacity) {
  // The corpus is outside anything the prior can segment.
  const std::string input = TempPath("cont_uni_alpha.txt");
  const std::string prior_path = TempPath("cont_uni_alpha.prior");
  const std::string result_path = TempPath("cont_uni_alpha.result");
  const std::string prefix = TempPath("cont_uni_alpha_model");
  ASSERT_TRUE(WriteProto(prior_path, MakeMultiPathUnigramPrior()));
  ASSERT_TRUE(WriteLines(input, std::vector<std::string>(20, "zzzz")));
  TrainerSpec trainer = UnigramContinuationSpec(input, "text", prior_path,
                                                result_path, prefix);
  trainer.set_vocab_size(6);
  EXPECT_FALSE(RunTrainer(trainer, NormalizerSpec()).ok());

  // Asking for a vocabulary smaller than the prior cannot be honoured: the
  // prior is the floor, and inherited pieces are never pruned.
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
