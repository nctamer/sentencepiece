// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "continuation_io.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_trainer.h"

namespace sentencepiece::continuation {
namespace {

std::string TempFile(absl::string_view leaf) {
  return std::string(::testing::TempDir()) + "/" + std::string(leaf);
}

NormalizerSpec Identity() {
  NormalizerSpec n;
  n.set_name("identity");
  n.set_add_dummy_prefix(false);
  n.set_remove_extra_whitespaces(false);
  n.set_escape_whitespaces(false);
  return n;
}

TEST(ContinuationMetadataTest, TsvMetaKeepsEqualSurfaceDifferentFactsSeparate) {
  const std::string path = TempFile("continuation_metadata.tsv");
  {
    std::ofstream out(path);
    out << "aa\t2\tgrammar-A\n";
    out << "aa\t3\tgrammar-B\n";
    out << "aa\t4\tgrammar-A\n";
  }
  TrainerSpec spec;
  spec.add_input(path);
  spec.set_input_format("tsv_meta");
  spec.set_input_sentence_size(0);
  spec.set_max_sentence_length(1024);

  PreparedCorpus corpus;
  ASSERT_TRUE(LoadPreparedCorpus(spec, Identity(), TrainerComponents{},
                                 &corpus).ok());
  ASSERT_EQ(2u, corpus.sentences.size());
  ASSERT_EQ(2u, corpus.metadata_keys.size());
  EXPECT_EQ("aa", corpus.sentences[0].first);
  EXPECT_EQ(6, corpus.sentences[0].second);
  EXPECT_EQ("grammar-A", corpus.metadata_keys[0]);
  EXPECT_EQ("aa", corpus.sentences[1].first);
  EXPECT_EQ(3, corpus.sentences[1].second);
  EXPECT_EQ("grammar-B", corpus.metadata_keys[1]);
  EXPECT_EQ(9, corpus.weighted_sentence_count);
}

TEST(ContinuationMetadataTest, LegacyTsvStillFoldsBySurfaceOnly) {
  const std::string path = TempFile("continuation_legacy.tsv");
  {
    std::ofstream out(path);
    out << "aa\t2\n";
    out << "aa\t4\n";
  }
  TrainerSpec spec;
  spec.add_input(path);
  spec.set_input_format("tsv");
  spec.set_input_sentence_size(0);
  spec.set_max_sentence_length(1024);

  PreparedCorpus corpus;
  ASSERT_TRUE(LoadPreparedCorpus(spec, Identity(), TrainerComponents{},
                                 &corpus).ok());
  ASSERT_EQ(1u, corpus.sentences.size());
  EXPECT_EQ(6, corpus.sentences[0].second);
  ASSERT_EQ(1u, corpus.metadata_keys.size());
  EXPECT_TRUE(corpus.metadata_keys[0].empty());
}

TEST(ContinuationMetadataTest, RealTrainerSeparatesSupportFromApplications) {
  const std::string prefix = TempFile("metadata_train");
  {
    std::ofstream out(prefix + ".tsv");
    out << "aa\t2\tA\naa\t3\tB\nbb\t4\tC\n";
  }
  {
    std::ofstream out(prefix + ".hier");
    out << "# sentencepiece-bpe-hierarchy-v2\n"
           "aa\tA\t\t0-2\naa\tB\t\t\nbb\tC\t\t0-2\n";
  }
  ExpansionSpec spec;
  spec.set_model_type(EXPANSION_BPE);
  spec.set_first_new_external_id(3);
  spec.set_requested_new_pieces(2);
  auto* p = spec.add_base_pieces();
  p->set_external_id(0); p->set_piece("<unk>");
  p->set_type(ModelProto::SentencePiece::UNKNOWN); p->set_mergeable(false);
  for (int id = 1; id <= 2; ++id) {
    p = spec.add_base_pieces();
    p->set_external_id(id); p->set_piece(id == 1 ? "a" : "b");
    p->set_atomic(true);
  }
  {
    std::ofstream out(prefix + ".spec", std::ios::binary);
    out << spec.SerializeAsString();
  }
  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE);
  ts.add_input(prefix + ".tsv"); ts.set_input_format("tsv_meta");
  ts.set_model_prefix(prefix); ts.set_vocab_size(5);
  ts.set_expansion_spec(prefix + ".spec");
  ts.set_expansion_result(prefix + ".expansion");
  ts.set_bpe_hierarchy_file(prefix + ".hier");
  ts.set_input_sentence_size(0);
  ts.set_split_by_whitespace(false); ts.set_split_by_unicode_script(false);
  ts.set_split_by_number(false); ts.set_bos_id(-1); ts.set_eos_id(-1);
  auto status = SentencePieceTrainer::Train(ts, Identity(), NormalizerSpec{});
  ASSERT_TRUE(status.ok()) << status;
  std::ifstream in(prefix + ".expansion", std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(2, result.learned_merges_size());
  EXPECT_EQ("b", result.learned_merges(0).left());
  EXPECT_EQ(4, result.learned_merges(0).weighted_count());
  EXPECT_EQ(4, result.learned_merges(0).application_count());
  EXPECT_EQ("a", result.learned_merges(1).left());
  EXPECT_EQ(2, result.learned_merges(1).weighted_count());
  EXPECT_EQ(5, result.learned_merges(1).application_count());
  EXPECT_EQ(9, result.training_final_weighted_tokens());
  // A v2 support range, just like a hierarchy cut, must align to UTF-8.
  spec.mutable_base_pieces(1)->set_piece("é");
  {
    std::ofstream out(prefix + ".spec", std::ios::binary);
    out << spec.SerializeAsString();
  }
  {
    std::ofstream out(prefix + ".tsv");
    out << "éé\t2\tA\n";
  }
  {
    std::ofstream out(prefix + ".hier");
    out << "# sentencepiece-bpe-hierarchy-v2\néé\tA\t\t1-4\n";
  }
  status = SentencePieceTrainer::Train(ts, Identity(), NormalizerSpec{});
  EXPECT_FALSE(status.ok());
  EXPECT_NE(std::string::npos, std::string(status.message()).find("UTF-8"));
}

}  // namespace
}  // namespace sentencepiece::continuation
