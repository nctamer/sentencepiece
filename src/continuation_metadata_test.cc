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

}  // namespace
}  // namespace sentencepiece::continuation
