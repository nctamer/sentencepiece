// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <gtest/gtest.h>

#include <vector>

#include "id_overlay_processor.h"
#include "sentencepiece_model.pb.h"

namespace sentencepiece::overlay {
namespace {

IdOverlayProgram Program() {
  IdOverlayProgram p;
  p.set_schema_version(1);
  p.set_base_identity_sha256("0123456789abcdef");
  p.set_base_vocab_size(12);
  p.set_model_vocab_size(16);
  p.set_first_overlay_id(12);
  p.set_open_fence_id(8);
  p.set_close_fence_id(9);
  p.add_protected_base_ids(7);
  p.set_max_input_ids(1000);
  p.set_max_expansion_ids(1000);

  auto add = [&](uint32_t rank, int left, int right, int child,
                 std::initializer_list<int> expansion) {
    auto* r = p.add_rules();
    r->set_rank(rank);
    r->set_left_id(left);
    r->set_right_id(right);
    r->set_child_id(child);
    for (int id : expansion) r->add_base_expansion(id);
  };
  add(0, 1, 1, 12, {1, 1});
  add(1, 2, 3, 13, {2, 3});
  add(2, 1, 2, 14, {1, 2});
  return p;
}

std::vector<int> Encode(const IdOverlayProcessor& proc,
                        std::initializer_list<int> input,
                        double dropout = 0.0, uint64_t seed = 0,
                        bool allow_unclosed = false) {
  std::vector<int> out;
  EXPECT_TRUE(proc.EncodeIds(std::vector<int>(input), dropout, seed,
                             allow_unclosed, &out).ok());
  return out;
}

TEST(IdOverlayProcessorTest, RankedMergesOnlyInsideFences) {
  IdOverlayProcessor proc;
  ASSERT_TRUE(proc.Load(Program()).ok());

  EXPECT_EQ((std::vector<int>{1, 2, 3, 8, 1, 13, 9, 1, 2}),
            Encode(proc, {1, 2, 3, 8, 1, 2, 3, 9, 1, 2}));
  // Same learned pair outside the active region is untouched.
  EXPECT_EQ((std::vector<int>{2, 3, 8, 13, 9, 2, 3}),
            Encode(proc, {2, 3, 8, 2, 3, 9, 2, 3}));
}

TEST(IdOverlayProcessorTest, RankBeatsPositionAndOccurrenceTieIsLeftmost) {
  IdOverlayProcessor proc;
  ASSERT_TRUE(proc.Load(Program()).ok());
  // (2,3) has rank 1 and wins over the earlier-position (1,2) rank 2.
  EXPECT_EQ((std::vector<int>{8, 1, 13, 9}),
            Encode(proc, {8, 1, 2, 3, 9}));
  // Two overlapping occurrences of rank-0 (1,1): leftmost must win.
  EXPECT_EQ((std::vector<int>{8, 12, 1, 9}),
            Encode(proc, {8, 1, 1, 1, 9}));
}

TEST(IdOverlayProcessorTest, MultipleAndEmptyBlocksPreserveFences) {
  IdOverlayProcessor proc;
  ASSERT_TRUE(proc.Load(Program()).ok());
  EXPECT_EQ((std::vector<int>{4, 8, 9, 5, 8, 12, 9, 6}),
            Encode(proc, {4, 8, 9, 5, 8, 1, 1, 9, 6}));
}

TEST(IdOverlayProcessorTest, ProtocolErrorsFailClosedButOpenPrefixIsAllowed) {
  IdOverlayProcessor proc;
  ASSERT_TRUE(proc.Load(Program()).ok());
  std::vector<int> out;
  EXPECT_FALSE(proc.EncodeIds({9}, 0.0, 0, false, &out).ok());
  EXPECT_FALSE(proc.EncodeIds({8, 8, 9}, 0.0, 0, false, &out).ok());
  EXPECT_FALSE(proc.EncodeIds({8, 1}, 0.0, 0, false, &out).ok());
  EXPECT_TRUE(proc.EncodeIds({8, 1, 1}, 0.0, 0, true, &out).ok());
  EXPECT_EQ((std::vector<int>{8, 12}), out);
  // Overlay/model-extension IDs are illegal outside the protocol.
  EXPECT_FALSE(proc.EncodeIds({12}, 0.0, 0, false, &out).ok());
}

TEST(IdOverlayProcessorTest, ExpansionIsExactAndValidatesProtocol) {
  IdOverlayProcessor proc;
  ASSERT_TRUE(proc.Load(Program()).ok());
  std::vector<int> out;
  ASSERT_TRUE(proc.ExpandIds({4, 8, 12, 13, 9, 5}, false, &out).ok());
  EXPECT_EQ((std::vector<int>{4, 8, 1, 1, 2, 3, 9, 5}), out);
  EXPECT_FALSE(proc.ExpandIds({12}, false, &out).ok());
}

TEST(IdOverlayProcessorTest, DropoutEndpointsAndSeedAreReproducible) {
  IdOverlayProcessor proc;
  ASSERT_TRUE(proc.Load(Program()).ok());
  EXPECT_EQ((std::vector<int>{8, 12, 9}),
            Encode(proc, {8, 1, 1, 9}, 0.0, 77));
  EXPECT_EQ((std::vector<int>{8, 1, 1, 9}),
            Encode(proc, {8, 1, 1, 9}, 1.0, 77));
  const auto a = Encode(proc, {8, 1, 1, 9}, 0.5, 1234);
  const auto b = Encode(proc, {8, 1, 1, 9}, 0.5, 1234);
  EXPECT_EQ(a, b);

  std::vector<int> expanded;
  ASSERT_TRUE(proc.ExpandIds(a, false, &expanded).ok());
  EXPECT_EQ((std::vector<int>{8, 1, 1, 9}), expanded);
}

TEST(IdOverlayProcessorTest, InvalidArtifactIsRejected) {
  {
    auto p = Program();
    p.mutable_rules(0)->set_left_id(8);  // opening fence as merge operand
    IdOverlayProcessor proc;
    EXPECT_FALSE(proc.Load(p).ok());
  }
  {
    auto p = Program();
    p.mutable_rules(0)->set_base_expansion(0, 2);  // false expansion witness
    IdOverlayProcessor proc;
    EXPECT_FALSE(proc.Load(p).ok());
  }
  {
    auto p = Program();
    p.mutable_rules(1)->set_rank(0);  // duplicate rank
    IdOverlayProcessor proc;
    EXPECT_FALSE(proc.Load(p).ok());
  }
}

}  // namespace
}  // namespace sentencepiece::overlay
