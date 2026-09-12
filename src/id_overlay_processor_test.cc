// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <limits>

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

TEST(IdOverlayProcessorTest, LoadIsTransactionalAndRequiredBeforeUse) {
  IdOverlayProcessor proc;
  std::vector<int> out;
  EXPECT_FALSE(proc.EncodeIds({}, 0, 0, false, &out).ok());
  EXPECT_FALSE(proc.ExpandIds({}, false, &out).ok());
  ASSERT_TRUE(proc.Load(Program()).ok());
  auto bad = Program();
  bad.mutable_rules(2)->set_child_id(12);
  EXPECT_FALSE(proc.Load(bad).ok());
  EXPECT_EQ((std::vector<int>{8, 13, 9}), Encode(proc, {8, 2, 3, 9}));
}

TEST(IdOverlayProcessorTest, MacroLimitIsNotTheWholeSequenceLimit) {
  auto p = Program();
  p.set_max_expansion_ids(2);
  p.set_max_input_ids(8);
  IdOverlayProcessor proc;
  ASSERT_TRUE(proc.Load(p).ok());
  std::vector<int> out;
  EXPECT_TRUE(proc.ExpandIds({8, 12, 12, 9}, false, &out).ok());
  EXPECT_EQ((std::vector<int>{8, 1, 1, 1, 1, 9}), out);
  EXPECT_FALSE(proc.ExpandIds({8, 12, 12, 12, 12, 9}, false, &out).ok());
  EXPECT_FALSE(proc.ExpandIds(std::vector<int>(9, 1), false, &out).ok());
}

TEST(IdOverlayProcessorTest, DuplicateExpandedSequenceIsRejected) {
  auto p = Program();
  p.set_model_vocab_size(17);
  auto* r = p.add_rules();
  r->set_rank(3); r->set_left_id(1); r->set_right_id(13); r->set_child_id(15);
  for (int id : {1, 2, 3}) r->add_base_expansion(id);
  r = p.add_rules();
  r->set_rank(4); r->set_left_id(14); r->set_right_id(3); r->set_child_id(16);
  for (int id : {1, 2, 3}) r->add_base_expansion(id);
  IdOverlayProcessor proc;
  EXPECT_FALSE(proc.Load(p).ok());
}

// Independent O(rules * tokens * replacements) reference: no production
// eligibility, links, pair lookup, heap, or mutation helper is reused.
std::vector<int> Reference(std::vector<int> ids, const IdOverlayProgram& p) {
  while (true) {
    size_t best = ids.size();
    uint32_t rank = std::numeric_limits<uint32_t>::max();
    int child = -1;
    bool inside = false;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] == p.open_fence_id()) { inside = true; continue; }
      if (ids[i] == p.close_fence_id()) { inside = false; continue; }
      if (!inside || i + 1 == ids.size()) continue;
      for (const auto& r : p.rules()) {
        if (r.left_id() == ids[i] && r.right_id() == ids[i + 1] &&
            (best == ids.size() || r.rank() < rank)) {
          best = i; rank = r.rank(); child = r.child_id();
        }
      }
    }
    if (best == ids.size()) return ids;
    ids[best] = child;
    ids.erase(ids.begin() + best + 1);
  }
}

TEST(IdOverlayProcessorTest, IndependentFullReplayRandomDifferential) {
  auto p = Program();
  p.set_model_vocab_size(18);
  auto* r = p.add_rules();
  r->set_rank(3); r->set_left_id(12); r->set_right_id(1); r->set_child_id(15);
  for (int id : {1, 1, 1}) r->add_base_expansion(id);
  r = p.add_rules();
  r->set_rank(4); r->set_left_id(15); r->set_right_id(13); r->set_child_id(16);
  for (int id : {1, 1, 1, 2, 3}) r->add_base_expansion(id);
  IdOverlayProcessor proc;
  ASSERT_TRUE(proc.Load(p).ok());
  std::mt19937 rng(348);
  for (int trial = 0; trial < 1000; ++trial) {
    std::vector<int> input{1, 2, 3};
    for (int block = 0; block < 3; ++block) {
      input.push_back(8);
      const auto n = rng() % 60;
      for (size_t j = 0; j < n; ++j) input.push_back(1 + rng() % 3);
      input.push_back(9);
      input.push_back(7);
    }
    std::vector<int> actual, expanded;
    ASSERT_TRUE(proc.EncodeIds(input, 0, trial, false, &actual).ok());
    EXPECT_EQ(Reference(input, p), actual) << trial;
    ASSERT_TRUE(proc.ExpandIds(actual, false, &expanded).ok());
    EXPECT_EQ(input, expanded);
    ASSERT_TRUE(proc.EncodeIds(input, .4, trial, false, &actual).ok());
    ASSERT_TRUE(proc.ExpandIds(actual, false, &expanded).ok());
    EXPECT_EQ(input, expanded);
  }
}

}  // namespace
}  // namespace sentencepiece::overlay
