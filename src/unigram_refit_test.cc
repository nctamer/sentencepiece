// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Acceptance for FIXED-VOCABULARY UNIGRAM SCORE REFIT.
//
// The properties here are the whole contract: support cannot move, special
// pieces keep their semantics, and the scores really are re-estimated from the
// WEIGHTED corpus by forward-backward EM rather than by a one-pass count.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "filesystem.h"
#include "sentencepiece_model.pb.h"
#include "unigram_model.h"
#include "unigram_refit.h"
#include "util.h"

namespace sentencepiece {
namespace {

using unigram_refit::RefitFixedVocabulary;
using unigram_refit::RefitOptions;
using unigram_refit::RefitStats;
using SP = ModelProto::SentencePiece;

std::string TempPath(absl::string_view leaf) {
  return filesystem::JoinPath(::testing::TempDir(), leaf);
}

bool WriteLines(absl::string_view path, const std::vector<std::string>& lines) {
  auto out = filesystem::NewWritableFile(path);
  if (!out->status().ok()) return false;
  for (const auto& l : lines) {
    if (!out->WriteLine(l)) return false;
  }
  return true;
}

NormalizerSpec IdentityNormalizer() {
  NormalizerSpec n;
  n.set_name("identity");
  n.set_add_dummy_prefix(false);
  n.set_remove_extra_whitespaces(false);
  n.set_escape_whitespaces(true);
  return n;
}

struct Entry {
  std::string piece;
  float score;
  SP::Type type;
};

ModelProto MakeModel(const std::vector<Entry>& entries) {
  ModelProto m;
  TrainerSpec* t = m.mutable_trainer_spec();
  t->set_model_type(TrainerSpec::UNIGRAM);
  t->set_vocab_size(static_cast<int>(entries.size()));
  t->set_unk_id(0);
  t->set_bos_id(-1);
  t->set_eos_id(-1);
  t->set_pad_id(-1);
  t->set_byte_fallback(false);
  t->set_split_by_whitespace(false);
  t->set_split_by_unicode_script(false);
  t->set_split_by_number(false);
  *m.mutable_normalizer_spec() = IdentityNormalizer();
  for (const Entry& e : entries) {
    auto* sp = m.add_pieces();
    sp->set_piece(e.piece);
    sp->set_score(e.score);
    sp->set_type(e.type);
  }
  return m;
}

// A three-way ambiguous all-NORMAL vocabulary: "ab" competes with "a"+"b".
ModelProto MakeAbModel(float sa = -1.0f, float sb = -1.0f, float sab = -1.0f) {
  return MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                    {"a", sa, SP::NORMAL},
                    {"b", sb, SP::NORMAL},
                    {"ab", sab, SP::NORMAL}});
}

TrainerSpec CorpusSpec(const std::string& path, const std::string& format) {
  TrainerSpec s;
  s.add_input(path);
  s.set_input_format(format);
  s.set_max_sentence_length(8192);
  return s;
}

absl::Status Refit(const ModelProto& in, const std::string& corpus,
                   const std::string& format, ModelProto* out,
                   RefitStats* stats, RefitOptions options = RefitOptions()) {
  TrainerComponents components;
  return RefitFixedVocabulary(in, CorpusSpec(corpus, format), components,
                              options, out, stats);
}

double ScoreOf(const ModelProto& m, absl::string_view piece) {
  for (const auto& p : m.pieces()) {
    if (p.piece() == piece) return p.score();
  }
  ADD_FAILURE() << "missing piece " << piece;
  return std::numeric_limits<double>::quiet_NaN();
}

bool BitEqual(float a, float b) {
  uint32_t x, y;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}

// --------------------------------------------------------------- Test 1

TEST(UnigramRefitTest, SupportIsImmutable) {
  const std::string corpus = TempPath("refit_support.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"ab\t10", "ba\t3"}));
  const ModelProto in = MakeAbModel();
  ModelProto out;
  RefitStats stats;
  ASSERT_TRUE(Refit(in, corpus, "tsv", &out, &stats).ok());

  ASSERT_EQ(in.pieces_size(), out.pieces_size());
  for (int i = 0; i < in.pieces_size(); ++i) {
    EXPECT_EQ(in.pieces(i).piece(), out.pieces(i).piece()) << "id " << i;
    EXPECT_EQ(in.pieces(i).type(), out.pieces(i).type()) << "id " << i;
  }
  EXPECT_EQ(4, stats.piece_count);
  EXPECT_EQ(3, stats.normal_piece_count);
  EXPECT_EQ(1, stats.special_piece_count);
}

// --------------------------------------------------------------- Test 2

TEST(UnigramRefitTest, NonNormalPiecesArePreservedBitIdentically) {
  // USER_DEFINED is scored GetUserDefinedScore(length) in the lattice, never
  // from its stored score, and CONTROL is not in the trie at all. Neither may
  // be reinterpreted as a trainable NORMAL probability.
  const ModelProto in = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                   {"<s>", -7.25f, SP::CONTROL},
                                   {"<X>", -3.5f, SP::USER_DEFINED},
                                   {"a", -1.0f, SP::NORMAL},
                                   {"b", -1.0f, SP::NORMAL},
                                   {"ab", -1.0f, SP::NORMAL}});
  const std::string corpus = TempPath("refit_special.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"a<X>b\t9", "ab\t4"}));

  ModelProto out;
  RefitStats stats;
  ASSERT_TRUE(Refit(in, corpus, "tsv", &out, &stats).ok());

  for (int i = 0; i < in.pieces_size(); ++i) {
    EXPECT_EQ(in.pieces(i).type(), out.pieces(i).type());
    EXPECT_EQ(in.pieces(i).piece(), out.pieces(i).piece());
    if (in.pieces(i).type() != SP::NORMAL) {
      EXPECT_TRUE(BitEqual(in.pieces(i).score(), out.pieces(i).score()))
          << "non-NORMAL score moved at id " << i << " ("
          << in.pieces(i).piece() << ")";
    }
  }
  EXPECT_EQ(3, stats.special_piece_count);

  // The NORMAL simplex is over NORMAL pieces only; the special pieces do not
  // take part in it and their stored scores are not renormalized into it.
  double mass = 0.0;
  for (const auto& p : out.pieces()) {
    if (p.type() == SP::NORMAL) mass += std::exp(static_cast<double>(p.score()));
  }
  EXPECT_NEAR(1.0, mass, 1e-5);
}

// --------------------------------------------------------------- Test 3

TEST(UnigramRefitTest, ScoresActuallyChangeInTheExpectedDirection) {
  // Flat input scores, corpus overwhelmingly "ab". "ab" must gain against its
  // two-piece decomposition. A no-op implementation fails here.
  const ModelProto in = MakeAbModel(-1.0f, -1.0f, -1.0f);
  const std::string corpus = TempPath("refit_direction.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"ab\t500", "ba\t1"}));

  ModelProto out;
  RefitStats stats;
  ASSERT_TRUE(Refit(in, corpus, "tsv", &out, &stats).ok());

  EXPECT_GT(stats.changed_normal_scores, 0);
  const double ab = ScoreOf(out, "ab");
  const double a = ScoreOf(out, "a");
  const double b = ScoreOf(out, "b");
  EXPECT_GT(ab, ScoreOf(in, "ab")) << "ab did not gain on an ab-dominated corpus";
  EXPECT_GT(ab, a + b) << "the merged piece did not beat its decomposition";
}

// --------------------------------------------------------------- Test 4

TEST(UnigramRefitTest, OriginalTsvCountsAreActuallyUsed) {
  const ModelProto in = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                   {"x", -1.0f, SP::NORMAL},
                                   {"y", -1.0f, SP::NORMAL}});
  const std::string a = TempPath("refit_wa.tsv");
  const std::string b = TempPath("refit_wb.tsv");
  ASSERT_TRUE(WriteLines(a, {"x\t1", "y\t1"}));
  ASSERT_TRUE(WriteLines(b, {"x\t100", "y\t1"}));

  ModelProto oa, ob;
  RefitStats sa, sb;
  ASSERT_TRUE(Refit(in, a, "tsv", &oa, &sa).ok());
  ASSERT_TRUE(Refit(in, b, "tsv", &ob, &sb).ok());

  // Balanced corpus -> p(x) = p(y) = 1/2.
  EXPECT_NEAR(std::log(0.5), ScoreOf(oa, "x"), 1e-5);
  EXPECT_NEAR(std::log(0.5), ScoreOf(oa, "y"), 1e-5);
  // Skewed corpus -> p(x) = 100/101, analytically, because each record is one
  // unambiguous token.
  EXPECT_NEAR(std::log(100.0 / 101.0), ScoreOf(ob, "x"), 1e-4);
  EXPECT_NEAR(std::log(1.0 / 101.0), ScoreOf(ob, "y"), 1e-4);
  EXPECT_GT(ScoreOf(ob, "x"), ScoreOf(oa, "x"));
}

// --------------------------------------------------------------- Test 5

TEST(UnigramRefitTest, WeightedTsvEqualsPhysicalRepetition) {
  const ModelProto in = MakeAbModel();
  const std::string tsv = TempPath("refit_rep.tsv");
  const std::string txt = TempPath("refit_rep.txt");
  ASSERT_TRUE(WriteLines(tsv, {"ab\t10", "ba\t3"}));
  std::vector<std::string> rep;
  for (int i = 0; i < 10; ++i) rep.push_back("ab");
  for (int i = 0; i < 3; ++i) rep.push_back("ba");
  ASSERT_TRUE(WriteLines(txt, rep));

  ModelProto o1, o2;
  RefitStats s1, s2;
  ASSERT_TRUE(Refit(in, tsv, "tsv", &o1, &s1).ok());
  ASSERT_TRUE(Refit(in, txt, "text", &o2, &s2).ok());

  EXPECT_EQ(s1.weighted_record_count, s2.weighted_record_count);
  EXPECT_EQ(s1.distinct_record_count, s2.distinct_record_count);
  EXPECT_DOUBLE_EQ(s1.final_objective, s2.final_objective);
  ASSERT_EQ(o1.pieces_size(), o2.pieces_size());
  for (int i = 0; i < o1.pieces_size(); ++i) {
    // Exact, not approximate: the loader folds identical normalized records
    // into one weighted entry, so both spellings take the same summation path.
    EXPECT_TRUE(BitEqual(o1.pieces(i).score(), o2.pieces(i).score()))
        << "id " << i << " " << o1.pieces(i).piece();
  }
}

// --------------------------------------------------------------- Test 6

TEST(UnigramRefitTest, ZeroPosteriorPieceKeepsItsSlot) {
  // "qq" has no occurrence at all. Standard Unigram RunMStep() would drop it
  // (expected count < 0.5). Refit must not: support is fixed.
  const ModelProto in = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                   {"a", -1.0f, SP::NORMAL},
                                   {"b", -1.0f, SP::NORMAL},
                                   {"ab", -1.0f, SP::NORMAL},
                                   {"qq", -1.0f, SP::NORMAL}});
  const std::string corpus = TempPath("refit_zero.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"ab\t50", "ba\t7"}));

  ModelProto out;
  RefitStats stats;
  ASSERT_TRUE(Refit(in, corpus, "tsv", &out, &stats).ok());

  ASSERT_EQ(5, out.pieces_size());
  EXPECT_EQ("qq", out.pieces(4).piece());
  EXPECT_EQ(SP::NORMAL, out.pieces(4).type());
  EXPECT_TRUE(std::isfinite(out.pieces(4).score()));
  EXPECT_LT(out.pieces(4).score(), -50.0f) << "a zero-count piece should be "
                                              "floored to a tiny probability, "
                                              "not deleted";
  ASSERT_FALSE(stats.per_iteration.empty());
  EXPECT_GE(stats.per_iteration.front().floored_normal, 1);
}

// --------------------------------------------------------------- Test 7

TEST(UnigramRefitTest, UnsupportedAlphabetFailsInsteadOfExpanding) {
  const ModelProto in = MakeAbModel();
  const std::string corpus = TempPath("refit_unsupported.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"ab\t5", "Zb\t5"}));

  ModelProto out;
  RefitStats stats;
  const absl::Status status = Refit(in, corpus, "tsv", &out, &stats);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;
  EXPECT_NE(std::string::npos, std::string(status.message()).find("Z"));
  // No model pretending success, and certainly no appended 'Z'.
  EXPECT_EQ(0, out.pieces_size());
}

// --------------------------------------------------------------- Test 8

TEST(UnigramRefitTest, RerunIsBitIdentical) {
  const ModelProto in = MakeAbModel();
  const std::string corpus = TempPath("refit_det.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"ab\t31", "ba\t17", "aab\t5"}));

  // FIXED-CONFIGURATION determinism: same model, same corpus, same flags,
  // same thread count. Cross-thread-count identity is NOT claimed and is not
  // tested, because the E-step partitions records by thread stride and sums
  // per-thread partials, so a different thread count takes a different (still
  // deterministic) summation path. Both configurations are pinned below.
  ModelProto o1, o2;
  RefitStats s1, s2;
  for (const int threads : {1, 4}) {
    RefitOptions options;
    options.num_threads = threads;
    ASSERT_TRUE(Refit(in, corpus, "tsv", &o1, &s1, options).ok());
    ASSERT_TRUE(Refit(in, corpus, "tsv", &o2, &s2, options).ok());
    ASSERT_EQ(o1.pieces_size(), o2.pieces_size());
    for (int i = 0; i < o1.pieces_size(); ++i) {
      EXPECT_TRUE(BitEqual(o1.pieces(i).score(), o2.pieces(i).score()))
          << "id " << i << " at num_threads=" << threads;
    }
    EXPECT_DOUBLE_EQ(s1.final_objective, s2.final_objective)
        << "at num_threads=" << threads;
  }

  ASSERT_EQ(o1.pieces_size(), o2.pieces_size());
  for (int i = 0; i < o1.pieces_size(); ++i) {
    EXPECT_EQ(o1.pieces(i).piece(), o2.pieces(i).piece());
    EXPECT_EQ(o1.pieces(i).type(), o2.pieces(i).type());
    EXPECT_TRUE(BitEqual(o1.pieces(i).score(), o2.pieces(i).score()))
        << "id " << i;
  }
  EXPECT_DOUBLE_EQ(s1.final_objective, s2.final_objective);
}

// --------------------------------------------------------------- Test 9

TEST(UnigramRefitTest, ObjectiveNeverWorsens) {
  // Ordinary fixed-support EM with an exact (non-Bayesian) M-step: the
  // weighted mean negative log-likelihood is non-increasing FROM ONE EM
  // ITERATION TO THE NEXT.
  //
  // The initial objective is deliberately NOT part of that chain. It is
  // measured with the input model's stored scores, and an input model is not
  // required to be normalized -- exp-sum over its NORMAL pieces can exceed 1,
  // which inflates every path likelihood and makes the pre-refit objective
  // look better than any proper distribution could be. The first M-step
  // projects onto the simplex and the objective may legitimately rise once.
  // See ObjectiveIsMonotoneFromANormalizedStart for the case where the
  // guarantee does hold from the very first step.
  const ModelProto in = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                   {"a", -0.2f, SP::NORMAL},
                                   {"b", -9.0f, SP::NORMAL},
                                   {"ab", -4.0f, SP::NORMAL},
                                   {"aab", -6.0f, SP::NORMAL}});
  const std::string corpus = TempPath("refit_obj.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"aab\t40", "ab\t25", "ba\t9", "a\t3"}));

  RefitOptions options;
  options.num_iterations = 12;
  ModelProto out;
  RefitStats stats;
  ASSERT_TRUE(Refit(in, corpus, "tsv", &out, &stats, options).ok());

  ASSERT_EQ(12, static_cast<int>(stats.per_iteration.size()));
  for (size_t i = 1; i < stats.per_iteration.size(); ++i) {
    const double prev = stats.per_iteration[i - 1].objective;
    const double cur = stats.per_iteration[i].objective;
    EXPECT_LE(cur, prev + 1e-6) << "objective worsened between EM iterations "
                                << i << " and " << i + 1 << ": " << prev
                                << " -> " << cur;
  }
}

TEST(UnigramRefitTest, ObjectiveIsMonotoneFromANormalizedStart) {
  // Same corpus, but the input model is itself a refit output, so its NORMAL
  // scores are a proper simplex. Now the guarantee holds from iteration 0 on,
  // and a second refit is a fixed point up to float tolerance.
  const std::string corpus = TempPath("refit_obj2.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"aab\t40", "ab\t25", "ba\t9", "a\t3"}));
  const ModelProto seed = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                     {"a", -0.2f, SP::NORMAL},
                                     {"b", -9.0f, SP::NORMAL},
                                     {"ab", -4.0f, SP::NORMAL},
                                     {"aab", -6.0f, SP::NORMAL}});
  ModelProto normalized;
  RefitStats s0;
  RefitOptions options;
  options.num_iterations = 20;
  ASSERT_TRUE(Refit(seed, corpus, "tsv", &normalized, &s0, options).ok());

  double mass = 0.0;
  for (const auto& p : normalized.pieces()) {
    if (p.type() == SP::NORMAL) mass += std::exp(static_cast<double>(p.score()));
  }
  ASSERT_NEAR(1.0, mass, 1e-5);

  ModelProto again;
  RefitStats s1;
  ASSERT_TRUE(Refit(normalized, corpus, "tsv", &again, &s1, options).ok());

  double prev = s1.initial_objective;
  for (size_t i = 0; i < s1.per_iteration.size(); ++i) {
    const double cur = s1.per_iteration[i].objective;
    EXPECT_LE(cur, prev + 1e-6) << "objective worsened at iteration " << i
                                << ": " << prev << " -> " << cur;
    prev = cur;
  }
  // Converged input -> refit is a fixed point.
  EXPECT_NEAR(s1.initial_objective, s1.final_objective, 1e-5);
}

// --------------------------------------------------------------- Test 10

TEST(UnigramRefitTest, RefitBeatsEqualWeightCalibrationOnWeightedData) {
  // THE REASON THIS FEATURE EXISTS. Stage 1 picks support from a corpus where
  // every distinct record has weight 1; those equal-weight scores are then
  // wrong for the real weighted corpus. Refit must fix exactly that.
  const std::string flat = TempPath("refit_flat.tsv");
  const std::string skewed = TempPath("refit_skewed.tsv");
  ASSERT_TRUE(WriteLines(flat, {"ab\t1", "ba\t1", "aab\t1"}));
  ASSERT_TRUE(WriteLines(skewed, {"ab\t1", "ba\t1", "aab\t900"}));

  const ModelProto seed = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                     {"a", -1.0f, SP::NORMAL},
                                     {"b", -1.0f, SP::NORMAL},
                                     {"ab", -1.0f, SP::NORMAL},
                                     {"aab", -1.0f, SP::NORMAL}});

  // 1. calibrate on the equal-weight corpus: this stands in for stage 1.
  ModelProto equal_weight;
  RefitStats s_flat;
  ASSERT_TRUE(Refit(seed, flat, "tsv", &equal_weight, &s_flat).ok());

  // 2. refit that exact support on the real weighted corpus.
  ModelProto refit;
  RefitStats s_skewed;
  ASSERT_TRUE(Refit(equal_weight, skewed, "tsv", &refit, &s_skewed).ok());

  // initial_objective is the equal-weight model measured on the WEIGHTED
  // corpus; final_objective is the refit model on the same corpus.
  EXPECT_LT(s_skewed.final_objective, s_skewed.initial_objective)
      << "refit did not improve weighted-corpus NLL over equal-weight "
         "calibration";
  EXPECT_GT(ScoreOf(refit, "aab"), ScoreOf(equal_weight, "aab"));
  // And support still did not move.
  ASSERT_EQ(equal_weight.pieces_size(), refit.pieces_size());
  for (int i = 0; i < refit.pieces_size(); ++i) {
    EXPECT_EQ(equal_weight.pieces(i).piece(), refit.pieces(i).piece());
    EXPECT_EQ(equal_weight.pieces(i).type(), refit.pieces(i).type());
  }
}


// ------------------------------------------------------------- UNK exclusion

// THE REGRESSION THAT EXPOSED A REAL MATHEMATICAL INCONSISTENCY.
//
// PopulateNodes() inserts an UNK node at every position lacking a
// single-character node -- including positions strictly INSIDE the span that
// the USER_DEFINED piece "<X>" covers, because '<', 'X' and '>' are not
// pieces. Those UNK nodes are scored min_score() - kUnkPenalty, and
// min_score() is derived from the trainable NORMAL scores. If they carried
// posterior mass then p_i = d_i / sum_NORMAL d_j would not be the exact
// maximizer of the objective, because moving a NORMAL score would also move
// the UNK-path score.
//
// The whole assertion is hand-computable, which is the point: with UNK
// excluded there is EXACTLY ONE path through "a<X>b", namely a + <X> + b, so
//
//     log Z = score(a) + GetUserDefinedScore(3) + score(b)
//           = score(a) + 0.1*(3-1) + score(b)
//
// Any implementation that lets UNK contribute posterior gets a strictly larger
// Z (more paths) and fails this test.
TEST(UnigramRefitTest, UnknownNodesContributeZeroPosterior) {
  const float kUserDefinedBonus = 0.1f * (3 - 1);  // "<X>" is 3 characters
  const ModelProto in = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                   {"<X>", -3.5f, SP::USER_DEFINED},
                                   {"a", -0.9f, SP::NORMAL},
                                   {"b", -1.7f, SP::NORMAL}});
  const std::string corpus = TempPath("refit_unkzero.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"a<X>b\t1"}));

  ModelProto out;
  RefitStats stats;
  RefitOptions options;
  options.num_iterations = 4;
  ASSERT_TRUE(Refit(in, corpus, "tsv", &out, &stats, options).ok())
      << "coverage must PASS: <X> covers the span its characters cannot";

  // 1. the pre-refit objective is exactly the single known path.
  const double expected_initial = -(-0.9 + kUserDefinedBonus + -1.7);
  EXPECT_NEAR(expected_initial, stats.initial_objective, 1e-5)
      << "UNK alternatives inside <X> are contributing to Z";

  // 2. one occurrence each of "a" and "b" -> p = 1/2 each, so the fixed point
  //    is exact and known in closed form.
  EXPECT_NEAR(std::log(0.5), ScoreOf(out, "a"), 1e-5);
  EXPECT_NEAR(std::log(0.5), ScoreOf(out, "b"), 1e-5);
  const double expected_final = -(std::log(0.5) + kUserDefinedBonus +
                                  std::log(0.5));
  EXPECT_NEAR(expected_final, stats.final_objective, 1e-5);

  // 3. the NORMAL scores are a proper simplex, and <X> is not in it.
  double mass = 0.0;
  for (const auto& p : out.pieces()) {
    if (p.type() == SP::NORMAL) mass += std::exp(static_cast<double>(p.score()));
  }
  EXPECT_NEAR(1.0, mass, 1e-5);
  EXPECT_TRUE(BitEqual(-3.5f, ScoreOf(out, "<X>")));

  // 4. monotone after the first feasible M-step.
  for (size_t i = 1; i < stats.per_iteration.size(); ++i) {
    EXPECT_LE(stats.per_iteration[i].objective,
              stats.per_iteration[i - 1].objective + 1e-6);
  }

  // 5. HOW BIG WAS THE ERROR THIS FIXED? Measured, not asserted from theory.
  //    The library's own PopulateMarginal keeps the UNK alternatives, so the
  //    difference between its log Z and the single known path is exactly the
  //    contamination the fix removes.
  //
  //    It is provably tiny, and that is worth stating plainly rather than
  //    overselling the fix: an UNK node scores min_score() - kUnkPenalty, and
  //    min_score() is the MINIMUM over NORMAL scores, so for a k-character
  //    span the UNK path costs at most k*(min - 10) while any covering known
  //    piece costs at least min. The gap is therefore at least
  //    10k + (k-1)*|min| >= 20 nats for k >= 2, i.e. a relative posterior
  //    below e^-20. So the pre-fix estimator was wrong in principle -- the
  //    M-step ignored the min_score dependence and so was not the exact
  //    maximizer -- but numerically negligible. The fix buys exactness, not
  //    a different answer.
  unigram::Model model(in);
  ASSERT_TRUE(model.status().ok());
  unigram::Lattice lattice;
  lattice.SetSentence("a<X>b");
  model.PopulateNodes(&lattice);
  std::vector<float> expected(in.pieces_size(), 0.0f);
  const double z_with_unk = lattice.PopulateMarginal(1.0f, &expected);
  const double z_known_only = -stats.initial_objective;
  const double contamination = z_with_unk - z_known_only;
  // MEASURED, and smaller than the arithmetic that carries it: the library
  // computes alpha/beta in float32 while the known-only path here is double,
  // so the float32 rounding of log Z (~7.5e-08 here) swamps the true
  // contamination and can even make the difference negative. That is the
  // honest finding -- the defect is real in principle and invisible in
  // practice.
  EXPECT_LT(std::abs(contamination), 1e-5)
      << "UNK contamination of log Z was " << contamination;
  // The UNK node did receive posterior under the library path -- that is the
  // defect -- even though it is ~1e-15.
  const int unk_id = 0;
  EXPECT_GT(expected[unk_id], 0.0f)
      << "the library path is expected to put SOME mass on UNK; if it does "
         "not, this test no longer demonstrates anything";
  EXPECT_LT(expected[unk_id], 1e-12f)
      << "UNK mass this large would change the estimate materially, which "
         "the min_score() - kUnkPenalty bound says cannot happen";
  LOG(INFO) << "UNK contamination of log Z = " << contamination
            << ", library UNK posterior = " << expected[unk_id];
}

TEST(UnigramRefitTest, MinScoreOfAnUnusedPieceCannotMoveTheResult) {
  // min_score() is the minimum NORMAL score, and it sets the UNK node score.
  // Two input models that differ ONLY in the score of a piece the corpus never
  // uses therefore have very different UNK-path scores. With UNK excluded,
  // that cannot reach the answer: the refit fixed point must be identical.
  const std::string corpus = TempPath("refit_minscore.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"a<X>b\t7", "ab\t3"}));

  auto run = [&](float unused_score) {
    const ModelProto in = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                     {"<X>", -3.5f, SP::USER_DEFINED},
                                     {"a", -0.9f, SP::NORMAL},
                                     {"b", -1.7f, SP::NORMAL},
                                     {"ab", -2.2f, SP::NORMAL},
                                     {"qq", unused_score, SP::NORMAL}});
    ModelProto out;
    RefitStats stats;
    RefitOptions options;
    options.num_iterations = 30;
    EXPECT_TRUE(Refit(in, corpus, "tsv", &out, &stats, options).ok());
    return out;
  };

  // -4 vs -400 moves min_score(), hence the UNK score, by 396 nats.
  const ModelProto shallow = run(-4.0f);
  const ModelProto deep = run(-400.0f);
  for (const char* piece : {"a", "b", "ab"}) {
    EXPECT_NEAR(ScoreOf(shallow, piece), ScoreOf(deep, piece), 1e-4)
        << "piece " << piece << " moved with min_score, so UNK alternatives "
        << "are still influencing the refit";
  }
}

TEST(UnigramRefitTest, NoCompleteKnownPathFailsEvenWhenEveryCharacterOccurs) {
  // Every character of "aab" appears inside a piece, so a per-character
  // coverage rule would wrongly pass this. There is still no complete
  // non-UNKNOWN segmentation: only "ab" exists.
  const ModelProto in = MakeModel({{"<unk>", 0.0f, SP::UNKNOWN},
                                   {"ab", -1.0f, SP::NORMAL}});
  const std::string corpus = TempPath("refit_nopath.tsv");
  ASSERT_TRUE(WriteLines(corpus, {"ab\t5", "aab\t2"}));

  ModelProto out;
  RefitStats stats;
  const absl::Status status = Refit(in, corpus, "tsv", &out, &stats);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;
  EXPECT_EQ(0, out.pieces_size());
}

TEST(UnigramRefitTest, ByteFallbackDoesNotWaiveTheCoverageRequirement) {
  // Byte fallback is applied AFTER the Unigram model emits an UNKNOWN span;
  // BYTE pieces are not forward-backward nodes in PopulateNodes(). So a corpus
  // that needs byte fallback for coverage cannot be refit under this
  // objective, and byte_fallback=true must not silently waive the check.
  std::vector<Entry> entries = {{"<unk>", 0.0f, SP::UNKNOWN},
                                {"a", -1.0f, SP::NORMAL},
                                {"b", -1.0f, SP::NORMAL}};
  for (int i = 0; i < 256; ++i) {
    entries.push_back({absl::StrFormat("<0x%02X>", i), -10.0f, SP::BYTE});
  }
  ModelProto in = MakeModel(entries);
  in.mutable_trainer_spec()->set_byte_fallback(true);

  const std::string ok_corpus = TempPath("refit_bytes_ok.tsv");
  const std::string bad_corpus = TempPath("refit_bytes_bad.tsv");
  ASSERT_TRUE(WriteLines(ok_corpus, {"ab\t4", "ba\t2"}));
  ASSERT_TRUE(WriteLines(bad_corpus, {"ab\t4", "aZb\t2"}));

  // A complete known path exists -> refit normally, byte fallback or not.
  ModelProto good;
  RefitStats gstats;
  EXPECT_TRUE(Refit(in, ok_corpus, "tsv", &good, &gstats).ok());
  ASSERT_EQ(in.pieces_size(), good.pieces_size());
  for (int i = 0; i < in.pieces_size(); ++i) {
    EXPECT_EQ(in.pieces(i).type(), good.pieces(i).type());
    if (in.pieces(i).type() != SP::NORMAL) {
      EXPECT_TRUE(BitEqual(in.pieces(i).score(), good.pieces(i).score()))
          << "BYTE/UNKNOWN score moved at id " << i;
    }
  }

  // Coverage would require the byte path -> refuse rather than optimize the
  // wrong objective.
  ModelProto bad;
  RefitStats bstats;
  const absl::Status status = Refit(in, bad_corpus, "tsv", &bad, &bstats);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(absl::StatusCode::kFailedPrecondition, status.code()) << status;
  EXPECT_NE(std::string::npos,
            std::string(status.message()).find("byte_fallback"));
  EXPECT_EQ(0, bad.pieces_size());
}

// ------------------------------------------------- guard, called in-process

TEST(UnigramRefitTest, ContractCheckerRejectsAMutatedSupport) {
  const ModelProto in = MakeAbModel();
  ModelProto out = in;
  EXPECT_TRUE(unigram_refit::VerifyFixedSupportContract(in, out).ok());

  ModelProto added = in;
  added.add_pieces()->set_piece("zz");
  EXPECT_FALSE(unigram_refit::VerifyFixedSupportContract(in, added).ok());

  ModelProto retyped = in;
  retyped.mutable_pieces(1)->set_type(SP::USER_DEFINED);
  EXPECT_FALSE(unigram_refit::VerifyFixedSupportContract(in, retyped).ok());

  ModelProto meta_moved = in;
  meta_moved.mutable_pieces(0)->set_score(-1.0f);
  EXPECT_FALSE(unigram_refit::VerifyFixedSupportContract(in, meta_moved).ok());

  ModelProto renamed = in;
  renamed.mutable_pieces(2)->set_piece("bb");
  EXPECT_FALSE(unigram_refit::VerifyFixedSupportContract(in, renamed).ok());
}

}  // namespace
}  // namespace sentencepiece
