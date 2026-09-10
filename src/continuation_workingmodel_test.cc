// Persistent working-model equivalence (defect B).
//
// A score-only update must be indistinguishable from rebuilding the model from
// the same scores. The trap is Model::min_score_, which is cached at
// construction and feeds the UNKNOWN fallback (min_score() - kUnkPenalty) in
// PopulateNodes. Mutating protobuf scores without refreshing it leaves the
// model scoring unknown spans against a stale minimum, which no ordinary
// segmentation test would notice.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "sentencepiece_model.pb.h"
#include "unigram_model.h"

namespace sentencepiece {
namespace unigram {
namespace {

void AddPiece(ModelProto* m, const std::string& p, float s,
              ModelProto::SentencePiece::Type t) {
  auto* sp = m->add_pieces();
  sp->set_piece(p);
  sp->set_score(s);
  sp->set_type(t);
}

ModelProto MakeProto(float a_score, float b_score, float ab_score) {
  ModelProto m;
  m.mutable_trainer_spec()->set_model_type(TrainerSpec::UNIGRAM);
  m.mutable_normalizer_spec()->set_name("identity");
  m.mutable_normalizer_spec()->set_add_dummy_prefix(false);
  AddPiece(&m, "<unk>", 0.0, ModelProto::SentencePiece::UNKNOWN);
  AddPiece(&m, "<s>", 0.0, ModelProto::SentencePiece::CONTROL);
  AddPiece(&m, "</s>", 0.0, ModelProto::SentencePiece::CONTROL);
  AddPiece(&m, "a", a_score, ModelProto::SentencePiece::NORMAL);
  AddPiece(&m, "b", b_score, ModelProto::SentencePiece::NORMAL);
  AddPiece(&m, "ab", ab_score, ModelProto::SentencePiece::NORMAL);
  m.mutable_trainer_spec()->set_vocab_size(m.pieces_size());
  return m;
}

TEST(ContinuationWorkingModelTest, ScoreRefreshMatchesReconstruction) {
  // Persistent model, built once at the initial scores.
  ModelProto live = MakeProto(-1.0, -2.0, -0.5);
  Model persistent(live);
  EXPECT_TRUE(persistent.status().ok());

  // Score-only mutation, deliberately moving the minimum NORMAL score a lot.
  live.mutable_pieces(3)->set_score(-9.0);    // "a"
  live.mutable_pieces(4)->set_score(-30.0);   // "b"  <- new minimum
  live.mutable_pieces(5)->set_score(-0.25);   // "ab"
  EXPECT_TRUE(persistent.RefreshScoreCache().ok());

  // Reference: a fresh model over exactly those scores.
  ModelProto same = MakeProto(-9.0, -30.0, -0.25);
  Model fresh(same);
  EXPECT_TRUE(fresh.status().ok());

  EXPECT_NEAR(persistent.min_score(), fresh.min_score(), 1e-6);

  for (const std::string& s : {"ab", "aab", "abab", "ba"}) {
    Lattice lp, lf;
    lp.SetSentence(s);
    lf.SetSentence(s);
    persistent.PopulateNodes(&lp);
    fresh.PopulateNodes(&lf);
    const auto vp = lp.Viterbi();
    const auto vf = lf.Viterbi();
    EXPECT_EQ(vp.first.size(), vf.first.size());
    for (size_t i = 0; i < vp.first.size(); ++i) {
      EXPECT_EQ(std::string(vp.first[i]->piece),
                std::string(vf.first[i]->piece));
    }
    EXPECT_NEAR(vp.second, vf.second, 1e-4);

    std::vector<float> ep(live.pieces_size(), 0.0f);
    std::vector<float> ef(same.pieces_size(), 0.0f);
    const float zp = lp.PopulateMarginal(1.0, &ep);
    const float zf = lf.PopulateMarginal(1.0, &ef);
    EXPECT_NEAR(zp, zf, 1e-4);
    for (int i = 0; i < live.pieces_size(); ++i) {
      EXPECT_NEAR(ep[i], ef[i], 1e-4);
    }
  }
}

TEST(ContinuationWorkingModelTest, StaleMinScoreIsDetectedOnUnknownFallback) {
  // The UNKNOWN path is scored from min_score(); 'z' is outside the support.
  ModelProto live = MakeProto(-1.0, -2.0, -0.5);
  Model stale(live);
  Model refreshed(live);
  EXPECT_TRUE(stale.status().ok());

  live.mutable_pieces(4)->set_score(-40.0);   // minimum moves far down
  EXPECT_TRUE(refreshed.RefreshScoreCache().ok());
  // `stale` deliberately NOT refreshed.

  ModelProto same = MakeProto(-1.0, -40.0, -0.5);
  Model fresh(same);
  EXPECT_TRUE(fresh.status().ok());

  EXPECT_NEAR(refreshed.min_score(), fresh.min_score(), 1e-6);
  // The regression this guards: without RefreshScoreCache the cached minimum
  // is the OLD one and unknown spans are mispriced.
  EXPECT_NE(stale.min_score(), fresh.min_score());

  Lattice lr, lf;
  lr.SetSentence("azb");
  lf.SetSentence("azb");
  refreshed.PopulateNodes(&lr);
  fresh.PopulateNodes(&lf);
  EXPECT_NEAR(lr.Viterbi().second, lf.Viterbi().second, 1e-4);
}


// SYNTHETIC ADMISSIBILITY (covered basis).
//
// Continuation admits an ordinary candidate only if its best decomposition
// through the COVERED BASIS -- inherited NORMAL pieces plus required atomic
// bootstrap characters -- uses basis symbols only. This is the predicate that
// once rejected every V-bearing candidate (because V was missing from the
// prior) and, after the bootstrap fix, must still reject a candidate needing a
// symbol that is in neither the prior nor the required set.
//
// The production corpus cannot exercise the reject branch: bootstrap adds every
// missing corpus character, so every corpus substring is representable and
// rejected==0 is the expected invariant. Hence this synthetic test.
TEST(ContinuationWorkingModelTest, CoveredBasisAdmissibility) {
  ModelProto basis;
  basis.mutable_trainer_spec()->set_model_type(TrainerSpec::UNIGRAM);
  basis.mutable_normalizer_spec()->set_name("identity");
  basis.mutable_normalizer_spec()->set_add_dummy_prefix(false);
  AddPiece(&basis, "<unk>", 0.0, ModelProto::SentencePiece::UNKNOWN);
  AddPiece(&basis, "<s>", 0.0, ModelProto::SentencePiece::CONTROL);
  AddPiece(&basis, "</s>", 0.0, ModelProto::SentencePiece::CONTROL);
  // Inherited NORMAL atoms.
  AddPiece(&basis, "\xe2\x96\x81", -1.0, ModelProto::SentencePiece::NORMAL);
  AddPiece(&basis, "a", -1.0, ModelProto::SentencePiece::NORMAL);
  AddPiece(&basis, ":", -1.0, ModelProto::SentencePiece::NORMAL);
  // REQUIRED bootstrap atom: present in the basis, absent from the prior.
  AddPiece(&basis, "V", -2.0, ModelProto::SentencePiece::NORMAL);
  basis.mutable_trainer_spec()->set_vocab_size(basis.pieces_size());

  Model model(basis);
  EXPECT_TRUE(model.status().ok());
  const int unk_id = 0;

  auto spellable = [&](const std::string& s) {
    Lattice lat;
    lat.SetSentence(s);
    model.PopulateNodes(&lat);
    for (const auto* node : lat.Viterbi().first) {
      if (node->id == unk_id) return false;
    }
    return true;
  };

  // Uses required V -> admissible.
  EXPECT_TRUE(spellable("\xe2\x96\x81V:a"));
  // Z is in neither the inherited basis nor the required set -> rejected.
  EXPECT_FALSE(spellable("\xe2\x96\x81Z:a"));
}

}  // namespace
}  // namespace unigram
}  // namespace sentencepiece
