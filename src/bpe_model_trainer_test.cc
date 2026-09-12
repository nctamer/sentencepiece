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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "continuation_io.h"
#include "filesystem.h"
#include "expansion_processor.h"
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


struct RefGate {
  int level = 0;
  std::vector<int> cuts;
};
struct RefToken {
  std::string piece;
  int begin = 0;
  int end = 0;
};
struct RefRow {
  std::string text;
  int64_t weight = 1;
  std::vector<RefGate> gates;
};
struct RefRule {
  std::string left;
  std::string right;
  uint64_t support = 0;
  int external_id = -1;
  std::string post_state_sha256;
};
struct RefRun {
  std::vector<RefRule> rules;
  std::vector<std::vector<RefToken>> final_rows;
  std::string final_sha256;
};

using RefKey = std::pair<std::string, std::string>;

bool RefSupportsMerge(const RefRow& row, const RefToken& left,
                      const RefToken& right) {
  if (left.end != right.begin) return false;
  const int begin = left.begin;
  const int end = right.end;
  for (const RefGate& gate : row.gates) {
    const int gate_begin = gate.cuts.front();
    const int gate_end = gate.cuts.back();
    if (end <= gate_begin || begin >= gate_end) continue;
    if (begin <= gate_begin && end >= gate_end) continue;

    bool crosses_internal = false;
    for (size_t i = 1; i + 1 < gate.cuts.size(); ++i) {
      if (begin < gate.cuts[i] && gate.cuts[i] < end) {
        crosses_internal = true;
        break;
      }
    }
    if (!crosses_internal) continue;

    if (begin < gate_begin || end > gate_end ||
        std::find(gate.cuts.begin(), gate.cuts.end(), begin) ==
            gate.cuts.end() ||
        std::find(gate.cuts.begin(), gate.cuts.end(), end) ==
            gate.cuts.end()) {
      return false;
    }
  }
  return true;
}

std::vector<RefToken> RefAtoms(const RefRow& row) {
  std::vector<RefToken> out;
  for (int i = 0; i < static_cast<int>(row.text.size()); ++i) {
    out.push_back({row.text.substr(i, 1), i, i + 1});
  }
  return out;
}

// Independent runtime oracle: restart from atoms and repeatedly apply the
// globally lowest matching rank, leftmost on ties.
std::vector<RefToken> RefReplayRow(const RefRow& row,
                                   const std::vector<RefRule>& rules) {
  std::vector<RefToken> tokens = RefAtoms(row);
  while (true) {
    int best_rank = -1;
    int best_left = -1;
    for (int i = 0; i + 1 < static_cast<int>(tokens.size()); ++i) {
      for (int rank = 0; rank < static_cast<int>(rules.size()); ++rank) {
        const RefRule& rule = rules[rank];
        if (rule.left == tokens[i].piece &&
            rule.right == tokens[i + 1].piece) {
          if (best_rank == -1 || rank < best_rank ||
              (rank == best_rank && i < best_left)) {
            best_rank = rank;
            best_left = i;
          }
          break;
        }
      }
    }
    if (best_rank == -1) break;
    tokens[best_left] = {
        tokens[best_left].piece + tokens[best_left + 1].piece,
        tokens[best_left].begin, tokens[best_left + 1].end};
    tokens.erase(tokens.begin() + best_left + 1);
  }
  return tokens;
}

std::string RefFinalSha256(const std::vector<RefRow>& rows,
                           const std::vector<std::vector<RefToken>>& tokens) {
  std::vector<size_t> order(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return rows[a].text < rows[b].text;
  });
  std::string canonical;
  for (size_t sid : order) {
    absl::StrAppend(&canonical, rows[sid].text.size(), ":", rows[sid].text,
                    "\t", rows[sid].weight, "\t", tokens[sid].size());
    for (const RefToken& token : tokens[sid]) {
      absl::StrAppend(&canonical, "\t", token.piece.size(), ":", token.piece);
    }
    canonical.push_back('\n');
  }
  return continuation::Sha256Hex(canonical);
}

int RefFirstNewId(const std::vector<RefRow>& rows) {
  std::set<char> atoms;
  for (const RefRow& row : rows) {
    for (char c : row.text) atoms.insert(c);
  }
  return 1 + static_cast<int>(atoms.size());  // id 0 is <unk>
}

std::string RefHierarchySpec(const RefRow& row) {
  std::vector<std::string> specs;
  for (const RefGate& gate : row.gates) {
    std::vector<std::string> cuts;
    for (int cut : gate.cuts) cuts.push_back(std::to_string(cut));
    specs.push_back(
        absl::StrCat(gate.level, ":", absl::StrJoin(cuts, ",")));
  }
  return absl::StrJoin(specs, ";");
}

RefRun RunFlatHierarchyReference(const std::vector<RefRow>& rows,
                                 int first_new_external_id,
                                 int requested_new_pieces) {
  std::set<std::string> pieces;
  for (const RefRow& row : rows) {
    for (char c : row.text) pieces.insert(std::string(1, c));
  }

  std::vector<RefRule> rules;
  while (static_cast<int>(rules.size()) < requested_new_pieces) {
    std::vector<std::vector<RefToken>> current;
    current.reserve(rows.size());
    for (const RefRow& row : rows) current.push_back(RefReplayRow(row, rules));

    std::set<RefKey> learned;
    for (const RefRule& rule : rules) learned.insert({rule.left, rule.right});
    std::map<RefKey, uint64_t> counts;

    for (size_t sid = 0; sid < rows.size(); ++sid) {
      const auto& toks = current[sid];
      std::map<RefKey, int> last_actual_right;
      for (int i = 0; i + 1 < static_cast<int>(toks.size()); ++i) {
        const RefKey key{toks[i].piece, toks[i + 1].piece};
        if (learned.count(key)) continue;
        const std::string child = key.first + key.second;
        if (pieces.count(child)) continue;  // fresh-child invariant

        if (key.first == key.second) {
          const auto prev = last_actual_right.find(key);
          if (prev != last_actual_right.end() && prev->second == i) continue;
          // Update regardless of hierarchy support: this is an actual flat
          // replacement and consumes an overlapping occurrence to its right.
          last_actual_right[key] = i + 1;
        }
        if (RefSupportsMerge(rows[sid], toks[i], toks[i + 1])) {
          counts[key] += static_cast<uint64_t>(rows[sid].weight);
        }
      }
    }
    if (counts.empty()) break;

    auto better = [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      if (a.first.first != b.first.first) return a.first.first < b.first.first;
      return a.first.second < b.first.second;
    };
    auto best = counts.begin();
    for (auto it = std::next(counts.begin()); it != counts.end(); ++it) {
      if (better(*it, *best)) best = it;
    }

    RefRule rule;
    rule.left = best->first.first;
    rule.right = best->first.second;
    rule.support = best->second;
    rule.external_id =
        first_new_external_id + static_cast<int>(rules.size());
    pieces.insert(rule.left + rule.right);
    rules.push_back(rule);

    std::vector<std::vector<RefToken>> post;
    for (const RefRow& row : rows) post.push_back(RefReplayRow(row, rules));
    rules.back().post_state_sha256 = RefFinalSha256(rows, post);
  }

  RefRun out;
  out.rules = rules;
  for (const RefRow& row : rows) out.final_rows.push_back(RefReplayRow(row, rules));
  out.final_sha256 = RefFinalSha256(rows, out.final_rows);
  return out;
}

int BuildRandomLaminarGates(int begin, int end, int depth,
                            std::mt19937* rng,
                            std::vector<RefGate>* gates) {
  if (end - begin < 2) return 0;
  std::uniform_int_distribution<int> split_dist(begin + 1, end - 1);
  const int split = split_dist(*rng);
  const int lh = BuildRandomLaminarGates(begin, split, depth + 1, rng, gates);
  const int rh = BuildRandomLaminarGates(split, end, depth + 1, rng, gates);
  const int level = std::max(lh, rh) + 1;
  std::bernoulli_distribution keep(depth == 0 ? 0.9 : 0.6);
  if (keep(*rng)) gates->push_back({level, {begin, split, end}});
  return level;
}

struct HierarchyTrainerRun {
  ExpansionResult result;
  std::vector<std::string> trace;
};

HierarchyTrainerRun TrainHierarchyFixture(const std::string& name,
                                          const std::vector<RefRow>& rows,
                                          int requested_new_pieces) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), name + "_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), name + ".spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), name + "_hier.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), name + "_model");
  const std::string result_path = prefix + ".expansion";
  const std::string trace_path = prefix + ".trace";

  {
    auto out = filesystem::NewWritableFile(input);
    for (const RefRow& row : rows) {
      EXPECT_TRUE(out->WriteLine(absl::StrCat(row.text, "\t", row.weight)));
    }
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    EXPECT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    for (const RefRow& row : rows) {
      EXPECT_TRUE(out->WriteLine(
          absl::StrCat(row.text, "\t", RefHierarchySpec(row))));
    }
  }

  std::set<char> atoms;
  for (const RefRow& row : rows) {
    for (char ch : row.text) atoms.insert(ch);
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(1 + static_cast<int>(atoms.size()));
  expansion.set_requested_new_pieces(requested_new_pieces);

  auto* unk = expansion.add_base_pieces();
  unk->set_external_id(0);
  unk->set_piece("<unk>");
  unk->set_type(ModelProto::SentencePiece::UNKNOWN);
  unk->set_mergeable(false);
  unk->set_atomic(false);
  int id = 1;
  for (char ch : atoms) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id++);
    p->set_piece(std::string(1, ch));
    p->set_type(ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(true);
    p->set_atomic(true);
  }
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    EXPECT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE);
  ts.add_input(input);
  ts.set_input_format("tsv");
  ts.set_model_prefix(prefix);
  ts.set_vocab_size(id + requested_new_pieces);
  ts.set_expansion_spec(spec_path);
  ts.set_expansion_result(result_path);
  ts.set_bpe_hierarchy_file(hierarchy);
  ts.set_bpe_reference_trace_file(trace_path);
  ts.set_input_sentence_size(0);
  ts.set_split_by_whitespace(false);
  ts.set_split_by_unicode_script(false);
  ts.set_split_by_number(false);
  ts.set_split_digits(false);
  ts.set_max_sentencepiece_length(64);
  ts.set_bos_id(-1);
  ts.set_eos_id(-1);
  ts.set_pad_id(-1);
  ts.set_hard_vocab_limit(true);

  NormalizerSpec ns;
  ns.set_name("identity");
  ns.set_add_dummy_prefix(false);
  ns.set_remove_extra_whitespaces(false);
  NormalizerSpec dns;
  EXPECT_TRUE(SentencePieceTrainer::Train(ts, ns, dns).ok());

  HierarchyTrainerRun out;
  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    EXPECT_TRUE(in->ReadAll(&bytes));
  }
  EXPECT_TRUE(out.result.ParseFromString(bytes));
  {
    auto in = filesystem::NewReadableFile(trace_path);
    std::string line;
    while (in->ReadLine(&line)) out.trace.push_back(line);
    EXPECT_TRUE(in->status().ok());
  }
  return out;
}

std::string RunTrainer(std::string RunTrainer(
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

TEST(BPETrainerTest, HierarchySupportSelectsFlatPairAndAppliesGlobally) {
  const std::vector<RefRow> rows = {
      {"ab", 10, {{1, {0, 1, 2}}}},
      // a+b is NOT hierarchy-supported here: the parent is a | bc.
      {"abc", 1, {{1, {0, 1, 3}}}},
  };
  const auto run = TrainHierarchyFixture("hier_flat_global", rows, 1);
  ASSERT_EQ(1, run.result.learned_merges_size());
  const auto& merge = run.result.learned_merges(0);
  EXPECT_EQ("a", merge.left());
  EXPECT_EQ("b", merge.right());
  EXPECT_EQ(10u, merge.weighted_count())
      << "weighted_count is hierarchy support, not global application weight";
  EXPECT_FALSE(merge.has_scope_level());
  EXPECT_TRUE(absl::StartsWith(
      run.result.boundary_policy(), "bpe_hierarchy_guided_training_v1:"));
  EXPECT_EQ(0, run.result.unreachable_pieces());

  expansion::ExpansionProcessor runtime;
  ASSERT_TRUE(runtime.Load(run.result).ok());
  EXPECT_FALSE(runtime.RequiresHierarchy());
  std::vector<expansion::TokenSpan> out;
  ASSERT_TRUE(runtime.Encode("abc", &out).ok());
  ASSERT_EQ(2u, out.size());
  EXPECT_EQ("ab", out[0].piece);
  EXPECT_EQ("c", out[1].piece);
}

TEST(BPETrainerTest, HierarchyCanRepairAFlatPartialSpan) {
  const std::vector<RefRow> rows = {
      {"ab", 10, {{1, {0, 1, 2}}}},
      {"abc", 1, {{1, {0, 1, 3}}}},
  };
  const auto run = TrainHierarchyFixture("hier_flat_repair", rows, 2);
  ASSERT_EQ(2, run.result.learned_merges_size());
  EXPECT_EQ("a", run.result.learned_merges(0).left());
  EXPECT_EQ("b", run.result.learned_merges(0).right());
  // Rank 0 also fires in abc, creating ab|c. The next merge is supported
  // because its RESULT completes the whole a|bc parent.
  EXPECT_EQ("ab", run.result.learned_merges(1).left());
  EXPECT_EQ("c", run.result.learned_merges(1).right());

  expansion::ExpansionProcessor runtime;
  ASSERT_TRUE(runtime.Load(run.result).ok());
  std::vector<expansion::TokenSpan> out;
  ASSERT_TRUE(runtime.Encode("abc", &out).ok());
  ASSERT_EQ(1u, out.size());
  EXPECT_EQ("abc", out[0].piece);
}

TEST(BPETrainerTest, OldSixToThreeAliasFailureIsUnrepresentable) {
  // Weight 3 and two disjoint a+b occurrences: support is exactly 6. The old
  // scoped trainer could later relearn a+b as an alias, create an existing
  // "ab", wake an older ab+a rank, and realize only 3. Flat training learns
  // a+b once, creates a fresh child, and globally obtains ab|ab.
  const std::vector<RefRow> rows = {
      {"abab", 3, {{1, {0, 1, 2}}, {1, {2, 3, 4}}}},
  };
  const auto run = TrainHierarchyFixture("hier_no_alias_6_to_3", rows, 1);
  ASSERT_EQ(1, run.result.learned_merges_size());
  EXPECT_EQ("a", run.result.learned_merges(0).left());
  EXPECT_EQ("b", run.result.learned_merges(0).right());
  EXPECT_EQ(6u, run.result.learned_merges(0).weighted_count());
  EXPECT_FALSE(run.result.learned_merges(0).has_scope_level());

  expansion::ExpansionProcessor runtime;
  ASSERT_TRUE(runtime.Load(run.result).ok());
  std::vector<expansion::TokenSpan> out;
  ASSERT_TRUE(runtime.Encode("abab", &out).ok());
  ASSERT_EQ(2u, out.size());
  EXPECT_EQ("ab", out[0].piece);
  EXPECT_EQ("ab", out[1].piece);
}

TEST(BPETrainerTest, RandomLaminarHierarchyMatchesFlatReferenceOracle) {
  std::mt19937 rng(0x51A7BEEF);
  std::uniform_int_distribution<int> weight_dist(1, 5);
  std::vector<RefRow> rows;
  rows.reserve(256);

  // Unique length-6 rows over abcd, so PreparedCorpus aggregation cannot hide
  // an oracle discrepancy.
  for (int n = 0; n < 256; ++n) {
    int x = n;
    std::string text(6, 'a');
    for (int i = 5; i >= 0; --i) {
      text[i] = static_cast<char>('a' + (x & 3));
      x >>= 2;
    }
    RefRow row;
    row.text = text;
    row.weight = weight_dist(rng);
    BuildRandomLaminarGates(0, 6, 0, &rng, &row.gates);
    rows.push_back(std::move(row));
  }

  constexpr int kPieces = 16;
  const int first_new_id = RefFirstNewId(rows);
  const RefRun oracle =
      RunFlatHierarchyReference(rows, first_new_id, kPieces);
  ASSERT_EQ(kPieces, static_cast<int>(oracle.rules.size()));

  const auto run = TrainHierarchyFixture("hier_flat_random", rows, kPieces);
  ASSERT_EQ(kPieces, run.result.learned_merges_size());
  ASSERT_EQ(kPieces, static_cast<int>(run.trace.size()));

  for (int i = 0; i < kPieces; ++i) {
    const auto& got = run.result.learned_merges(i);
    const auto& want = oracle.rules[i];
    EXPECT_EQ(want.left, got.left()) << "rank " << i;
    EXPECT_EQ(want.right, got.right()) << "rank " << i;
    EXPECT_EQ(want.support, got.weighted_count()) << "rank " << i;
    EXPECT_EQ(want.external_id, got.external_id()) << "rank " << i;
    EXPECT_FALSE(got.has_scope_level()) << "rank " << i;

    const std::vector<std::string> fields =
        absl::StrSplit(run.trace[i], '\t');
    ASSERT_EQ(8u, fields.size()) << "rank " << i;
    EXPECT_EQ(want.post_state_sha256, fields[7]) << "rank " << i;
  }
  EXPECT_EQ(oracle.final_sha256,
            run.result.training_final_segmentation_sha256());
  EXPECT_EQ(0, run.result.unreachable_pieces());
}

TEST(BPETrainerTest, HierarchyNeverVetoesInheritedMergeReplay)TEST(BPETrainerTest, HierarchyNeverVetoesInheritedMergeReplay) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "hier_inherited_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "hier_inherited.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "hier_inherited.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "hier_inherited_model");
  const std::string result_path = prefix + ".expansion";
  { auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abc\t5")); }
  { auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    ASSERT_TRUE(out->WriteLine("abc\t1:0,1,3")); }  // a | bc

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(5);
  expansion.set_requested_new_pieces(1);
  auto add = [&](int id, absl::string_view piece, bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id); p->set_piece(std::string(piece));
    p->set_type(id == 0 ? ModelProto::SentencePiece::UNKNOWN
                        : ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(id != 0); p->set_atomic(atomic);
  };
  add(0, "<unk>", false); add(1, "a", true); add(2, "b", true);
  add(3, "c", true); add(4, "ab", false);
  auto* inherited = expansion.add_base_merges();
  inherited->set_rank(0); inherited->set_left("a"); inherited->set_right("b");
  inherited->set_external_id(4);
  { auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString())); }

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE); ts.add_input(input);
  ts.set_input_format("tsv"); ts.set_model_prefix(prefix); ts.set_vocab_size(6);
  ts.set_expansion_spec(spec_path); ts.set_expansion_result(result_path);
  ts.set_bpe_hierarchy_file(hierarchy); ts.set_input_sentence_size(0);
  ts.set_split_by_whitespace(false); ts.set_split_by_unicode_script(false);
  ts.set_split_by_number(false); ts.set_split_digits(false);
  ts.set_bos_id(-1); ts.set_eos_id(-1); ts.set_pad_id(-1);
  ts.set_hard_vocab_limit(true);
  NormalizerSpec ns; ns.set_name("identity"); ns.set_add_dummy_prefix(false);
  ns.set_remove_extra_whitespaces(false);
  NormalizerSpec dns;
  ASSERT_TRUE(SentencePieceTrainer::Train(ts, ns, dns).ok());

  std::string bytes;
  { auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes)); }
  ExpansionResult result; ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(1, result.base_merges_size());
  ASSERT_EQ(1, result.learned_merges_size());
  EXPECT_EQ("a", result.base_merges(0).left());
  EXPECT_EQ("b", result.base_merges(0).right());
  EXPECT_EQ("ab", result.learned_merges(0).left());
  EXPECT_EQ("c", result.learned_merges(0).right())
      << "base a+b must replay even though the later InterMo gate says a|bc";
}




}  // namespace
}  // namespace bpe
}  // namespace sentencepiece
