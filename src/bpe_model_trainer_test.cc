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
  int scope = 0;
  uint64_t count = 0;
  int external_id = -1;
  bool allocated = false;
  std::string post_state_sha256;
};
struct RefRun {
  std::vector<RefRule> rules;
  std::vector<std::vector<RefToken>> final_rows;
  std::string final_sha256;
};

using RefKey = std::tuple<std::string, std::string, int>;

int RefScope(const RefRow& row, const RefToken& left, const RefToken& right) {
  if (left.end != right.begin) return -1;
  for (const RefGate& gate : row.gates) {
    bool boundary = false;
    for (size_t i = 1; i + 1 < gate.cuts.size(); ++i) {
      if (gate.cuts[i] == left.end) {
        boundary = true;
        break;
      }
    }
    if (!boundary) continue;
    const bool left_begin =
        std::find(gate.cuts.begin(), gate.cuts.end(), left.begin) !=
        gate.cuts.end();
    const bool right_end =
        std::find(gate.cuts.begin(), gate.cuts.end(), right.end) !=
        gate.cuts.end();
    if (!left_begin || !right_end || left.begin < gate.cuts.front() ||
        right.end > gate.cuts.back()) {
      return -1;
    }
    return gate.level;
  }
  return 0;
}

std::vector<RefToken> RefAtoms(const RefRow& row) {
  std::vector<RefToken> out;
  for (int i = 0; i < static_cast<int>(row.text.size()); ++i) {
    out.push_back({row.text.substr(i, 1), i, i + 1});
  }
  return out;
}

std::vector<RefToken> RefReplayRow(const RefRow& row,
                                   const std::vector<RefRule>& rules) {
  std::vector<RefToken> tokens = RefAtoms(row);
  while (true) {
    int best_rank = -1;
    int best_left = -1;
    for (int i = 0; i + 1 < static_cast<int>(tokens.size()); ++i) {
      const int scope = RefScope(row, tokens[i], tokens[i + 1]);
      if (scope < 0) continue;
      for (int rank = 0; rank < static_cast<int>(rules.size()); ++rank) {
        const RefRule& rule = rules[rank];
        if (rule.scope == scope && rule.left == tokens[i].piece &&
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
    RefToken merged;
    merged.piece = tokens[best_left].piece + tokens[best_left + 1].piece;
    merged.begin = tokens[best_left].begin;
    merged.end = tokens[best_left + 1].end;
    tokens[best_left] = std::move(merged);
    tokens.erase(tokens.begin() + best_left + 1);
  }
  return tokens;
}

std::string RefFinalSha256(const std::vector<RefRow>& rows,
                           const std::vector<std::vector<RefToken>>& tokens) {
  std::vector<size_t> order(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return rows[a].text < rows[b].text;  // generated oracle rows are ASCII.
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

RefRun RunScopedReference(const std::vector<RefRow>& rows,
                          int first_new_external_id,
                          int requested_new_pieces) {
  // Intentionally dumb correctness oracle:
  //   * restart every row from atoms after every learned operation;
  //   * replay the ENTIRE scoped program by repeated full scans;
  //   * rescan every adjacent occurrence to count the next candidates.
  // It shares no live links, occurrence indexes, candidate heap or incremental
  // state with ContinuationTrainer.
  std::map<std::string, int> piece_id;
  std::map<std::string, std::pair<std::string, std::string>> ancestry;
  int next_id = first_new_external_id;
  for (const RefRow& row : rows) {
    for (char c : row.text) {
      const std::string atom(1, c);
      if (!piece_id.count(atom)) {
        // Test specs assign atom IDs separately; only existence matters until
        // a learned child is allocated below.
        piece_id[atom] = -1;
      }
    }
  }

  std::vector<RefRule> rules;
  int allocated = 0;
  while (allocated < requested_new_pieces) {
    std::vector<std::vector<RefToken>> current;
    current.reserve(rows.size());
    for (const RefRow& row : rows) current.push_back(RefReplayRow(row, rules));

    std::map<RefKey, uint64_t> counts;
    std::set<RefKey> learned;
    for (const RefRule& rule : rules) {
      learned.emplace(rule.left, rule.right, rule.scope);
    }

    for (size_t sid = 0; sid < rows.size(); ++sid) {
      const auto& toks = current[sid];
      std::map<RefKey, int> last_counted_right;
      for (int i = 0; i + 1 < static_cast<int>(toks.size()); ++i) {
        const int scope = RefScope(rows[sid], toks[i], toks[i + 1]);
        if (scope < 0) continue;
        const RefKey key{toks[i].piece, toks[i + 1].piece, scope};
        if (learned.count(key)) continue;
        const std::string child = toks[i].piece + toks[i + 1].piece;
        const auto existing = piece_id.find(child);
        if (existing != piece_id.end()) {
          const auto a = ancestry.find(child);
          if (a == ancestry.end() ||
              a->second != std::make_pair(toks[i].piece, toks[i + 1].piece)) {
            continue;
          }
        }
        if (toks[i].piece == toks[i + 1].piece) {
          const auto prev = last_counted_right.find(key);
          if (prev != last_counted_right.end() && prev->second == i) {
            continue;
          }
          last_counted_right[key] = i + 1;
        }
        counts[key] += static_cast<uint64_t>(rows[sid].weight);
      }
    }
    if (counts.empty()) break;

    auto better = [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      if (std::get<2>(a.first) != std::get<2>(b.first)) {
        return std::get<2>(a.first) < std::get<2>(b.first);
      }
      if (std::get<0>(a.first) != std::get<0>(b.first)) {
        return std::get<0>(a.first) < std::get<0>(b.first);
      }
      return std::get<1>(a.first) < std::get<1>(b.first);
    };
    auto best = counts.begin();
    for (auto it = std::next(counts.begin()); it != counts.end(); ++it) {
      if (better(*it, *best)) best = it;
    }

    RefRule rule;
    rule.left = std::get<0>(best->first);
    rule.right = std::get<1>(best->first);
    rule.scope = std::get<2>(best->first);
    rule.count = best->second;
    const std::string child = rule.left + rule.right;
    auto id = piece_id.find(child);
    if (id == piece_id.end()) {
      rule.external_id = next_id++;
      rule.allocated = true;
      piece_id[child] = rule.external_id;
      ancestry[child] = {rule.left, rule.right};
      ++allocated;
    } else {
      rule.external_id = id->second;
      rule.allocated = false;
    }
    rules.push_back(std::move(rule));

    // This is deliberately expensive: reconstruct the entire corpus after
    // EVERY operation so the optimized trainer can be checked iteration by
    // iteration, not only at the end.
    std::vector<std::vector<RefToken>> post;
    post.reserve(rows.size());
    for (const RefRow& row : rows) post.push_back(RefReplayRow(row, rules));
    rules.back().post_state_sha256 = RefFinalSha256(rows, post);
  }

  RefRun out;
  out.rules = rules;
  for (const RefRow& row : rows) {
    out.final_rows.push_back(RefReplayRow(row, rules));
  }
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

std::string RunTrainer(
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

TEST(BPETrainerTest, CompletionHierarchyUnlocksOnlyWholeChildren) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "hier_bpe_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "hier_bpe.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "hier_bpe.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "hier_bpe_model");
  const std::string result_path = prefix + ".expansion";

  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abcd\t5"));
  }
  {
    // One parent with children "ab" and "cd".  The byte boundary at 2 may be
    // crossed only when the current left token starts at 0 and the current
    // right token ends at 4.  Thus a+b and c+d must happen before ab+cd.
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    ASSERT_TRUE(out->WriteLine("abcd\t1:0,2,4"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(5);
  expansion.set_requested_new_pieces(3);

  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type, bool mergeable,
                 bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  add(2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  add(3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  add(4, "d", ModelProto::SentencePiece::NORMAL, true, true);
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input);
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_prefix(prefix);
  trainer_spec.set_vocab_size(8);
  trainer_spec.set_expansion_spec(spec_path);
  trainer_spec.set_expansion_result(result_path);
  trainer_spec.set_bpe_hierarchy_file(hierarchy);
  trainer_spec.set_input_sentence_size(0);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_split_digits(false);
  trainer_spec.set_bos_id(-1);
  trainer_spec.set_eos_id(-1);
  trainer_spec.set_pad_id(-1);
  trainer_spec.set_hard_vocab_limit(true);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  normalizer_spec.set_remove_extra_whitespaces(false);
  NormalizerSpec denormalizer_spec;

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  trainer_spec, normalizer_spec, denormalizer_spec)
                  .ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(3, result.learned_merges_size());

  EXPECT_EQ("a", result.learned_merges(0).left());
  EXPECT_EQ("b", result.learned_merges(0).right());
  EXPECT_EQ(0, result.learned_merges(0).grammar_level());
  EXPECT_EQ(0, result.learned_merges(0).scope_level());
  EXPECT_EQ(5, result.learned_merges(0).weighted_count());

  EXPECT_EQ("c", result.learned_merges(1).left());
  EXPECT_EQ("d", result.learned_merges(1).right());
  EXPECT_EQ(0, result.learned_merges(1).grammar_level());
  EXPECT_EQ(0, result.learned_merges(1).scope_level());
  EXPECT_EQ(5, result.learned_merges(1).weighted_count());

  EXPECT_EQ("ab", result.learned_merges(2).left());
  EXPECT_EQ("cd", result.learned_merges(2).right());
  EXPECT_EQ(1, result.learned_merges(2).grammar_level());
  EXPECT_EQ(1, result.learned_merges(2).scope_level());
  EXPECT_EQ(5, result.learned_merges(2).weighted_count());
  EXPECT_TRUE(absl::StartsWith(
      result.boundary_policy(), "bpe_hierarchical_completion_v1:"));
  EXPECT_EQ(result.contract().bpe_hierarchy_sha256(),
            result.boundary_policy().substr(
                std::string("bpe_hierarchical_completion_v1:").size()));
}



TEST(BPETrainerTest, CompletionHierarchyUnlocksNestedParentsBottomUp) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "hier_nested_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "hier_nested.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "hier_nested.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "hier_nested_model");
  const std::string result_path = prefix + ".expansion";

  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abcdef\t5"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    // Level 1: [2,6] = "cd" | "ef".
    // Level 2: [0,6] = "ab" | "cdef".
    // Therefore cdef must complete before the level-2 boundary can unlock.
    ASSERT_TRUE(out->WriteLine("abcdef\t1:2,4,6;2:0,2,6"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(7);
  expansion.set_requested_new_pieces(5);
  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type, bool mergeable,
                 bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  add(2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  add(3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  add(4, "d", ModelProto::SentencePiece::NORMAL, true, true);
  add(5, "e", ModelProto::SentencePiece::NORMAL, true, true);
  add(6, "f", ModelProto::SentencePiece::NORMAL, true, true);
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input);
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_prefix(prefix);
  trainer_spec.set_vocab_size(12);
  trainer_spec.set_expansion_spec(spec_path);
  trainer_spec.set_expansion_result(result_path);
  trainer_spec.set_bpe_hierarchy_file(hierarchy);
  trainer_spec.set_input_sentence_size(0);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_split_digits(false);
  trainer_spec.set_bos_id(-1);
  trainer_spec.set_eos_id(-1);
  trainer_spec.set_pad_id(-1);
  trainer_spec.set_hard_vocab_limit(true);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  normalizer_spec.set_remove_extra_whitespaces(false);
  NormalizerSpec denormalizer_spec;

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  trainer_spec, normalizer_spec, denormalizer_spec)
                  .ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(5, result.learned_merges_size());

  int nested_rank = -1;
  int outer_rank = -1;
  for (const auto& merge : result.learned_merges()) {
    const std::string piece = absl::StrCat(merge.left(), merge.right());
    if (piece == "cdef") {
      nested_rank = merge.rank();
      EXPECT_EQ(1, merge.grammar_level());
      EXPECT_EQ(5, merge.weighted_count());
    } else if (piece == "abcdef") {
      outer_rank = merge.rank();
      EXPECT_EQ(2, merge.grammar_level());
      EXPECT_EQ(5, merge.weighted_count());
    }
  }
  ASSERT_GE(nested_rank, 0);
  ASSERT_GE(outer_rank, 0);
  EXPECT_LT(nested_rank, outer_rank);
}

TEST(BPETrainerTest, CompletionHierarchyKeepsUserDefinedFrozen) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "hier_ud_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "hier_ud.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "hier_ud.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "hier_ud_model");
  const std::string result_path = prefix + ".expansion";

  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("aa<X>bb\t9"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    // Direct children are "aa", frozen "<X>", and "bb".
    ASSERT_TRUE(out->WriteLine("aa<X>bb\t1:0,2,5,7"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(4);
  expansion.set_requested_new_pieces(2);
  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type, bool mergeable,
                 bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "<X>", ModelProto::SentencePiece::USER_DEFINED, false, false);
  add(2, "a", ModelProto::SentencePiece::NORMAL, true, true);
  add(3, "b", ModelProto::SentencePiece::NORMAL, true, true);
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input);
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_prefix(prefix);
  trainer_spec.set_vocab_size(6);
  trainer_spec.set_expansion_spec(spec_path);
  trainer_spec.set_expansion_result(result_path);
  trainer_spec.set_bpe_hierarchy_file(hierarchy);
  trainer_spec.set_input_sentence_size(0);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_split_digits(false);
  trainer_spec.set_bos_id(-1);
  trainer_spec.set_eos_id(-1);
  trainer_spec.set_pad_id(-1);
  trainer_spec.set_hard_vocab_limit(true);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  normalizer_spec.set_remove_extra_whitespaces(false);
  NormalizerSpec denormalizer_spec;

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  trainer_spec, normalizer_spec, denormalizer_spec)
                  .ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(2, result.learned_merges_size());
  ASSERT_GE(result.base_pieces_size(), 2);
  EXPECT_EQ(1, result.base_pieces(1).external_id());
  EXPECT_EQ("<X>", result.base_pieces(1).piece());
  EXPECT_EQ(ModelProto::SentencePiece::USER_DEFINED,
            result.base_pieces(1).type());

  for (const auto& merge : result.learned_merges()) {
    EXPECT_NE("<X>", merge.left());
    EXPECT_NE("<X>", merge.right());
    EXPECT_EQ(std::string::npos,
              absl::StrCat(merge.left(), merge.right()).find("<X>"));
  }
}

TEST(BPETrainerTest, CompletionHierarchyCountsOnlyEligibleOccurrences) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "hier_occurrence_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "hier_occurrence.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "hier_occurrence.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "hier_occurrence_model");
  const std::string result_path = prefix + ".expansion";

  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abcd\t20"));
    ASSERT_TRUE(out->WriteLine("xabcd\t7"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    // After a+b and c+d, ab+cd is a complete-child crossing in "abcd" but
    // the same surface pair is only a suffix of the unfinished left child
    // "xab" in "xabcd". The eligible occurrence must still be learnable.
    ASSERT_TRUE(out->WriteLine("abcd\t1:0,2,4"));
    ASSERT_TRUE(out->WriteLine("xabcd\t1:0,3,5"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(6);
  expansion.set_requested_new_pieces(3);
  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type, bool mergeable,
                 bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "a", ModelProto::SentencePiece::NORMAL, true, true);
  add(2, "b", ModelProto::SentencePiece::NORMAL, true, true);
  add(3, "c", ModelProto::SentencePiece::NORMAL, true, true);
  add(4, "d", ModelProto::SentencePiece::NORMAL, true, true);
  add(5, "x", ModelProto::SentencePiece::NORMAL, true, true);
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input);
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_prefix(prefix);
  trainer_spec.set_vocab_size(9);
  trainer_spec.set_expansion_spec(spec_path);
  trainer_spec.set_expansion_result(result_path);
  trainer_spec.set_bpe_hierarchy_file(hierarchy);
  trainer_spec.set_input_sentence_size(0);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_split_digits(false);
  trainer_spec.set_bos_id(-1);
  trainer_spec.set_eos_id(-1);
  trainer_spec.set_pad_id(-1);
  trainer_spec.set_hard_vocab_limit(true);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  normalizer_spec.set_remove_extra_whitespaces(false);
  NormalizerSpec denormalizer_spec;

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  trainer_spec, normalizer_spec, denormalizer_spec)
                  .ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(3, result.learned_merges_size());

  bool saw_ab_cd = false;
  for (const auto& merge : result.learned_merges()) {
    if (merge.left() == "ab" && merge.right() == "cd") {
      saw_ab_cd = true;
      EXPECT_EQ(20, merge.weighted_count());
      EXPECT_EQ(1, merge.grammar_level());
    }
  }
  EXPECT_TRUE(saw_ab_cd);
}

TEST(BPETrainerTest, CompletionHierarchyDoesNotShadowShortDenominator) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "hier_prefix_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "hier_prefix.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "hier_prefix.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "hier_prefix_model");
  const std::string result_path = prefix + ".expansion";

  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("/12\t20"));
    ASSERT_TRUE(out->WriteLine("/128\t7"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    ASSERT_TRUE(out->WriteLine("/12\t1:0,1,3"));
    ASSERT_TRUE(out->WriteLine("/128\t1:0,1,4"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(5);
  expansion.set_requested_new_pieces(4);
  auto add = [&](int id, absl::string_view piece,
                 ModelProto::SentencePiece::Type type, bool mergeable,
                 bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(type);
    p->set_mergeable(mergeable);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", ModelProto::SentencePiece::UNKNOWN, false, false);
  add(1, "/", ModelProto::SentencePiece::NORMAL, true, true);
  add(2, "1", ModelProto::SentencePiece::NORMAL, true, true);
  add(3, "2", ModelProto::SentencePiece::NORMAL, true, true);
  add(4, "8", ModelProto::SentencePiece::NORMAL, true, true);
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec trainer_spec;
  trainer_spec.set_model_type(TrainerSpec::BPE);
  trainer_spec.add_input(input);
  trainer_spec.set_input_format("tsv");
  trainer_spec.set_model_prefix(prefix);
  trainer_spec.set_vocab_size(9);
  trainer_spec.set_expansion_spec(spec_path);
  trainer_spec.set_expansion_result(result_path);
  trainer_spec.set_bpe_hierarchy_file(hierarchy);
  trainer_spec.set_input_sentence_size(0);
  trainer_spec.set_split_by_whitespace(false);
  trainer_spec.set_split_by_unicode_script(false);
  trainer_spec.set_split_by_number(false);
  trainer_spec.set_split_digits(false);
  trainer_spec.set_bos_id(-1);
  trainer_spec.set_eos_id(-1);
  trainer_spec.set_pad_id(-1);
  trainer_spec.set_hard_vocab_limit(true);

  NormalizerSpec normalizer_spec;
  normalizer_spec.set_name("identity");
  normalizer_spec.set_add_dummy_prefix(false);
  normalizer_spec.set_remove_extra_whitespaces(false);
  NormalizerSpec denormalizer_spec;

  ASSERT_TRUE(SentencePieceTrainer::Train(
                  trainer_spec, normalizer_spec, denormalizer_spec)
                  .ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));
  ASSERT_EQ(4, result.learned_merges_size());

  EXPECT_EQ("1", result.learned_merges(0).left());
  EXPECT_EQ("2", result.learned_merges(0).right());
  EXPECT_EQ(27, result.learned_merges(0).weighted_count());
  ASSERT_TRUE(result.learned_merges(0).has_scope_level());
  EXPECT_EQ(0, result.learned_merges(0).scope_level());

  EXPECT_EQ("/", result.learned_merges(1).left());
  EXPECT_EQ("12", result.learned_merges(1).right());
  EXPECT_EQ(20, result.learned_merges(1).weighted_count());
  EXPECT_EQ(1, result.learned_merges(1).grammar_level());
  ASSERT_TRUE(result.learned_merges(1).has_scope_level());
  EXPECT_EQ(1, result.learned_merges(1).scope_level());

  EXPECT_EQ("12", result.learned_merges(2).left());
  EXPECT_EQ("8", result.learned_merges(2).right());
  EXPECT_EQ(7, result.learned_merges(2).weighted_count());
  EXPECT_EQ(0, result.learned_merges(2).grammar_level());
  ASSERT_TRUE(result.learned_merges(2).has_scope_level());
  EXPECT_EQ(0, result.learned_merges(2).scope_level());

  EXPECT_EQ("/", result.learned_merges(3).left());
  EXPECT_EQ("128", result.learned_merges(3).right());
  EXPECT_EQ(7, result.learned_merges(3).weighted_count());
  EXPECT_EQ(1, result.learned_merges(3).grammar_level());
  ASSERT_TRUE(result.learned_merges(3).has_scope_level());
  EXPECT_EQ(1, result.learned_merges(3).scope_level());

  // The artifact produced by training must replay with the same occurrence-
  // local decision. This is the regression the first flat runtime was missing.
  expansion::ExpansionProcessor runtime;
  ASSERT_TRUE(runtime.Load(result).ok());
  expansion::CompletionGate short_gate;
  short_gate.level = 1;
  short_gate.cuts = {0, 1, 3};
  std::vector<expansion::TokenSpan> short_out;
  ASSERT_TRUE(runtime.EncodeWithHierarchy("/12", {short_gate}, &short_out).ok());
  ASSERT_EQ(1u, short_out.size());
  EXPECT_EQ("/12", short_out[0].piece);

  expansion::CompletionGate long_gate;
  long_gate.level = 1;
  long_gate.cuts = {0, 1, 4};
  std::vector<expansion::TokenSpan> long_out;
  ASSERT_TRUE(runtime.EncodeWithHierarchy("/128", {long_gate}, &long_out).ok());
  ASSERT_EQ(1u, long_out.size());
  EXPECT_EQ("/128", long_out[0].piece)
      << "the earlier /+12 rank is blocked locally, allowing 12+8 and then "
         "/+128 to reproduce the training construction";
}


TEST(BPETrainerTest, HierarchyKeepsPairCountsSeparatedByExactScope) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "scope_pool_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "scope_pool.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "scope_pool.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "scope_pool_model");
  const std::string result_path = prefix + ".expansion";
  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abx\t5"));
    ASSERT_TRUE(out->WriteLine("ab\t5"));
    ASSERT_TRUE(out->WriteLine("cd\t8"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    ASSERT_TRUE(out->WriteLine("abx\t1:0,2,3"));  // a+b is ordinary/internal.
    ASSERT_TRUE(out->WriteLine("ab\t1:0,1,2"));   // a+b is level-1 crossing.
    ASSERT_TRUE(out->WriteLine("cd\t"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(6);
  expansion.set_requested_new_pieces(1);
  auto add = [&](int id, absl::string_view piece) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(id == 0 ? ModelProto::SentencePiece::UNKNOWN
                        : ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(id != 0);
    p->set_atomic(id != 0);
  };
  add(0, "<unk>"); add(1, "a"); add(2, "b");
  add(3, "x"); add(4, "c"); add(5, "d");
  { auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString())); }

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE); ts.add_input(input);
  ts.set_input_format("tsv"); ts.set_model_prefix(prefix); ts.set_vocab_size(7);
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
  ASSERT_EQ(1, result.learned_merges_size());
  const auto& first = result.learned_merges(0);
  EXPECT_EQ("c", first.left());
  EXPECT_EQ("d", first.right());
  EXPECT_EQ(8, first.weighted_count());
  ASSERT_TRUE(first.has_scope_level());
  EXPECT_EQ(0, first.scope_level())
      << "a+b must remain two candidates of count 5, not one pooled count 10";
}

TEST(BPETrainerTest, RepeatedPairCountEqualsNonOverlappingReplacements) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "overlap_count_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "overlap_count.spec");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "overlap_count_model");
  const std::string result_path = prefix + ".expansion";
  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("aaa\t2"));
    ASSERT_TRUE(out->WriteLine("bc\t3"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(4);
  expansion.set_requested_new_pieces(1);
  auto add = [&](int id, absl::string_view piece) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id); p->set_piece(std::string(piece));
    p->set_type(id == 0 ? ModelProto::SentencePiece::UNKNOWN
                        : ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(id != 0); p->set_atomic(id != 0);
  };
  add(0, "<unk>"); add(1, "a"); add(2, "b"); add(3, "c");
  { auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString())); }

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE); ts.add_input(input);
  ts.set_input_format("tsv"); ts.set_model_prefix(prefix); ts.set_vocab_size(5);
  ts.set_expansion_spec(spec_path); ts.set_expansion_result(result_path);
  ts.set_input_sentence_size(0);
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
  ASSERT_EQ(1, result.learned_merges_size());
  EXPECT_EQ("b", result.learned_merges(0).left());
  EXPECT_EQ("c", result.learned_merges(0).right());
  EXPECT_EQ(3, result.learned_merges(0).weighted_count())
      << "aaa weight 2 contains one non-overlapping aa replacement, not two";
}

TEST(BPETrainerTest, ExactCountTiesPreferLowerGrammarScope) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "scope_tie_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "scope_tie.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "scope_tie.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "scope_tie_model");
  const std::string result_path = prefix + ".expansion";
  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("zz\t5"));
    ASSERT_TRUE(out->WriteLine("ab\t5"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    ASSERT_TRUE(out->WriteLine("zz\t"));
    ASSERT_TRUE(out->WriteLine("ab\t1:0,1,2"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(4);
  expansion.set_requested_new_pieces(1);
  auto add = [&](int id, absl::string_view piece) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id); p->set_piece(std::string(piece));
    p->set_type(id == 0 ? ModelProto::SentencePiece::UNKNOWN
                        : ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(id != 0); p->set_atomic(id != 0);
  };
  add(0, "<unk>"); add(1, "z"); add(2, "a"); add(3, "b");
  { auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString())); }

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE); ts.add_input(input);
  ts.set_input_format("tsv"); ts.set_model_prefix(prefix); ts.set_vocab_size(5);
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
  ASSERT_EQ(1, result.learned_merges_size());
  EXPECT_EQ("z", result.learned_merges(0).left());
  EXPECT_EQ("z", result.learned_merges(0).right());
  EXPECT_EQ(5, result.learned_merges(0).weighted_count());
  EXPECT_EQ(0, result.learned_merges(0).scope_level())
      << "exact count ties must prefer ordinary/lower scope";
}


TEST(BPETrainerTest, EqualSurfaceAncestriesUsePairBytesAsFinalTieBreak) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "ancestry_tie_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "ancestry_tie.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "ancestry_tie.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "ancestry_tie_model");
  const std::string result_path = prefix + ".expansion";
  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abcX\t5"));
    ASSERT_TRUE(out->WriteLine("Yabc\t5"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    // abcX: parent is a | bc, so b+c is ordinary; a+b is blocked.
    ASSERT_TRUE(out->WriteLine("abcX\t1:0,1,3"));
    // Yabc: parent is ab | c, so a+b is ordinary; b+c is blocked.
    ASSERT_TRUE(out->WriteLine("Yabc\t1:1,3,4"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(6);
  expansion.set_requested_new_pieces(3);
  auto add_normal = [&](int id, absl::string_view piece) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id); p->set_piece(std::string(piece));
    p->set_type(id == 0 ? ModelProto::SentencePiece::UNKNOWN
                        : ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(id != 0); p->set_atomic(id != 0);
  };
  add_normal(0, "<unk>"); add_normal(1, "a");
  add_normal(2, "b"); add_normal(3, "c");
  for (const auto& [id, piece] :
       std::vector<std::pair<int, std::string>>{{4, "X"}, {5, "Y"}}) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id); p->set_piece(piece);
    p->set_type(ModelProto::SentencePiece::USER_DEFINED);
    p->set_mergeable(false); p->set_atomic(false);
  }
  { auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString())); }

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE); ts.add_input(input);
  ts.set_input_format("tsv"); ts.set_model_prefix(prefix); ts.set_vocab_size(9);
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
  ASSERT_EQ(3, result.learned_pieces_size());
  ASSERT_EQ(3, result.learned_merges_size());

  EXPECT_EQ("a", result.learned_merges(0).left());
  EXPECT_EQ("b", result.learned_merges(0).right());
  EXPECT_EQ("b", result.learned_merges(1).left());
  EXPECT_EQ("c", result.learned_merges(1).right());

  // At this point a+bc and ab+c both have count 5, scope 1, and both
  // concatenate to "abc". Concatenated-surface tie-breaking cannot order
  // them; the child pair itself must. ("a","bc") sorts before ("ab","c").
  EXPECT_EQ("a", result.learned_merges(2).left());
  EXPECT_EQ("bc", result.learned_merges(2).right());
  EXPECT_EQ(1, result.learned_merges(2).scope_level());
  EXPECT_EQ(5, result.learned_merges(2).weighted_count());
}

TEST(BPETrainerTest, SamePairAtTwoScopesSharesOneTokenId) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "scope_alias_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "scope_alias.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "scope_alias.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "scope_alias_model");
  const std::string result_path = prefix + ".expansion";
  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abX\t12"));
    ASSERT_TRUE(out->WriteLine("ab\t11"));
    ASSERT_TRUE(out->WriteLine("cd\t1"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    ASSERT_TRUE(out->WriteLine("abX\t1:0,2,3"));
    ASSERT_TRUE(out->WriteLine("ab\t1:0,1,2"));
    ASSERT_TRUE(out->WriteLine("cd\t"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(6);
  expansion.set_requested_new_pieces(2);
  auto add_normal = [&](int id, absl::string_view piece) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id); p->set_piece(std::string(piece));
    p->set_type(id == 0 ? ModelProto::SentencePiece::UNKNOWN
                        : ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(id != 0); p->set_atomic(id != 0);
  };
  add_normal(0, "<unk>"); add_normal(1, "a"); add_normal(2, "b");
  add_normal(3, "c"); add_normal(4, "d");
  auto* frozen = expansion.add_base_pieces();
  frozen->set_external_id(5); frozen->set_piece("X");
  frozen->set_type(ModelProto::SentencePiece::USER_DEFINED);
  frozen->set_mergeable(false); frozen->set_atomic(false);
  { auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString())); }

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE); ts.add_input(input);
  ts.set_input_format("tsv"); ts.set_model_prefix(prefix); ts.set_vocab_size(8);
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
  ASSERT_EQ(2, result.learned_pieces_size());
  ASSERT_GE(result.learned_merges_size(), 3);
  EXPECT_EQ("a", result.learned_merges(0).left());
  EXPECT_EQ("b", result.learned_merges(0).right());
  EXPECT_EQ(0, result.learned_merges(0).scope_level());
  EXPECT_EQ("a", result.learned_merges(1).left());
  EXPECT_EQ("b", result.learned_merges(1).right());
  EXPECT_EQ(1, result.learned_merges(1).scope_level());
  EXPECT_EQ(result.learned_merges(0).external_id(),
            result.learned_merges(1).external_id())
      << "scope belongs to the operation; ab remains one token ID";
}



TEST(BPETrainerTest, RandomLaminarHierarchyMatchesIndependentReplayOracle) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "oracle_random_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "oracle_random.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "oracle_random.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "oracle_random_model");
  const std::string result_path = prefix + ".expansion";
  const std::string trace_path =
      filesystem::JoinPath(::testing::TempDir(), "oracle_random.trace");

  constexpr int kRows = 2048;
  constexpr int kLength = 6;
  constexpr int kRequested = 24;
  std::mt19937 rng(0x1A2B3C4Du);
  std::uniform_int_distribution<int> weight_dist(1, 5);

  std::vector<RefRow> rows;
  rows.reserve(kRows);
  for (int n = 0; n < kRows; ++n) {
    int x = n;
    std::string text(kLength, 'a');
    for (int i = kLength - 1; i >= 0; --i) {
      text[i] = "abcd"[x & 3];
      x >>= 2;
    }
    RefRow row;
    row.text = text;
    row.weight = weight_dist(rng);
    BuildRandomLaminarGates(0, kLength, 0, &rng, &row.gates);
    rows.push_back(std::move(row));
  }

  {
    auto out = filesystem::NewWritableFile(input);
    for (const RefRow& row : rows) {
      ASSERT_TRUE(out->WriteLine(
          absl::StrCat(row.text, "\t", row.weight)));
    }
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    for (const RefRow& row : rows) {
      std::vector<std::string> encoded;
      for (const RefGate& gate : row.gates) {
        encoded.push_back(absl::StrCat(
            gate.level, ":", absl::StrJoin(gate.cuts, ",")));
      }
      ASSERT_TRUE(out->WriteLine(
          absl::StrCat(row.text, "\t", absl::StrJoin(encoded, ";"))));
    }
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(5);
  expansion.set_requested_new_pieces(kRequested);
  auto add = [&](int id, absl::string_view piece, bool atomic) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(id == 0 ? ModelProto::SentencePiece::UNKNOWN
                        : ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(id != 0);
    p->set_atomic(atomic);
  };
  add(0, "<unk>", false);
  add(1, "a", true); add(2, "b", true);
  add(3, "c", true); add(4, "d", true);
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  // Run the deliberately slow oracle BEFORE reading trainer output. It starts
  // from atoms and full-replays the complete scoped prefix after every rank.
  const RefRun reference =
      RunScopedReference(rows, /*first_new_external_id=*/5, kRequested);
  ASSERT_EQ(kRequested,
            std::count_if(reference.rules.begin(), reference.rules.end(),
                          [](const RefRule& r) { return r.allocated; }));

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE);
  ts.add_input(input);
  ts.set_input_format("tsv");
  ts.set_model_prefix(prefix);
  ts.set_vocab_size(5 + kRequested);
  ts.set_expansion_spec(spec_path);
  ts.set_expansion_result(result_path);
  ts.set_bpe_hierarchy_file(hierarchy);
  ts.set_bpe_reference_trace_file(trace_path);
  ts.set_input_sentence_size(0);
  ts.set_split_by_whitespace(false);
  ts.set_split_by_unicode_script(false);
  ts.set_split_by_number(false);
  ts.set_split_digits(false);
  ts.set_max_sentencepiece_length(50);
  ts.set_bos_id(-1); ts.set_eos_id(-1); ts.set_pad_id(-1);
  ts.set_hard_vocab_limit(true);
  NormalizerSpec ns;
  ns.set_name("identity");
  ns.set_add_dummy_prefix(false);
  ns.set_remove_extra_whitespaces(false);
  NormalizerSpec dns;

  ASSERT_TRUE(SentencePieceTrainer::Train(ts, ns, dns).ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));

  ASSERT_EQ(reference.rules.size(),
            static_cast<size_t>(result.learned_merges_size()));
  for (size_t rank = 0; rank < reference.rules.size(); ++rank) {
    const RefRule& want = reference.rules[rank];
    const ExpansionMerge& got =
        result.learned_merges(static_cast<int>(rank));
    EXPECT_EQ(static_cast<int>(rank), got.rank()) << "rank " << rank;
    EXPECT_EQ(want.left, got.left()) << "rank " << rank;
    EXPECT_EQ(want.right, got.right()) << "rank " << rank;
    EXPECT_EQ(want.scope, got.scope_level()) << "rank " << rank;
    EXPECT_EQ(want.count, got.weighted_count()) << "rank " << rank;
    EXPECT_EQ(want.external_id, got.external_id()) << "rank " << rank;
  }
  ASSERT_EQ(kRequested, result.learned_pieces_size());

  // Exact per-iteration state: operation, selected count, allocation/ID, and
  // full restarted-corpus segmentation digest.
  std::vector<std::string> trace_lines;
  {
    auto trace = filesystem::NewReadableFile(trace_path);
    ASSERT_TRUE(trace->status().ok());
    std::string blob;
    ASSERT_TRUE(trace->ReadAll(&blob));
    for (absl::string_view line : absl::StrSplit(blob, '\n')) {
      if (!line.empty()) trace_lines.emplace_back(line);
    }
  }
  ASSERT_EQ(reference.rules.size(), trace_lines.size());
  for (size_t rank = 0; rank < reference.rules.size(); ++rank) {
    const RefRule& want = reference.rules[rank];
    EXPECT_EQ(
        absl::StrCat(rank, "\t", want.left, "\t", want.right, "\t",
                     want.scope, "\t", want.count, "\t",
                     want.allocated ? 1 : 0, "\t", want.external_id, "\t",
                     want.post_state_sha256),
        trace_lines[rank]) << "iteration " << rank;
  }

  ASSERT_TRUE(result.has_training_final_segmentation_sha256());
  EXPECT_EQ(reference.final_sha256,
            result.training_final_segmentation_sha256())
      << "optimized trainer final state differs from restart-and-replay oracle";

  // Runtime is a third implementation. Check every generated tree, not only
  // the final aggregate token count.
  expansion::ExpansionProcessor runtime;
  ASSERT_TRUE(runtime.Load(result).ok());
  for (size_t sid = 0; sid < rows.size(); ++sid) {
    std::vector<expansion::CompletionGate> gates;
    for (const RefGate& g : rows[sid].gates) {
      expansion::CompletionGate gate;
      gate.level = g.level;
      gate.cuts = g.cuts;
      gates.push_back(std::move(gate));
    }
    std::vector<expansion::TokenSpan> got;
    ASSERT_TRUE(runtime.EncodeWithHierarchy(
        rows[sid].text, gates, &got).ok()) << "sid=" << sid;
    std::vector<std::string> pieces;
    for (const auto& token : got) pieces.push_back(token.piece);
    std::vector<std::string> want;
    for (const RefToken& token : reference.final_rows[sid]) {
      want.push_back(token.piece);
    }
    EXPECT_EQ(want, pieces) << "sid=" << sid << " text=" << rows[sid].text;
  }
}

TEST(BPETrainerTest, ScopedAliasReplaysEarlierRanksToFixedPoint) {
  const std::string input =
      filesystem::JoinPath(::testing::TempDir(), "alias_cascade_input.tsv");
  const std::string spec_path =
      filesystem::JoinPath(::testing::TempDir(), "alias_cascade.spec");
  const std::string hierarchy =
      filesystem::JoinPath(::testing::TempDir(), "alias_cascade.tsv");
  const std::string prefix =
      filesystem::JoinPath(::testing::TempDir(), "alias_cascade_model");
  const std::string result_path = prefix + ".expansion";

  {
    auto out = filesystem::NewWritableFile(input);
    ASSERT_TRUE(out->WriteLine("abcX\t10"));
    ASSERT_TRUE(out->WriteLine("abcY\t2"));
    ASSERT_TRUE(out->WriteLine("deZ\t1"));
  }
  {
    auto out = filesystem::NewWritableFile(hierarchy);
    ASSERT_TRUE(out->WriteLine("# sentencepiece-bpe-hierarchy-v1"));
    // X row: a+b is ordinary inside child "ab"; ab+c is a level-2 crossing.
    ASSERT_TRUE(out->WriteLine("abcX\t2:0,2,3"));
    // Y row: a+b is a level-1 crossing; once it fires, ab+c is level 2.
    ASSERT_TRUE(out->WriteLine("abcY\t1:0,1,2;2:0,2,3"));
    ASSERT_TRUE(out->WriteLine("deZ\t"));
  }

  ExpansionSpec expansion;
  expansion.set_schema_version(1);
  expansion.set_model_type(EXPANSION_BPE);
  expansion.set_preserve_base_ids(true);
  expansion.set_first_new_external_id(9);
  expansion.set_requested_new_pieces(3);

  auto atom = [&](int id, absl::string_view piece) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(std::string(piece));
    p->set_type(id == 0 ? ModelProto::SentencePiece::UNKNOWN
                        : ModelProto::SentencePiece::NORMAL);
    p->set_mergeable(id != 0);
    p->set_atomic(id != 0);
  };
  atom(0, "<unk>");
  atom(1, "a"); atom(2, "b"); atom(3, "c");
  atom(4, "d"); atom(5, "e");

  for (const auto& [id, piece] :
       std::vector<std::pair<int, std::string>>{
           {6, "X"}, {7, "Y"}, {8, "Z"}}) {
    auto* p = expansion.add_base_pieces();
    p->set_external_id(id);
    p->set_piece(piece);
    p->set_type(ModelProto::SentencePiece::USER_DEFINED);
    p->set_mergeable(false);
    p->set_atomic(false);
  }
  {
    auto out = filesystem::NewWritableFile(spec_path, true);
    ASSERT_TRUE(out->Write(expansion.SerializeAsString()));
  }

  TrainerSpec ts;
  ts.set_model_type(TrainerSpec::BPE);
  ts.add_input(input);
  ts.set_input_format("tsv");
  ts.set_model_prefix(prefix);
  ts.set_vocab_size(12);
  ts.set_expansion_spec(spec_path);
  ts.set_expansion_result(result_path);
  ts.set_bpe_hierarchy_file(hierarchy);
  ts.set_input_sentence_size(0);
  ts.set_split_by_whitespace(false);
  ts.set_split_by_unicode_script(false);
  ts.set_split_by_number(false);
  ts.set_split_digits(false);
  ts.set_bos_id(-1); ts.set_eos_id(-1); ts.set_pad_id(-1);
  ts.set_hard_vocab_limit(true);
  NormalizerSpec ns;
  ns.set_name("identity");
  ns.set_add_dummy_prefix(false);
  ns.set_remove_extra_whitespaces(false);
  NormalizerSpec dns;

  ASSERT_TRUE(SentencePieceTrainer::Train(ts, ns, dns).ok());

  std::string bytes;
  {
    auto in = filesystem::NewReadableFile(result_path, true);
    ASSERT_TRUE(in->ReadAll(&bytes));
  }
  ExpansionResult result;
  ASSERT_TRUE(result.ParseFromString(bytes));

  // Three new token IDs, but four operations: rank 2 is the scoped alias.
  ASSERT_EQ(3, result.learned_pieces_size());
  ASSERT_EQ(4, result.learned_merges_size());

  EXPECT_EQ("a", result.learned_merges(0).left());
  EXPECT_EQ("b", result.learned_merges(0).right());
  EXPECT_EQ(0, result.learned_merges(0).scope_level());
  EXPECT_EQ(10, result.learned_merges(0).weighted_count());

  EXPECT_EQ("ab", result.learned_merges(1).left());
  EXPECT_EQ("c", result.learned_merges(1).right());
  EXPECT_EQ(2, result.learned_merges(1).scope_level());
  EXPECT_EQ(10, result.learned_merges(1).weighted_count());

  EXPECT_EQ("a", result.learned_merges(2).left());
  EXPECT_EQ("b", result.learned_merges(2).right());
  EXPECT_EQ(1, result.learned_merges(2).scope_level());
  EXPECT_EQ(2, result.learned_merges(2).weighted_count());
  EXPECT_EQ(result.learned_merges(0).external_id(),
            result.learned_merges(2).external_id());

  EXPECT_EQ("d", result.learned_merges(3).left());
  EXPECT_EQ("e", result.learned_merges(3).right());
  EXPECT_EQ(1, result.learned_merges(3).weighted_count());

  // The critical assertion: after rank 2 creates "ab" in abcY, rank 1 must
  // replay there. Without alias closure the Y row remains ab|c|Y and the
  // weighted final token count is 28 rather than 26.
  ASSERT_TRUE(result.has_training_final_weighted_tokens());
  EXPECT_EQ(26u, result.training_final_weighted_tokens());

  // The serialized runtime must execute the same backward cascade.
  expansion::ExpansionProcessor runtime;
  ASSERT_TRUE(runtime.Load(result).ok());
  expansion::CompletionGate y_l1;
  y_l1.level = 1; y_l1.cuts = {0, 1, 2};
  expansion::CompletionGate y_l2;
  y_l2.level = 2; y_l2.cuts = {0, 2, 3};
  std::vector<expansion::TokenSpan> y_out;
  ASSERT_TRUE(runtime.EncodeWithHierarchy(
      "abcY", {y_l1, y_l2}, &y_out).ok());
  std::vector<std::string> y_pieces;
  for (const auto& x : y_out) y_pieces.push_back(x.piece);
  EXPECT_EQ(std::vector<std::string>({"abc", "Y"}), y_pieces);
}

TEST(BPETrainerTest, HierarchyNeverVetoesInheritedMergeReplay) {
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
