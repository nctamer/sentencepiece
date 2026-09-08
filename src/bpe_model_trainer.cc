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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "filesystem.h"
#include "ret_check.h"
#include "util.h"

namespace sentencepiece::bpe {

std::string Trainer::Symbol::ToString() const {
  return string_util::UnicodeTextToUTF8(chars);
}

Trainer::Symbol* Trainer::GetCharSymbol(char32_t c) {
  const auto req_it = required_chars_.find(c);
  const uint64_t freq = (req_it != required_chars_.end()) ? req_it->second : 1;
  CHECK_GT(freq, uint64_t{0});
  const auto it = symbols_cache_.find(c);
  if (it != symbols_cache_.end()) {
    return it->second;
  }
  auto s = std::make_unique<Symbol>();
  s->is_unk = (kUNKChar == c);
  s->fp = c;
  s->chars.push_back(c);
  s->freq = freq;
  symbols_cache_.emplace(s->fp, s.get());
  Symbol* s_ptr = s.get();
  allocated_.push_back(std::move(s));
  return s_ptr;
}

Trainer::Symbol* Trainer::GetPairSymbol(const Symbol* left,
                                        const Symbol* right) {
  if (left == nullptr || right == nullptr || left->is_unk || right->is_unk) {
    return nullptr;
  }

  const uint64_t fp = absl::HashOf(left->fp, right->fp);
  const auto it = symbols_cache_.find(fp);
  if (it != symbols_cache_.end()) {
    return it->second;
  }

  CHECK(!left->chars.empty());
  CHECK(!right->chars.empty());
  string_util::UnicodeText ut;
  for (const char32_t c : left->chars) {
    ut.push_back(c);
  }
  for (const char32_t c : right->chars) {
    ut.push_back(c);
  }

  // Do not make an invalid piece.
  if (!IsValidSentencePiece(ut)) {
    return nullptr;
  }

  auto s = std::make_unique<Symbol>();
  s->fp = fp;
  s->left = left;
  s->right = right;
  s->chars = ut;
  symbols_cache_.emplace(s->fp, s.get());
  Symbol* s_ptr = s.get();
  allocated_.push_back(std::move(s));
  return s_ptr;
}

void Trainer::ComputeFreq(Symbol* symbol) const {
  if (!symbol->needs_recomputation) {
    return;
  }
  symbol->freq = 0;
  for (auto it = symbol->positions.begin(); it != symbol->positions.end();) {
    const Position pos = DecodePos(*it);
    // symbols_[sid][left] and symbols_[sid]right] must store
    // the same symbols in symbol->left and symbols->right.
    if (symbol->left != symbols_[pos.sid][pos.left] ||
        symbol->right != symbols_[pos.sid][pos.right]) {
      it = symbol->positions.erase(it);
    } else {
      symbol->freq += sentences_[pos.sid].second;
      ++it;
    }
  }
  symbol->needs_recomputation = false;
}

int Trainer::GetNextIndex(int sid, int index) const {
  for (size_t i = index + 1; i < symbols_[sid].size(); ++i) {
    if (symbols_[sid][i] == nullptr) {
      continue;
    }
    return i;
  }
  return -1;
}

int Trainer::GetPrevIndex(int sid, int index) const {
  for (int i = index - 1; i >= 0; --i) {
    if (symbols_[sid][i] == nullptr) {
      continue;
    }
    return i;
  }
  return -1;
}

void Trainer::AddNewPair(int sid, int left, int right) {
  if (left == -1 || right == -1) {
    return;
  }
  auto* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr) {
    symbol->positions.insert(EncodePos(sid, left, right));
    if (!symbol->pending) {
      symbol->pending = true;
      pending_queue_.push_back(symbol);
    }
  }
}

void Trainer::ResetFreq(int sid, int left, int right, const Symbol* best) {
  if (left == -1 || right == -1) {
    return;
  }
  auto* symbol = GetPairSymbol(symbols_[sid][left], symbols_[sid][right]);
  if (symbol != nullptr && symbol != best) {
    symbol->needs_recomputation = true;
  }
}

absl::Status Trainer::AcceptSymbol(Symbol* symbol) {
  // Add new bigrams which are created after symbol replacement.
  // We do not need to scan all characters, but scan the neighbors in
  // best_symbol.
  for (const uint64_t& encoded_pos : symbol->positions) {
    const Position pos = DecodePos(encoded_pos);

    if (symbols_[pos.sid][pos.left] == nullptr) {
      // left index might be NULL (set in the previous iteration)
      // when left_symbol == right_symbol.
      continue;
    }
    RET_CHECK(symbols_[pos.sid][pos.right]);

    // We have three bigrams [prev, left], [left, right], [right, next],
    // which are affected with this symbol replacement.
    const int next = GetNextIndex(pos.sid, pos.right);
    const int prev = GetPrevIndex(pos.sid, pos.left);

    // Resets the frequencies of bigrams [prev, left] and [right, next].
    ResetFreq(pos.sid, prev, pos.left, symbol);
    ResetFreq(pos.sid, pos.right, next, symbol);

    // Merges two symbols.
    symbols_[pos.sid][pos.left] = symbol;
    symbols_[pos.sid][pos.right] = nullptr;

    // Makes new symbol bigrams [prev, left] and [left, next].
    AddNewPair(pos.sid, prev, pos.left);
    AddNewPair(pos.sid, pos.left, next);
  }

  // Removes best_symbol so it is not selected again.
  symbols_cache_.erase(symbol->fp);
  symbol->active = false;

  return absl::OkStatus();
}

absl::Status Trainer::Train() {
  ABSL_RETURN_IF_ERROR(status());

  RET_CHECK(normalizer_spec_.escape_whitespaces());
  RET_CHECK_EQ(TrainerSpec::BPE, trainer_spec_.model_type());

  symbols_.clear();
  allocated_.clear();
  symbols_cache_.clear();
  pq_ = decltype(pq_)();
  pending_queue_.clear();

  // Load all sentences
  ABSL_RETURN_IF_ERROR(LoadSentences());
  if (trainer_spec_.split_by_whitespace() ||
      trainer_spec_.GetExtension(::sentencepiece::split_by_interval) ||
      trainer_spec_.GetExtension(::sentencepiece::split_by_barline)) {
    SplitSentencesByWhitespace();
  }

  // Initializes symbols_. symbols_[sid][i] stores an unary symbol.
  symbols_.resize(sentences_.size());
  for (size_t i = 0; i < sentences_.size(); ++i) {
    for (const char32_t c :
         string_util::UTF8ToUnicodeText(sentences_[i].first)) {
      symbols_[i].push_back(GetCharSymbol(c));
    }
  }

  // Makes all bigram symbols.
  for (size_t sid = 0; sid < symbols_.size(); ++sid) {
    for (size_t i = 1; i < symbols_[sid].size(); ++i) {
      AddNewPair(sid, i - 1, i);
    }
  }

  for (Symbol* symbol : pending_queue_) {
    symbol->pending = false;
    ComputeFreq(symbol);
    pq_.push({symbol->freq, symbol});
  }
  pending_queue_.clear();

  const int vocab_size =
      trainer_spec_.vocab_size() - meta_pieces_.size() - required_chars_.size();
  RET_CHECK_GE(vocab_size, 0);

  RET_CHECK(final_pieces_.empty());

  // LEGACY protected_pieces_file: pre-load protected strings into
  // final_pieces_ before the merge loop, so BPE learns new merges alongside
  // them. Single characters are skipped because they are already in
  // required_chars_. This is protected-vocabulary fresh training, NOT BPE
  // continuation: no external ID ABI and no merge provenance for these pieces.
  const std::string& protected_file =
      trainer_spec_.GetExtension(::sentencepiece::protected_pieces_file);
  if (!protected_file.empty()) {
    absl::flat_hash_set<std::string> required_strs;
    for (const auto& w : required_chars_) {
      required_strs.insert(string_util::UnicodeCharToUTF8(w.first));
    }
    auto input = filesystem::NewReadableFile(protected_file);
    RET_CHECK(input->status().ok())
        << "Cannot open protected_pieces_file: " << protected_file;
    std::string line;
    while (input->ReadLine(&line)) {
      if (!line.empty() && !required_strs.count(line)) {
        final_pieces_.emplace_back(line,
                                   -static_cast<float>(final_pieces_.size()));
      }
    }
    LOG(INFO) << "Loaded " << final_pieces_.size()
              << " protected pieces for BPE";
  }

  ABSL_RETURN_IF_ERROR(ApplySeedMerges());

  // The pair each learned piece was merged from, in learned order. Upstream
  // discards this at Save(), forcing every consumer that needs a merge list to
  // guess the split back from the piece string, which is lossy.
  std::vector<std::pair<std::string, std::string>> merges;

  // We may see duplicated pieces that are extracted with different path.
  // In real segmentation phase, we can consider them as one symbol.
  // e.g., "aaa" => "aa" + "a" or "a" + "aa".
  absl::flat_hash_set<std::string> dup;

  // Pre-populate dup with protected pieces so BPE does not re-add them.
  for (const auto& p : final_pieces_) {
    dup.insert(p.first);
  }

  // Main loop. final_pieces_ may already carry protected pieces, so (unlike
  // upstream) it is not required to be empty here.
  while (final_pieces_.size() < static_cast<size_t>(vocab_size)) {
    Symbol* best_symbol = nullptr;
    while (!pq_.empty()) {
      QueueEntry entry = pq_.top();
      Symbol* symbol = entry.symbol;
      if (!symbol->active) {
        pq_.pop();
        continue;
      }
      if (entry.freq != symbol->freq) {
        pq_.pop();
        continue;
      }
      if (symbol->needs_recomputation) {
        pq_.pop();
        ComputeFreq(symbol);
        pq_.push({symbol->freq, symbol});
        continue;
      }
      best_symbol = symbol;
      pq_.pop();
      break;
    }

    if (best_symbol == nullptr) {
      LOG(WARNING) << "No valid symbol found";
      break;
    }

    if (!dup.insert(best_symbol->ToString()).second) {
      // Removes best_symbol so it is not selected again.
      symbols_cache_.erase(best_symbol->fp);
      best_symbol->active = false;
      continue;
    }

    // Stores the best_symbol in the final output.
    final_pieces_.emplace_back(best_symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));

    // Both halves are current segments, so each is either a required char or
    // an already-accepted piece - never something absent from the vocabulary.
    if (best_symbol->IsBigram()) {
      merges.emplace_back(best_symbol->left->ToString(),
                          best_symbol->right->ToString());
    }

    if (final_pieces_.size() % 20 == 0) {
      LOG(INFO) << "Added: freq=" << best_symbol->freq
                << " size=" << final_pieces_.size()
                << " all=" << symbols_cache_.size() << " active=" << pq_.size()
                << " piece=" << best_symbol->ToString();
    }

    ABSL_RETURN_IF_ERROR(AcceptSymbol(best_symbol));

    for (Symbol* symbol : pending_queue_) {
      symbol->pending = false;
      if (symbol->active) {
        ComputeFreq(symbol);
        pq_.push({symbol->freq, symbol});
      }
    }
    pending_queue_.clear();
  }  // end of main loop

  // Adds required_chars_
  for (const auto& w : Sorted(required_chars_)) {
    const Symbol* symbol = GetCharSymbol(w.first);
    final_pieces_.emplace_back(symbol->ToString(),
                               -static_cast<float>(final_pieces_.size()));
  }

  ABSL_RETURN_IF_ERROR(SaveMerges(merges));

  allocated_.clear();
  symbols_cache_.clear();

  return Save();
}

absl::Status Trainer::ApplySeedMerges() {
  // LEGACY seed_merges_file. Replays the seed tokenizer's merges onto the
  // corpus, in rank order, so that learning starts from the segmentation the
  // seed actually produces. Nothing is added to the vocabulary here - the
  // pieces are already in it via protected_pieces_file; what changes is only
  // how the corpus is segmented.
  //
  // A merge whose halves are not currently present, or whose pair does not
  // occur in this corpus, simply does not fire. That is not an error: a seed
  // trained on other data will always carry merges this corpus cannot
  // exercise, and the piece keeps its vocabulary slot regardless.
  //
  // This is the legacy approximation of BPE continuation. --expansion_spec
  // supersedes it with an authoritative ID ABI and exact provenance.
  const std::string& filename =
      trainer_spec_.GetExtension(::sentencepiece::seed_merges_file);
  if (filename.empty()) return absl::OkStatus();

  auto input = filesystem::NewReadableFile(filename);
  RET_CHECK(input->status().ok())
      << "Cannot open seed_merges_file: " << filename;

  // The symbol currently representing each piece string. Seeded with the
  // characters, extended by each merge that fires.
  absl::flat_hash_map<std::string, Symbol*> by_string;
  for (const auto& w : required_chars_) {
    Symbol* symbol = GetCharSymbol(w.first);
    by_string[symbol->ToString()] = symbol;
  }

  std::string line;
  int applied = 0, skipped = 0;
  while (input->ReadLine(&line)) {
    const auto tab = line.find('\t');
    if (tab == std::string::npos) continue;
    const auto left = by_string.find(line.substr(0, tab));
    const auto right = by_string.find(line.substr(tab + 1));
    if (left == by_string.end() || right == by_string.end()) {
      ++skipped;
      continue;
    }
    Symbol* symbol = GetPairSymbol(left->second, right->second);
    if (symbol == nullptr || !symbol->active) {
      ++skipped;
      continue;
    }
    // Positions recorded at scan time go stale as earlier merges consume them.
    // ComputeFreq drops the dead ones; without it AcceptSymbol walks into a
    // half-merged position and trips its RET_CHECK.
    symbol->needs_recomputation = true;
    ComputeFreq(symbol);
    if (symbol->freq == 0) {
      ++skipped;
      continue;
    }
    by_string[symbol->ToString()] = symbol;
    ABSL_RETURN_IF_ERROR(AcceptSymbol(symbol));
    ++applied;

    for (Symbol* pending : pending_queue_) {
      pending->pending = false;
      if (pending->active) {
        ComputeFreq(pending);
        pq_.push({pending->freq, pending});
      }
    }
    pending_queue_.clear();
  }

  LOG(INFO) << "Applied " << applied << " seed merges (" << skipped
            << " did not fire on this corpus)";
  return absl::OkStatus();
}

absl::Status Trainer::SaveMerges(
    const std::vector<std::pair<std::string, std::string>>& merges) const {
  // Written beside .model and .vocab, in learned order: one merge per line,
  // "left<TAB>right". Tab-separated because a piece may contain a space (the
  // whitespace marker is a piece character), which the usual space-separated
  // merges.txt cannot express.
  if (trainer_spec_.model_prefix().empty()) return absl::OkStatus();
  const std::string filename = trainer_spec_.model_prefix() + ".merges";
  LOG(INFO) << "Saving merges: " << filename;
  auto output = filesystem::NewWritableFile(filename);
  ABSL_RETURN_IF_ERROR(output->status());
  for (const auto& merge : merges) {
    RET_CHECK(output->WriteLine(merge.first + "\t" + merge.second));
  }
  return absl::OkStatus();
}

}  // namespace sentencepiece::bpe
