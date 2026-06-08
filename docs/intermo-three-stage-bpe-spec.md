# Three-Stage BPE Tokenization for intermo

## 1. Goal

Train a tokenizer for intermo music notation using three rounds of BPE (or unigram), where each round relaxes the pretokenization boundary to allow progressively larger multi-token merges:

- **Round 1 (EventPiece)**: Whitespace is a hard boundary. Learn subword merges within events.
- **Round 2 (MomentPiece)**: Only whitespace followed by digit or `|` is a hard boundary. Learn cross-event merges within moments.
- **Round 3 (BarPiece)**: Only whitespace followed by `|` is a hard boundary. Learn cross-moment merges within bars.

Each round inherits (freezes) all merges from prior rounds.

Target: 8192 total vocab.

## 2. intermo Format

Example: `|4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5`

Token types:
- **Barlines**: `|4/4k0`, `|3/4k-2` (always start with `|`)
- **Intervals**: `1/4`, `1/8`, `3/16` (always start with digit)
- **Pitches (onset)**: `C5`, `D#4`, `A-3` (uppercase letter). May have modifiers appended without whitespace: `C5tr` (trill), `C5.` (staccato), `C5>` (accent)
- **Pitches (offset)**: `c5`, `d#4`, `a-3` (lowercase letter)
- **Staff labels**: `PR:`, `PL:`, `Ob:` (end with `:`)
- **Slur markers**: `+slur` (slur start, begins with `+`), `-slur` (slur end, begins with `-`)
- **Skip edge markers**: `<|skip:on|>`, `<|skip:off|>` (begin with `<`)

Note on regex interactions: `+slur` and `-slur` start with `+`/`-`, and `<|skip:on|>` starts with `<`. None of these characters are `|` or digits, so they do NOT trigger MomentPiece or BarPiece boundaries. The `|` characters embedded inside `<|skip:on|>` are not preceded by whitespace, so they also do not trigger boundaries in the whitespace-aware split logic. Pitch modifier suffixes (e.g., `tr`, `.`, `>`) are part of the same whitespace-delimited token as the pitch, so they do not affect boundary logic.

### 2.1 Initial-Rest Edge Case

When a barline has a non-zero initial rest, the text_writer emits the interval immediately after the barline within the same whitespace-delimited context: `|4/4k0 1/8 PR: C5 ...`. Here the `1/8` follows the barline and starts a new MomentPiece boundary (since it starts with a digit). This is correct behavior -- the initial rest interval IS a separate moment from the barline's time signature declaration.

### 2.2 Format Variants

The `im_midilike` format variant uses `@` for velocity annotations (e.g., `C4@88`). The `@` character is neither `|` nor a digit, so it does not affect boundary logic. If the tokenizer will handle `im_midilike` data, no additional regex modifications are needed -- the `@` and subsequent digits are part of the same whitespace-delimited token.

## 3. Three Boundary Levels

### 3.1 EventPiece (Round 1 boundary)

Split at every whitespace. Each whitespace-delimited token is a pretokenization unit.

```
Input:  |4/4k0 PR: C5 1/4 PL: A-3 C4 F4
Units:  [|4/4k0] [PR:] [C5] [1/4] [PL:] [A-3] [C4] [F4]
```

BPE operates within each unit.

**Token representation differs by implementation:**
- **SentencePiece (Implementations B/C)**: Tokens include leading `_` (U+2581): `_|4/4k0`, `_PR:`, `_C5`, `_1/4`, etc. The whitespace marker is prefixed to the following word.
- **HF-SuperBPE (Implementation A)**: With `Split(regex="\S+", behavior="isolated") + ByteLevel(add_prefix_space=False, use_regex=False)`, whitespace becomes standalone `G` (U+0120) tokens between words. BPE units are `[|4/4k0] [G] [PR:] [G] [C5]`, NOT `[G|4/4k0] [GPR:] [GC5]`. The space is a separate isolated piece, not a prefix on the next token.

If leading-space-on-token behavior is desired for HF-SuperBPE (to match SentencePiece semantics), change the regex from `\S+` to `\s?\S+` or use a GPT-2-style regex that captures optional leading space with each word.

### 3.2 MomentPiece (Round 2 boundary)

Split only before whitespace+digit or whitespace+`|`. Events within a moment stay together.

**Simple regex approximation** (used by build_corpus.py and HF-SuperBPE):
```python
re.split(r'\s+(?=[|\d])', text)
```

```
Input:  |4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5
Units:  [|4/4k0 PR: C5] [1/4 PL: A-3 C4 F4] [1/8 PR: c5 D-5]
```

BPE can now merge across whitespace within a moment:
- `_PR:_C5`, `_PL:_A-3_C4`, etc.

**C++ consumed_by_barline edge case (SentencePiece pretokenizer only):**

The C++ pretokenizer in `pretokenizer_for_training.cc` implements additional stateful logic that cannot be replicated by a stateless regex. When an interval (digit-prefixed token) immediately precedes a barline (pipe-prefixed token), the interval does NOT create a new boundary -- it stays with the previous moment. The barline then starts a fresh segment.

Example: `... PR: C5 1/4 |3/4k0 D5` produces `[... PR: C5 1/4] [|3/4k0 D5]`

The interval `1/4` is absorbed BACKWARD into the previous segment, not grouped forward with the barline. This is the opposite of what a naive regex split would produce (which would be `[...PR: C5] [1/4] [|3/4k0 D5]` -- three segments).

Additional sub-case: if the consumed interval would be the FIRST token (the current segment buffer is empty), it is dropped entirely rather than creating a zero-content segment.

**Divergence between implementations:** The Python regex `re.split(r'\s+(?=[|\d])', text)` is a simpler approximation that diverges from the C++ in two cases:
1. Interval-before-barline is NOT merged backward (the regex has no lookahead for what follows)
2. Leading consumed intervals are NOT dropped

This difference is acceptable for BPE training because the mismatch is rare (only occurs at measure boundaries where the final event duration happens to be stated just before the barline). The training corpus quality is not significantly affected.

### 3.3 BarPiece (Round 3 boundary)

Split only before whitespace+`|`. Entire measures stay together.

```
Input:  |4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5 |3/4k0 ...
Units:  [|4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5] [|3/4k0 ...]
```

BPE can merge across moments within a bar:
- `_1/4_PR:_C5_c5`, `_1/8_PL:_a-3_c4`, etc.

**C++ BarPiece behavior vs regex:** The C++ implementation (`std::getline(iss, segment, '|')`) STRIPS the leading `|` from each segment. The resulting segments are `[4/4k0 PR: C5 1/4 ...]`, `[4/4k0 D5 1/4 ...]` -- without the pipe character. The spec's regex `\|[^|]*` PRESERVES the leading `|`. For BPE training, preserving the `|` is preferable since it retains information (the barline token identity). The intended behavior for this spec is to PRESERVE the `|` as part of the segment (matching the regex, not the C++ getline behavior). The C++ BarPiece output format is NOT the reference for this spec.

## 4. Implementation A: HuggingFace-SuperBPE

### 4.1 Architecture

Uses superbpe's existing HF tokenizers Rust fork (`tokenizers_superbpe/`). Three scripts instead of two.

### 4.2 Pretokenizer Setup

From `utils.py`, the pretokenizer is:
```python
pretokenizers = [
    Split(pattern=Regex(regex_string), behavior="isolated", invert=False),
    ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
]
```

`behavior="isolated"` means: each regex match becomes a separate pretokenization unit. Text between matches is also a separate unit. BPE can only merge within one unit.

ByteLevel is kept (per user requirement) -- it maps bytes to the GPT-2 unicode charset. Space (0x20) becomes `G` (U+0120).

### 4.3 Three Regex Patterns

**Round 1 (EventPiece)**: Match each whitespace-delimited token.
```
\S+
```
Each non-whitespace run is isolated. Whitespace between runs is a separate (discarded) piece. Note: with `behavior="isolated"`, the whitespace between matches becomes its own unit containing `G` byte-level encoded characters. BPE operates independently within each unit. Stage 1 has NO constraints beyond the whitespace split -- no `:G` rejection, no word count limit. This is correct because with the `\S+` regex, no `G` characters can appear inside tokens.

**Round 2 (MomentPiece)**: Match each moment (interval/barline + following events).
```
[|\d]\S*(?:\s+(?![|\d])\S+)*
```
Breakdown:
- `[|\d]\S*` -- starts with `|` or digit, followed by rest of that token
- `(?:\s+(?![|\d])\S+)*` -- zero or more: whitespace + non-boundary-starting token

Example matches on `|4/4k0 PR: C5 1/4 PL: A-3`:
- Match 1: `|4/4k0 PR: C5` (starts with `|`, then `PR:` and `C5` don't start with `|`/digit)
- Match 2: `1/4 PL: A-3` (starts with `1`, then `PL:` and `A-3` don't start with `|`/digit)

**Important limitation:** This regex CANNOT replicate the C++ consumed_by_barline logic. The regex is stateless -- `1/4` always matches `[|\d]\S*` and starts a new unit regardless of what follows. There is no way to express "match digit-start UNLESS next token starts with |" in a single Split regex. The regex produces 3 units `[|4/4k0 PR: C5] [1/4] [|3/4k0 D5]` while the C++ would produce 2 units `[|4/4k0 PR: C5 1/4] [|3/4k0 D5]`.

**Mitigation options (choose one):**
- **(a) Pre-process the corpus** (RECOMMENDED): Before applying the regex, replace ` <digit>/<digit> |` patterns with a placeholder (e.g., replace the space before the interval with a non-breaking marker), split with the regex, then restore. Concretely: `re.sub(r'(\d+/\d+)\s+(\|)', r'\1\x00\2', text)` then split, then restore `\x00` to space.
- **(b) Accept the divergence**: Document that the HF-SuperBPE implementation intentionally does NOT replicate consumed_by_barline. The tradeoff is that occasional intervals may form their own tiny BPE unit rather than being absorbed into the previous moment. This has minimal impact on training quality.
- **(c) Custom PreTokenizer**: Write a custom PreTokenizer in Rust/Python that implements the stateful lookahead logic.

For this spec, option (b) is the default unless benchmarking shows quality degradation.

**Interaction with `<|skip:on|>` tokens:** The `<|skip:on|>` token contains `|` characters, but since the full token is `<|skip:on|>` (no whitespace before the internal `|`), the regex `[|\d]\S*` would only match if `<|skip:on|>` were preceded by whitespace and started with `|`. Since it starts with `<`, it will be captured as a non-boundary-starting token in the `(?![|\d])\S+` part. No special handling needed.

**Round 3 (BarPiece)**: Match each bar (barline + everything until next barline).
```
\|\S*(?:\s+(?!\|)\S+)*
```
This handles the case where text doesn't start with `|` by leaving any prefix text as a separate unmatched unit (which `behavior="isolated"` handles).

Note: The `<|skip:on|>` token contains `|` but is a single `\S+` unit. The lookahead `(?!\|)` only fires after `\s+`, so internal `|` characters within a non-whitespace token do not trigger false boundaries.

### 4.4 Training Scripts

**Stage 1** (`train_eventpiece.sh`):
```bash
vocab_size=8192
num_inherit_merges=4000  # ~50% of final vocab for stage 1
regex_string="\S+"
python -m train_tokenizer \
    --output_dir $output_dir \
    --corpus_dir $corpus_dir \
    --vocab_size $vocab_size \
    --regex_string "$regex_string"
```

Note: Stage 1 uses `do_train_original` which has NO constraints -- no `:G` rejection, no word count limit. This is correct for Stage 1 with the `\S+` regex (no `G` can appear in matched tokens since `\S` excludes spaces).

**Stage 2** (`extend_momentpiece.sh`):
```bash
orig_tokenizer_dir=...  # stage 1 output
num_inherit_merges=4000
vocab_size=8192  # total target (will add ~2400 new merges)
regex_string="[|\d]\S*(?:\s+(?![|\d])\S+)*"

mkdir -p $output_dir
head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt
cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size $vocab_size \
    --regex_string "$regex_string"
```

**Stage 3** (`extend_barpiece.sh`):
```bash
orig_tokenizer_dir=...  # stage 2 output
num_inherit_merges=6400  # all merges from stages 1+2
vocab_size=8192
regex_string="\|\S*(?:\s+(?!\|)\S+)*"

mkdir -p $output_dir
head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt
cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size $vocab_size \
    --regex_string "$regex_string"
```

### 4.5 Rust Modifications to `do_train_extend()`

**Current behavior** (superbpe for NLP):
- `:G` constraint: reject tokens containing `:G`
- 4-word constraint: `split("G").filter(!empty).count() > 4` -> reject

**Required changes for intermo**:
- Remove `:G` constraint (intermo has `PR:` followed by space legitimately)
- Replace 4-word constraint with `max_N_events` constraint:
  - Count events as `split("G").filter(!empty).count()`
  - Stage 2: max 6 events (one moment shouldn't span too many events)
  - Stage 3: max 20 events (one bar can have many events)
  - Pass as a configurable parameter: add `--max_word_count <N>` to the train_tokenizer CLI, thread through to `do_train_extend()` in trainer.rs
- Optional: Add intermo-specific dropout that only drops merges bridging across the CURRENT stage's boundary (not needed for correctness, but may improve quality)

**Inherited merge handling in `do_train_extend()`**: When an inherited merge pair is not found in the new corpus (i.e., the pair's constituent symbols never appear adjacently given the new pretokenization boundaries), the merge is silently skipped (trainer.rs lines 570-588). This is unlikely for intermo (Stage 2 units are strictly supersets of Stage 1 units at the character level) but should be noted as a potential issue if the corpus changes between stages.

### 4.6 Vocab Budget and Inheritance

Stage 1 trains with `vocab_size=8192`, which produces approximately 7936 merges (8192 minus ~256 byte-level base vocab). Only `num_inherit_merges=4000` of these are inherited by Stage 2 (via `head -n 4000 merges.txt`). This means more than half of Stage 1 merges are discarded.

Justification: The first N merges are the most frequent and universal subword combinations. Later merges become increasingly corpus-specific and less likely to generalize well to the wider-boundary regime. Alternatively, if this trimming seems wasteful, set Stage 1 `vocab_size` to approximately 4256 (4000 merges + 256 base vocab) so that all Stage 1 merges are inherited.

### 4.7 Corpus

One score per line (full intermo text). The regex handles the pretokenization. `max_sentence_length` is not an issue for HF tokenizers (no such limit).

## 5. Implementation B: SentencePiece-BPE

### 5.1 Approach

Add flags to sentencepiece:
- `split_by_interval` (bool, default false): boundary at whitespace + digit or whitespace + `|`. **Already implemented** in the current fork (proto field 55, SplitIntoWords, IsValidSentencePiece, unigram seed generation).
- `split_by_barline` (bool, default false): boundary at whitespace + `|` only. **NOT YET IMPLEMENTED** -- must be added (see Section 8).
- `initial_merges_file` (string, default ""): path to file of merge pairs to inherit. **NOT YET IMPLEMENTED**.

These affect `SplitIntoWords()`, `IsValidSentencePiece()`, and the BPE trainer's boundary mechanism.

**Mutual exclusivity:** When `split_by_interval=true` or `split_by_barline=true`, `split_by_whitespace` should be set to `false` (the proto comment at field 55 says "split_by_whitespace is effectively false"). However, the current code does NOT enforce mutual exclusivity automatically. The user must explicitly pass `--split_by_whitespace=false` alongside `--split_by_interval=true` or `--split_by_barline=true`. A future improvement could add validation logic that errors if both `split_by_whitespace=true` and `split_by_interval=true` are set.

### 5.2 `SplitIntoWords()` Changes

File: `src/model_interface.cc`, function at line 159.

**Current implementation (split_by_interval=true):**
- The function takes a `bool split_by_interval` parameter
- Uses a `is_interval_boundary()` lambda that checks if the character after whitespace is `0-9` or `|`
- When `split_by_interval` is true AND a whitespace symbol is found, only creates a boundary if `is_interval_boundary()` returns true
- Otherwise, whitespace is treated as a non-boundary character within the unit

**Needed for split_by_barline=true:**
- Add a `bool split_by_barline` parameter to the function signature
- Add a `is_barline_boundary()` lambda that checks only for `|` after whitespace (not digits)
- When `split_by_barline` is true, use the barline-only boundary check

The `SplitSentencesByWhitespace()` function (trainer_interface.cc line 610-623) calls `SplitIntoWords()` and must pass the `split_by_barline` flag. Currently it only passes `split_by_interval`.

### 5.3 `IsValidSentencePiece()` Changes

File: `src/trainer_interface.cc`, function at line 215.

**Current implementation for split_by_interval (lines 267-273):**
```cpp
if (trainer_spec_.split_by_interval()) {
    if (pos > 0 && pos + 1 < sentencepiece.size()) {
        const char32 next_c = sentencepiece[pos + 1];
        if ((next_c >= '0' && next_c <= '9') || next_c == '|') {
            return false;
        }
    }
}
```

This rejects pieces containing whitespace (U+2581) followed by digit/`|` at INTERNAL positions. The conditions are:
- `pos > 0`: whitespace at the start of a piece is allowed (it's the leading space marker)
- `pos + 1 < sentencepiece.size()`: whitespace at the very end is allowed (there's no following character to form a boundary). This is correct because a trailing whitespace alone cannot indicate a boundary without knowing what follows.

**Edge case documentation:** If `treat_whitespace_as_suffix` were combined with `split_by_interval`, a piece like `C5_` (trailing whitespace) would be allowed by this check since `pos + 1 == sentencepiece.size()`. For intermo, `treat_whitespace_as_suffix` is NOT used, so this is not a concern. If it were ever combined, additional logic would be needed to reject suffix-whitespace pieces that could form boundaries with the next piece.

**Needed for split_by_barline=true:** Add a parallel check:
```cpp
if (trainer_spec_.split_by_barline()) {
    if (pos > 0 && pos + 1 < sentencepiece.size()) {
        const char32 next_c = sentencepiece[pos + 1];
        if (next_c == '|') {
            return false;
        }
    }
}
```

### 5.4 BPE Boundary Mechanism

**How boundaries work in the BPE trainer:**

The BPE trainer's boundary enforcement does NOT use `kUPPBoundaryStr` (TAB character) for the whitespace-split path. The mechanism is:

1. `SplitSentencesByWhitespace()` (line 610-623) calls `SplitIntoWords()` to split each sentence at boundaries, then REPLACES the `sentences_` vector with individual words (each word becomes a separate sentence entry with its frequency).
2. The BPE trainer (lines 257-263) initializes a separate symbol array per sentence.
3. Bigrams are created WITHIN each sentence only (lines 265-269). Since each "sentence" is now a single word/moment/bar, bigrams cannot cross boundaries.

The `kUPPBoundaryStr` path is ONLY used when `pretokenizer` or `pretokenization_delimiter` is set (lines 243-254). It replaces delimiter characters with TAB within each sentence string, and the BPE symbol initialization treats TAB as an uncrossable barrier. This is a DIFFERENT mechanism from `SplitSentencesByWhitespace()`.

**BUG IN CURRENT CODE:** Line 235 of `bpe_model_trainer.cc` only checks `split_by_whitespace()`:
```cpp
if (trainer_spec_.split_by_whitespace()) {
    SplitSentencesByWhitespace();
}
```

When running with `--split_by_interval=true --split_by_whitespace=false`, `SplitSentencesByWhitespace()` is NEVER called. The interval-aware `SplitIntoWords` logic inside it never executes during BPE training. BPE operates on entire lines as single sentences with no boundaries.

**Required fix (both code paths):**

In `Train()` at line 235:
```cpp
if (trainer_spec_.split_by_whitespace() ||
    trainer_spec_.split_by_interval() ||
    trainer_spec_.split_by_barline()) {
    SplitSentencesByWhitespace();
}
```

In `TrainFast()` at line 352 (under `#ifdef SPM_NLCODEC_BPE`):
```cpp
if (trainer_spec_.split_by_whitespace() ||
    trainer_spec_.split_by_interval() ||
    trainer_spec_.split_by_barline()) {
    SplitSentencesByWhitespace();
}
```

Both code paths have the identical condition and need the same fix.

### 5.5 Merge Inheritance (BPE)

SentencePiece currently has NO mechanism to inherit merges from a prior training run.

**Proposed feature:** Add `--initial_merges_file` flag. The BPE trainer loads these merges, applies them to the corpus FIRST (like superbpe's `do_train_extend()`), then continues greedy BPE for new merges.

Proto field:
```proto
optional string initial_merges_file = 57 [default = ""];
```

**Merge file format:**
One merge pair per line, TAB-separated:
```
left_piece\tright_piece
```
Same format as HuggingFace merges.txt (but with TAB instead of space separator, since pieces themselves may contain encoded spaces). Lines are ordered: first merge applied first. Example:
```
_|	4
_1	/
4/	4
_|4/4	k0
_P	R:
```

**Implementation complexity -- force-applying merges:**

The BPE trainer's internal representation uses `Symbol*` objects stored in per-sentence vectors (`symbols_[sid][i]`). Each Symbol has:
- `chars`: the Unicode character sequence
- `fp`: fingerprint (hash)
- `left`/`right`: child symbols (for tree structure)
- Positions tracked via `active_symbols_` and the bigram frequency system

Force-applying a merge requires:
1. For each loaded merge pair (left_piece, right_piece):
   a. Create the merged Symbol via `GetPairSymbol(left_sym, right_sym)`
   b. Scan ALL sentences to find adjacent symbol pairs matching (left, right)
   c. For each match, execute the merge (equivalent to `AcceptSymbol()` logic)
   d. Add the merged token to `final_pieces_`
2. After all inherited merges are applied, continue the normal greedy BPE loop for the remaining vocab budget

**Pseudocode for force-applying one merge:**
```cpp
// For each sentence sid:
for (size_t sid = 0; sid < symbols_.size(); ++sid) {
    for (size_t i = 0; i + 1 < symbols_[sid].size(); ++i) {
        if (symbols_[sid][i] == nullptr) continue;  // already merged away
        // Find next non-null symbol
        size_t j = i + 1;
        while (j < symbols_[sid].size() && symbols_[sid][j] == nullptr) ++j;
        if (j >= symbols_[sid].size()) break;
        
        if (symbols_[sid][i]->fp == left_fp && symbols_[sid][j]->fp == right_fp) {
            // Merge: replace symbols_[sid][i] with merged symbol,
            // set symbols_[sid][j] = nullptr
            symbols_[sid][i] = merged_symbol;
            symbols_[sid][j] = nullptr;
        }
    }
}
```

**Simpler alternative:** Instead of using the full `active_symbols_`/positions/`AcceptSymbol()` machinery, iterate over sentences and manually merge adjacent symbols matching each merge pair. This avoids the complexity of populating frequency tables for inherited merges. After all inherited merges are applied, rebuild the bigram frequency structures from the current symbol state before entering the greedy BPE loop.

**Edge cases:**
- An inherited merge pair may not be found in the new corpus (the pair's constituent symbols never appear adjacently given the new pretokenization boundaries). This merge is silently skipped. This is unlikely for intermo but should be logged as a warning.
- Inherited merges from a narrower boundary (Stage 1) may produce symbols that span what would be a boundary in the current stage. This is expected and correct -- the inherited merge is a frozen fact from the prior stage.
- **Vocab budget accounting:** `vocab_size` includes inherited tokens. If inheriting 4000 merges and `vocab_size=8192`, the remaining budget for new merges is approximately `8192 - 4000 - meta_pieces_.size() - required_chars_.size()`.

**Guard clause:** Add validation similar to `seed_sentencepieces_file`:
```cpp
if (!trainer_spec.initial_merges_file().empty()) {
    RET_CHECK(trainer_spec.model_type() == TrainerSpec::BPE)
        << "initial_merges_file is only supported for BPE model.";
}
```

### 5.6 Interaction with Normalization

The normalization step (trainer_interface.cc lines 436-469) replaces meta_pieces with `kUPPBoundaryStr`. If a pretokenizer is ALSO set alongside `split_by_interval`, both mechanisms would be active:
- Meta pieces are replaced with TAB characters in the sentence strings
- `SplitSentencesByWhitespace()` splits at interval/barline boundaries

These do not conflict because `kUPPBoundaryStr` (TAB) is handled during symbol initialization (BPE treats TAB as uncrossable), while `SplitSentencesByWhitespace()` splits sentences into separate entries before symbol initialization. However, for clarity, intermo training should NOT set a pretokenizer or `pretokenization_delimiter` when using `split_by_interval`/`split_by_barline`. The interval/barline split logic is sufficient.

### 5.7 Inference Behavior

**BPE inference:** The BPE model at inference time (bpe_model.cc) does NOT call `SplitIntoWords()`. It operates on the full normalized string and replays the merge table greedily left-to-right. Boundary enforcement is implicit:
- No merge was ever learned that crosses an interval/barline boundary (because training only created bigrams within split words)
- Therefore the merge table inherently respects boundaries for any input that was in the training distribution

For unseen combinations at inference time, the greedy merge replay will still respect boundaries because no crossing-merge exists in the table. The worst case is that an unseen combination is segmented into smaller pieces (character-level), which is correct behavior.

**Unigram inference:** The Viterbi lattice only contains pieces from the trained vocabulary. Since no pieces crossing boundaries were ever added to the vocabulary (due to `IsValidSentencePiece()` rejection), the lattice naturally respects boundaries.

**No explicit inference-time boundary enforcement is needed** for either BPE or unigram.

### 5.8 Corpus

Feed full scores or full bars as lines. Set `--max_sentence_length=500000` to avoid filtering.

With `split_by_interval=true`, sentencepiece internally splits each line at moment boundaries via `SplitSentencesByWhitespace()`, correctly counting frequencies of each moment across the corpus.

## 6. Implementation C: SentencePiece-Unigram

### 6.1 The Convergence Problem

Without protection/freezing of stage-1 pieces, EM will converge to approximately the same point regardless of initialization. Multi-stage training only matters if prior-stage pieces are constrained.

### 6.2 Proposed Solution: Protected Vocab

Add field:
```proto
optional string protected_pieces_file = 58 [default = ""];
```

**File format:** One piece per line (the string representation, e.g., `_PR:_C5`). No scores -- scores are determined by EM. Lines starting with `#` are comments.

When set:
- Load pieces from file
- These pieces are ALWAYS included in the model -- never pruned by EM
- They occupy fixed vocab slots, leaving remaining budget for new pieces
- EM can adjust their probabilities but cannot remove them

**Required modifications in `unigram_model_trainer.cc`:**

**(1) In `MakeSeedSentencePieces()` (line ~305):** Union the seed set with protected pieces. After the suffix-array-based seed generation, inject all protected pieces that are not already in the seed:
```cpp
// After normal seed generation:
if (!trainer_spec_.protected_pieces_file().empty()) {
    auto protected_pieces = LoadProtectedPieces(trainer_spec_.protected_pieces_file());
    for (const auto& piece : protected_pieces) {
        seed_sentencepieces.emplace_back(piece, /* initial_freq */ 0.0);
    }
}
```

**(2) In `RunMStep()` (lines 417-424):** This is a CRITICAL location that the naive approach would miss. `RunMStep()` filters out pieces with expected frequency below `kExpectedFrequencyThreshold` (0.5). If a protected piece has very low expected usage in the E-step, it would be silently dropped from `new_sentencepieces`:
```cpp
for (size_t i = 0; i < expected.size(); ++i) {
    const float freq = expected[i];
    constexpr float kExpectedFrequencyThreshold = 0.5;
    if (freq < kExpectedFrequencyThreshold && !is_protected_[i]) {
        continue;  // Skip low-freq non-protected pieces
    }
    new_sentencepieces.emplace_back(sentencepieces[i].first, freq);
    sum += freq;
}
```

**(3) In `PruneSentencePieces()`:** Mark protected piece indices and skip them in the pruning candidate list. The pruning loop removes pieces whose removal has the least impact on likelihood -- protected pieces must never be candidates:
```cpp
// Build protected set
absl::flat_hash_set<std::string> protected_set(protected_pieces_.begin(), protected_pieces_.end());

// In the pruning loop:
for (size_t i = 0; i < candidates.size(); ++i) {
    if (protected_set.contains(candidates[i].first)) {
        // Never prune: assign maximum loss so it's never selected for removal
        candidates[i].second = std::numeric_limits<float>::max();
    }
}
```

**(4) In `FinalizeSentencePieces()`:** Reserve vocab slots for protected pieces FIRST, then fill remaining slots from EM-selected pieces:
```cpp
// Always include protected pieces
for (const auto& piece : protected_pieces_) {
    final_pieces.emplace_back(piece, model.GetScore(piece));
}
// Fill remaining budget from EM-ranked pieces (excluding already-added protected ones)
size_t remaining = vocab_size - final_pieces.size();
for (const auto& piece : em_ranked_pieces) {
    if (remaining == 0) break;
    if (!protected_set.contains(piece.first)) {
        final_pieces.emplace_back(piece);
        --remaining;
    }
}
```

### 6.3 Three-Stage Unigram Training

**Stage 1**: Train with `split_by_whitespace=true` (EventPiece boundaries).
- Produces vocab V1 (~4000 pieces): character n-grams within events
- Example pieces: `_|4/4k0`, `_PR:`, `_C5`, `_1/`, `4`, `_A-3`

**Stage 2**: Train with `split_by_interval=true` (MomentPiece boundaries).
- Load V1 as protected vocab
- Seed generation now finds cross-whitespace candidates within moments
- EM allocates probability to both protected pieces and new cross-event pieces
- Produces V2 (~6400 pieces): V1 + new moment-level pieces like `_PR:_C5`

**Stage 3**: Train with `split_by_barline=true` (BarPiece boundaries).
- Load V2 as protected vocab
- Seed generation finds cross-moment candidates within bars
- Produces V3 (8192 pieces): V2 + new bar-level pieces

### 6.4 Seed Generation Changes

In `MakeSeedSentencePieces()` (line 305-312):
```cpp
auto split_into_pieces = [&](absl::string_view w) -> std::vector<absl::string_view> {
    if (trainer_spec_.split_by_whitespace() || 
        trainer_spec_.split_by_interval() ||
        trainer_spec_.split_by_barline()) {
        return SplitIntoWords(w, trainer_spec_.treat_whitespace_as_suffix(),
                              trainer_spec_.allow_whitespace_only_pieces(),
                              trainer_spec_.split_by_interval(),
                              trainer_spec_.split_by_barline());
    }
    return {w};
};
```

When `split_by_interval=true`: suffix array candidates are split at moment boundaries only. Cross-whitespace candidates within moments are valid seeds.

### 6.5 Corpus

Same as BPE: full scores or bars as lines, `max_sentence_length` set high.

## 7. Vocab Budget

For 8192 total:
- Round 1: 4000 merges/pieces (EventPiece level)
- Round 2: 2400 additional (MomentPiece level)
- Round 3: 1792 additional (BarPiece level)

These are initial estimates. The actual split should be tuned based on encoding efficiency (bytes-per-token) on a held-out set.

## 8. Files to Modify

### HF-SuperBPE:
- `superbpe/scripts/` -- add three new training scripts
- `superbpe/tokenizers_superbpe/tokenizers/src/models/bpe/trainer.rs` -- remove `:G` constraint, adjust multiword limit to configurable `max_word_count` parameter

### SentencePiece:

#### 8.1 Proto and Builtin_pb Changes

**`src/sentencepiece_model.proto`:**
- Update comment `// Next id: 55` to `// Next id: 59`
- `split_by_interval = 55` already exists
- Add `optional bool split_by_barline = 56 [default = false];` with comment explaining barline-only boundary
- Add `optional string initial_merges_file = 57 [default = ""];` with comment
- Add `optional string protected_pieces_file = 58 [default = ""];` with comment

**`src/builtin_pb/sentencepiece_model.pb.h` and `.pb.cc` -- Manual Patching:**

The builtin_pb files are pre-generated and checked in. They are NOT regenerated by protoc during the normal CMake build. Adding new proto fields requires manually patching these files.

**Procedure (copy the pattern from `split_by_interval` field 55):**

For each new bool field (e.g., `split_by_barline`), add to `.pb.h`:
1. Comment block: `// optional bool split_by_barline = 56 [default = false];`
2. Method declarations: `has_split_by_barline()`, `_internal_has_split_by_barline()`, `clear_split_by_barline()`, `split_by_barline()`, `set_split_by_barline(bool value)`, `_internal_split_by_barline()`, `_internal_set_split_by_barline(bool value)`
3. Member variable: `bool split_by_barline_;` in the class fields section
4. `_has_bits_` index: assign the next available bit index
5. Inline accessor implementations (at bottom of .pb.h)

For each new string field (e.g., `initial_merges_file`), add:
1. Method declarations: `has_initial_merges_file()`, `clear_initial_merges_file()`, `initial_merges_file()`, `set_initial_merges_file(...)`, `mutable_initial_merges_file()`
2. Member variable: `std::string initial_merges_file_;`
3. Default value initialization
4. Inline accessor implementations

In `.pb.cc`, add for each field:
1. `set_has_X()` function in `_Internal` class
2. Default initialization in constructor and `Clear()` 
3. Wire format deserialization in `_InternalParse()` (tag number, wire type, read logic)
4. Serialization in `_InternalSerialize()` (write field if has_bit set)
5. `ByteSizeLong()` contribution

**Alternative (one-time protoc generation):**
```bash
protoc --cpp_out=src/builtin_pb/ src/sentencepiece_model.proto
```
This requires matching the protoc version used to generate the existing files. Check with `grep "protobuf" src/builtin_pb/sentencepiece_model.pb.h | head -3` for the version string. WARNING: The builtin_pb files may have custom modifications beyond standard protoc output -- verify the diff carefully after regeneration.

The manual approach (copying the split_by_interval pattern) is SAFER and RECOMMENDED.

#### 8.2 Source File Changes

- **`src/model_interface.h`** -- Update `SplitIntoWords()` signature to add `bool split_by_barline = false` parameter
- **`src/model_interface.cc`** -- Implement `split_by_barline` logic in `SplitIntoWords()`: add `is_barline_boundary()` lambda that only checks for `|` after whitespace. When `split_by_barline` is true, use this instead of `is_interval_boundary()`
- **`src/trainer_interface.cc`**:
  - `IsValidSentencePiece()`: add `split_by_barline` check (reject whitespace+`|` internally)
  - `SplitSentencesByWhitespace()` (line 618): pass `split_by_barline` flag to `SplitIntoWords()`
- **`src/bpe_model_trainer.cc`**:
  - Line 235 (`Train()`): change condition to `if (split_by_whitespace() || split_by_interval() || split_by_barline())`
  - Line 352 (`TrainFast()`): same condition change
  - Add merge inheritance logic (load and force-apply merges from `initial_merges_file`)
- **`src/unigram_model_trainer.cc`**:
  - Add protected pieces logic in `RunMStep()`, `PruneSentencePieces()`, `FinalizeSentencePieces()`
  - Update `split_into_pieces` lambda in `MakeSeedSentencePieces()` to include `split_by_barline`
- **`src/spm_train_main.cc`** -- Register new flags:
  ```cpp
  ABSL_FLAG(bool, split_by_barline, false, "Split at whitespace followed by | only");
  ABSL_FLAG(std::string, initial_merges_file, "", "BPE merge pairs to inherit");
  ABSL_FLAG(std::string, protected_pieces_file, "", "Pieces to protect from pruning in unigram");
  // ... and SetTrainerSpecFromFlag() calls
  ```
- **`src/spec_parser.h`** -- Add entries:
  ```cpp
  PRINT_PARAM(split_by_barline);
  PRINT_PARAM(initial_merges_file);
  PRINT_PARAM(protected_pieces_file);
  // ...
  PARSE_BOOL(split_by_barline);
  PARSE_STRING(initial_merges_file);
  PARSE_STRING(protected_pieces_file);
  ```

## 9. Three-Stage SentencePiece Orchestration

### 9.1 Training Commands

**Stage 1 (EventPiece):**
```bash
spm_train \
    --model_type=bpe \
    --input=corpus.txt \
    --model_prefix=stage1 \
    --vocab_size=4256 \
    --split_by_whitespace=true \
    --max_sentence_length=500000 \
    --num_threads=8
```

Note: `vocab_size=4256` produces approximately 4000 merges (4256 minus meta_pieces and required_chars). This ensures all Stage 1 merges are usable for inheritance.

**Stage 2 (MomentPiece):**
```bash
# First, extract merges from Stage 1 model (see Section 9.2)
python extract_merges.py --model=stage1.model --output=stage1.merges

spm_train \
    --model_type=bpe \
    --input=corpus.txt \
    --model_prefix=stage2 \
    --vocab_size=6656 \
    --split_by_whitespace=false \
    --split_by_interval=true \
    --initial_merges_file=stage1.merges \
    --max_sentence_length=500000 \
    --num_threads=8
```

**Stage 3 (BarPiece):**
```bash
python extract_merges.py --model=stage2.model --output=stage2.merges

spm_train \
    --model_type=bpe \
    --input=corpus.txt \
    --model_prefix=stage3 \
    --vocab_size=8192 \
    --split_by_whitespace=false \
    --split_by_barline=true \
    --initial_merges_file=stage2.merges \
    --max_sentence_length=500000 \
    --num_threads=8
```

### 9.2 Merge Extraction Utility

A utility script is needed to extract merge pairs from a trained `.model` file into the `initial_merges_file` format:

```python
#!/usr/bin/env python3
"""Extract BPE merge pairs from a trained SentencePiece .model file."""
import argparse
import sentencepiece_model_pb2 as sp_model

def extract_merges(model_path, output_path):
    model = sp_model.ModelProto()
    with open(model_path, 'rb') as f:
        model.ParseFromString(f.read())
    
    # BPE pieces are stored in order of merge priority (after meta pieces)
    # Each piece's score is -merge_rank (negative, so first merge has highest score)
    meta_count = sum(1 for p in model.pieces if p.type != 1)  # type 1 = NORMAL
    
    with open(output_path, 'w') as f:
        for piece in model.pieces:
            if piece.type != 1:  # skip meta pieces (UNK, BOS, EOS, etc.)
                continue
            # For BPE, we need to find how this piece was formed
            # The piece string itself is the merged result
            # We need left+right components -- stored implicitly by merge order
            # Alternative: output piece strings and reconstruct during loading
            f.write(f"{piece.piece}\n")
    
    print(f"Extracted {meta_count} meta + {len(model.pieces) - meta_count} normal pieces")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    extract_merges(args.model, args.output)
```

**Important note:** SentencePiece's .model file stores the RESULT of each merge (the merged piece string) but does NOT store the left/right components explicitly (unlike HuggingFace's merges.txt which stores "left right" pairs). To reconstruct merge pairs:
- Option A: During BPE training, also output a merges.txt-style file alongside the .model (requires modifying `Save()` in bpe_model_trainer.cc)
- Option B: Reconstruct pairs by replaying: for each piece in merge order, find its two longest subpieces (from earlier merges) that concatenate to form it
- Option A is STRONGLY RECOMMENDED for simplicity.

### 9.3 Unigram Three-Stage Commands

**Stage 1:**
```bash
spm_train \
    --model_type=unigram \
    --input=corpus.txt \
    --model_prefix=stage1_uni \
    --vocab_size=4256 \
    --split_by_whitespace=true \
    --max_sentence_length=500000
```

**Stage 2:**
```bash
python extract_pieces.py --model=stage1_uni.model --output=stage1.pieces

spm_train \
    --model_type=unigram \
    --input=corpus.txt \
    --model_prefix=stage2_uni \
    --vocab_size=6656 \
    --split_by_whitespace=false \
    --split_by_interval=true \
    --protected_pieces_file=stage1.pieces \
    --max_sentence_length=500000
```

**Stage 3:**
```bash
python extract_pieces.py --model=stage2_uni.model --output=stage2.pieces

spm_train \
    --model_type=unigram \
    --input=corpus.txt \
    --model_prefix=stage3_uni \
    --vocab_size=8192 \
    --split_by_whitespace=false \
    --split_by_barline=true \
    --protected_pieces_file=stage2.pieces \
    --max_sentence_length=500000
```

## 10. Testing Strategy

### 10.1 Unit Tests

**File: `src/model_interface_test.cc`**

Add tests for `SplitIntoWords` with interval and barline modes:

```cpp
TEST(SplitIntoWordsTest, SplitByInterval) {
    // Basic interval split
    auto result = SplitIntoWords(
        "\xe2\x96\x81|4/4k0\xe2\x96\x81PR:\xe2\x96\x81C5\xe2\x96\x811/4\xe2\x96\x81PL:",
        /*treat_ws_as_suffix=*/false,
        /*allow_ws_only_pieces=*/false,
        /*split_by_interval=*/true);
    // Should split at _| and _1 but not at _P
    ASSERT_EQ(result.size(), 2);  // [_|4/4k0_PR:_C5] [_1/4_PL:]
}

TEST(SplitIntoWordsTest, SplitByBarline) {
    auto result = SplitIntoWords(
        "\xe2\x96\x81|4/4k0\xe2\x96\x81PR:\xe2\x96\x81C5\xe2\x96\x811/4\xe2\x96\x81|3/4k0",
        /*treat_ws_as_suffix=*/false,
        /*allow_ws_only_pieces=*/false,
        /*split_by_interval=*/false,
        /*split_by_barline=*/true);
    // Should only split at _| not at _1
    ASSERT_EQ(result.size(), 2);  // [_|4/4k0_PR:_C5_1/4] [_|3/4k0]
}
```

**File: `src/trainer_interface_test.cc`**

Add tests for `IsValidSentencePiece` with new modes:

```cpp
TEST(IsValidSentencePieceTest, IntervalMode) {
    TrainerSpec spec;
    spec.set_split_by_interval(true);
    // Piece with internal whitespace+digit should be rejected
    // _PR:_1 -> has _1 internally -> INVALID
    EXPECT_FALSE(IsValidSentencePiece(U"_PR:_1/4", spec));
    // _PR:_C5 -> has _C internally -> VALID (C is not digit/pipe)
    EXPECT_TRUE(IsValidSentencePiece(U"_PR:_C5", spec));
    // Leading whitespace is always OK
    EXPECT_TRUE(IsValidSentencePiece(U"_1/4", spec));
}

TEST(IsValidSentencePieceTest, BarlineMode) {
    TrainerSpec spec;
    spec.set_split_by_barline(true);
    // _PR:_|4/4 -> has _| internally -> INVALID
    EXPECT_FALSE(IsValidSentencePiece(U"_PR:_|4/4", spec));
    // _PR:_1/4 -> has _1 internally -> VALID (only | is boundary)
    EXPECT_TRUE(IsValidSentencePiece(U"_PR:_1/4", spec));
}
```

### 10.2 Sample Test Corpus

Create `test/intermo_sample.txt`:
```
|4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5 |4/4k0 PR: E5 1/2 PL: G3 B3 D4 1/2 PR: e5
|3/4k0 PR: C5 1/4 D5 1/4 E5 1/4 c5 d5 e5 |3/4k0 PR: F5 3/4 f5
|4/4k-2 PR: A4 1/8 B4 1/8 C5 1/4 PL: F3 A3 C4 1/2 PR: a4 b4 c5 |4/4k-2 PR: D5 1/1 d5
|6/8k0 PR: C5 1/8 D5 1/8 E5 1/8 c5 d5 e5 1/8 F5 1/8 G5 1/8 f5 g5
|4/4k0 PR: C5. 1/4 +slur D5 1/4 E5 1/4 -slur F5> 1/4 c5 d5 e5 f5
```

### 10.3 Integration Tests

```bash
# Build
cd build && cmake .. -DSPM_ENABLE_TESTS=ON && make -j8

# Run unit tests
ctest --test-dir build --output-on-failure

# Integration: Stage 1 (EventPiece)
./build/src/spm_train \
    --model_type=bpe \
    --input=test/intermo_sample.txt \
    --model_prefix=/tmp/test_stage1 \
    --vocab_size=100 \
    --split_by_whitespace=true \
    --max_sentence_length=500000

# Verify: no piece should contain internal whitespace
python3 -c "
import sentencepiece_model_pb2 as m
model = m.ModelProto()
model.ParseFromString(open('/tmp/test_stage1.model','rb').read())
for p in model.pieces:
    if p.type == 1:  # NORMAL
        assert '\xe2\x96\x81' not in p.piece[3:], f'Boundary in piece: {p.piece}'
print('PASS: No pieces cross whitespace boundary')
"

# Integration: Stage 2 (MomentPiece)
./build/src/spm_train \
    --model_type=bpe \
    --input=test/intermo_sample.txt \
    --model_prefix=/tmp/test_stage2 \
    --vocab_size=100 \
    --split_by_whitespace=false \
    --split_by_interval=true \
    --max_sentence_length=500000

# Verify: pieces can cross whitespace but not interval/barline boundaries
python3 -c "
import sentencepiece_model_pb2 as m
model = m.ModelProto()
model.ParseFromString(open('/tmp/test_stage2.model','rb').read())
ws = '\xe2\x96\x81'
for p in model.pieces:
    if p.type == 1 and len(p.piece) > 3:
        # Find internal whitespace positions
        idx = 3  # skip leading ws
        while True:
            idx = p.piece.find(ws, idx)
            if idx == -1: break
            after = p.piece[idx+3:idx+4] if idx+3 < len(p.piece) else ''
            assert after not in '0123456789|', f'Boundary violation: {p.piece}'
            idx += 3
print('PASS: No pieces cross interval/barline boundary')
"
```

### 10.4 Expected Behavior

For the sample corpus with `split_by_interval=true`:
- `SplitIntoWords` on `_|4/4k0_PR:_C5_1/4_PL:_A-3_C4_F4_1/8_PR:_c5_D-5` should produce:
  - `[_|4/4k0_PR:_C5]` (starts with `|`, continues through non-boundary tokens)
  - `[_1/4_PL:_A-3_C4_F4]` (starts with digit, continues through non-boundary tokens)
  - `[_1/8_PR:_c5_D-5]` (starts with digit)
- Early BPE merges should include frequent character pairs within events: `P`+`R`, `C`+`5`, `/`+`4`
- Later merges should include cross-event combinations: `_PR:`+`_C5`

### 10.5 Round-Trip Verification

```bash
# Encode and decode should be lossless
echo "|4/4k0 PR: C5 1/4 PL: A-3 C4 F4" | \
    ./build/src/spm_encode --model=/tmp/test_stage1.model | \
    ./build/src/spm_decode --model=/tmp/test_stage1.model
# Should output: |4/4k0 PR: C5 1/4 PL: A-3 C4 F4
```

## 11. Implementation Status and Priority

### Already Implemented:
- `split_by_interval` proto field (55), builtin_pb accessors, flag registration, spec_parser entry
- `SplitIntoWords()` with `split_by_interval` logic
- `IsValidSentencePiece()` with `split_by_interval` logic  
- Unigram seed generation `split_into_pieces` lambda with `split_by_interval`
- `SplitSentencesByWhitespace()` passes `split_by_interval` to `SplitIntoWords()`

### Critical Bug to Fix:
- **bpe_model_trainer.cc line 235**: condition must include `split_by_interval()` (and later `split_by_barline()`)
- **bpe_model_trainer.cc line 352** (TrainFast): same fix needed

### Not Yet Implemented (in priority order):
1. **BPE trainer condition fix** (lines 235, 352) -- trivial one-line fix, unblocks BPE+interval training
2. **split_by_barline** -- proto field, builtin_pb, flag, all code paths (medium complexity, follows split_by_interval pattern exactly)
3. **initial_merges_file** -- proto field, builtin_pb, flag, BPE trainer merge-loading and force-apply logic (HIGH complexity)
4. **protected_pieces_file** -- proto field, builtin_pb, flag, unigram trainer integration for seed/prune/finalize/RunMStep (HIGH complexity)
5. **Merge extraction utility** -- Python script or C++ tool to extract merges/pieces from .model files
6. **Orchestration scripts** -- shell scripts for three-stage pipeline
7. **Unit tests** -- for all new functionality
