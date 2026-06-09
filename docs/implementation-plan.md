# Implementation Plan: Three-Stage BPE in SentencePiece

## Architecture Decision

**Progressive Constraint** (inspired by BoundlessBPE) instead of merge file inheritance.

Single `spm_train` command produces a three-stage tokenizer:
```bash
spm_train --model_type=bpe --split_by_barline=true \
  --phase1_merge_budget=4000 --phase2_merge_budget=2400 \
  --vocab_size=8192 --split_by_whitespace=false \
  --max_sentence_length=500000 --split_by_unicode_script=false \
  --split_by_number=false
```

- First 4000 merges: within-event only (no internal ▁)
- Next 2400 merges: cross-event within-moment (no internal ▁+digit/|)
- Remaining ~1792 merges: cross-moment within-bar (no internal ▁+|)

## Proto Fields

```proto
optional bool split_by_interval = 55 [default = false];
optional bool split_by_barline = 56 [default = false];
optional int32 phase1_merge_budget = 57 [default = 0];
optional int32 phase2_merge_budget = 58 [default = 0];
```

## Steps

### Step 1: Proto fields (Trivial)
- File: `src/sentencepiece_model.proto`
- Add fields 55-58 after `seed_sentencepieces_file`
- Update `Next id` comment to 59
- **Test**: N/A (needs Step 2 to compile)

### Step 2: Regenerate builtin_pb (Medium)
- Install protoc 3.14.0 (matches bundled protobuf-lite)
- Run: `protoc --cpp_out=src/builtin_pb/ -I src/ src/sentencepiece_model.proto`
- Verify diff is minimal
- **Test**: `cd build && cmake .. && make -j8` compiles

### Step 3: Register flags (Trivial)
- Files: `src/spm_train_main.cc`, `src/spec_parser.h`
- Add ABSL_FLAG, SetTrainerSpecFromFlag, PRINT_PARAM, PARSE_BOOL/INT32
- **Test**: `spm_train --help | grep split_by_interval`

### Step 4: SplitIntoWords interval/barline (Medium)
- Files: `src/model_interface.h`, `src/model_interface.cc`
- Add `split_by_interval` and `split_by_barline` bool params
- Only split at ▁ when followed by digit/| (interval) or | (barline)
- **Test**: Unit test in `model_interface_test.cc`

### Step 5: Pass flags through SplitSentencesByWhitespace (Trivial)
- File: `src/trainer_interface.cc` line 604
- Pass `split_by_interval()` and `split_by_barline()` to SplitIntoWords
- **Test**: Build succeeds

### Step 6: IsValidSentencePiece boundary checks (Medium)
- File: `src/trainer_interface.cc` around line 253
- When `split_by_interval`: reject pieces with internal ▁+digit/|
- When `split_by_barline`: reject pieces with internal ▁+|
- **Test**: Unit test verifying rejection/acceptance of specific pieces

### Step 7: Fix BPE trainer split gate (Trivial, Critical)
- File: `src/bpe_model_trainer.cc` lines 235 and 352
- Change `if (split_by_whitespace())` to include `|| split_by_interval() || split_by_barline()`
- **Test**: Train BPE with `--split_by_barline=true --split_by_whitespace=false` on tiny corpus

### Step 8: Progressive constraint in BPE merge loop (Hard, Core)
- File: `src/bpe_model_trainer.cc` lines 282-331
- After finding `best_symbol`, check if merge crosses current phase boundary
- Phase 1 (merges < phase1_budget): reject internal ▁
- Phase 2 (merges < phase1+phase2): reject internal ▁+digit/|
- Phase 3: no extra constraint (barline enforced by IsValidSentencePiece)
- Deferred symbols reactivated at phase transitions
- **Test**: Train 3-stage, verify pieces respect phase boundaries

### Step 9: Unigram trainer updates (Medium)
- File: `src/unigram_model_trainer.cc`
- Update `split_into_pieces` lambda (line 305)
- Update `SplitSentencesByWhitespace` gate (line 624)
- **Test**: Train unigram with `--split_by_interval=true`, verify cross-event pieces

### Step 10: Validation guards (Trivial)
- File: `src/trainer_interface.cc`
- Error if phase budgets set without BPE + split_by_barline
- Error if split_by_whitespace + split_by_interval both true
- **Test**: Verify error messages

## Key Design Decisions

1. **IsValidSentencePiece enforces the PERMANENT boundary** (barline). The progressive constraint in the merge loop enforces TEMPORARY tighter boundaries (whitespace, interval) that relax per phase.

2. **Deferred symbols**: When a merge is rejected by the current phase, the symbol is removed from active_symbols_ and stored in a deferred list. At phase transitions, all deferred symbols are reactivated. If active_symbols_ is exhausted within a phase (all deferred), force a phase transition with a warning.

3. **No auto-setting**: User must explicitly set `--split_by_whitespace=false` when using interval/barline modes.

## Critical Path

```
Step 1 → Step 2 → Step 3 (parallel with 4, 6) → Step 5 → Step 7 → Step 8
                   Step 4 ──────────────���──────────┘
                   Step 6 ─────────────────────────┘
```

Step 8 depends on everything else being correct.

## Protoc Version

The builtin_pb files use protobuf 3.14.0 (from `GOOGLE_PROTOBUF_VERSION 3014000` in bundled headers). Must use matching protoc for regeneration.
