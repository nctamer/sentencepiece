# intermo Three-Stage Tokenizer Training

Train BPE or unigram tokenizers with three hierarchical pretokenization levels for intermo music notation. Each stage relaxes the boundary, building on pieces from the prior stage via `protected_pieces_file`.

## The Three Stages

| Stage | Boundary | Flag | What tokens can span |
|-------|----------|------|---------------------|
| 1. EventPiece | Every whitespace | `--split_by_whitespace=true` | Nothing — each event is atomic |
| 2. IntervalPiece | Whitespace before digit or `\|` | `--split_by_interval=true` | Across whitespace within a moment |
| 3. BarPiece | Whitespace before `\|` | `--split_by_barline=true` | Across moments within a bar |

## Workflow (same for BPE and Unigram)

Three training runs. Each inherits pieces from the previous via `--protected_pieces_file`.

```
Stage 1 (EventPiece) → extract vocab → Stage 2 (IntervalPiece, protect Stage 1) → extract vocab → Stage 3 (BarPiece, protect Stage 2)
```

---

## BPE — All Three Stages

### Stage 1: EventPiece (BPE)

```bash
spm_train \
  --model_type=bpe \
  --input=corpus.txt \
  --model_prefix=bpe_stage1 \
  --vocab_size=500 \
  --split_by_whitespace=true \
  --split_by_unicode_script=false \
  --split_by_number=false \
  --character_coverage=1.0 \
  --max_sentence_length=500000
```

Extract vocab:
```bash
awk -F'\t' 'NR>3 {print $1}' bpe_stage1.vocab > stage1_pieces.txt
```

### Stage 2: IntervalPiece (BPE)

```bash
spm_train \
  --model_type=bpe \
  --input=corpus.txt \
  --model_prefix=bpe_stage2 \
  --vocab_size=3000 \
  --split_by_whitespace=false \
  --split_by_interval=true \
  --split_by_unicode_script=false \
  --split_by_number=false \
  --character_coverage=1.0 \
  --max_sentence_length=500000 \
  --protected_pieces_file=stage1_pieces.txt
```

Extract vocab:
```bash
awk -F'\t' 'NR>3 {print $1}' bpe_stage2.vocab > stage2_pieces.txt
```

### Stage 3: BarPiece (BPE)

```bash
spm_train \
  --model_type=bpe \
  --input=corpus.txt \
  --model_prefix=bpe_stage3 \
  --vocab_size=4096 \
  --split_by_whitespace=false \
  --split_by_barline=true \
  --split_by_unicode_script=false \
  --split_by_number=false \
  --character_coverage=1.0 \
  --max_sentence_length=500000 \
  --protected_pieces_file=stage2_pieces.txt
```

---

## Unigram — All Three Stages

### Stage 1: EventPiece (Unigram)

```bash
spm_train \
  --model_type=unigram \
  --input=corpus.txt \
  --model_prefix=unigram_stage1 \
  --vocab_size=500 \
  --split_by_whitespace=true \
  --split_by_unicode_script=false \
  --split_by_number=false \
  --character_coverage=1.0 \
  --max_sentence_length=500000 \
  --max_sentencepiece_length=32
```

Extract vocab:
```bash
awk -F'\t' 'NR>3 {print $1}' unigram_stage1.vocab > stage1_pieces.txt
```

### Stage 2: IntervalPiece (Unigram)

```bash
spm_train \
  --model_type=unigram \
  --input=corpus.txt \
  --model_prefix=unigram_stage2 \
  --vocab_size=3000 \
  --split_by_whitespace=false \
  --split_by_interval=true \
  --split_by_unicode_script=false \
  --split_by_number=false \
  --character_coverage=1.0 \
  --max_sentence_length=500000 \
  --max_sentencepiece_length=64 \
  --protected_pieces_file=stage1_pieces.txt
```

Extract vocab:
```bash
awk -F'\t' 'NR>3 {print $1}' unigram_stage2.vocab > stage2_pieces.txt
```

### Stage 3: BarPiece (Unigram)

```bash
spm_train \
  --model_type=unigram \
  --input=corpus.txt \
  --model_prefix=unigram_stage3 \
  --vocab_size=4096 \
  --split_by_whitespace=false \
  --split_by_barline=true \
  --split_by_unicode_script=false \
  --split_by_number=false \
  --character_coverage=1.0 \
  --max_sentence_length=500000 \
  --max_sentencepiece_length=128 \
  --protected_pieces_file=stage2_pieces.txt
```

---

## What Each Stage Produces

Example input: `|4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5`

| Stage | Example tokens | What happened |
|-------|---------------|---------------|
| 1 (EventPiece) | `▁\|4/4k0` `▁PR:` `▁C5` `▁1/4` `▁PL:` `▁A-3` `▁C4` `▁F4` `▁1/8` `▁PR:` `▁c5` `▁D-5` | Each event is one token |
| 2 (IntervalPiece) | `▁\|4/4k0` `▁PR:▁C5` `▁1/4▁PL:▁A-3▁C4▁F4` `▁1/8▁PR:▁c5▁D-5` | Events within a moment merge |
| 3 (BarPiece) | `▁\|4/4k0▁PR:▁C5▁1/4▁PL:▁A-3▁C4▁F4▁1/8▁PR:▁c5▁D-5` | Moments within a bar merge |

## How `protected_pieces_file` Works

- Protected pieces are guaranteed to exist in the final vocabulary
- They have normal scores — the model decides when to use them based on probability
- They serve as fallback for rare events (e.g., `A--0` → `A-` + `-0` instead of individual characters)
- They are NOT forced (unlike `user_defined_symbols` which always win)
- Works identically for BPE and unigram

## Flags Reference

| Flag | Values | Description |
|------|--------|-------------|
| `--split_by_interval` | bool | Boundary at `▁` + digit or `▁` + `\|` |
| `--split_by_barline` | bool | Boundary at `▁` + `\|` only |
| `--protected_pieces_file` | path | One piece per line to protect from pruning |
| `--phase1_merge_budget` | int | (BPE only) Merges constrained to within-whitespace |
| `--phase2_merge_budget` | int | (BPE only) Merges constrained to within-interval |

Note: `phase1/2_merge_budget` is an alternative single-command BPE approach. The multi-run `protected_pieces_file` approach above is preferred as it works identically for both BPE and unigram.

## Building

```bash
git clone git@github.com:nctamer/sentencepiece.git
cd sentencepiece
mkdir build && cd build
cmake .. -DSPM_ENABLE_SHARED=OFF
make -j8
```

The fork is compatible with upstream Google sentencepiece. Custom fields use proto extensions (200+) and do not conflict with upstream changes.
