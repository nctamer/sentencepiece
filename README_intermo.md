# intermo Three-Phase Tokenizer Training

Train BPE or unigram tokenizers for intermo music notation as a **curriculum of phases**. Each phase is one `spm_train` run that (1) relaxes the pretokenization boundary by one level and (2) protects the vocabulary learned in the previous phase. The vocabulary is built bottom-up — atomic events first, then cross-event groupings, then cross-moment groupings — so the model learns musical primitives before it is ever allowed to merge across them.

## Why train in phases?

intermo notation is hierarchical. The whitespace-separated stream nests three levels:

- **Event** — a single note, duration, or marker (e.g. `▁C5`, `▁1/4`, `▁PR:`).
- **Moment** — everything sounding together; events separated by whitespace up to the next duration/barline.
- **Bar** — a measure, delimited by a barline `▁|...`.

A single greedy pass over the raw stream would happily form huge, rare **cross-bar** pieces before it ever learns the common **event-level** pieces — spending vocabulary budget badly and leaving rare inputs without good fallbacks. Phased training prevents this: each phase only unlocks the next level of merging, and freezes everything learned so far.

## How a phase works

Each phase is a plain, **unmodified** `spm_train` run differing from the last in just two arguments:

1. **Relax the boundary** with one `split_by_*` flag, so progressively larger spans become mergeable. The boundary is a *static pretokenization constraint* — pieces simply cannot cross it during that run.
2. **Protect the prior vocab** with `--protected_pieces_file=<prev_stage_pieces>.txt`. Protected pieces are guaranteed to survive into this run's vocabulary, so the primitives from earlier phases remain available as fallbacks.

> Note: an earlier single-run approach used in-loop `--phase1_merge_budget`/`--phase2_merge_budget` to switch boundaries mid-training. It was removed — the `split_by_*` flags below express the same boundary constraints more simply, keep the trainer identical to upstream, and work for unigram too (which prunes rather than merges, so it has no merge budget).

## The three phases

| Phase | Piece type | Boundary | Flag | What tokens can span |
|-------|-----------|----------|------|---------------------|
| 1 | EventPiece | Every whitespace | `--split_by_whitespace=true` | Nothing — each event is atomic |
| 2 | IntervalPiece | Whitespace before digit or `\|` | `--split_by_interval=true` | Across whitespace within a moment |
| 3 | BarPiece | Whitespace before `\|` | `--split_by_barline=true` | Across moments within a bar |

## Workflow (same for BPE and Unigram)

Three training runs; each inherits the previous run's vocabulary via `--protected_pieces_file`:

```
Phase 1 (EventPiece)
  → extract vocab → stage1_pieces.txt
Phase 2 (IntervalPiece, protect stage1_pieces.txt)
  → extract vocab → stage2_pieces.txt
Phase 3 (BarPiece, protect stage2_pieces.txt)
```

---

## Corpus format: deduplicate to TSV (do NOT sample)

**Always feed the trainer a deduplicated TSV corpus**, for both BPE and unigram.
Music moments repeat heavily (~5–8×), so writing one moment per line wastes the
whole pipeline on duplicates: a 34M-line corpus (713M chars) makes the unigram
suffix array explode to ~635M nodes and — worse — makes EM re-scan all 34M lines
on every one of its ~40 sub-iterations (hours).

Instead, count unique moments and emit `<moment>\t<count>` (one tab per line,
`count` an integer ≥ 1), then pass `--input_format=tsv`. This is **lossless**:
both BPE pair-counts and unigram EM weight by the `count` field, so the learned
vocabulary is identical to the raw repeated corpus — just ~5× fewer rows
(e.g. 34M → 6.65M), unigram suffix nodes 635M → 306M, and EM ~5× faster. The
same TSV corpus is reused for all three phases.

Build it with a few lines of Python:
```python
from collections import Counter
import re
def split_moments(t):  # one moment per element; strip timestamps first if any
    segs = (s.strip().replace('\t', ' ').replace('\r', ' ')
            for s in re.split(r'\s+(?=[|\d])', t))
    return [s for s in segs if s]                    # no tab/CR inside a moment
counts = Counter()
for line in open('raw_moments_or_jsonl'):            # accumulate over your source
    counts.update(split_moments(line))
with open('corpus.tsv', 'w') as f:
    for moment, c in counts.items():
        f.write(f"{moment}\t{c}\n")
```

> **Do not use `--input_sentence_size` / `--shuffle_input_sentence` to tame a
> large corpus** — that randomly *discards* data (and silently changes the
> model). Dedup keeps every moment and its true frequency.
>
> Also keep lines at the **moment level** (split at `\s+(?=[|\d])`), not bar- or
> piece-level. Longer lines re-inflate the unigram suffix array because
> `split_by_unicode_script=false` and `split_by_number=false` disable the
> candidate pruning that keeps standard NLP tokenization fast on long sentences.

---

## BPE — All Three Phases

### Phase 1: EventPiece (BPE)

```bash
spm_train \
  --model_type=bpe \
  --input=corpus.tsv \
  --input_format=tsv \
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

### Phase 2: IntervalPiece (BPE)

```bash
spm_train \
  --model_type=bpe \
  --input=corpus.tsv \
  --input_format=tsv \
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

### Phase 3: BarPiece (BPE)

```bash
spm_train \
  --model_type=bpe \
  --input=corpus.tsv \
  --input_format=tsv \
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

## Unigram — All Three Phases

### Phase 1: EventPiece (Unigram)

```bash
spm_train \
  --model_type=unigram \
  --input=corpus.tsv \
  --input_format=tsv \
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

### Phase 2: IntervalPiece (Unigram)

```bash
spm_train \
  --model_type=unigram \
  --input=corpus.tsv \
  --input_format=tsv \
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

### Phase 3: BarPiece (Unigram)

```bash
spm_train \
  --model_type=unigram \
  --input=corpus.tsv \
  --input_format=tsv \
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

## What Each Phase Produces

Example input: `|4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5`

| Phase | Example tokens | What happened |
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

All three are proto extensions (field numbers 200+); they do not exist in upstream sentencepiece. See [How a phase works](#how-a-phase-works) for how they combine into the phased curriculum.

## Building

```bash
git clone git@github.com:nctamer/sentencepiece.git
cd sentencepiece
mkdir build && cd build
cmake .. -DSPM_ENABLE_SHARED=OFF
make -j8
```

The fork is compatible with upstream Google sentencepiece. Custom fields use proto extensions (200+) and do not conflict with upstream changes.
