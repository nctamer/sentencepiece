# SuperBPE Three-Stage Training for intermo (Experiment Log)

## What Was Done

Trained a three-stage BPE tokenizer for intermo music notation using the superbpe framework (HuggingFace tokenizers Rust fork). Each stage progressively relaxes the pretokenization boundary, allowing merges across wider structural spans.

## Setup

### Environment
```bash
cd /Users/nctamer/repos/superbpe
python3 -m venv .venv
source .venv/bin/activate
pip install maturin pysimdjson tqdm filelock click huggingface-hub
pip install transformers --no-deps  # only needed if using construct_hf_tokenizer
cd tokenizers_superbpe/bindings/python
pip install -e . --no-deps
```

### Rust Modification

Removed the `:Ġ` constraint in `tokenizers_superbpe/tokenizers/src/models/bpe/trainer.rs` (line 724-728). This constraint rejects any merge producing a token containing `:Ġ` — critical for NLP (prevents `Mr.` + ` the` merges) but breaks intermo where `PR:` is always followed by a space.

Also increased `max_words` from 4 to 8 (line 731) to allow larger multi-event tokens for music.

### Corpus

Source: OpenScore Lieder corpus, 1173 scores converted to intermo format.
- Raw: `/Users/nctamer/repos/intermo/data/lieder_train.txt` (14.8M chars, one score per line)
- Preprocessed: `/Users/nctamer/repos/intermo/data/lieder_train_ws.txt` (spaces replaced with `▁`)

Preprocessing:
```python
with open('lieder_train.txt') as f, open('lieder_train_ws.txt', 'w') as out:
    for line in f:
        out.write(line.rstrip('\n').replace(' ', '▁') + '\n')
```

## Why `▁` Instead of ByteLevel

SuperBPE's standard approach uses `ByteLevel` encoding which maps space (0x20) to `Ġ` (U+0120). This is a GPT-2 convention needed for arbitrary byte sequences.

For intermo (clean ASCII, ~50 unique characters), ByteLevel adds unnecessary complexity. Instead:
- Replace spaces with `▁` (U+2581, the sentencepiece convention) in the corpus
- Use regex patterns that split on `▁` instead of whitespace
- Tokens are directly human-readable: `▁PR:▁C5` means "space PR: space C5"

## Three-Stage Training

### Stage 1: EventPiece (within-event merges only)

**Regex**: `▁?[^▁]+` — matches each `▁`-delimited token.

```
Input:   |4/4k0▁PR:▁C5▁1/4▁PL:▁A-3▁C4
Units:   [|4/4k0] [▁PR:] [▁C5] [▁1/4] [▁PL:] [▁A-3] [▁C4]
```

BPE merges characters within each unit. Learns tokens like `▁PR:`, `▁1/8`, `|4/4k0`, `▁C#5`.

```python
from tokenizers import Tokenizer, pre_tokenizers, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(show_progress=True, vocab_size=4256)
tokenizer.pre_tokenizer = Split(pattern=Regex(r'▁?[^▁]+'), behavior='isolated', invert=False)
tokenizer.train([corpus_path], trainer)
tokenizer.model.save('.')  # produces merges.txt + vocab.json
tokenizer.save('tokenizer.json')
```

Result: 1510 vocab (corpus too small to reach 4256), 1459 merges.

### Stage 2: MomentPiece (cross-event merges within moments)

**Regex**: `▁?[|\d][^▁]*(?:▁(?![|\d])[^▁]*)*` — matches each moment (starts with `|` or digit, captures following events until next `|`/digit boundary).

```
Input:   |4/4k0▁PR:▁C5▁1/4▁PL:▁A-3▁C4▁F4
Units:   [|4/4k0▁PR:▁C5] [▁1/4▁PL:▁A-3▁C4▁F4]
```

**Inheritance**: Copy `merges.txt` from Stage 1 into Stage 2 output directory. The Rust `do_train_extend()` detects its presence and force-applies all inherited merges before learning new ones.

```bash
mkdir -p $stage2_dir
cp $stage1_dir/merges.txt $stage2_dir/merges.txt
# Remove empty lines (bug in merges output):
sed -i '' '/^$/d' $stage2_dir/merges.txt
```

```python
os.chdir(stage2_dir)  # do_train_extend() looks for ./merges.txt
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(show_progress=True, vocab_size=6400)
tokenizer.pre_tokenizer = Split(pattern=Regex(r'▁?[|\d][^▁]*(?:▁(?![|\d])[^▁]*)*'), behavior='isolated', invert=False)
tokenizer.train([corpus_path], trainer)
```

The Rust fork:
1. Reads `merges.txt` (1459 inherited merges)
2. Initializes BPE alphabet from the new corpus
3. Force-applies each inherited merge in order (skips if pair not found in new tokenization)
4. Continues greedy BPE for remaining vocab budget

Result: 6400 vocab. New merges like `▁1/8▁PR:`, `▁A4▁PL:`, `▁A-3▁C4▁F4`.

### Stage 3: BarPiece (cross-moment merges within bars)

**Regex**: `▁?\|[^▁]*(?:▁(?!\|)[^▁]*)*` — matches each bar (starts with `|`, captures everything until next `|`).

```
Input:   |4/4k0▁PR:▁C5▁1/4▁PL:▁A-3▁1/8▁PR:▁c5▁|3/4k0▁...
Units:   [|4/4k0▁PR:▁C5▁1/4▁PL:▁A-3▁1/8▁PR:▁c5] [▁|3/4k0▁...]
```

```bash
cp $stage2_dir/merges.txt $stage3_dir/merges.txt
sed -i '' '/^$/d' $stage3_dir/merges.txt
```

```python
os.chdir(stage3_dir)
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(show_progress=True, vocab_size=8192)
tokenizer.pre_tokenizer = Split(pattern=Regex(r'▁?\|[^▁]*(?:▁(?!\|)[^▁]*)*'), behavior='isolated', invert=False)
tokenizer.train([corpus_path], trainer)
```

Result: 8192 vocab. New merges like `▁D-3▁1/8▁d-3`, `▁|4/4k0▁Ob:▁b4`.

## Output Example

Input: `|4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5 |3/4k0 PR: G4 1/4 PL: B-3 D4`

(Encoded as: `|4/4k0▁PR:▁C5▁1/4▁PL:▁A-3▁C4▁F4▁1/8▁PR:▁c5▁D-5▁|3/4k0▁PR:▁G4▁1/4▁PL:▁B-3▁D4`)

| Stage | Tokens | Count |
|-------|--------|-------|
| 1 (EventPiece) | `\|4/4k0` `▁PR:` `▁C5` `▁1/4` `▁PL:` `▁A-3` `▁C4` `▁F4` `▁1/8` `▁PR:` `▁c5` `▁D-5` `▁\|3/4k0` `▁PR:` `▁G4` `▁1/4` `▁PL:` `▁B-3` `▁D4` | 19 |
| 2 (MomentPiece) | `\|4/4k0` `▁PR:` `▁C5` `▁1/4` `▁PL:` `▁A-3▁C4▁F4` `▁1/8▁PR:▁c5` `▁D-5` `▁\|3/4k0▁PR:` `▁G4` `▁1/4` `▁PL:` `▁B-3▁D4` | 13 |
| 3 (BarPiece) | `\|4/4k0` `▁PR:` `▁C5▁1/4` `▁PL:` `▁A-3▁C4▁F4` `▁1/8▁PR:▁c5` `▁D-5` `▁\|3/4k0▁PR:` `▁G4▁1/4` `▁PL:` `▁B-3▁D4` | 11 |

## Merge Statistics

| Range | Stage | What it learns | Example |
|-------|-------|----------------|---------|
| 0–50 | Base | Single characters | `▁`, `\|`, `0`-`9`, `A`-`G`, `#`, `-` |
| 51–1459 | 1 (EventPiece) | Subwords within events | `▁PR:`, `▁1/8`, `\|4/4k0`, `▁C#5` |
| 1460–6349 | 2 (MomentPiece) | Cross-event within moments | `▁1/8▁PR:`, `▁A4▁PL:`, `▁A-3▁C4▁F4` |
| 6350–8141 | 3 (BarPiece) | Cross-moment within bars | `▁D-3▁1/8▁d-3`, `▁G3.▁D4.▁F4.` |

## Artifacts

```
/Users/nctamer/repos/intermo/data/
├── lieder.jsonl                    # 1173 scores as {id, intermo}
├── lieder_train.txt                # One score per line (raw spaces)
├── lieder_train_ws.txt             # Spaces replaced with ▁
├── superbpe_final_s1/
│   ├── tokenizer.json              # Stage 1 tokenizer
│   ├── merges.txt                  # 1459 within-event merges
│   └── vocab.json                  # 1510 tokens
├── superbpe_final_s2/
│   ├── tokenizer.json              # Stage 2 tokenizer
│   ├── merges.txt                  # 6349 merges (inherited + cross-event)
│   └── vocab.json                  # 6400 tokens
└── superbpe_final_s3/
    ├── tokenizer.json              # Stage 3 final tokenizer
    ├── merges.txt                  # 8141 merges (all three stages)
    ├── vocab.json                  # 8192 tokens
    ├── vocab_full.tsv              # Full vocab with merge origins
    └── vocab_freq.tsv              # Token frequencies from corpus encoding
```

## Known Issues / Notes

1. **Empty lines in merges.txt**: The tokenizer occasionally writes empty lines in `merges.txt`. These cause a panic in `do_train_extend()` (line 477 unwrap on None). Fix: strip empty lines before inheriting.

2. **Skipped merges**: When Stage 1 merges reference pairs that don't exist under Stage 2's wider pretokenization boundaries (e.g., a merge like `1/8▁|` that crossed a moment boundary), they're silently skipped. This is expected — ~100 merges are skipped typically.

3. **Corpus size**: The Lieder corpus (14.8M chars) is small for BPE training. Stage 1 only produced 1510 vocab instead of the target 4256. A larger corpus (OpenScore full, or synthetic augmentation) would fill the vocab budget.

4. **No ByteLevel decoder needed**: Since we use `▁` directly in the corpus, decoding is just `token.replace('▁', ' ')`. No byte-level mapping tables.

## Reproducing

```bash
cd /Users/nctamer/repos/superbpe
source .venv/bin/activate

# Ensure the Rust fork is built with :Ġ constraint removed
cd tokenizers_superbpe/bindings/python && pip install -e . --no-deps && cd ../../..

python -c "
import os, shutil
from tokenizers import Tokenizer, pre_tokenizers, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

corpus = '/Users/nctamer/repos/intermo/data/lieder_train_ws.txt'

def train_stage(out_dir, regex, vocab_size, inherit_from=None):
    os.makedirs(out_dir, exist_ok=True)
    if inherit_from:
        shutil.copy(f'{inherit_from}/merges.txt', f'{out_dir}/merges.txt')
        # Remove empty lines
        with open(f'{out_dir}/merges.txt') as f:
            lines = [l for l in f.readlines() if l.strip()]
        with open(f'{out_dir}/merges.txt', 'w') as f:
            f.writelines(lines)
    os.chdir(out_dir)
    t = Tokenizer(BPE())
    t.pre_tokenizer = Split(pattern=Regex(regex), behavior='isolated', invert=False)
    trainer = BpeTrainer(show_progress=True, vocab_size=vocab_size)
    t.train([corpus], trainer)
    t.model.save('.')
    t.save('tokenizer.json')
    return out_dir

base = '/Users/nctamer/repos/intermo/data'
s1 = train_stage(f'{base}/superbpe_final_s1', r'▁?[^▁]+', 4256)
s2 = train_stage(f'{base}/superbpe_final_s2', r'▁?[|\d][^▁]*(?:▁(?![|\d])[^▁]*)*', 6400, s1)
s3 = train_stage(f'{base}/superbpe_final_s3', r'▁?\|[^▁]*(?:▁(?!\|)[^▁]*)*', 8192, s2)
"
```
