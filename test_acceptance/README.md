# Acceptance runs against real tokenizers

These are not unit tests. They need a real pretrained tokenizer on disk, so
they are opt-in and are not wired into CTest or Bazel: CI has no Qwen
checkout, and a test that silently skips when its data is missing is worse
than one you have to ask for.

## `real_bytelevel_bpe_acceptance.py`

Continues a genuine Qwen3 byte-level BPE (151,643 pieces / 151,387 merges)
over a small domain corpus and checks the contract on real inherited state
rather than on a fixture: inherited IDs, strings, types and merge order
unchanged; new IDs strictly after the occupied range; exact `(left, right)`
provenance; zero unreachable additions, verified independently by replaying
the serialized merge table; compression improves on the domain corpus while
ordinary language tokenizes identically; and no native `.model` is emitted,
because a real byte-level vocabulary does not satisfy the exactness
conditions in `README_expansion.md`.

    UNITS_TSV=<units.tsv> N_NEW=512 \
      python3 test_acceptance/real_bytelevel_bpe_acceptance.py <output-dir>

Expects `build/src/spm_train` and a Qwen3 snapshot in the HuggingFace cache.
`UNITS_TSV` is a `<unit><TAB><count>` corpus from the real pretokenizer; the
run falls back to a few built-in lines without it. `N_NEW` sets how many
pieces to learn, `MAX_UNITS` caps the corpus. Compression is measured on
held-out units, not on the training set. Run it twice into different
directories and compare `*.expansion` to check determinism.

Measured on 20,000 real intermo units (528,927 occurrences) continuing a
151,643-piece Qwen3 with 512 new pieces: every inherited ID, string, type and
merge rank unchanged, exact provenance, zero unreachable, held-out domain
text 69.3% shorter, and the prose lines identical at 31 tokens either way.

## `unigram_prior_continuation_acceptance.py`

Mirrors the violin/piano flow without needing its data: trains a piano-only
Stage-1 Unigram model, then continues it with `--unigram_prior_model` over a
corpus containing both piano and violin.

Checks that every Stage-1 ID keeps its string and type, that non-NORMAL scores
are untouched, and that a single additive-length gauge governs every inherited
NORMAL score - the per-piece shift/length spread is the measurement, and it
comes out around 1e-7. Violin specializations win extension slots, weighted
TSV equals physical repetition, and reruns are byte-identical.

The segmentation check is the careful one. Extensions are *allowed* to win
where they are better, so it does not demand that Stage 2 segment everything
as Stage 1 did. It requires the narrower property the gauge actually
guarantees: where no extension piece takes part, the segmentation is exactly
Stage 1's, and every difference is explained by an extension piece winning.

    STAGE1_TSV=<piano.tsv> STAGE2_TSV=<piano+violin.tsv> \
      python3 test_acceptance/unigram_prior_continuation_acceptance.py <out>

Needs `build/src/spm_train` and the `sentencepiece` Python package importable.
Without the two TSVs it falls back to synthetic corpora. `S1_VOCAB` and
`TARGET_VOCAB` set the two stage sizes (1200 -> 1500 on real input, matching
the violin/piano flow).

One thing real data teaches that synthetic data does not: Stage 2 carried
Greek staff labels (`Vn:` + a Greek instrument name) that a piano-only Stage 1
had never seen, and the trainer refused the corpus as unrepresentable without
`<unk>` - correctly. A deployment fixes that by giving Stage 1 the coverage,
or `byte_fallback`; the script reports how many units it had to drop.

Measured on real violin/piano data, 1200 -> 1500: per-piece shift/length
spread 1.2e-07 across every inherited NORMAL piece, 459 of 541 probes stay
entirely on inherited pieces and all match Stage 1 exactly, and weighted TSV
is byte-identical to physical repetition.
