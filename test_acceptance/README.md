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

    python3 test_acceptance/real_bytelevel_bpe_acceptance.py <output-dir>

Expects `build/src/spm_train` and a Qwen3 snapshot in the HuggingFace cache.
`N_NEW` sets how many pieces to learn (default 24). Run it twice into
different directories and compare `*.expansion` to check determinism.

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

    python3 test_acceptance/unigram_prior_continuation_acceptance.py <output-dir>

Needs `build/src/spm_train` and the `sentencepiece` Python package importable.
