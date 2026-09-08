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
