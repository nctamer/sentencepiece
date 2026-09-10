# Tokenizer curriculum: 200 -> 200(refit) -> 1000 -> 1500

Status: PLAN for stages 1, 3 and 4. Stage 2 is IMPLEMENTED (`spm_refit_unigram`).

Baselines this plan builds on, neither of which it modifies:

- frozen continuation repair `a3c0deb94d61e6043559f73377a69019713caa05`
  (`fix/unigram-continuation-repair`, PR #4), plus the cleanup branch
  `fix/unigram-continuation-cleanup`, which is portability/consistency only and
  changes no modelling;
- fixed-vocabulary refit `feat/unigram-fixed-vocab-refit` (PR #5).

Freeze record for the validated 1500-piece artifact:
`/gscratch/scrubbed/nctamer/pianotok_work/pianovn1500cont_fixed/FREEZE.md`

## The four stages

### 1. Base support selection (~200 pieces) -- record-balanced

Combine piano + violin records, then give **equal weight per distinct
normalized record**. This is NOT literal token-uniform or type-uniform
frequency, and the pipeline must not be described that way.

The preprocessing is deliberately not "normalize -> write normalized text ->
spm_train", because that applies SentencePiece normalization twice: once by
hand and once inside the trainer. Instead:

    raw record
      -> authoritative normalized KEY (used ONLY for deduplication)
      -> deduplicate by that key
      -> emit ONE deterministic raw representative per key, with count 1
      -> spm_train normalizes exactly once, as usual

`split_by_whitespace=true`. This stage chooses structural support vocabulary,
not final probabilities.

Context-switch tokens that must be retained are ordinary NORMAL pieces, never
USER_DEFINED, because stages 3 and 4 must be allowed to learn pieces containing
them. In this corpus that is already the case: `PL:`, `PR:` and `Vn:` are plain
text and appear only inside NORMAL pieces.

### 2. Fixed-vocabulary score refit -- IMPLEMENTED

    spm_refit_unigram --model=<stage1.model> --input=<weighted.tsv> \
                      --input_format=tsv --model_prefix=<stage2> \
                      --num_iterations=8 --num_threads=4

The implemented contract, which supersedes the earlier sketch in this document
(it claimed "every piece is a free parameter" and proposed checking
`sum(exp(score))` over the whole serialized vocabulary -- both wrong for
SentencePiece special pieces):

- NORMAL support fixed; only NORMAL scores are re-estimated;
- piece count, IDs, strings and types fixed; non-NORMAL scores bit-identical;
- the simplex is over NORMAL pieces only. Proto scores are NOT normalized
  across all serialized piece types: USER_DEFINED nodes are scored
  `GetUserDefinedScore(length)`, the UNK node is scored
  `min_score() - kUnkPenalty`, and CONTROL/BYTE are not lattice nodes at all;
- known-only forward-backward: UNKNOWN contributes exactly zero posterior, so
  the NORMAL M-step is the exact maximizer;
- byte fallback does NOT waive known-support coverage. A corpus needing the
  byte path fails rather than being optimized under the wrong objective.

Why it is mandatory here: continuation freezes the score geometry it inherits
(up to one global gauge), so stage 1's record-balanced weights would otherwise
become permanent probability geometry.

### 3. Context-independent continuation (~200 -> ~1000) -- NOT IMPLEMENTED

Real weighted full combined corpus, `split_by_whitespace=false`. Context
switches remain present in the E-step corpus and are never deleted from any
corpus; the fence changes candidate ADMISSIBILITY only.

Goal: let note bigrams, chords, rhythm/event compounds and other
context-independent musical structure claim capacity first.

### 4. Context-conditioned continuation (~1000 -> 1500) -- NOT IMPLEMENTED

Same corpus, fences removed, candidates containing context switches admissible
(`Vn:`+pitch, `PL:`+event). Adds the remaining ~500 pieces.

## What the corpus measurably contains

`/gscratch/cse/nctamer/pianotok_work/pianovn1500/corpus.tsv`, rows are
`<normalized record>\t<count>`:

    |3/4k-1 PL: B-3 PR: D4 Vn: D4	4
    1/12 PR: d4 C#4	16

The context switches `PL:`, `PR:` and `Vn:` are ordinary text. In the
1200-piece prior they appear only inside NORMAL pieces (`▁PR:▁c5`,
`▁1/8▁PL:▁g3`, and a bare `:` piece). The prior's 130 USER_DEFINED pieces are
`<|vel:N|>` and `<|CC64:on|off|>`, none of which occur in this corpus.

Consequence: the stage-3 fence CANNOT reuse the inherited-meta fence, which
keys on piece TYPE. It must key on normalized corpus SURFACE STRINGS.

## G3: the context fence -- IMPLEMENTED

Flag name: `--continuation_fence_strings` (not `fence_pieces`) -- these are
normalized corpus surface strings, and they need not be standalone vocabulary
pieces.

The caller supplies LOGICAL strings:

    --continuation_fence_strings=PL:,PR:,Vn:

and the trainer converts them to authoritative normalized surfaces using the
PRIOR's normalizer before matching. Callers must not be asked to type internal
`▁` forms by hand.

Semantics, deliberately narrow:

- candidate GENERATION only: an enumerated candidate may neither contain nor
  straddle a fence occurrence;
- the corpus is untouched; fence strings stay in the E-step text and keep
  accruing posterior through the pieces that already cover them;
- inherited pieces containing a fence string are unaffected: the prior is
  immutable;
- stage 4 simply omits the flag.

**Fence correctness is an EXTRACTION property**, and the tests are written at
that level (`SPM_DUMP_CANDIDATES`, paired with `SPM_STOP_AFTER_INIT` for an
extraction-only gate):

    fence ON  -> tracked context-containing/crossing candidates are ABSENT
                 from the extracted candidate pool
    fence OFF -> those same candidates are PRESENT in extraction

Do NOT require fence-OFF candidates to survive final pruning. Selection is an
optimizer outcome, not a fence-correctness condition.

The matcher **unions all overlapping fence intervals**: every character
boundary is probed as a match start and a match only ever ADDS to the mask.
Skipping ahead by the previous match length would miss overlapping starts --
with fences `{abc, bcd}` over `abcd` it marks `[0,3)` and leaves the final
character admissible. Both fence sources share one `FenceMask` builder, whose
byte<->character mapping is exact: matches start only at character boundaries,
and a match whose end is not a boundary is refused rather than rounded.

Diagnostics distinguish the two sources. Explicit context fences are NOT
`meta_fenced_spans`:

    METAFENCE symbols=... present_in_corpus=... matcher=...
    FENCE logical="Vn:" normalized="<boundary-bearing surface>"
    FENCE logical_count=... normalized_unique=... matcher=...
    FENCE inherited_meta_matches=... explicit_string_matches=...
          fenced_characters=... records_with_explicit_fence=...
          malformed_boundary_matches=...

## Gates before any full run

1. Stage-1 output: distinct record count, and every count equal to 1.
2. Stage 2: piece strings and IDs identical to stage 1; NORMAL scores changed;
   non-NORMAL scores bit-identical; NORMAL simplex sums to 1.
3. G3: with the fence on, tracked context candidates are absent from
   EXTRACTION and `explicit_string_matches > 0`; with it off, present. Same
   corpus, same seed -- the flag is the only variable. Do NOT require
   `meta_fenced_spans > 0`: `PL:`/`PR:`/`Vn:` are ordinary NORMAL surface text,
   not inherited typed meta symbols, so they are counted by
   `explicit_string_matches` and the meta counter can legitimately stay 0.
4. Only then stages 3 and 4 end to end, with the same contract checks the
   baseline passes and the same 40k-row token-cost evaluation against the
   frozen 1500 baseline.

## Explicitly NOT in scope

An external review proposed several mathematically interesting changes. They
are recorded here as separate experimental hypotheses and are NOT part of this
curriculum; the validated baseline must remain comparable.

- likelihood / deletion-loss pruning;
- exact masked-partition deletion loss near K;
- an extension-mass initialization parameter `eta`;
- a required-mass parameter `rho`;
- removing the current `BaseMass(0)` initialization limitation;
- additive per-character / per-script / per-feature gauges;
- joint re-pruning of previous curriculum-stage pieces;
- replacing the append-only curriculum with a reoptimizable one;
- a generic BoundaryPolicy architecture.

Also unchanged by the cleanup: expected-posterior-count pruning, the shrinking
schedule, candidate ranking, tied candidate initialization, the scalar additive
-length gauge, the continuation M-step, the append-only inherited contract, the
200/800/500 allocation, and corpus semantics.

## A note on what the gauge proves

The scalar gauge preserves **inherited-submodel segmentation geometry**: for
two segmentations of the same normalized surface using inherited NORMAL pieces
only, total length is equal, the `lambda*length` terms cancel, and the score
difference is unchanged. It does NOT mean the expanded tokenizer emits the old
token sequence -- an extension is deliberately allowed to beat the inherited
segmentation. The theorem also concerns the inherited NORMAL probabilistic
submodel only; USER_DEFINED and UNKNOWN have special lattice semantics and do
not take part in the cancellation.
