# Tokenizer curriculum: 200 -> 200(refit) -> 1000 -> 1500

Status: PLAN. No code written yet. Branched from the frozen repaired
continuation at `a3c0deb94d61e6043559f73377a69019713caa05`
(`fix/unigram-continuation-repair`, PR #4), which stays the stable baseline and
is not modified by this experiment.

Freeze record for the baseline:
`/gscratch/scrubbed/nctamer/pianotok_work/pianovn1500cont_fixed/FREEZE.md`

## The four stages

### 1. Base support selection (~200 pieces)

- Combine piano + violin records.
- Normalize, then deduplicate identical normalized records.
- **Ignore original occurrence counts**: every distinct normalized record has
  weight 1. This is a support-selection stage, so type frequency, not token
  frequency, decides what structure exists.
- `split_by_whitespace=true`.
- Chooses structural support vocabulary, NOT final probabilities.
- Context-switch tokens that must be retained are ordinary NORMAL pieces, never
  USER_DEFINED, because stages 3 and 4 must be allowed to learn pieces
  containing them.

### 2. Fixed-vocabulary score refit (MANDATORY, before any continuation)

- Keep exactly the selected piece strings and their IDs.
- Add nothing, prune nothing.
- Re-estimate the Unigram scores on the **real weighted** combined corpus.
- Purpose: stage 1's artificial type-uniform (weight-1) frequencies must not
  become permanently pinned inherited score geometry. Continuation freezes
  inherited scores up to one global gauge, so whatever stage 2 produces is what
  every later stage inherits.

### 3. Context-independent continuation (~200 -> ~1000)

- Real weighted full combined corpus.
- `split_by_whitespace=false`.
- Context switches remain **present in the E-step corpus**. They are never
  deleted from any corpus; the fence changes candidate ADMISSIBILITY only.
- Context switches act as candidate-generation fences: no new candidate may
  contain or cross one.
- Goal: let note bigrams, chords, rhythm/event compounds and other
  context-independent musical structure claim capacity first.

### 4. Context-conditioned continuation (~1000 -> 1500)

- Same real weighted corpus.
- Remove the context fences.
- Candidates containing context switches are now admissible (`Vn:`+pitch,
  `PL:`+event, ...).
- Adds the remaining ~500 pieces.

## What the corpus actually looks like (measured, not assumed)

Corpus: `/gscratch/cse/nctamer/pianotok_work/pianovn1500/corpus.tsv`, rows are
`<normalized record>\t<count>`:

    |3/4k-1 PL: B-3 PR: D4 Vn: D4	4
    1/12 PR: d4 C#4	16

The context switches are `PL:`, `PR:` and `Vn:` -- **ordinary text**. In the
1200-piece prior they appear only inside NORMAL pieces (`▁PR:▁c5`,
`▁1/8▁PL:▁g3`, and a bare `:` piece). The prior's 130 USER_DEFINED pieces are
`<|vel:N|>` and `<|CC64:on|off|>`, none of which occur in this corpus
(`METAFENCE symbols=131 present_in_corpus=no`).

Consequence: **the stage-3 fence cannot reuse defect D's fence.** That one is
keyed on piece TYPE (USER_DEFINED / CONTROL / UNKNOWN / BYTE). Context switches
here are NORMAL, so stage 3 needs a fence keyed on explicit piece STRINGS.

## What has to be built

Three gaps, in dependency order.

### G1. Weight-1 deduplicated stage-1 corpus (script only, no trainer change)

Read the combined corpus, normalize, deduplicate, emit each distinct record
once with count 1. A TSV with every count set to 1 is exactly this, and the
frozen implementation already treats weighted TSV as equivalent to physical
repetition (`WeightedTsvMatchesPhysicalRepetition`).

### G2. Fixed-vocabulary refit mode (NEW trainer capability)

The existing zero-extension path is NOT this. `ZeroExtensionReturnsThePriorUnchanged`
pins that a continuation with no new slots returns the prior **unchanged** --
scores included. Stage 2 needs the opposite: same pieces, same IDs, scores
re-estimated.

Shape: EM over a FIXED support -- no candidate extraction, no pruning, no
extension budget, no gauge. Every piece is a free parameter; the normalization
is the ordinary Unigram one over the fixed vocabulary. Roughly the frozen
E-step plus an unconstrained M-step, with the piece table held constant.

Open question for sign-off: whether this lives as a mode of the continuation
trainer (`--unigram_refit_only`) or as a separate small entry point. It shares
the E-step but none of the continuation invariants (no inherited/extension
split, no lambda, no required coverage), so folding it into the continuation
trainer risks entangling code that is now frozen and tested.

### G3. String-keyed candidate fence (NEW trainer capability)

A flag naming the fence pieces, e.g.

    --continuation_fence_pieces=▁PL:,▁PR:,▁Vn:

Semantics, deliberately narrow:

- affects candidate GENERATION only -- an enumerated candidate may neither
  contain nor straddle a fence occurrence;
- the corpus is untouched; fence strings stay in the E-step text and keep
  accruing posterior mass through the pieces that already cover them;
- inherited pieces that contain a fence string are unaffected -- the prior is
  immutable, and stage 3 inherits stage 2's table as-is;
- stage 4 simply omits the flag.

Implementation note: defect D already builds a `normalizer::PrefixMatcher` over
a set of strings and marks fenced byte positions, with a precheck that skips
the matcher when no fence symbol occurs. G3 is the same mechanism with the
string set supplied by flag instead of derived from piece type, so the two
should share one fenced-span builder rather than growing a second one.

## Gates before any full run

1. G1 output: distinct record count, and that every count is 1.
2. G2: piece strings and IDs bit-identical to stage 1; scores CHANGED; the
   refit model normalizes (sum of exp(score) over the fixed vocabulary).
3. G3: with the fence on, `meta_fenced_spans > 0` and no learned piece contains
   `PL:`, `PR:` or `Vn:`; with it off, such pieces appear. Same corpus, same
   seed -- the flag is the only variable.
4. Only then stages 3 and 4 end to end, with the same contract checks the
   baseline passes (inherited IDs/strings/types immutable, non-NORMAL scores
   bit-identical, one global gauge) and the same 40k-row token-cost evaluation
   against the frozen 1500 baseline.

## Not in scope

No optimizer or architecture change to the frozen continuation implementation.
No deletion-loss pruning. No vocabulary quotas, phrase bonuses, violin-specific
heuristics, or new initializer. No deleting context tokens from any corpus.
