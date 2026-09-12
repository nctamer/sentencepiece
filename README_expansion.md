# Tokenizer continuation (expansion)

Continuation extends an **existing, pretrained tokenizer** with new pieces
learned from a new corpus, without disturbing anything the old tokenizer
already decided. It is a different operation from fresh training, and this
document is the contract.

Continuation is generic. Nothing in `src/*continuation*` knows about a
particular tokenizer, byte alphabet, pretokenizer regex, or domain grammar.
That knowledge belongs in an **adapter** outside SentencePiece, which produces
an `ExpansionSpec` and consumes an `ExpansionResult`.

---

## Five things continuation is not

These are the confusions that cost real work, so they are stated first.

**`protected vocabulary != BPE continuation`.**
`--protected_pieces_file` (legacy, below) puts strings into the vocabulary and
protects them from pruning. It does not replay any merge, so a BPE trainer
keeps segmenting from raw characters and learns merges that *contradict* the
seed's own. Such a merge can never fire in a rank-ordered tokenizer built from
the result, because the seed merge outranks it: the piece holds an embedding
row no input can produce. Continuation replays the inherited merge program
first, so everything learned on top of it composes inherited pieces.

**`BPE continuation != Unigram continuation`.**
They share only the name. BPE continuation inherits an *ordered merge program*
and appends ranks. Unigram continuation inherits a *score geometry* and shifts
it by one global gauge. Neither implementation is a special case of the other.

**`external base IDs != SentencePiece internal score sorting`.**
An external tokenizer's ID space is an ABI: ID 42 addresses an embedding row.
SentencePiece's own `.model` orders pieces however training left them. The
`ExpansionSpec`/`ExpansionResult` external IDs are authoritative; a native
`.model` is emitted only when its index order can carry them exactly.

**`base text pretokenizer != expansion hard-fence policy`.**
The pretokenizer the base tokenizer was trained with (a GPT-2-style regex, say)
is a property of the *base*, recorded as `pretokenizer_sha256` for audit. The
fence a continuation run trains under is separate: **one input record is one
training unit**, and no merge or piece may cross a record boundary.
`boundary_policy` names the adapter's domain policy for the audit trail; the
generic trainer does not parse it.

**`append-only learned merges != explicit rank-prepended bootstrap merges`.**
Learned merges always append after the whole inherited program, so inherited
segmentation is untouched. Bootstrap merges are different: with
`allow_rank_prepend=true` they are placed *before* the inherited ranks, which
deliberately changes how the base tokenizer segments text. That is sometimes
what you want, and it is never accidental — it must be requested, and the
change it causes should be measured.

---

## Architecture

### External ID ABI

Every piece carries an `external_id`. Inherited IDs are immutable: same ID,
same string, same type, from the spec through to the result. New pieces are
allocated strictly after the maximum occupied base ID (or at
`first_new_external_id` when the adapter pins it). The trainer refuses a spec
whose new-ID region overlaps occupied IDs, and refuses to allocate past the
representable integer range.

The external ID space may be sparse. A native SentencePiece `ModelProto` cannot
represent that — its piece index *is* its ID — so a sparse space is one of the
reasons native emission is refused and `ExpansionResult` stays authoritative.

### `ExpansionSpec` (input)

Adapter-produced, authoritative. Generic code never reconstructs any of it from
piece strings.

| Field | Meaning |
|---|---|
| `model_type` | `EXPANSION_BPE` or `EXPANSION_UNIGRAM` |
| `base_pieces` | inherited pieces: external ID, string, type, mergeable, atomic |
| `base_merges` | inherited BPE merges, `(left, right)` in rank order |
| `bootstrap_pieces` / `bootstrap_merges` | extra pieces/merges introduced by this run |
| `preserve_base_ids` | must be true |
| `allow_rank_prepend` | place bootstrap merges before inherited ranks |
| `first_new_external_id` | where the appended region starts (default: max + 1) |
| `requested_new_pieces` | how many pieces to LEARN (see Budget) |
| `*_sha256`, `boundary_policy` | provenance recorded into the result |

**Atomic pieces** are the reversible alphabet the corpus is segmented into. A
valid spec induces **exactly one** segmentation of every training record into
atoms: zero parses means the adapter omitted an atom, and two parses mean the
alphabet is ambiguous. Both are rejected before any merge is replayed.

### `ExpansionResult` (output)

Self-auditing, and the source of truth for external tokenizers. It carries the
three piece groups and the three merge groups separately, the effective ranks
actually serialized, the ID region, budget accounting
(`requested_new_pieces` / `actual_new_pieces`), `unreachable_pieces`,
`rank_prepend`, the provenance hashes, and — for Unigram — the prior's hash,
piece count, solved `unigram_lambda`, and `unigram_score_gauge_max_error`.

It is deliberately enough to audit a run **without the trainer logs**.

### Budget

`requested_new_pieces` counts only pieces the run may **learn**. Inherited and
bootstrap pieces are inherited state and never spend it.

It is read by **field presence**. An explicit `0` is a replay-only run and
stays distinguishable from "unset", which falls back to filling
`trainer_spec.vocab_size`. An inherited merge that never fires on the
continuation corpus is still exported and still costs nothing.

Every newly learned BPE rank allocates exactly one fresh NORMAL piece. A pair
already present in the inherited/learned program is never selected again, and a
candidate whose concatenated child string already exists is retired. This
unique-construction rule is not cosmetic: because a later rank always creates a
fresh symbol, it cannot manufacture an operand of an earlier rank. Sequential
training application is therefore equivalent to ordinary ranked flat replay.

---

## BPE continuation

    --model_type=bpe --expansion_spec=<ExpansionSpec.pb>

1. Validate the spec: IDs, types, atom alphabet, and that the inherited merge
   program is well-formed **on its own** (rank-prepended bootstrap state may not
   retroactively make a malformed inherited tokenizer constructible).
2. Segment the corpus: every `USER_DEFINED` occurrence first (longest prefix
   match, leftmost), each one a frozen unit; every run of text between them
   into declared atoms — exactly one parse per run.
3. Replay the effective inherited prefix in serialized order (bootstrap first
   when `allow_rank_prepend`, otherwise after the inherited ranks).
4. Learn new merges, appending ranks, recording exact `(left, right)`
   provenance **at acceptance time**, before the corpus is mutated.
5. Verify every learned piece is reachable under the final serialized table.
   Zero unreachable pieces is a precondition for success, not a statistic.

### `USER_DEFINED` pieces are frozen units, in training and at inference

Native SentencePiece gives a `USER_DEFINED` piece one role: at inference it is
recognized by longest prefix match **before** any merge runs and is frozen on
both sides (`bpe_model.cc`); native training reaches the same statistics by
replacing every occurrence with a pretokenization boundary
(`trainer_interface.cc`). Continuation and the explicit runtime implement that
role, not an approximation of it:

* a `USER_DEFINED` base piece must be declared `mergeable=false, atomic=false`
  — anything else is refused, not corrected;
* in the corpus, each occurrence becomes one frozen symbol that forms no pair
  on either side, so no merge is ever learned inside it or across it, and it
  is absent from the replay table (a merge naming it is rejected);
* its external ID and type are inherited verbatim through every stage;
* `ExpansionProcessor::Encode` runs the same longest-prefix match first and
  skips every pair with a frozen side; `Load` rejects a program in which any
  merge names a `USER_DEFINED` string as a child.

`USER_DEFINED` is **not** `--protected_pieces_file`. The latter is a Unigram
pruning shield: the listed pieces stay `NORMAL`, keep learned scores, and
merge/segment like any other piece. The two mechanisms are unrelated.

### Explicit fences (`--continuation_fence_strings`) apply to BPE too

The same flag, normalization and semantics as the Unigram G3 fence: each
logical string is normalized with the effective (inherited) normalizer, every
byte boundary of every record is probed, overlapping matches are unioned, and
**no merge may overlap a fenced character** — nothing starts inside, ends
inside, contains or spans a fence occurrence, in either direction. The fence
unit itself (e.g. an inherited `▁PL:`) is replayed whole. Provenance:
`ExpansionResult.boundary_policy = bpe_explicit_fences_v1:[...]`, sorted and
%XX-escaped like the Unigram form. With `PL:,PR:,Vn:` a record segments as
`▁1/12 | ▁PL: | ▁d3▁F#3 | ▁PR: | ▁a3▁D4 | ▁Vn: | ▁f#4▁D4` and a stage learns only
inside those units.

### Training-only hierarchy guidance (`--bpe_hierarchy_file`)

BPE may optionally receive an adapter-produced
`sentencepiece-bpe-hierarchy-v1` sidecar. The sidecar is **training metadata
only**. It changes which ordinary surface pair wins the next rank; it is not
serialized as runtime scope and it is not required by inference.

The trainer maintains exactly one evolving state: the ordinary flat BPE
segmentation produced by the ranked prefix learned so far. Candidate identity is
just `(left,right)`. For a candidate, the trainer first determines the exact
left-to-right, non-overlapping occurrence set that ordinary flat BPE would
replace if that pair were selected next. An occurrence contributes its row
weight to the candidate's support iff the RESULTING source span is compatible
with the immutable laminar hierarchy.

Compatibility is span based. For every grammar parent partially traversed by
the resulting span, the span must be a consecutive union of complete children
of that parent. Fully containing a nested parent is allowed: at an ancestor
level the completed nested parent is one child. This permits ordinary flat BPE
to temporarily form a partial token and later repair it by completing a larger
parent.

The selected pair is the pair with maximum hierarchy-supported weighted count;
ties are resolved by `left` bytes then `right` bytes. After selection, the
rule is appended as an ordinary unscoped rank and is applied **globally** to
every flat occurrence, including occurrences that contributed no hierarchy
support. Thus `ExpansionMerge.weighted_count` is a training support statistic,
not necessarily the total application weight.

This distinction is deliberate. For example, if `a+b` is supported in one
row but occurs inside an incomplete grammar child in another, support comes only
from the first row; if it wins, flat BPE still merges `a+b` in both rows.
The next iteration is counted from that real flat state.

The following invariants make this mathematically equivalent to the serialized
flat program:

1. one learned operation per ordinary `(left,right)` pair;
2. every learned operation creates a previously nonexistent NORMAL child;
3. application uses the same deterministic left-to-right non-overlap rule as
   inference;
4. after every rank, the trainer state is ordinary replay of the learned prefix.

The fresh-child invariant is what removes the former scoped-alias failure: a
later operation cannot create a symbol named by an older rule, so it cannot
wake an older rank between two applications of the newly selected rule.
There is no backward closure and no hierarchy-scoped alias operation.

Inherited/base/bootstrap ranks replay first and are never vetoed by a newly
introduced training hierarchy. Only selection of appended ranks is guided by
the sidecar.

The sidecar SHA-256 remains in the continuation contract for training
provenance. The result boundary policy is
`bpe_hierarchy_guided_training_v1:<sha256>`, but the emitted merge table is an
ordinary flat table and `ExpansionProcessor::Encode` requires no hierarchy.

### Shape options are about NEW pieces### Shape options are about NEW pieces

`max_sentencepiece_length`, `split_by_whitespace`, `split_by_unicode_script`,
`split_by_number` and `split_digits` are fresh-training heuristics, applied to
merge candidates. A setting that cannot express an *inherited* piece is a
configuration error and fails before the corpus is opened.

This matters in practice: `max_sentencepiece_length` defaults to **16**, and
real byte-level vocabularies contain longer pieces.

`treat_whitespace_as_suffix`, training pretokenizer callbacks,
`pretokenization_delimiter`, `input_sentence_size` sampling, and the legacy
`split_by_interval` / `split_by_barline` flags are all refused: continuation's
fence is one record per training unit.

### The native `.model` is a second, weaker artifact

**Native SentencePiece BPE inference does not read a merge table.** It merges
whichever adjacent pair has the best-scoring **concatenation** in the
vocabulary. An explicit program merges the declared pair `(left, right)`. These
rules are not the same rule.

Concretely: with vocabulary `{a, b, c, ab, bc, abc}`, `"abc"` recorded as
`(ab, c)`, and `(b, c)` outranking `(a, b)`, the program produces `[a, bc]` —
it has no rule for `(a, bc)` — while native inference finds `"abc"` in the
vocabulary and merges anyway.

So a native model is emitted **only** when it can be proven exact:

- every atom is a single Unicode scalar (native inference starts from
  characters, and no merge may build an atom);
- every piece admits exactly one split into two vocabulary pieces, and that
  split is its recorded merge;
- no `NORMAL` piece is marked non-mergeable (only `CONTROL`/`USER_DEFINED` are
  excluded from merging by native code);
- external IDs are contiguous from zero with exactly one `UNKNOWN`;
- the rank count is small enough for float32 scores to order exactly (2^24).

When it is emitted, scores are the program: a merge child scores `-rank`, so
"best score" and "lowest rank" are the same relation, and atoms score `0`.

When any condition fails, **no `.model` is written**, the offending piece is
named, and `ExpansionResult` plus `.merges` are authoritative. The adapter is
then responsible for reconstructing the real tokenizer.

> An intermediate SentencePiece `.model` is a drop-in tokenizer **only** when
> the conditions above hold. For an arbitrary external byte-level BPE they
> generally do not. Do not describe it as one otherwise.

---

## Unigram continuation

    --model_type=unigram --unigram_prior_model=<prior.model>

The prior `ModelProto` is authoritative.

- Inherited IDs, strings and types are immutable.
- Inherited **NORMAL** scores may move only by one global additive-length
  gauge: `s_i' = s_i + lambda * length(i)`.
- Inherited non-NORMAL pieces keep their scores bit-identical.
- New pieces append after the entire prior. Only extension candidates may be
  pruned.

### Why one gauge

A single `lambda` shared by every inherited piece cancels out of any comparison
between two segmentations of the same string. For `"ab"`:

    (s_ab + 2L) - ((s_a + L) + (s_b + L)) = s_ab - s_a - s_b

The inherited model's *relative* preferences are therefore preserved exactly,
while the gauge frees the probability mass that new pieces need. This is
verified before anything is written: a per-piece drift beyond one float32
rounding fails the run.

### Learning

New mass is learned separately from inherited geometry, by a **constrained**
M-step. Ordinary `RunMStep()` over the combined vocabulary is never called;
inherited pieces are not independently re-estimated, get no hidden `0.5`
pseudo-count, and are never given a synthetic `0.0` score.

Both lambdas are roots of monotone objectives, found by deterministic bracketed
bisection. An unbracketed root, a non-finite objective, an interval that never
closes, or an inherited mass outside `(0, 1]` all fail loudly. No general-purpose
optimizer is involved, and no endpoint is returned because a loop cap expired.

### Normalization

The prior's normalizer and denormalizer are authoritative — inherited scores
are only meaningful over the text the prior was fitted on. Authoritative is not
silent:

| Caller supplied | Result |
|---|---|
| nothing, or the CLI default | inherit the prior's |
| exactly the prior's | accepted |
| anything else | **rejected** before training |

`treat_whitespace_as_suffix` follows the same rule, because it decides what the
inherited pieces mean. The check is by **value**, not field presence: the CLI
sets every normalization field on every run, so presence would report
"explicit" for a command line that never mentioned normalization.

---

## Weighted input

`--input_format=tsv` reads `<unit><TAB><count>`. A weighted corpus must be
exactly equivalent to physical repetition of each record `count` times — this
is a tested contract for both model types.

Frequency arithmetic is checked, not wrapped. The corpus's weighted position
mass is proven to fit in `uint64_t` before any candidate is summed. Candidate
frequency is the weighted number of **non-overlapping replacements that can
actually be applied** at that exact scope. Acceptance checks count == applied
weight and fails with an internal error on any discrepancy.

For hierarchy-guided correctness the optimized trainer is not treated as its
own oracle. Tests include a separate restart-and-replay implementation that
rebuilds tiny corpora from atoms after every operation, full-replays the ordinary
flat ranked program, recomputes hierarchy support from source spans, and
recounts candidates from scratch. With the opt-in
`TrainerSpec.bpe_reference_trace_file`, the optimized trainer emits a
test-only per-rank state trace; generated laminar-tree tests compare operation,
count, allocation/ID and full-corpus segmentation SHA at every iteration.
Production must leave this field unset.

`ExpansionResult.training_final_segmentation_sha256` independently commits to
the final canonical PreparedCorpus segmentation (normalized row, aggregated
weight and ordered final token strings). A production runtime must reproduce
that digest exactly; token-count equality alone is insufficient.

---

## Reproducibility

A fixed spec, corpus and build produce byte-identical artifacts. Ordering that
carries meaning uses deterministic containers; hash maps are used for lookup
only, never as authoritative order. Thread count is a scheduling detail and
does not change the model.

`vocab_sha256`, `merges_sha256`, `tokenizer_sha256`, `pretokenizer_sha256` and
(for Unigram) `prior_model_sha256` are recorded so a result can be tied back to
the exact base tokenizer it continued.

---

## Legacy compatibility shims

`TrainerSpec` extensions 200/201/204/205 predate first-class continuation. They
keep their original wire meanings so existing models and scripts keep working,
and they are **not** the continuation model.

| # | Field | Status |
|---|---|---|
| 200 | `split_by_interval` | legacy intermo pretokenization |
| 201 | `split_by_barline` | legacy intermo pretokenization |
| 202, 203 | *retired* | progressive BPE phase budgets; excluded from the extension range so protoc refuses to reuse them |
| 204 | `protected_pieces_file` | fresh-training protected vocabulary (BPE and Unigram) |
| 205 | `seed_merges_file` | BPE-only seed merge replay; writes `<model_prefix>.merges` |

`protected_pieces_file` never denotes inherited state. Combining it — or
`seed_merges_file` — with `--expansion_spec` or `--unigram_prior_model` is
ambiguous rather than additive, and is rejected.

`protected_pieces_file` + `seed_merges_file` is the legacy *approximation* of
BPE continuation: it has no external ID ABI, no exact merge provenance and no
reachability proof. `--expansion_spec` supersedes it.

---

## Artifacts

| File | Contents |
|---|---|
| `<prefix>.expansion` | `ExpansionResult` — authoritative |
| `<prefix>.merges` | effective ordinary flat merge table in rank order (`left<TAB>right`) |
| `<prefix>.vocab` | pieces in external ID order |
| `<prefix>.model` | native `ModelProto` — **only when proven exact** |

Tab-separated merges, because a piece may contain the whitespace marker, which
a space-separated `merges.txt` cannot express. `ExpansionResult` remains
authoritative; the optional third column exists so the human-readable table does
not erase scoped-operation identity.


## Self-describing continuation artifacts and the explicit BPE runtime

`ExpansionSpec.contract` / `ExpansionResult.contract` (`ContinuationContract`)
make an expansion artifact readable without its training log. It records the
authoritative normalizer and denormalizer, the pretokenizer policy that decided
which pieces were possible (`max_sentencepiece_length`, every `split_*`,
`hard_vocab_limit`, `input_format`, `input_sentence_size`), the inherited
special-token ABI (`unk_id`/`bos_id`/`eos_id`/`pad_id`/`unk_piece`/
`unk_surface`/`byte_fallback`), the inherited ordered-ID-map SHA-256, and the
corpus identity. `character_coverage` is recorded for provenance only:
continuation takes its atomic alphabet from the inherited piece table, never
from coverage.

BPE continuation now reconciles the inherited text pipeline the way Unigram
continuation always has: an omitted/default caller normalizer inherits the
base, an equivalent one is accepted, and an explicitly conflicting one fails
before training instead of silently retraining under a different pipeline.

`BuildNativeModel()` no longer copies the caller's `TrainerSpec`. The CLI
defaults `bos_id=1`/`eos_id=2` would otherwise declare ordinary inherited
pieces at IDs 1 and 2 to be BOS and EOS. The emitted ABI is derived from the
piece table and the inherited contract, and emission fails rather than writing
a semantically false model. `ExpansionResult.native_model_emitted` and
`native_model_refusal` record the outcome.

`expansion::ExpansionProcessor` (`src/expansion_processor.h`) is the
authoritative explicit-program runtime. Ordinary expansion artifacts normalize
with the artifact's own normalizer and replay their ranked merge program.
Hierarchical artifacts additionally require the adapter-produced per-input
completion gates and use `EncodeWithHierarchy`: inherited/base ranks replay
first without grammar veto, then appended ranks replay with occurrence-local
eligibility. External IDs never change.

The implementation uses live-neighbour links plus a priority heap rather than
rescanning the token sequence for every rank. It provides encode/decode,
`IdToPiece`/`PieceToId`, exact per-token UTF-8 byte spans over the normalized
text, and `IdMapSha256()`. `spm_expansion_cli --hierarchy_file=...` exposes
the same hierarchical runtime for cross-language corpus/parity checks.
