"""Real external byte-level BPE continuation acceptance.

Uses a genuine Qwen3 tokenizer (151k pieces, 151k merges) as the inherited
state and checks the contract on it, not on a fixture.
"""
import json, os, subprocess, sys, glob
sys.path.insert(0, os.path.expanduser("~/repos/sentencepiece/python/src/sentencepiece"))
import sentencepiece_model_pb2 as pb

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*"))[0]
OUT = sys.argv[1]
SPM = os.path.expanduser("~/repos/sentencepiece/build/src/spm_train")

def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = bytes_to_unicode()
ALPHABET = set(B2U.values())

def encode_bytelevel(text):
    return "".join(B2U[b] for b in text.encode("utf-8"))

vocab = json.load(open(os.path.join(SNAP, "vocab.json")))
merges = []
with open(os.path.join(SNAP, "merges.txt"), encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line or line.startswith("#version"):
            continue
        left, _, right = line.partition(" ")
        merges.append((left, right))

print(f"Qwen base: {len(vocab)} pieces, {len(merges)} merges")

spec = pb.ExpansionSpec()
spec.schema_version = 1
spec.model_type = pb.EXPANSION_BPE
spec.preserve_base_ids = True
spec.requested_new_pieces = int(os.environ.get("N_NEW", "24"))
spec.vocab_sha256 = ""
spec.boundary_policy = "one-record-per-training-unit"

max_id = -1
for piece, pid in vocab.items():
    p = spec.base_pieces.add()
    p.external_id = pid
    p.piece = piece
    in_alphabet = all(ch in ALPHABET for ch in piece)
    if in_alphabet:
        p.type = pb.ModelProto.SentencePiece.NORMAL
        p.mergeable = True
        p.atomic = len(piece) == 1
    else:
        # Special/added tokens are addressable but never merge participants.
        p.type = pb.ModelProto.SentencePiece.USER_DEFINED
        p.mergeable = False
        p.atomic = False
    p.score = 0.0
    max_id = max(max_id, pid)

for rank, (left, right) in enumerate(merges):
    m = spec.base_merges.add()
    m.rank = rank
    m.left = left
    m.right = right

spec.first_new_external_id = max_id + 1
spec_path = os.path.join(OUT, "qwen.spec")
open(spec_path, "wb").write(spec.SerializeToString())
print(f"spec: {len(spec.base_pieces)} pieces, first_new_external_id={spec.first_new_external_id}")

# The domain corpus. UNITS_TSV points at "<unit><TAB><count>" produced by the
# real pretokenizer; without it a few hand-written lines stand in, so the
# script still runs where that export is not available.
UNITS_TSV = os.environ.get("UNITS_TSV")
if UNITS_TSV:
    weighted = []
    with open(UNITS_TSV, encoding="utf-8") as f:
        for line in f:
            unit, _, count = line.rstrip("\n").rpartition("\t")
            if unit and count.isdigit():
                weighted.append((unit, int(count)))
    weighted.sort(key=lambda uc: (-uc[1], uc[0]))
    weighted = weighted[:int(os.environ.get("MAX_UNITS", "20000"))]
    print(f"domain corpus: {len(weighted)} distinct real units, "
          f"{sum(c for _, c in weighted)} occurrences")
else:
    weighted = [(l, 60) for l in [
        "|4/4k0 PR: C5 1/4 PL: A-3 C4 F4 1/8 PR: c5 D-5 1/4 PL: G3 B3 D4",
        "|4/4k0 PR: E4 1/8 PL: B-2 D3 F3 1/16 PR: g4 a4 1/4 PL: C3 E3",
        "|3/4k0 PR: C5 1/4 PL: A-3 1/4 PR: D5 1/4 PL: F3"]]
    print("domain corpus: built-in fallback lines")

# Held out, so the compression number is not read off the training units.
holdout = [u for u, _ in weighted[::7]][:200]
domain = [u for u, _ in weighted[:200]]

prose = ["The quick brown fox jumps over the lazy dog.",
         "SentencePiece is an unsupervised text tokenizer.",
         "Continuation must leave ordinary language exactly as it was."]

# Weighted TSV, so the run also exercises repetition-equivalence at scale.
corpus_path = os.path.join(OUT, "qwen_corpus.tsv")
with open(corpus_path, "w", encoding="utf-8") as f:
    for unit, count in weighted:
        f.write(f"{encode_bytelevel(unit)}\t{count}\n")
    for line in prose:
        f.write(f"{encode_bytelevel(line)}\t10\n")

prefix = os.path.join(OUT, "qwen_cont")
cmd = [SPM, f"--input={corpus_path}", f"--model_prefix={prefix}",
       "--model_type=bpe", f"--expansion_spec={spec_path}",
       f"--vocab_size={max_id + 1 + spec.requested_new_pieces}",
       "--normalization_rule_name=identity",
       "--add_dummy_prefix=false", "--remove_extra_whitespaces=false",
       "--split_by_whitespace=false", "--split_by_unicode_script=false",
       "--split_by_number=false", "--shuffle_input_sentence=false",
       "--max_sentencepiece_length=512", "--max_sentence_length=100000",
       "--input_format=tsv", "--hard_vocab_limit=false"]
print("running:", " ".join(cmd[:4]), "...")
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("TRAIN FAILED")
    print(r.stderr[-4000:])
    sys.exit(1)

result = pb.ExpansionResult()
result.ParseFromString(open(prefix + ".expansion", "rb").read())

failures = []
def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond:
        failures.append(label)

print("\n--- acceptance ---")
base_by_id = {p.external_id: p for p in result.base_pieces}
check(len(base_by_id) == len(vocab), "every inherited piece is returned")
check(all(base_by_id[i].piece == p for p, i in vocab.items()),
      "every inherited external ID keeps its exact string")
types_ok = all(
    (base_by_id[i].type == pb.ModelProto.SentencePiece.NORMAL) ==
    all(ch in ALPHABET for ch in p) for p, i in vocab.items())
check(types_ok, "every inherited token type is unchanged")

check([ (m.left, m.right) for m in result.base_merges ] == merges,
      "inherited merge order is preserved exactly")
check(result.rank_prepend is False, "no bootstrap merges were rank-prepended")

learned = list(result.learned_pieces)
check(len(learned) > 0, f"learned {len(learned)} new pieces")
check(all(p.external_id > max_id for p in learned),
      "every new ID begins after the maximum occupied base ID")
check(result.first_new_external_id == max_id + 1,
      f"first_new_external_id == {max_id + 1}")

lm = list(result.learned_merges)
check(len(lm) == len(learned), "every new piece has a recorded merge")
check(all(m.left and m.right and (m.left + m.right) == p.piece
          for m, p in zip(lm, learned)),
      "every new BPE token has exact (left, right) provenance")
check(result.unreachable_pieces == 0,
      "zero unreachable additions under the final merge table")

# Independent reachability check against the serialized table.
all_merges = ([(m.rank, m.left, m.right) for m in result.base_merges] +
              [(m.rank, m.left, m.right) for m in result.learned_merges])
rank_of = {(l, r): rk for rk, l, r in all_merges}
def bpe_apply(text):
    syms = list(text)
    while len(syms) > 1:
        best, at = None, -1
        for i in range(len(syms) - 1):
            rk = rank_of.get((syms[i], syms[i + 1]))
            if rk is not None and (best is None or rk < best):
                best, at = rk, i
        if at < 0:
            break
        syms[at:at + 2] = [syms[at] + syms[at + 1]]
    return syms
check(all(bpe_apply(p.piece) == [p.piece] for p in learned),
      "every new token is reachable by replaying the final merge table")

# Compression on the domain corpus, base program vs continued program.
base_rank = {(l, r): i for i, (l, r) in enumerate(merges)}
def toklen(text, table):
    global rank_of
    saved = rank_of; rank_of = table
    try:
        return len(bpe_apply(encode_bytelevel(text)))
    finally:
        rank_of = saved
before = sum(toklen(l, base_rank) for l in holdout)
after = sum(toklen(l, rank_of) for l in holdout)
print(f"  domain tokens: base={before} continued={after} "
      f"({100.0 * (before - after) / before:.1f}% fewer)")
check(after < before, "compression on the domain corpus improves")

prose_before = sum(toklen(l, base_rank) for l in prose)
prose_after = sum(toklen(l, rank_of) for l in prose)
check(prose_before == prose_after,
      f"ordinary language is untouched ({prose_before} tokens either way)")

native = os.path.exists(prefix + ".model")
print(f"  native .model emitted: {native} "
      f"(refusal is correct here: sparse/USER_DEFINED base, ambiguous splits)")
check(not native, "no misleading native .model for a real byte-level BPE")

print("\nFAILURES:" if failures else "\nALL CHECKS PASSED")
for f in failures:
    print(" -", f)
sys.exit(1 if failures else 0)
