"""Representative Stage-1 -> Stage-2 Unigram continuation acceptance.

Mirrors the violin/piano flow: a piano-only Stage-1 model is the authoritative
prior, and Stage 2 adds violin over a corpus that contains both. The point is
that Stage 1's score geometry survives while Stage 2 specializes.
"""
import os, random, subprocess, sys, glob
sys.path.insert(0, os.path.expanduser("~/repos/sentencepiece/python/src/sentencepiece"))
import sentencepiece_model_pb2 as pb

OUT = sys.argv[1]
BIN = os.path.expanduser("~/repos/sentencepiece/build/src")
SPM_TRAIN = os.path.join(BIN, "spm_train")

rng = random.Random(20260908)
PITCH_P = ["C3", "E3", "G3", "A-3", "B3", "D4", "F4", "C4"]
PITCH_V = ["E5", "A5", "D5", "G5", "B-5", "c5"]
DUR = ["1/4", "1/8", "1/16"]

def piano_line():
    parts = ["|4/4k0"]
    for _ in range(rng.randint(3, 6)):
        parts += ["PL:", rng.choice(PITCH_P), rng.choice(DUR)]
    return " ".join(parts)

def violin_line():
    parts = ["|4/4k0"]
    for _ in range(rng.randint(3, 6)):
        parts += ["Vn:", rng.choice(PITCH_V), rng.choice(DUR)]
    return " ".join(parts)

def write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# Stage 1: piano only. A few alphabet lines so character coverage includes the
# letters violin will need - the prior must be able to segment Stage 2 at all.
stage1 = [piano_line() for _ in range(3000)]
stage1 += ["Vn ABCDEFG abcdefg |/-0123456789: " for _ in range(20)]
s1_corpus = os.path.join(OUT, "stage1.txt")
write(s1_corpus, stage1)

s1_prefix = os.path.join(OUT, "stage1")
COMMON = ["--model_type=unigram", "--character_coverage=1.0",
          "--split_by_whitespace=true", "--split_by_unicode_script=false",
          "--split_by_number=false", "--num_threads=1",
          "--hard_vocab_limit=false"]
r = subprocess.run([SPM_TRAIN, f"--input={s1_corpus}",
                    f"--model_prefix={s1_prefix}", "--vocab_size=200"] + COMMON,
                   capture_output=True, text=True)
if r.returncode != 0:
    print("stage 1 failed\n", r.stderr[-3000:]); sys.exit(1)

prior = pb.ModelProto()
prior.ParseFromString(open(s1_prefix + ".model", "rb").read())
print(f"Stage 1: {len(prior.pieces)} pieces")

# Stage 2: piano AND violin. Violin is frequent enough to deserve pieces.
stage2 = [piano_line() for _ in range(1500)] + [violin_line() for _ in range(1500)]
rng.shuffle(stage2)
s2_corpus = os.path.join(OUT, "stage2.txt")
write(s2_corpus, stage2)

# The same corpus as a weighted TSV: physical repetition must be equivalent.
from collections import Counter
counts = Counter(stage2)
s2_tsv = os.path.join(OUT, "stage2.tsv")
write(s2_tsv, [f"{line}\t{n}" for line, n in sorted(counts.items())])
s2_rep = os.path.join(OUT, "stage2_rep.txt")
rep = []
for line, n in sorted(counts.items()):
    rep += [line] * n
write(s2_rep, rep)

TARGET = 260
def continue_run(tag, corpus, fmt):
    prefix = os.path.join(OUT, "stage2_" + tag)
    cmd = [SPM_TRAIN, f"--input={corpus}", f"--model_prefix={prefix}",
           f"--vocab_size={TARGET}", f"--unigram_prior_model={s1_prefix}.model",
           f"--input_format={fmt}"] + COMMON
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"stage 2 ({tag}) failed\n", r.stderr[-3000:]); sys.exit(1)
    m = pb.ModelProto(); m.ParseFromString(open(prefix + ".model", "rb").read())
    return prefix, m

p_tsv, m_tsv = continue_run("tsv", s2_tsv, "tsv")
p_rep, m_rep = continue_run("rep", s2_rep, "text")
p_a, m_a = continue_run("a", s2_corpus, "text")
p_b, m_b = continue_run("b", s2_corpus, "text")

failures = []
def check(cond, label):
    print(("  PASS  " if cond else "  FAIL  ") + label)
    if not cond: failures.append(label)

print("\n--- acceptance ---")
n_prior = len(prior.pieces)
check(len(m_a.pieces) > n_prior, f"extended {n_prior} -> {len(m_a.pieces)} pieces")

ids_ok = all(m_a.pieces[i].piece == prior.pieces[i].piece
             for i in range(n_prior))
types_ok = all(m_a.pieces[i].type == prior.pieces[i].type
               for i in range(n_prior))
check(ids_ok, "every Stage-1 ID keeps its exact string")
check(types_ok, "every Stage-1 token type is unchanged")
check(all(p.type == pb.ModelProto.SentencePiece.NORMAL
          for p in m_a.pieces[n_prior:]), "new IDs append after the whole prior")

lam = m_a.expansion_result.unigram_lambda
shifts = []
for i in range(n_prior):
    if prior.pieces[i].type != pb.ModelProto.SentencePiece.NORMAL:
        check(m_a.pieces[i].score == prior.pieces[i].score,
              f"non-NORMAL piece {prior.pieces[i].piece!r} score untouched")
        continue
    length = len(prior.pieces[i].piece)
    shifts.append((m_a.pieces[i].score - prior.pieces[i].score) / length)
spread = max(shifts) - min(shifts)
print(f"  lambda={lam:.6f}  per-piece shift/length spread={spread:.3e}")
check(spread < 1e-4, "one global additive-length gauge governs every inherited score")
check(abs(sum(shifts) / len(shifts) - lam) < 1e-3,
      "the measured gauge equals the reported lambda")
check(m_a.expansion_result.unigram_score_gauge_max_error < 1e-3,
      "reported gauge error is within float32 tolerance")

sp_new = [p.piece for p in m_a.pieces[n_prior:]]
violin_pieces = [p for p in sp_new if "Vn" in p or any(v in p for v in PITCH_V)]
print(f"  learned {len(sp_new)} pieces, {len(violin_pieces)} violin-specific")
check(len(violin_pieces) > 0,
      "frequent Stage-2 specializations win extension slots")

# Tail behaviour: a string the extension never covers must fall back exactly
# on Stage-1 geometry, i.e. the same pieces the prior would have chosen.
sys.path.insert(0, os.path.join(os.path.expanduser("~"), ".nonexistent"))
import sentencepiece  # from the venv
sp1 = sentencepiece.SentencePieceProcessor(model_file=s1_prefix + ".model")
sp2 = sentencepiece.SentencePieceProcessor(model_file=p_a + ".model")
# Extensions are ALLOWED to win where they are better - that is what they are
# for. The inherited guarantee is narrower and is what is tested here: where
# no extension piece takes part, the segmentation must be exactly Stage 1's,
# because the gauge cancels out of every comparison between inherited paths.
probes = ["|4/4k0 PL: A-3 1/16", "|4/4k0 PL: B3 1/8 PL: F4 1/4",
          "|4/4k0 Vn: E5 1/4"]
probes += ["".join(rng.choice("ABCDEFGabcdefg-0123456789:/| ") for _ in range(12))
           for _ in range(400)]

inherited_only = 0
mismatched = []
explained = 0
for t in probes:
    ids2 = sp2.encode(t)
    a = sp1.encode(t, out_type=str)
    b = sp2.encode(t, out_type=str)
    if all(i < n_prior for i in ids2):
        inherited_only += 1
        if a != b:
            mismatched.append((t, a, b))
    elif a != b:
        explained += 1

print(f"  probes: {len(probes)}, inherited-only {inherited_only}, "
      f"differing-but-uses-extensions {explained}")
check(inherited_only > 0, "some probe stays entirely on inherited pieces")
check(not mismatched,
      "where no extension piece takes part, segmentation is exactly Stage 1's")
for t, a, b in mismatched[:3]:
    print("    ", repr(t), a, "!=", b)

# And a difference is never unexplained: every changed segmentation used an
# extension piece.
check(all(sp1.encode(t, out_type=str) == sp2.encode(t, out_type=str) or
          any(i >= n_prior for i in sp2.encode(t)) for t in probes),
      "every changed segmentation is explained by an extension piece winning")

check([p.piece for p in m_tsv.pieces] == [p.piece for p in m_rep.pieces],
      "weighted TSV equals physical repetition (piece table)")
check(all(abs(a.score - b.score) < 1e-4
          for a, b in zip(m_tsv.pieces, m_rep.pieces)),
      "weighted TSV equals physical repetition (scores)")

check([p.piece for p in m_a.pieces] == [p.piece for p in m_b.pieces],
      "reruns are deterministic (piece table)")
check(all(a.score == b.score for a, b in zip(m_a.pieces, m_b.pieces)),
      "reruns are deterministic (scores)")

print("\nFAILURES:" if failures else "\nALL CHECKS PASSED")
for f in failures: print(" -", f)
sys.exit(1 if failures else 0)
