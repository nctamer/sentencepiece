"""Local scenario tests for tools/gen_stubs.py.

Runs the generator under controlled mutations of the imported ``sentencepiece``
module to prove its key guarantees without touching CI:

  A. The committed stub matches a fresh generation (the CI --check gate).
  B. A newly added public method is auto-discovered, typed permissively,
     flagged on stderr, and makes --check fail (drift detection).
  C. A removed method disappears and makes --check fail.
  D. A new property is emitted as ``Any`` and flagged.
  E. A leaked module-level import is ignored.
  F. Output is deterministic.

Run from the repo root:  python python/tools/gen_stubs_scenarios.py
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_stubs  # noqa: E402
import sentencepiece as spm  # noqa: E402

STUB_PATH = os.path.join(
    HERE, os.pardir, "src", "sentencepiece", "__init__.pyi"
)

PASS = "PASS"
_failures = []


def check(label, condition):
  status = PASS if condition else "FAIL"
  if not condition:
    _failures.append(label)
  print(f"  [{status}] {label}")


def main():
  with open(STUB_PATH, encoding="utf-8") as f:
    committed = f.read()
  spp = spm.SentencePieceProcessor

  print("A: committed stub is in sync with a fresh generation")
  text, _ = gen_stubs.generate()
  check("committed __init__.pyi == generate() output (CI --check passes)",
        text == committed)

  print("B: a newly added upstream method is handled")
  spp.BrandNewParallelThing = lambda self: None
  try:
    text_b, uncurated_b = gen_stubs.generate()
    check("new method 'BrandNewParallelThing' auto-discovered in output",
          "def BrandNewParallelThing(" in text_b)
    check("derived alias is NOT invented (only real runtime names emitted)",
          "def brand_new_parallel_thing(" not in text_b)
    check("new method flagged as uncurated on stderr",
          any("BrandNewParallelThing" in u for u in uncurated_b))
    check("new method gets a permissive fallback signature",
          "def BrandNewParallelThing(self, *args: Any, **kwargs: Any) -> Any: ..."
          in text_b)
    check("drift detected: --check would now FAIL",
          text_b != committed)
  finally:
    del spp.BrandNewParallelThing
  text_b2, _ = gen_stubs.generate()
  check("removing the method returns output to in-sync state",
        text_b2 == committed)

  print("C: a removed method is handled")
  original_encode = spp.Encode
  del spp.Encode
  try:
    text_c, _ = gen_stubs.generate()
    check("'def Encode(' no longer emitted after removal",
          "def Encode(" not in text_c)
    check("drift detected on removal: --check would FAIL",
          text_c != committed)
  finally:
    spp.Encode = original_encode
  text_c2, _ = gen_stubs.generate()
  check("restoring the method returns output to in-sync state",
        text_c2 == committed)

  print("D: a new property is handled")
  spp.brand_new_prop = property(lambda self: 1)
  try:
    text_d, uncurated_d = gen_stubs.generate()
    check("new property emitted as '-> Any'",
          "def brand_new_prop(self) -> Any: ..." in text_d)
    check("new property flagged as uncurated",
          any("brand_new_prop" in u for u in uncurated_d))
  finally:
    del spp.brand_new_prop

  print("E: a leaked module-level import is ignored")
  spm.some_leaked_module = types.ModuleType("some_leaked_module")
  spm.some_leaked_constant = 1234
  try:
    text_e, _ = gen_stubs.generate()
    check("leaked module not emitted",
          "some_leaked_module" not in text_e)
    check("leaked non-callable constant not emitted",
          "some_leaked_constant" not in text_e)
  finally:
    del spm.some_leaked_module
    del spm.some_leaked_constant

  print("F: output is deterministic")
  first, _ = gen_stubs.generate()
  second, _ = gen_stubs.generate()
  check("two consecutive generations are byte-identical", first == second)

  print()
  if _failures:
    print(f"{len(_failures)} scenario(s) FAILED:")
    for name in _failures:
      print(f"  - {name}")
    return 1
  print("All scenarios passed.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
