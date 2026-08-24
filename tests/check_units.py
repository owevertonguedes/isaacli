#!/usr/bin/env python3
"""Tests for the one place that turns bytes into a size a person reads.

The point is not that the arithmetic is right, which is a division. It is that
there is one answer to "how precise is a size" per kind of place it appears,
and that no module goes back to deciding that for itself.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import units


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


GIB = 1024 ** 3

check(units.BYTES_PER_GIB == GIB, "a GiB is 2^30 bytes, not 10^9")
check(units.gib(2 * GIB) == "2.00" and units.gib_short(2 * GIB) == "2.0",
      "the same size reads with two decimals in prose and one in a cell")
check(units.gib(0) == "0.00" and units.gib_short(0) == "0.0",
      "zero bytes is a size like any other, and the caller decides if it is shown")
check(units.gib(None) == "0.00" and units.gib_short(None) == "0.0",
      "an absent size does not raise here: nothing is a bad reason for a traceback")

# The distinction the two functions exist for. Two neighbouring quantizations
# of one model, 33 MiB apart: the screen that chooses between them has to show
# a difference, and one decimal does not.
small, large = 4_700_000_000, 4_735_000_000
check(units.gib(small) != units.gib(large),
      "two quantizations 33 MiB apart are told apart by the prose form")
check(units.gib_short(small) == units.gib_short(large),
      "and are not by the cell form, which is why choosing between them uses prose")

check(units.gib(1_500_000_000) == "1.40" and units.gib_short(1_500_000_000) == "1.4",
      "both forms round, they do not truncate")

# No module decides this for itself again. Bytes to GiB was hand-written ten
# times across five modules in two precisions, which is one quantity with two
# answers and nowhere to change either.
BY_HAND = re.compile(r"1024 ?\*\* ?3|1024 ?\* ?1024 ?\* ?1024|1073741824")
handwritten = []
for source in sorted((HERE.parent / "tool_harness").glob("*.py")):
    if source.name == "units.py":
        continue
    body = source.read_text(encoding="utf-8")
    for number, line in enumerate(body.splitlines(), 1):
        if BY_HAND.search(line):
            handwritten.append(f"{source.name}:{number}")
check(not handwritten,
      "no module turns bytes into GiB by hand: " + (
          ", ".join(handwritten) or "none does"))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC UNITS OK: one size, one precision per place it is read")
