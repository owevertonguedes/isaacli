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

# Throughput, the same split for the same reason. The catalog's own measured
# numbers, so a change here is visible against the file that holds them.
check(units.tps(33.02) == "33.0" and units.tps_short(33.02) == "33",
      "a measured throughput keeps its tenth in prose and loses it in a cell")
check(units.tps_short(29.37) == "29" and units.tps_short(29.6) == "30",
      "the cell form rounds to whole tokens rather than truncating downward")
check(units.tps(0) == "0.0" and units.tps_short(0) == "0",
      "zero is a number here; whether a zero is worth showing is the caller's")

# No module decides this for itself again. Bytes to GiB was hand-written ten
# times across five modules in two precisions, which is one quantity with two
# answers and nowhere to change either.
BY_HAND = re.compile(r"1024 ?\*\* ?3|1024 ?\* ?1024 ?\* ?1024|1073741824")
# The same for throughput: a rate formatted at the call site is how the two
# precisions of one number got three call sites apart from each other.
# Either the rate is the thing being formatted, or it is the placeholder being
# filled. Matching the whole line instead caught `rate_limit_wait` filling a
# number of seconds, which is a different quantity that happens to share four
# letters: a ruler that fails correct code is a defect of the ruler.
RATE_BY_HAND = re.compile(
    r"\{[^{}]*(rate|tps|tokens_per_second)[^{}]*:\.\df\}"
    r'|\b(rate|tps)\s*=\s*f"[^"]*:\.\df\}"')
handwritten = []
rates = []
for source in sorted((HERE.parent / "tool_harness").glob("*.py")):
    if source.name == "units.py":
        continue
    body = source.read_text(encoding="utf-8")
    for number, line in enumerate(body.splitlines(), 1):
        if BY_HAND.search(line):
            handwritten.append(f"{source.name}:{number}")
        if RATE_BY_HAND.search(line):
            rates.append(f"{source.name}:{number}")
check(not handwritten,
      "no module turns bytes into GiB by hand: " + (
          ", ".join(handwritten) or "none does"))
check(not rates,
      "no module formats a throughput by hand: " + (
          ", ".join(rates) or "none does"))

# Which of the two a place gets is the rule, so it is checked and not trusted:
# the short form belongs to a table cell, and every screen draws its cells
# through one function.
short_callers = []
for source in sorted((HERE.parent / "tool_harness").glob("*.py")):
    if source.name == "units.py":
        continue
    for number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1):
        if "gib_short(" in line:
            short_callers.append(f"{source.name}:{number}")
check(len(short_callers) == 1 and short_callers[0].startswith("model_discovery.py:"),
      "the one-decimal form is only used where a table cell is built: "
      + (", ".join(short_callers) or "nowhere at all"))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC UNITS OK: one size, one precision per place it is read")
