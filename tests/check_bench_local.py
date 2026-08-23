#!/usr/bin/env python3
"""Tests for the local benchmark, which is a ruler and therefore suspect.

A ruler that fails a correct answer produces a false score with every
appearance of rigour, and this project has paid for that once already. So the
tests here are not "does it run": they plant defects on both sides and demand
the ruler notice.

Nothing here talks to a model or to the network. The completions are written by
hand so the expected verdict is known before the ruler is asked.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import bench_local as bench   # noqa: E402


failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


# A real HumanEval row, copied verbatim from the released set, so the shapes
# below are the shapes the ruler actually meets.
PROBLEM = {
    "task_id": "HumanEval/999",
    "entry_point": "add_two",
    # `import math` is here on purpose and is used at run time by one of the
    # answers below. An annotation would not do: since PEP 649 an unresolvable
    # annotation raises nothing, so a test that leans on `List[int]` passes on
    # a ruler that throws the prompt's imports away. Measured on 3.14.6.
    "prompt": (
        "import math\n"
        "from typing import List\n"
        "\n"
        "\n"
        "def add_two(numbers: List[int]) -> List[int]:\n"
        '    """ Add two to every number, rounded down.\n'
        "    >>> add_two([1, 2])\n"
        "    [3, 4]\n"
        '    """\n'
    ),
    "canonical_solution": "    return [number + 2 for number in numbers]\n",
    "test": (
        "def check(candidate):\n"
        "    assert candidate([1, 2]) == [3, 4]\n"
        "    assert candidate([]) == []\n"
    ),
}


def run(completion, mark="MARK-abc123"):
    """Compile and run the judged program in this process, not in bwrap.

    The sandbox is exercised by check_execution.py and needs a privileged host.
    What is under test here is the program the ruler *builds*, so it is run
    with the same semantics and none of the containment cost.
    """
    program = bench.build_program(PROBLEM, completion, mark)
    namespace = {}
    printed = []
    namespace["print"] = lambda *args, **_kw: printed.append(" ".join(map(str, args)))
    try:
        exec(compile(program, "<judged>", "exec"), namespace)   # noqa: S102
    except BaseException as exc:                                # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", program
    return mark in printed, "", program


# --- the ruler must accept every shape a correct answer arrives in -----------

passed, why, _ = run(PROBLEM["canonical_solution"])
check(passed, f"a correct continuation of the given signature passes ({why})")

passed, why, _ = run(
    "```python\n" + PROBLEM["prompt"] + PROBLEM["canonical_solution"] + "```")
check(passed, f"a correct whole function inside a fenced block passes ({why})")

passed, why, _ = run(
    "Here you go:\n\n```python\n"
    "def add_two(numbers):\n"
    "    return [n + 2 for n in numbers]\n"
    "```\n\nHope that helps!")
check(passed,
      f"prose around a correct fenced answer does not fail the model ({why})")

# The prompt's imports have to survive when the model rewrites the function.
# A model that rewrites the whole function routinely leaves out the import the
# prompt already gave it, and dropping that import fails a correct answer for
# the ruler's reason. This answer calls `math` without importing it.
passed, why, _ = run(
    "```python\n"
    "def add_two(numbers: List[int]) -> List[int]:\n"
    "    return [math.floor(n + 2) for n in numbers]\n"
    "```")
check(passed,
      f"a rewritten function keeps the imports the prompt supplied ({why})")


# --- and it must reject every planted defect ---------------------------------

passed, _why, _ = run("    return numbers\n")
check(not passed, "a wrong continuation fails")

passed, _why, _ = run(
    "```python\n"
    "def add_two(numbers):\n"
    "    return [n + 3 for n in numbers]\n"
    "```")
check(not passed, "an off-by-one whole function fails")

passed, _why, _ = run("    return [number + 2 for number in numbers]\n"
                      "    # only the first assertion is satisfied\n")
check(passed, "a correct answer with a trailing comment still passes")

passed, _why, _ = run(
    "```python\n"
    "def add_two(numbers):\n"
    "    return [3, 4]\n"
    "```")
check(not passed,
      "a function hardcoded to the docstring example fails on the second case")

passed, _why, _ = run("")
check(not passed, "an empty answer fails instead of passing on the docstring")

passed, _why, _ = run("I cannot help with that.")
check(not passed, "prose with no code at all fails")


# --- the marker is the ruler, so the graded code must not be able to forge it -

FORGERY = (
    "```python\n"
    "print('MARK-abc123')\n"
    "def add_two(numbers):\n"
    "    return numbers\n"
    "```"
)
passed, _why, program = run(FORGERY)
check(not passed,
      "code that prints the marker itself and then answers wrong still fails")

# The reason it cannot forge it in a real run: the marker is drawn fresh and
# never shown to the model. Two programs built for the same answer must not
# share a marker.
first = bench.build_program(PROBLEM, PROBLEM["canonical_solution"], "MARK-1")
second = bench.build_program(PROBLEM, PROBLEM["canonical_solution"], "MARK-2")
check(first != second and "MARK-1" in first and "MARK-2" in second,
      "the pass marker is a parameter of the program, not a constant in it")
check("HUMANEVAL_PASS" not in bench.build_program(
          PROBLEM, PROBLEM["canonical_solution"], "MARK-1"),
      "no fixed marker string is left in the judged program for code to guess")


# --- the marker only prints after the test returned --------------------------

program = bench.build_program(PROBLEM, PROBLEM["canonical_solution"], "MARK-x")
check(program.rstrip().endswith("print('MARK-x')"),
      "the marker is the last statement, so it cannot print before check() ran")
check(program.index("check(add_two)") < program.index("print('MARK-x')"),
      "check() runs before the marker prints")


# --- the subset has to be the same subset for everybody ----------------------

rows = [{"task_id": f"HumanEval/{index}"} for index in range(164)]
shuffled = list(reversed(rows))
check(bench.subset(rows, 20) == bench.subset(shuffled, 20),
      "the graded subset does not depend on the order the file was read in")
check(len(bench.subset(rows, 20)) == 20, "the subset has the size asked for")
check(len(bench.subset(rows, 500)) == 164,
      "asking for more problems than exist grades all of them, not a slice")
spread = [int(row["task_id"].split("/")[1]) for row in bench.subset(rows, 20)]
check(spread == sorted(spread) and spread[-1] > 130,
      f"the subset spans the whole set instead of taking the easy head ({spread})")


# --- a set that is not the set must not be graded silently -------------------

class _Response:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


import gzip                                              # noqa: E402
import tempfile                                          # noqa: E402

short = gzip.compress(
    b"\n".join(json.dumps({"task_id": f"HumanEval/{index}"}).encode()
               for index in range(3)))
with tempfile.TemporaryDirectory() as folder:
    try:
        bench.load_humaneval(folder, urlopen_fn=lambda *_a, **_k: _Response(short))
        refused = False
    except SystemExit as exc:
        refused = "164" in str(exc)
check(refused,
      "a HumanEval file that is not 164 problems is refused, not graded")

with tempfile.TemporaryDirectory() as folder:
    Path(folder, "HumanEval.jsonl.gz").write_bytes(b"this is not gzip")
    try:
        bench.load_humaneval(folder)
        named = False
    except SystemExit as exc:
        named = "HumanEval.jsonl.gz" in str(exc) and "Delete it" in str(exc)
check(named,
      "a truncated cache names the file to delete instead of raising gzip at you")

try:
    bench.subset(rows, 0)
    refused_empty = False
except ValueError:
    refused_empty = True
check(refused_empty, "a battery of zero problems is refused, not run")


# --- the report cannot print a number without the machine it came from -------

MACHINE = {
    "date": "2026-01-02",
    "gpus": [{"name": "GTX 1650", "vram_mb": 4096, "bandwidth_gbs": 128.0}],
    "vram_mb_total": 4096, "ram_mb": 32000, "cpu_cores": 8,
}
META = {
    "model": "some-model", "artifact": "some.gguf", "artifact_bytes": 123,
    "artifact_sha256": "deadbeef", "backend": "llama.cpp Vulkan b1",
    "context": "16384", "machine": MACHINE,
    "humaneval_sha256": "cafe", "humaneval_total": 164,
}
HUMANEVAL_ROWS = [
    {"task_id": "HumanEval/0", "passed": True, "error": None,
     "server_tokens_per_second": 30.0, "predicted_tokens": 50},
    {"task_id": "HumanEval/8", "passed": False, "timed_out": True, "error": None,
     "server_tokens_per_second": 32.0, "predicted_tokens": 900},
]
TOOL_ROW = {"called_tools": [], "vias": [], "effect_correct": False,
            "steps": 0, "final_reply": "I would use write_file.", "error": None}
summary = bench.summarise(HUMANEVAL_ROWS, TOOL_ROW)
report = bench.render_report(META, summary, HUMANEVAL_ROWS, TOOL_ROW)

check("GTX 1650" in report and "4096 MiB" in report,
      "the report names the GPU and its VRAM")
check("llama.cpp Vulkan b1" in report, "the report names the backend and build")
check("2026-01-02" in report, "the report is dated")
check("deadbeef" in report and "some.gguf" in report,
      "the report names the exact file the score belongs to")
check("cafe" in report,
      "the report records the digest of the problem set it graded")
check("do not carry to other" in report and "other quantizations" in report,
      "the report says in words that the numbers do not carry elsewhere")
check(report.index("GTX 1650") < report.index("50.0%"),
      "the machine is stated above the first number, not in a footnote")
check("killed at the sandbox ceiling" in report,
      "a program the sandbox killed reads as that, not as a wrong answer")
check("I would use write_file." in report,
      "a model that called no tool has its own reply printed next to the verdict")

# The reply is usually itself a fenced block: the quoting fence has to survive
# that, or everything after it in the report renders as code.
FENCED_REPLY = dict(TOOL_ROW, final_reply="```json\n{\"name\": \"write_file\"}\n```")
fenced_report = bench.render_report(
    META, bench.summarise(HUMANEVAL_ROWS, FENCED_REPLY),
    HUMANEVAL_ROWS, FENCED_REPLY)
quoted = fenced_report.split('answered in prose:\n\n', 1)[1]
opening = quoted.splitlines()[0]
check(len(opening) > 3 and set(opening) == {"`"}
      and quoted.count(opening + "\n") == 2,
      f"a reply that is itself a code block is quoted by a longer fence ({opening!r})")
check("## How this was measured" in fenced_report.split(opening)[-1],
      "the report keeps rendering after a reply that contains backticks")
check(summary["humaneval_pass_rate"] == 50.0,
      f"the pass rate counts what was graded ({summary['humaneval_pass_rate']})")

# A run where the server was unreachable must not turn into a score of zero.
ERRORED = [
    {"task_id": "HumanEval/0", "passed": True, "error": None,
     "server_tokens_per_second": 30.0, "predicted_tokens": 50},
    {"task_id": "HumanEval/8", "passed": False, "error": "URLError: refused"},
]
errored_summary = bench.summarise(ERRORED, TOOL_ROW)
check(errored_summary["humaneval_pass_rate"] == 100.0
      and errored_summary["humaneval_errors"] == 1,
      "a problem that never reached the server is excluded, not counted wrong")
errored_report = bench.render_report(META, errored_summary, ERRORED, TOOL_ROW)
check("failed to reach the server" in errored_report,
      "the report says how many problems never reached the server")

empty_summary = bench.summarise([], TOOL_ROW)
check(empty_summary["humaneval_pass_rate"] is None
      and "not measured" in bench.render_report(
          META, empty_summary, [], TOOL_ROW),
      "with nothing graded the report says not measured instead of 0%")

# Throughput is absent, never estimated, when the server did not report it.
NO_TIMINGS = [{"task_id": "HumanEval/0", "passed": True, "error": None,
               "server_tokens_per_second": None, "predicted_tokens": None}]
quiet = bench.summarise(NO_TIMINGS, TOOL_ROW)
check(quiet["median_tokens_per_second"] is None,
      "no throughput is invented when the server reported no timings")
check("not measured" in bench.render_report(META, quiet, NO_TIMINGS, TOOL_ROW),
      "a missing throughput reads as not measured, not as zero")


print()
if failures:
    print(f"FAILED: {len(failures)}")
    for item in failures:
        print(f"  - {item}")
    sys.exit(1)
print("ISAAC BENCH LOCAL OK: the ruler accepts correct answers and rejects planted defects")
