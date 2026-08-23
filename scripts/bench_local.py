#!/usr/bin/env python3
"""Measure a model on THIS machine and write the report that justifies listing it.

Why this exists: every score the catalogue carries was measured by somebody
else, on somebody else's hardware, inside somebody else's agent harness. That
answers "where did the number come from". It never answered "why is this model
on this list", and a recommendation without a measurement is a claim about a
machine nobody looked at.

So this script produces the missing half: throughput and behaviour of one
concrete artifact, on the card in this box, through the backend actually
installed, on a date. Nothing here is estimated. When a number cannot be
measured it is absent, not guessed.

Two rulers run, and they measure different things on purpose:

* **HumanEval**, the public set from `openai/human-eval`, pinned by the digest
  of the file that was downloaded. The judge is the dataset's own test, run
  under the project's real containment, never on the host. This is somebody
  else's ruler, which is the point: our own cases would grade our own taste.
* **Tool calls**, through the real `agent.py` loop with the real `SCHEMA`.
  HumanEval says nothing about this and it is the thing that decides whether a
  model is usable here at all: a model that writes perfect Python and cannot
  emit a tool call does no work in isaacli. Judged by effect on the file, never
  by the model saying it did it.

The report is a markdown file whose first section is the machine, because the
numbers below it mean nothing without it, and which says in words that they
describe hardware like this one and not the reader's.

Usage:

    scripts/bench_local.py --base-url http://127.0.0.1:8080/v1 \\
        --model qwen2.5-coder-3b-instruct-q4_k_m \\
        --artifact /path/to/the.gguf --backend "llama.cpp Vulkan b10502" \\
        --cases 20 --out docs/benchmarks
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tool_harness"))

import agent            # noqa: E402
import execution        # noqa: E402
import hardware         # noqa: E402
import tools            # noqa: E402

# The canonical release of the set, not a mirror and not a leaderboard page.
# A leaderboard is HTML that changes shape without telling anyone; this is a
# gzipped JSONL that has not moved since the paper. The digest is recorded in
# every report so a future run can prove it graded the same 164 problems.
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/"
    "HumanEval.jsonl.gz"
)
HUMANEVAL_TOTAL = 164

# Deterministic and frozen. A stride rather than the first N, because the first
# N of HumanEval are the easiest and shortest and would flatter every model
# equally. Same subset for every model or the comparison is worthless.
def subset(rows, count):
    """`count` problems spread evenly over the set, in a fixed order."""
    if count < 1:
        raise ValueError("a battery of no problems measures nothing")
    ordered = sorted(rows, key=lambda row: int(row["task_id"].split("/")[1]))
    if count >= len(ordered):
        return ordered
    stride = len(ordered) / float(count)
    return [ordered[int(index * stride)] for index in range(count)]


def load_humaneval(cache_dir, urlopen_fn=urllib.request.urlopen):
    """The problems plus the digest of the exact bytes they were read from."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "HumanEval.jsonl.gz"
    if cached.is_file():
        raw = cached.read_bytes()
    else:
        request = urllib.request.Request(
            HUMANEVAL_URL, headers={"User-Agent": "isaacli-bench"})
        with urlopen_fn(request, timeout=60) as response:
            raw = response.read()
        cached.write_bytes(raw)
    try:
        lines = gzip.decompress(raw).splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, EOFError, ValueError) as error:
        # A half-written cache from an interrupted run is the likely cause, and
        # a bare gzip traceback would send the reader looking at the network.
        raise SystemExit(
            f"{cached} is not a readable HumanEval archive ({error}). "
            "Delete it and run again."
        ) from error
    if len(rows) != HUMANEVAL_TOTAL:
        raise SystemExit(
            f"HumanEval should hold {HUMANEVAL_TOTAL} problems, this file holds "
            f"{len(rows)}. Refusing to grade against a set I cannot recognise."
        )
    return rows, hashlib.sha256(raw).hexdigest()


FENCE = re.compile(r"```(?:python|py)?\n(.*?)(?:```|\Z)", re.S)


def extract_code(text):
    """The code out of a chat reply, whatever wrapping the model chose.

    A chat model answers HumanEval with prose and a fenced block far more often
    than with a bare continuation, and grading the prose as Python marks a
    correct model wrong. That failure looks exactly like incompetence and is
    ours, so the first fenced block wins when there is one.
    """
    match = FENCE.search(text or "")
    return (match.group(1) if match else (text or "")).rstrip()


def _preamble(prompt, entry_point):
    """Everything in the prompt above the signature: the imports it needs."""
    marker = f"def {entry_point}"
    head = prompt.split(marker)[0] if marker in prompt else ""
    return head


def build_program(problem, completion, mark):
    """The exact file the judge runs, for either shape of answer.

    Two shapes exist and both are legitimate. A completion model continues the
    signature it was given; a chat model rewrites the whole function. Grading
    only one of them would reject the other for a formatting difference, so the
    shape is detected and the prompt's imports are carried across either way.

    `mark` is printed only after `check()` has returned, and is drawn fresh for
    every program. A fixed marker would be a ruler the graded code can forge:
    a model that writes `print("HUMANEVAL_PASS")` anywhere in its answer would
    score a pass on a function that never ran. The model cannot print a string
    it has not been shown.
    """
    entry_point = problem["entry_point"]
    code = extract_code(completion)
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.M):
        body = _preamble(problem["prompt"], entry_point) + "\n" + code
    else:
        body = problem["prompt"] + code
    return (
        f"{body}\n\n\n{problem['test']}\n\n"
        f"check({entry_point})\n"
        f"print({mark!r})\n"
    )


def judge(problem, completion, workspace, run_command=None, mark=None):
    """Run the dataset's own test under the project's containment.

    Never on the host: this executes code a model wrote, which is the exact
    thing `execution.py` exists to contain. The verdict is the fresh marker
    printed after `check()` returned, not the exit code alone, because a
    sandbox refusal also exits non-zero and must not read as a wrong answer.
    """
    run_command = run_command or execution.run_command
    mark = mark or "HUMANEVAL_PASS_" + secrets.token_hex(8)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    name = problem["task_id"].replace("/", "_") + ".py"
    (workspace / name).write_text(build_program(problem, completion, mark),
                                  encoding="utf-8")
    previous_root = tools.SANDBOX_ROOT
    tools.SANDBOX_ROOT = workspace
    try:
        output = run_command(f"python3 {name}", authorized=True)
    finally:
        tools.SANDBOX_ROOT = previous_root
    return mark in output, output


# --- the machine, which is the part that makes the numbers mean anything -----

def machine():
    detected = hardware.detect()
    gpus = detected.get("gpus") or []
    return {
        "date": dt.date.today().isoformat(),
        "gpus": [
            {
                "name": gpu.get("name"),
                "vram_mb": gpu.get("vram_mb"),
                "bandwidth_gbs": gpu.get("bandwidth_gbs"),
            }
            for gpu in gpus
        ],
        "vram_mb_total": sum(gpu.get("vram_mb", 0) for gpu in gpus),
        "ram_mb": detected.get("ram_mb"),
        "cpu_cores": detected.get("cpu_cores"),
    }


# --- talking to the server ---------------------------------------------------

def complete(base_url, model, prompt, max_tokens, timeout,
             urlopen_fn=urllib.request.urlopen):
    """One deterministic completion, with the server's own timings kept.

    Throughput is taken from the server rather than from a stopwatch here,
    because a stopwatch on this side also times the network and the client, and
    the report has to be able to say which of the two it printed.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urlopen_fn(request, timeout=timeout) as response:
        body = json.loads(response.read())
    wall = time.monotonic() - started
    return {
        "text": (body.get("choices") or [{}])[0].get("message", {}).get(
            "content") or "",
        "timings": body.get("timings") or {},
        "usage": body.get("usage") or {},
        "wall_seconds": wall,
    }


HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Reply with the complete function "
    "in a single ```python code block and nothing else. Do not write tests, "
    "explanations or examples.\n\n```python\n{prompt}```\n"
)


def run_humaneval(args, problems, run_dir):
    rows = []
    for index, problem in enumerate(problems, start=1):
        prompt = HUMANEVAL_INSTRUCTION.format(prompt=problem["prompt"])
        try:
            answer = complete(args.base_url, args.model, prompt,
                              args.max_tokens, args.request_timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # An unreachable server is not a wrong answer, and recording it as
            # one would invent a score. It is recorded as an error and the
            # report prints how many there were next to the total.
            rows.append({"task_id": problem["task_id"], "passed": False,
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{index}/{len(problems)}] {problem['task_id']}: ERROR {exc}")
            continue
        passed, output = judge(
            problem, answer["text"], run_dir / "humaneval" / problem["task_id"].replace("/", "_"))
        timings = answer["timings"]
        rows.append({
            "task_id": problem["task_id"],
            "passed": passed,
            # A program the sandbox killed at its ceiling is still a failure,
            # but it is a different failure from a wrong answer, and a reader
            # who cannot tell them apart cannot tell a slow solution from a
            # broken one.
            "timed_out": (not passed) and "TIMED OUT" in output,
            "error": None,
            "server_tokens_per_second": timings.get("predicted_per_second"),
            "predicted_tokens": timings.get("predicted_n"),
            "prompt_tokens": timings.get("prompt_n"),
            "wall_seconds": round(answer["wall_seconds"], 3),
            "judge_tail": output.strip().splitlines()[-3:] if not passed else [],
        })
        print(f"  [{index}/{len(problems)}] {problem['task_id']}: "
              f"{'pass' if passed else 'FAIL'}")
    return rows


# --- the tool-call ruler -----------------------------------------------------

TOOL_CASE = {
    "prompt": (
        "Create the file result.txt, which does not exist yet, containing "
        "exactly two lines: alpha and beta. The file must end with a newline. "
        "Use the appropriate file tool, do not use a terminal command."
    ),
    "path": "result.txt",
    "expected": b"alpha\nbeta\n",
}


def run_tool_case(args, run_dir):
    """Does this model drive the harness at all, and by effect not by claim."""
    workspace = run_dir / "toolcall"
    workspace.mkdir(parents=True, exist_ok=True)
    previous_root = tools.SANDBOX_ROOT
    tools.SANDBOX_ROOT = workspace
    calls = []
    error = None
    response = None
    try:
        response = agent.run(
            TOOL_CASE["prompt"], args.model, max_steps=args.max_steps,
            verbose=False, history=[],
            provider={"provider": "openai_compatible", "base_url": args.base_url},
            on_tool=lambda name, raw, result, via: calls.append((name, via)),
            temperature=0,
        )
    except Exception as exc:                     # noqa: BLE001 - reported, not hidden
        error = f"{type(exc).__name__}: {exc}"
    finally:
        tools.SANDBOX_ROOT = previous_root
    target = workspace / TOOL_CASE["path"]
    actual = target.read_bytes() if target.is_file() else None
    # The model's own last words are kept because "no tool was called" has two
    # very different causes: the harness never offered the tools, or the model
    # answered in prose. Only the reply itself tells them apart, and blaming
    # the wrong one is how a setup defect gets recorded as a model defect.
    final = ((response or {}).get("final") or "").strip()
    return {
        "called_tools": [name for name, _via in calls],
        "vias": [via for _name, via in calls],
        "effect_correct": actual == TOOL_CASE["expected"],
        "steps": (response or {}).get("steps"),
        "final_reply": final[:600],
        "error": error,
    }


# --- the report --------------------------------------------------------------

DISCLAIMER = (
    "These numbers describe **this machine and this file**, on the date above. "
    "They are not a ranking of the models and they do not carry to other "
    "hardware, other quantizations of the same weights, or other runtimes. "
    "Read a row as: on a machine like this one, this artifact behaved like "
    "this. A newer or larger model may well be better and simply not run here."
)


def _median(values):
    kept = sorted(value for value in values if value)
    if not kept:
        return None
    middle = len(kept) // 2
    if len(kept) % 2:
        return kept[middle]
    return (kept[middle - 1] + kept[middle]) / 2


def summarise(humaneval_rows, tool_row):
    graded = [row for row in humaneval_rows if not row.get("error")]
    errored = [row for row in humaneval_rows if row.get("error")]
    speeds = [row.get("server_tokens_per_second") for row in graded]
    return {
        "humaneval_graded": len(graded),
        "humaneval_errors": len(errored),
        "humaneval_passed": sum(1 for row in graded if row["passed"]),
        "humaneval_pass_rate": (
            round(100.0 * sum(1 for row in graded if row["passed"]) / len(graded), 1)
            if graded else None
        ),
        "median_tokens_per_second": (
            round(_median(speeds), 2) if _median(speeds) else None
        ),
        "tokens_per_second_samples": len([value for value in speeds if value]),
        "tool_call_effect_correct": tool_row["effect_correct"],
        "tool_calls_made": tool_row["called_tools"],
    }


def render_report(meta, summary, humaneval_rows, tool_row):
    lines = []
    lines.append(f"# {meta['model']} on {meta['machine']['gpus'][0]['name'] if meta['machine']['gpus'] else 'CPU'}, {meta['machine']['date']}")
    lines.append("")
    lines.append("## The machine these numbers came from")
    lines.append("")
    lines.append("| | |")
    lines.append("| --- | --- |")
    for gpu in meta["machine"]["gpus"] or [{"name": "none detected", "vram_mb": 0}]:
        lines.append(f"| GPU | {gpu['name']} |")
        lines.append(f"| VRAM | {gpu['vram_mb']} MiB |")
    lines.append(f"| System RAM | {meta['machine']['ram_mb']} MiB |")
    lines.append(f"| CPU cores | {meta['machine']['cpu_cores']} |")
    lines.append(f"| Backend | {meta['backend']} |")
    lines.append(f"| Context served | {meta['context']} |")
    lines.append(f"| Date | {meta['machine']['date']} |")
    lines.append("")
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append("## The artifact")
    lines.append("")
    lines.append("| | |")
    lines.append("| --- | --- |")
    lines.append(f"| File | `{meta['artifact']}` |")
    lines.append(f"| Bytes | {meta['artifact_bytes']} |")
    lines.append(f"| SHA-256 | `{meta['artifact_sha256']}` |")
    lines.append(f"| Served as | `{meta['model']}` |")
    lines.append("")
    lines.append(
        "The score belongs to this file. It does not belong to the weights it "
        "was quantized from, to another quantization of them, or to a fine-tune "
        "of them."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append("| ruler | result |")
    lines.append("| --- | --- |")
    rate = summary["humaneval_pass_rate"]
    lines.append(
        f"| HumanEval, {summary['humaneval_graded']} problems of "
        f"{meta['humaneval_total']} | "
        + (f"{summary['humaneval_passed']}/{summary['humaneval_graded']} "
           f"= {rate}% pass@1" if rate is not None else "not measured")
        + " |"
    )
    speed = summary["median_tokens_per_second"]
    lines.append(
        "| Generation throughput | "
        + (f"{speed} tok/s median over {summary['tokens_per_second_samples']} "
           "generations, reported by the server itself"
           if speed else "not measured")
        + " |"
    )
    lines.append(
        "| Native tool call | "
        + ("the file landed with the exact expected bytes, tools called: "
           f"`{'`, `'.join(summary['tool_calls_made'])}`"
           if summary["tool_call_effect_correct"]
           else "**no**, the file did not land as asked"
                + (f" (error: {tool_row['error']})" if tool_row.get("error") else "")
                + (f" (tools called: `{'`, `'.join(tool_row['called_tools'])}`)"
                   if tool_row["called_tools"] else " (no tool was called)"))
        + " |"
    )
    if not summary["tool_call_effect_correct"] and tool_row.get("final_reply"):
        lines.append("")
        lines.append(
            "The model's own reply to that request, so the reader can see "
            "whether the harness failed to offer the tools or the model "
            "answered in prose:"
        )
        lines.append("")
        # The reply is very often itself a fenced code block, so the fence that
        # quotes it has to be longer than the longest run of backticks inside
        # it. A three-backtick fence closes on the model's own fence and the
        # rest of the report renders as code.
        longest = max((len(run) for run in re.findall(
            r"`+", tool_row["final_reply"])), default=0)
        fence = "`" * max(3, longest + 1)
        lines.append(fence)
        lines.append(tool_row["final_reply"])
        lines.append(fence)
    if summary["humaneval_errors"]:
        lines.append(
            f"| Not graded | {summary['humaneval_errors']} problems failed to "
            "reach the server and are excluded from the rate above |"
        )
    lines.append("")
    lines.append("## How this was measured")
    lines.append("")
    lines.append(
        f"- HumanEval was read from `{HUMANEVAL_URL}`, whose bytes hash to "
        f"`{meta['humaneval_sha256']}`."
    )
    lines.append(
        f"- The {summary['humaneval_graded'] + summary['humaneval_errors']} "
        "problems are a fixed stride over the 164 sorted task ids, so every "
        "model is graded on the same ones."
    )
    lines.append(
        "- The judge is the dataset's own `check()`, run inside the project's "
        "sandbox (`execution.run_command`), never on the host."
    )
    lines.append("- Temperature 0.")
    lines.append(
        "- Throughput is the server's own `predicted_per_second`, so it "
        "excludes this client and the socket."
    )
    lines.append(
        "- The tool-call row is judged by the bytes on disk, not by what the "
        "model said it did."
    )
    lines.append("")
    lines.append("## Every problem")
    lines.append("")
    lines.append("| task | verdict | tok/s | generated tokens |")
    lines.append("| --- | --- | --- | --- |")
    for row in humaneval_rows:
        if row.get("error"):
            verdict = f"not graded: {row['error']}"
        elif row["passed"]:
            verdict = "pass"
        elif row.get("timed_out"):
            verdict = f"fail, killed at the sandbox ceiling of {execution.TIMEOUT_SECONDS}s"
        else:
            verdict = "fail"
        speed = row.get("server_tokens_per_second")
        lines.append(
            f"| {row['task_id']} | {verdict} | "
            f"{round(speed, 2) if speed else ''} | "
            f"{row.get('predicted_tokens') or ''} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", required=True,
                        help="the name the server answers to")
    parser.add_argument("--artifact", required=True,
                        help="path to the GGUF actually being served")
    parser.add_argument("--backend", required=True,
                        help='e.g. "llama.cpp Vulkan b10502"')
    parser.add_argument("--context", required=True,
                        help="the context the server was started with")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "benchmarks"))
    parser.add_argument("--work", default=None,
                        help="scratch directory (default: a temp dir)")
    args = parser.parse_args(argv)

    artifact = Path(args.artifact)
    if not artifact.is_file():
        raise SystemExit(f"no such artifact: {artifact}")

    run_dir = Path(args.work) if args.work else Path(
        f"/tmp/isaacli-bench-{dt.datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading HumanEval into {run_dir}")
    rows, humaneval_sha = load_humaneval(run_dir / "cache")
    problems = subset(rows, args.cases)

    print(f"grading {len(problems)} problems against {args.model}")
    humaneval_rows = run_humaneval(args, problems, run_dir)
    print("running the tool-call case")
    tool_row = run_tool_case(args, run_dir)

    summary = summarise(humaneval_rows, tool_row)
    meta = {
        "model": args.model,
        "artifact": artifact.name,
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": _sha256_file(artifact),
        "backend": args.backend,
        "context": args.context,
        "machine": machine(),
        "humaneval_sha256": humaneval_sha,
        "humaneval_total": HUMANEVAL_TOTAL,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^a-z0-9]+", "-", args.model.casefold()).strip("-")
    report_path = out_dir / f"{meta['machine']['date']}-{stem}.md"
    report_path.write_text(
        render_report(meta, summary, humaneval_rows, tool_row), encoding="utf-8")
    (out_dir / f"{meta['machine']['date']}-{stem}.json").write_text(
        json.dumps({"meta": meta, "summary": summary,
                    "humaneval": humaneval_rows, "toolcall": tool_row},
                   indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {report_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
