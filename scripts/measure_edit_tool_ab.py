#!/usr/bin/env python3
"""Measure file-edit tool choice with and without replace_between.

The script talks to an already running OpenAI-compatible endpoint. Benchmark
sessions are written below the run directory, never to cli_sessions.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tool_harness"))

import agent
import tools
from cli_sessions import _build_history


TEMPERATURE = 0.7
SEED_BASE = 21_000
CONDITION_TOOLS = {
    "A": tuple(item["function"]["name"] for item in tools.SCHEMA),
    "B": tuple(
        item["function"]["name"] for item in tools.SCHEMA
        if item["function"]["name"] != "replace_between"
    ),
}


def _filtered_schema(names):
    """The subset of tools one condition offers the model.

    Lives here rather than in `tools`, which is imported by every session: this
    measurement is the only thing that has ever narrowed the schema, and the
    program itself always offers all of it.
    """
    allowed = set(names)
    return [item for item in tools.SCHEMA if item["function"]["name"] in allowed]


CASES = {
    "write": {
        "prompt": (
            "Crie o arquivo resultado.txt, que não existe, com o conteúdo exato "
            "de duas linhas: alpha e beta. O arquivo deve terminar com quebra de "
            "linha. Use a ferramenta de arquivo apropriada, não use comando de terminal."
        ),
        "path": "resultado.txt",
        "initial": None,
        "expected": b"alpha\nbeta\n",
        "tool": "write_file",
    },
    "append": {
        "prompt": (
            "Acrescente exatamente a linha fim ao final de registro.txt, preservando "
            "todo o conteúdo existente. O arquivo deve terminar com quebra de linha. "
            "Use a ferramenta de arquivo apropriada, não use comando de terminal."
        ),
        "path": "registro.txt",
        "initial": b"inicio\n",
        "expected": b"inicio\nfim\n",
        "tool": "append_file",
    },
    "replace": {
        "prompt": (
            "No arquivo config.txt, troque o trecho exato conhecido cor=azul por "
            "cor=verde, preservando todo o restante. Use a ferramenta de arquivo "
            "apropriada, não use comando de terminal."
        ),
        "path": "config.txt",
        "initial": b"cor=azul\nmodo=teste\n",
        "expected": b"cor=verde\nmodo=teste\n",
        "tool": "replace_text",
    },
    "between": {
        "prompt": (
            "No arquivo secoes.txt, substitua todo o miolo entre os marcadores "
            "SECTION_START e SECTION_END por exatamente duas linhas: novo e conteudo. "
            "Preserve os marcadores, o cabeçalho e o rodapé. Você não conhece o miolo "
            "atual e não precisa reproduzi-lo. Use a ferramenta de arquivo apropriada, "
            "não use comando de terminal."
        ),
        "path": "secoes.txt",
        "initial": None,
        "expected": (
            b"cabecalho\n<!-- SECTION_START -->\nnovo\nconteudo\n"
            b"<!-- SECTION_END -->\nrodape\n"
        ),
        "tool": {"A": "replace_between", "B": "replace_text"},
    },
}


def _default_output() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return cache / "isaacli-021" / stamp


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _event(kind: str, common: dict, **fields) -> dict:
    return {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "type": kind,
        "benchmark": "task-021",
        **common,
        **fields,
    }


def _prepare_workspace(workspace: Path, case_name: str, repetition: int) -> None:
    case = CASES[case_name]
    workspace.mkdir(parents=True)
    initial = case["initial"]
    if case_name == "between":
        initial = (
            "cabecalho\n<!-- SECTION_START -->\n"
            f"conteudo-oculto-021-{repetition}\nnao-reproduzir-{repetition}\n"
            "<!-- SECTION_END -->\nrodape\n"
        ).encode()
    if initial is not None:
        (workspace / case["path"]).write_bytes(initial)


def _expected_tool(case_name: str, condition: str) -> str:
    expected = CASES[case_name]["tool"]
    return expected[condition] if isinstance(expected, dict) else expected


def _run_one(args, run_dir: Path, condition: str, case_name: str,
             repetition: int, seed: int) -> dict:
    case = CASES[case_name]
    label = f"{condition}-{case_name}-{repetition}"
    workspace = run_dir / "workspaces" / label
    _prepare_workspace(workspace, case_name, repetition)
    tools.SANDBOX_ROOT = workspace
    session_id = str(uuid.uuid4())
    common = {
        "session_id": session_id,
        "model": args.model,
        "workspace": str(workspace),
        "condition": condition,
        "case": case_name,
        "repetition": repetition,
        "temperature": args.temperature,
        "seed": seed,
    }
    events = [_event("meta", common, event="start")]
    events.append(_event("user", common, content=case["prompt"]))

    def before(name, raw_args):
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            parsed = raw_args
        events.append(_event("tool_start", common, name=name, args=parsed))

    def after(name, raw_args, result, via):
        events.append(_event("tool_result", common, name=name, result=result, via=via))

    error = None
    response = None
    try:
        response = agent.run(
            case["prompt"], args.model, max_steps=args.max_steps, verbose=False,
            history=_build_history(workspace),
            tools_schema=_filtered_schema(CONDITION_TOOLS[condition]),
            provider={"provider": "openai_compatible", "base_url": args.base_url},
            require_change=True, on_tool_before=before, on_tool=after,
            temperature=args.temperature, seed=seed,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        events.append(_event("error", common, error=error))

    response = response or {}
    events.append(_event(
        "assistant_final", common, content=response.get("final") or "",
        usage=response.get("usage") or {}, calls=len(response.get("calls") or []),
        steps=response.get("steps"), error=error,
    ))
    _write_jsonl(run_dir / "sessions" / f"{session_id}.jsonl", events)

    target = workspace / case["path"]
    actual = target.read_bytes() if target.is_file() else None
    changing = [call for call in response.get("calls") or []
                if call[0] in agent.CHANGING_TOOLS]
    chosen = changing[0][0] if changing else None
    effect_correct = actual == case["expected"]
    trace = [chosen, response.get("steps"), effect_correct]
    return {
        **common,
        "chosen": chosen,
        "expected_tool": _expected_tool(case_name, condition),
        "choice_correct": chosen == _expected_tool(case_name, condition),
        "effect_correct": effect_correct,
        "steps": response.get("steps"),
        "calls": len(response.get("calls") or []),
        "constrained_calls": sum(call[3] == "constrained"
                                 for call in response.get("calls") or []),
        "tools": [call[0] for call in response.get("calls") or []],
        "vias": [call[3] for call in response.get("calls") or []],
        "trace": trace,
        "workspace_sha256": hashlib.sha256(actual or b"").hexdigest(),
        "actual_hex": None if actual is None else actual.hex(),
        "error": error,
        "session_log": str(run_dir / "sessions" / f"{session_id}.jsonl"),
    }


def _print_summary(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["case"])].append(row)
    print("condition\tcase\truns\tdistinct_traces\tdistinct_outputs\tchoice\teffect")
    for condition in ("A", "B"):
        for case_name in CASES:
            cell = grouped[(condition, case_name)]
            traces = {json.dumps(row["trace"], sort_keys=True) for row in cell}
            outputs = {row["workspace_sha256"] for row in cell}
            choice = sum(row["choice_correct"] for row in cell)
            effect = sum(row["effect_correct"] for row in cell)
            print(
                f"{condition}\t{case_name}\t{len(cell)}\t{len(traces)}\t"
                f"{len(outputs)}\t{choice}/{len(cell)}\t{effect}/{len(cell)}"
            )
            variants = Counter(tuple(row["trace"]) for row in cell)
            for trace, count in sorted(variants.items(), key=lambda item: repr(item[0])):
                print(f"  trace={trace!r}\tcount={count}")

    print("\ncondition\truns\tchoice\teffect\tmean_steps\tcalls\tconstrained")
    for condition in ("A", "B"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        steps = [row["steps"] for row in condition_rows
                 if isinstance(row["steps"], int)]
        print(
            f"{condition}\t{len(condition_rows)}\t"
            f"{sum(row['choice_correct'] for row in condition_rows)}/{len(condition_rows)}\t"
            f"{sum(row['effect_correct'] for row in condition_rows)}/{len(condition_rows)}\t"
            f"{sum(steps) / len(steps):.3f}\t"
            f"{sum(row['calls'] for row in condition_rows)}\t"
            f"{sum(row['constrained_calls'] for row in condition_rows)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default="qwen3-4b-task021")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be above zero for this benchmark")

    run_dir = (args.output or _default_output()).expanduser().resolve()
    if run_dir.exists():
        parser.error(f"output directory already exists: {run_dir}")
    (run_dir / "sessions").mkdir(parents=True)
    metadata = {
        "benchmark": "task-021",
        "model": args.model,
        "base_url": args.base_url,
        "repetitions": args.repetitions,
        "temperature": args.temperature,
        "seed_base": args.seed_base,
        "seed_design": "same seed for each paired A/B run; distinct by case and repetition",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    results_path = run_dir / "results.jsonl"
    rows = []
    for repetition in range(1, args.repetitions + 1):
        conditions = ("A", "B") if repetition % 2 else ("B", "A")
        for case_index, case_name in enumerate(CASES):
            seed = args.seed_base + case_index * 1_000 + repetition
            for condition in conditions:
                row = _run_one(
                    args, run_dir, condition, case_name, repetition, seed,
                )
                rows.append(row)
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"{condition}-{case_name}-{repetition}: seed={seed} "
                    f"tool={row['chosen']} effect={row['effect_correct']} "
                    f"steps={row['steps']} error={row['error']}",
                    flush=True,
                )

    print()
    _print_summary(rows)
    print(f"results={results_path}")
    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
