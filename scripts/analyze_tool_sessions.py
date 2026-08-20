#!/usr/bin/env python3
"""Summarize tool usage recorded in isaacli JSONL session logs."""

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_DIR = PROJECT_ROOT / "tool_harness" / "cli_sessions"
DIRECT_SCHEMA_TOOLS = (
    "read_file",
    "write_file",
    "append_file",
    "replace_between",
    "replace_text",
    "fetch_url",
    "list_dir",
)
ALL_TOOLS = DIRECT_SCHEMA_TOOLS + ("run_command",)


SESSION_KINDS = ("test", "benchmark", "real")


def _session_kind(events: list[dict], workspace: str) -> str:
    """Separate automated tests, benchmarks and ordinary user sessions."""
    if any(event.get("benchmark") for event in events):
        return "benchmark"
    try:
        path = Path(workspace).expanduser()
    except TypeError:
        return "real"
    parts = path.parts
    if any(part.startswith("isaacli-021") for part in parts):
        return "benchmark"
    if len(parts) >= 2 and (
        parts[1] == "tmp" or (len(parts) >= 3 and parts[1:3] == ("var", "tmp"))
    ):
        return "test"
    return "real"


def _read_events(path: Path) -> list[dict]:
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
        if isinstance(event, dict):
            events.append(event)
    return events


def analyze(session_dir: Path) -> tuple[dict, list[dict]]:
    totals = {
        "tools": {kind: Counter() for kind in SESSION_KINDS},
        "via": {kind: Counter() for kind in SESSION_KINDS},
        "sessions": Counter(),
        "finals": Counter(),
        "invalid": [],
    }
    rows = []
    for path in sorted(session_dir.glob("*.jsonl")):
        try:
            events = _read_events(path)
        except ValueError as error:
            totals["invalid"].append(str(error))
            continue
        workspace = next(
            (str(event.get("workspace")) for event in reversed(events)
             if event.get("workspace")),
            "",
        )
        kind = _session_kind(events, workspace)
        results = [event for event in events if event.get("type") == "tool_result"]
        finals = [event for event in events if event.get("type") == "assistant_final"]
        for event in results:
            totals["tools"][kind][event.get("name") or "<missing-name>"] += 1
            totals["via"][kind][event.get("via") or "<missing>"] += 1
        exact_steps = next(
            (event.get("steps") for event in reversed(finals)
             if isinstance(event.get("steps"), int)),
            None,
        )
        totals["sessions"][kind] += 1
        totals["finals"][(kind, bool(finals))] += 1
        rows.append({
            "id": path.stem,
            "kind": kind,
            "workspace": workspace,
            "steps": exact_steps,
            "calls": len(results),
            "assistant_final": bool(finals),
        })
    return totals, rows


def _print_summary(totals: dict, rows: list[dict]) -> None:
    print(f"Session files: {len(rows)}")
    print(f"Invalid files: {len(totals['invalid'])}")
    print("\nSessions by class and completion:")
    print("class\tsessions\tassistant_final\twithout_final\texact_steps_logged")
    for kind in SESSION_KINDS:
        sessions = totals["sessions"][kind]
        finals = totals["finals"][(kind, True)]
        exact = sum(row["steps"] is not None for row in rows if row["kind"] == kind)
        print(f"{kind}\t{sessions}\t{finals}\t{sessions - finals}\t{exact}")

    names = list(ALL_TOOLS)
    extras = sorted(
        set().union(*(set(totals["tools"][kind]) for kind in SESSION_KINDS))
        - set(names)
    )
    print("\nCompleted calls by tool:")
    print("tool\ttest\tbenchmark\treal\ttotal")
    for name in names + extras:
        values = [totals["tools"][kind][name] for kind in SESSION_KINDS]
        print(f"{name}\t" + "\t".join(map(str, values)) + f"\t{sum(values)}")

    vias = sorted(set().union(*(set(totals["via"][kind]) for kind in SESSION_KINDS)))
    print("\nCall acquisition via:")
    print("via\ttest\tbenchmark\treal\ttotal")
    for via in vias:
        values = [totals["via"][kind][via] for kind in SESSION_KINDS]
        print(f"{via}\t" + "\t".join(map(str, values)) + f"\t{sum(values)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", nargs="?", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument(
        "--sessions", action="store_true",
        help="print one TSV row per session after the aggregate summary",
    )
    args = parser.parse_args()
    totals, rows = analyze(args.session_dir)
    _print_summary(totals, rows)
    if args.sessions:
        print("\nPer session:")
        print("id\tclass\tsteps\tcalls\tassistant_final\tworkspace")
        for row in rows:
            steps = row["steps"] if row["steps"] is not None else "unknown"
            final = "yes" if row["assistant_final"] else "no"
            print(
                f"{row['id']}\t{row['kind']}\t{steps}\t{row['calls']}\t"
                f"{final}\t{row['workspace']}"
            )
    for error in totals["invalid"]:
        print(error)
    return 1 if totals["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
