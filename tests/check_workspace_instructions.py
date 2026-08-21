#!/usr/bin/env python3
"""Focused checks for workspace-root project instructions."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tool_harness"))

import agent
import cli_sessions
import workspace_instructions as wi

failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


with tempfile.TemporaryDirectory() as temp:
    base = Path(temp)
    workspace = base / "workspace"
    workspace.mkdir()

    missing = wi.load_workspace_instructions(workspace)
    check(not missing.prompt and not missing.warning_key,
          "a missing AGENTS.md is silent")

    (workspace / "CLAUDE.md").write_text("claude-only", encoding="utf-8")
    parent_agents = base / "AGENTS.md"
    parent_agents.write_text("parent-only", encoding="utf-8")
    aliases = wi.load_workspace_instructions(workspace)
    check(not aliases.prompt,
          "CLAUDE.md and parent instructions are not implicit aliases")

    agents = workspace / "AGENTS.md"
    agents.write_text("run the focused check", encoding="utf-8")
    loaded = wi.load_workspace_instructions(workspace)
    check("run the focused check" in loaded.prompt
          and str(agents) in loaded.prompt
          and "untrusted text" in loaded.prompt,
          "a valid root AGENTS.md has an explicit source and trust boundary")
    history = cli_sessions._build_history(workspace, loaded)
    check(history[0]["content"].index(agent.TOOLS_KNOWLEDGE)
          < history[0]["content"].index("WORKSPACE PROJECT INSTRUCTIONS"),
          "built-in rules precede workspace text")

    agents.write_bytes(b"x" * wi.MAX_INSTRUCTIONS_BYTES)
    check(bool(wi.load_workspace_instructions(workspace).prompt),
          "the exact byte limit is accepted")
    agents.write_bytes(b"x" * (wi.MAX_INSTRUCTIONS_BYTES + 1))
    check(wi.load_workspace_instructions(workspace).warning_key
          == "cli.workspace.instructions.too_large",
          "one byte above the limit is rejected without truncation")

    agents.write_bytes(b"\xff")
    check(wi.load_workspace_instructions(workspace).warning_key
          == "cli.workspace.instructions.invalid_utf8",
          "invalid UTF-8 is rejected")

    agents.write_text("unreadable", encoding="utf-8")
    with patch.object(wi.os, "open", side_effect=PermissionError("denied")):
        unreadable = wi.load_workspace_instructions(workspace)
    check(unreadable.warning_key == "cli.workspace.instructions.read_failed",
          "a read failure becomes a warning")

    agents.unlink()
    agents.mkdir()
    check(wi.load_workspace_instructions(workspace).warning_key
          == "cli.workspace.instructions.not_file",
          "a directory named AGENTS.md is rejected")
    agents.rmdir()

    inside = workspace / "instructions.txt"
    inside.write_text("inside link", encoding="utf-8")
    agents.symlink_to(inside)
    check("inside link" in wi.load_workspace_instructions(workspace).prompt,
          "a symlink whose target stays inside the workspace is accepted")
    real_open = wi.os.open

    def swap_before_open(path, flags):
        inside.unlink()
        inside.symlink_to(outside)
        return real_open(path, flags)

    outside = base / "outside.txt"
    outside.write_text("outside link", encoding="utf-8")
    with patch.object(wi.os, "open", side_effect=swap_before_open):
        raced = wi.load_workspace_instructions(workspace)
    check(not raced.prompt and bool(raced.warning_key),
          "a target swapped to an outside symlink before open is never read")
    agents.unlink()
    agents.symlink_to(outside)
    check(wi.load_workspace_instructions(workspace).warning_key
          == "cli.workspace.instructions.outside",
          "a symlink outside the workspace is rejected")
    agents.unlink()

    sessions = base / "sessions"
    sessions.mkdir()
    original_sessions = cli_sessions.SESSIONS_DIR
    cli_sessions.SESSIONS_DIR = sessions
    try:
        session_id = "2026-08-07-123456-abcdef"
        session_path = sessions / f"{session_id}.jsonl"
        session_path.write_text(json.dumps({
            "type": "meta", "workspace": str(workspace), "model": "model",
        }) + "\n", encoding="utf-8")
        agents.write_text("current at resume", encoding="utf-8")
        resumed = cli_sessions._load_session(session_id)
    finally:
        cli_sessions.SESSIONS_DIR = original_sessions
    check("current at resume" in resumed["history"][0]["content"],
          "resume loads the current instructions rather than stale session data")
    check("current at resume" not in session_path.read_text(encoding="utf-8"),
          "instruction contents are not copied into the session log")

if failures:
    raise SystemExit(f"{len(failures)} workspace instruction check(s) failed")
print("WORKSPACE INSTRUCTIONS OK")
