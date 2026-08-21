"""Session identifiers, the on-disk JSONL log, and rebuilding a conversation
from it.

CLI_KNOWLEDGE lives here rather than with the slash commands (as the task
inventory first placed it) because its only real consumer is
_build_history(); keeping it next to that avoids a cli_commands <-> here
import cycle (cli_commands needs FEEDBACK_DIR from here for save_feedback).
"""
import datetime as dt
import json
import os
import re
import shlex
import shutil
import uuid
from pathlib import Path

import agent
import terminal_ui
import workspace_instructions
from cli_i18n import t
from cli_presentation import _color, _format_markdown_terminal

HERE = Path(__file__).resolve().parent
SESSIONS_DIR = HERE / "cli_sessions"
FEEDBACK_DIR = HERE / "feedback"
SESSION_ID_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
SESSION_ID_LEGACY = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{6}-[a-f0-9]{6}"
)

CLI_KNOWLEDGE = """You are isaacli running as a local CLI in the user's terminal.

OPERATING CONTEXT:
- Always answer in the same language as the user's latest message. If the user
  writes in Portuguese, answer in Brazilian Portuguese; do not switch to English.
- The current working directory is: {workspace}
- File and terminal tools are confined to that directory.
- To read public web content (pages, documentation, links or HTTP APIs), use
  fetch_url. It is a general web-reading tool, not a GitHub-only workaround.
- For structured, read-only GitHub queries, you may use `gh issue view`,
  `gh pr view`, `gh repo view`, `gh release view`, `gh run view` or `gh search`.
  Prefer fetch_url for public links; use gh when its GitHub-specific structure or
  authenticated access is useful. If gh reports missing/invalid authentication
  for a public link, use fetch_url immediately; do not inspect tokens, environment
  variables or credential files, and do not clone a repository just to read it.
- Before asking the user to clarify a local file, directory or project target,
  try to resolve it with list_dir, find, grep or read_file. If the user says
  "the txt file", "the config" or similar and the workspace can identify it,
  inspect the workspace instead of asking for an exact name.
- To inspect the project, use run_command with short commands: git status,
  git diff, ls, find, wc, pytest, python3.
- run_command executes exactly one program without a shell. Never use pipes,
  redirections, `&&`, `||`, `;`, `cd`, `$VARIABLE` or `2>/dev/null`; make separate
  tool calls instead.
- If `graphify-out/graph.json` exists and the user asks where a flow, resource,
  module, test or architectural relation lives, look it up first with
  `graphify query "question" --graph graphify-out/graph.json --budget 700`.
  Graphify is for locating context; after that read the files and verify before
  declaring success. If there is no graph, fall back to local search with
  find/rg, and do not edit before locating.
- To delete a file or perform another operation not covered by a specialized
  file tool, call run_command with the exact terminal command (for example,
  `rm hello-world.txt`). The CLI, not you, handles user approval.
- Never claim that you created, edited, deleted, committed, tested or otherwise
  changed something unless at least one tool actually performed that action in
  this turn and its result confirms success.
- For git: run git status and git diff before proposing a commit.
- You may use git add, git commit and git push when the user asks for it; the
  CLI's own approval step is what actually gates execution, not this prompt.
- Before proposing a command that is destructive or hard to reverse (delete,
  overwrite, force flags, push, reset, and the like), say so plainly in your
  message so the user is approving something they understood, not reflexively
  hitting enter.
- If any tool returns a non-zero exit code, that is a failure. NEVER say a
  commit, push or test worked when the output showed an error.
- If git commit fails, stop and explain the error before trying to push.
- Keep decisions and results short. The terminal shows a summary of the commands
  and keeps the full output in the session log.
"""


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _new_session_id():
    return str(uuid.uuid4())


def _valid_session_id(session_id):
    return bool(SESSION_ID_UUID.fullmatch(session_id)
                or SESSION_ID_LEGACY.fullmatch(session_id))


def _resume_command(session_id):
    launcher = HERE.parent / "isaacli"
    global_on_path = shutil.which("isaacli")
    if global_on_path:
        try:
            if Path(global_on_path).resolve() == launcher.resolve():
                return f"isaacli --resume {session_id}"
        except OSError:
            pass
    return f"{shlex.quote(str(launcher))} --resume {session_id}"


def _build_history(workspace, instructions=None):
    instructions = instructions or workspace_instructions.load_workspace_instructions(workspace)
    content = (
        agent.TOOLS_KNOWLEDGE + "\n\n" +
        CLI_KNOWLEDGE.format(workspace=str(workspace))
    )
    if instructions.prompt:
        content += "\n\n" + instructions.prompt
    return [{"role": "system", "content": content}]


def _workspace_transition(workspace):
    return {"role": "system", "content": (
        f"The working directory is now: {workspace}\n"
        "Instructions from the previous workspace no longer apply."
    )}


def _load_session(session_id):
    """Rebuild the conversation and tool calls from a local JSONL by exact ID."""
    if not _valid_session_id(session_id):
        raise ValueError(t("cli.session.invalid_id"))
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.is_file():
        raise ValueError(t("cli.session.not_found", id=session_id))
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError(t("cli.session.too_large"))

    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(t("cli.session.invalid_log", line=number)) from e
        if isinstance(event, dict):
            events.append(event)
    if not events:
        raise ValueError(t("cli.session.empty"))

    workspace = Path(events[-1].get("workspace") or os.getcwd()).expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(t("cli.session.workspace_gone", path=workspace))

    # Logs written before the identifiers were translated use Portuguese field
    # names. Reading both keeps `--resume` working on sessions already on disk;
    # dropping it would silently rebuild those sessions as empty.
    def field(event, name, legacy):
        value = event.get(name)
        return event.get(legacy) if value is None else value

    model = next((field(e, "model", "modelo") for e in reversed(events)
                  if field(e, "model", "modelo")), None)
    instructions = workspace_instructions.load_workspace_instructions(workspace)
    history = _build_history(workspace, instructions)
    transcript = []
    pending_tool = None
    tool_number = 0
    for event in events:
        kind = field(event, "type", "tipo")
        if kind == "meta" and field(event, "event", "evento") == "clear":
            history = _build_history(workspace, instructions)
            transcript = []
        elif kind == "user" and isinstance(event.get("content"), str):
            history.append({"role": "user", "content": event["content"]})
            transcript.append(("user", event["content"]))
        elif kind == "tool_start":
            tool_number += 1
            name = field(event, "name", "nome") or "unknown"
            args = event.get("args")
            if args is None and name == "run_command":
                args = {"cmd": event.get("cmd", "")}
            pending_tool = f"resume-tool-{tool_number}"
            history.append({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": pending_tool, "type": "function",
                                "function": {"name": name, "arguments": args or {}}}],
            })
            transcript.append(("tool_start", {
                "name": name, "args": args or {}, "cmd": event.get("cmd"),
            }))
        elif kind == "permission":
            transcript.append(("permission", {
                "cmd": event.get("cmd"),
                "decision": field(event, "decision", "decisao"),
            }))
        elif kind == "tool_result" and isinstance(field(event, "result", "resultado"), str):
            result = field(event, "result", "resultado")
            history.append({"role": "tool", "tool_call_id": pending_tool or "resume-tool",
                            "content": result})
            transcript.append(("tool_result", {
                "name": field(event, "name", "nome") or "unknown",
                "code": field(event, "code", "codigo"),
                "result": result,
            }))
            pending_tool = None
        elif kind == "assistant_final" and isinstance(event.get("content"), str):
            history.append({"role": "assistant", "content": event["content"]})
            if event["content"]:
                transcript.append(("assistant", event["content"]))
    return {"id": session_id, "path": path, "workspace": workspace,
            "model": model, "history": history, "transcript": transcript,
            "workspace_instructions": instructions}


class SessionsMixin:
    def _log(self, kind, **data):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": _now(),
            "type": kind,
            "session_id": self.session_id,
            "model": self.model,
            "workspace": str(self.workspace),
            **data,
        }
        with self.session_path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def new_session(self):
        previous_id = self.session_id
        previous_path = self.session_path
        new_id = _new_session_id()
        self._log("meta", event="new_session", next_session=new_id)

        self.session_id = new_id
        self.session_path = SESSIONS_DIR / f"{new_id}.jsonl"
        self.feedback_path = FEEDBACK_DIR / f"{new_id}.jsonl"
        self.turns = 0
        self.failures = 0
        self.commands = []
        self.total_usage = {"prompt_eval_count": 0, "eval_count": 0,
                            "total_duration": 0, "eval_duration": 0}
        self.last_answer = ""
        self.ratings = 0
        self.resume_transcript = []
        self._working_visible = False
        self._assistant_label_pending = True
        self._token_buffer = []
        self._output_block = False
        self.set_workspace(self.workspace, reset=True)
        self._log("meta", event="start", pid=os.getpid(), model=self.model,
                  workspace=str(self.workspace), previous_session=previous_id)

        terminal_ui.clear()
        print(_color(t("cli.new.session", id=new_id), "assistant"))
        print(_color(t("cli.new.previous", path=previous_path), "dim"))
        print(_color(t("cli.new.resume", command=_resume_command(previous_id)), "dim"))
        self._show_workspace_instruction_warning()

    def list_sessions(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(SESSIONS_DIR.glob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print(t("cli.sessions.none"))
            return
        for p in files[:12]:
            stat = p.stat()
            modified = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            current = t("cli.sessions.current") if p == self.session_path else ""
            print(f"{p.stem}  {modified}  {stat.st_size} bytes{current}")

    def _history_text(self):
        events = list(self.resume_transcript)
        try:
            events.extend(_load_session(self.session_id)["transcript"])
        except ValueError:
            pass
        lines = []
        for role, content in events:
            if role == "user":
                lines.extend(["", f"❯ {content}"])
            elif role == "assistant":
                lines.extend(["", f"isaac: {content}"])
            elif role == "tool_start":
                name = content.get("name") or "unknown"
                if name == "run_command":
                    cmd = content.get("cmd") or (content.get("args") or {}).get("cmd", "")
                    lines.extend(["", f"$ {cmd}"])
                else:
                    args = json.dumps(content.get("args") or {}, ensure_ascii=False)
                    lines.extend(["", f"[{name}] → {args}"])
            elif role == "permission":
                lines.append(t("cli.history.permission",
                               decision=content.get("decision") or t("cli.history.unknown")))
            elif role == "tool_result":
                lines.append(content.get("result") or "")
        return "\n".join(lines).strip() or t("cli.history.empty")

    def redraw_session(self, message=None):
        """Report back to the conversation after a full-screen menu closed.

        The menu is the only thing that uses the alternate buffer, so leaving it
        already restores the conversation exactly as it was. Reprinting the
        transcript here would duplicate it in the scrollback; only the outcome
        of the menu is announced.
        """
        if not terminal_ui.interactive():
            if message:
                print(message)
            return
        if message:
            print(_color(message, "dim"))
        print()

    def show_history(self, _movement=""):
        # Printed normally, no full screen and no mouse capture: it stays in the
        # terminal's native scrollback, with formatted, copyable markdown.
        print(_format_markdown_terminal(self._history_text()))
