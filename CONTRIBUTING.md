# Contributing to isaacli

`isaacli` is a local-first, offline-capable CLI coding agent for small
language models: an alternative to Aider, Codex CLI and other AI coding
assistants for anyone running Ollama, llama.cpp, or another local LLM
runtime on consumer or budget GPU hardware. Contributions, bug reports and
hardware reproductions are welcome.

## What's most useful right now

- **Reproductions on other hardware.** Different GPU, a different quantized
  model, AMD/ROCm, Apple Silicon, or CPU-only. Open an issue with your setup
  and what did or didn't work.
- **Bug reports against the sandbox.** `tool_harness/execution.py` is the
  security-relevant part of this project: no-shell execution, the command
  allowlist, and the `bwrap` layer. A file or command that escapes its
  intended boundary is a priority-one report.
- **New OpenAI-compatible provider profiles**, small model recommendations
  for `tool_harness/model_catalog.json`, and translations alongside the
  existing `tool_harness/locales/en.json` and `pt-BR.json`.

## Before you start

1. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module map
   and the security invariants that must not be weakened: no-shell
   execution, the `bwrap` sandbox, and the permission/approval flow.
2. Check `git status` / `git log` rather than assuming state from an
   earlier issue or PR discussion.
3. Look for an existing check in `tests/check_*.py` that already
   covers the area you're touching.

## Development setup

Requires [Ollama](https://ollama.com), Python 3.10+, and `bwrap`
(`bubblewrap`) for the sandbox.

```bash
git clone https://github.com/owevertonguedes/isaacli.git
cd isaacli
./isaacli install
isaacli setup
```

The checks are standalone scripts, not pytest modules: they run their
assertions at import time, which is why they are `check_*` and not
`test_*` (pytest collection alone would execute them).

## Running the tests

```bash
python3 tests/check_cli.py
python3 tests/check_agent_config.py
python3 tests/check_setup.py
python3 tests/check_tools.py
python3 tests/check_sandbox.py
python3 tests/check_execution.py
```

Run `check_execution.py` outside a nested sandbox: `bwrap` needs to create
its own loopback interface, which fails inside another sandboxed
environment (for example, inside a container without that capability).

## Ground rules

- **Do not weaken the sandbox.** No-shell execution (`execution.py`),
  `bwrap` isolation, or the approval flow are not on the table for
  simplification, even for a UI improvement.
- **Prefer measurement over claims.** If a change is about performance,
  reliability, or model behavior, include the numbers and how they were
  produced.
- **English for identifiers, comments and docstrings**, and every
  user-facing string through `i18n.py` with a key in both
  `locales/en.json` and `locales/pt-BR.json`. Text the *model* reads
  (system prompts, tool descriptions, tool results, sandbox refusals)
  stays English regardless of the interface language: it is a contract
  with the model, not a user preference.
- **Small, coherent commits.** One commit per unit of work is easier to
  review and to revert if needed.

## Submitting a pull request

Open an issue first for anything larger than a small fix, so the approach
can be agreed on before the work is done. For small fixes, a pull request
directly is fine.

By submitting a pull request you agree to license your contribution under
AGPLv3, and you grant the maintainer the right to include it in commercial
licenses of this project as well. See [LICENSING.md](LICENSING.md).

## Reporting a security issue

If you find a way to escape the sandbox (file access outside the
workspace root, command execution outside the allowlist without
approval, or a `bwrap` bypass), please open an issue describing the
reproduction steps. There is no separate private disclosure channel yet;
until there is, treat sandbox-escape reports as high priority regardless
of the channel used.
