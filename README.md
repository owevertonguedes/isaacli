# isaacli

**A local-first CLI coding agent, built and measured for models that fit in 4 GB
of VRAM.**

*[Leia em português](README.pt-BR.md)*

Most agent harnesses assume a frontier cloud model. Point them at a small local
model and they tend to do one of two things: invent tool names that do not
exist, or describe the work instead of doing it. `isaacli` is a small harness
built the other way around: the starting question was which models run well on
budget hardware, and the harness was designed around what they need to work
reliably. It is not restricted to 4 GB; any Ollama-served model with native tool
calling, or any OpenAI-compatible endpoint, works the same way.

It reads and writes files, runs shell commands inside a three-layer sandbox, and
either finishes the task or tells you it failed. Nothing leaves the machine
unless you configure a remote API.

> [AGPLv3](LICENSE). Free to use, study and modify. Offering it as a closed
> network service requires a commercial license: see [LICENSING.md](LICENSING.md).

## Quickstart

Requires [Ollama](https://ollama.com), Python 3.10+, and `bwrap` (`bubblewrap`)
for the sandbox.

```bash
git clone https://github.com/owevertonguedes/isaacli.git
cd isaacli

./isaacli setup                    # choose model, context and reasoning effort

./isaacli                          # interactive REPL
./isaacli "run git status and tell me what is pending"
./isaacli --workspace /path/to/project
./isaacli --resume <session-id>
```

The first interactive run opens the setup automatically when no profile exists.
Setup can also configure any OpenAI-compatible endpoint (Groq, for example); the
API key is stored in `~/.config/isaacli/secrets.json` with mode `0600`, never in
the workspace or in a session log.

The interface speaks English and Brazilian Portuguese, selected during setup and
changeable at any time with `/language`.

Inside the REPL, `/help` lists every command. See
**[docs/USAGE.md](docs/USAGE.md)** for the commands, the permission modes,
session resuming and the setup flow in detail.

## Why it works

Nothing exotic. Four decisions, each of which is a failure mode avoided:

1. **Native `/api/chat`**, not an OpenAI-compat translation layer. Ollama drops
   `options.num_ctx` on its OpenAI-compatible `/v1` endpoint and honours it on
   the native one, so the compat layer costs you the context window before a
   single token is generated.
2. **A short tool schema.** Seven file and shell tools, so the list stays inside
   what a small model can hold and match against.
3. **A model with native tool calling**, picked by measurement rather than by
   parameter count. See the reasoning in
   [`Modelfile.isaac-granite.tmpl`](tool_harness/Modelfile.isaac-granite.tmpl).
4. **`num_ctx` and `temperature` set explicitly**, so they travel with the
   profile instead of depending on how the server was started. Ollama's default
   context truncates the tool schema silently, and a model that cannot see its
   tools invents plausible ones.

## The sandbox

Command execution is contained in three independent layers, in
[`tool_harness/execution.py`](tool_harness/execution.py):

- **Direct execve**, no shell, so there is no injection through `;`, `&&` or `$()`
- **A short default allowlist**, with explicit user approval required to widen it
- **`bwrap`** with the whole disk read-only, networking closed, and only the
  working directory writable

File tools refuse to escape their root, including through absolute paths and
`..`. Both are tested by trying to actually escape, with bait planted outside
the directory, in [`check_sandbox.py`](tool_harness/check_sandbox.py) and
[`check_execution.py`](tool_harness/check_execution.py), rather than by checking
that a refusal message appeared.

Approval never bypasses `bwrap`, the no-shell parser, or the block on
force-push. This part is reusable on its own, in any project that executes
model-generated code, local or cloud.

## Honest limits

- A 2 GB model is a 2 GB model. Raw capability comes from pretraining and you
  download it finished. What this repo adds is reliability and specialisation,
  not intelligence.
- This targets file and shell work through a small, fixed tool schema. It does
  not aim to replace what Aider or Codex are built for: diff-based editing
  across a large repository, deep git integration, or driving frontier cloud
  models.

## Documentation

- [docs/USAGE.md](docs/USAGE.md): commands, permissions, sessions, setup
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): module map, main flow, invariants
- [CONTRIBUTING.md](CONTRIBUTING.md): setup, tests and ground rules

## Contributing

Issues and pull requests are welcome, particularly reproductions on other
hardware, and particularly another measurement bug that slipped through.

By submitting a pull request you agree to license your contribution under AGPLv3
and grant the maintainer the right to include it in commercial licenses of this
project. See [LICENSING.md](LICENSING.md).
