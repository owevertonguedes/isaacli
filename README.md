# isaacli

Para entender rapidamente a implementação atual, seus limites de segurança e o
mapa dos módulos, consulte [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**A local-first CLI coding agent, built and measured for models that fit in 4 GB
of VRAM.**

Most agent harnesses assume a frontier cloud model. Point them at a small local
model and they tend to do one of two things: invent tool names that do not exist,
or describe the work instead of doing it. `isaacli` is a small harness built the
other way around: the starting question was which models run well on that kind
of budget hardware, and the harness was designed around what they need to work
reliably. It is not restricted to 4 GB; any Ollama-served model with native tool
calling, or any OpenAI-compatible endpoint, works the same way.

It reads and writes files, runs shell commands inside a three-layer sandbox, and
finishes the task or tells you it failed. Nothing leaves the machine.

> [AGPLv3](LICENSE). Free to use, study and modify. Offering it as a closed
> network service requires a commercial license: see [LICENSING.md](LICENSING.md).

---

## Quickstart

Requires [Ollama](https://ollama.com), Python 3.10+, and `bwrap`
(`bubblewrap`) for the sandbox.

```bash
git clone https://github.com/owevertonguedes/isaacli.git
cd isaacli

./isaacli setup                         # choose model, context and reasoning

./isaacli                                 # interactive REPL
./isaacli "run git status and tell me what is pending"
./isaacli --workspace /path/to/project
./isaacli --resume 2026-08-07-200018-589c03
```

On the first interactive run, isaacli also opens the setup automatically when no
profile exists. It reads installed tags from the live Ollama API. The model menu
shows a small curated recommendation section (Qwen3.6 35B-A3B UD-IQ1_M first)
and a second section built live from the local tags returned by Ollama. Models
already shown as recommendations are not repeated there, and legacy aliases
known to be context-only copies are collapsed into their base model. Long lists
scroll inside the selector instead of overflowing the terminal. A
recommended tag that is not registered in Ollama is shown as not installed and
is downloaded only after confirmation. The setup detects model capabilities and
context from Ollama instead of assigning behavior from its catalog. Context and
reasoning are separate. Recommendations are data in
`tool_harness/model_catalog.json`, not model-specific branches in the setup.
In a terminal, setup choices use the arrow keys and Enter;
manual context accepts friendly values such as `12K` (the safe minimum is `8K`).
After setup, the interactive Isaac prompt opens immediately.

The same setup can create a generic OpenAI-compatible API profile. It asks for
provider label, base endpoint, exact model ID, API key and reasoning mode; no
provider or model is compiled into the adapter. The key is stored separately in
`~/.config/isaacli/secrets.json` with mode `0600`, never in the workspace,
session log or regular `config.json`. For example, Groq can be configured with
base endpoint `https://api.groq.com/openai/v1` and model
`openai/gpt-oss-20b`.

Setup text comes from JSON catalogs in `tool_harness/locales/`. Portuguese and
English are available, and the agent is instructed to answer in the language
used by the user.

The advanced `build-model.sh` flow remains available for automation. It reads
`BASE_MODEL`, `MODEL_NAME`, `NUM_CTX` and `TEMPERATURE` from the environment and
refuses a base that does not advertise the `tools` capability.

```bash
BASE_MODEL=granite4:micro NUM_CTX=16384 ./scripts/build-model.sh
```

Inside the REPL, `/help` lists the commands, `/model` selects configured profiles,
`/tools` shows which tools and binaries are allowed, and `/status` reports token
usage for the session. On exit, isaacli prints the exact `--resume` command. A
resumed run restores the workspace, model, messages and tool results into a new
session log; recent messages, tool calls, results and permission decisions are
also redrawn in the terminal. The original JSONL remains unchanged.

The interactive REPL uses an isolated alternate screen, keeping the shell's
scrollback out of the Isaac interface. When the conversation exceeds the screen,
the mouse wheel opens the same transcript as an integrated viewport and scrolls
it in both directions; `/history` opens it explicitly. Arrow keys at the prompt
navigate previously submitted messages and are never repurposed as transcript
navigation. Mouse reporting is enabled only while scrolling is available;
Shift-drag keeps the terminal's native text selection in that state.
Leaving a menu or the history viewer redraws the recent conversation instead of
returning to a cleared screen; leaving the CLI restores the shell exactly as it
was. During streaming, a single transient `Trabalhando…` line reports approximate live
tokens/second, including hidden thinking emitted by Ollama; the reasoning text
itself remains hidden unless a model puts its entire answer there and leaves the
visible response empty. After each response Ollama's exact evaluation duration
is used when available. Mouse-wheel and arrow input received while the agent is
busy is discarded instead of leaking terminal escape sequences into the output.

The welcome panel shows the application version (`isaacli --version`), active
model, engine and workspace. Development builds use a `-dev` suffix; repository
release tags use the matching `vMAJOR.MINOR.PATCH` version. Assistant responses
render common Markdown in the terminal (headings, emphasis, inline and fenced
code, lists, checkboxes, quotes and links), while session logs keep the original
text unchanged.

Multiple Isaac sessions share an Ollama server started by the CLI. Closing one
session keeps that server available to the others; the last registered session
stops it. An Ollama server that was started outside Isaac remains under the
user's control and is never stopped automatically by the CLI.

Typing `/` on an empty prompt opens the command palette immediately. Results are
filtered while typing; use the arrow keys to select an entry and `Tab` to insert
it. When `prompt_toolkit` is unavailable, the CLI keeps the existing GNU
Readline `Tab` completion as a fallback. `Alt+Enter` inserts a line break without
sending the message. `/clear` resets only the current conversation context;
`/new` closes the current session log and starts a fresh session with a new ID.
New session IDs are full UUIDv4 values; legacy date-based IDs remain resumable.
`/model` first selects the source. Ollama then shows the curated recommendations
and every model reported as locally installed by the live server; configured
OpenAI-compatible APIs and the API setup entry appear as separate choices. After
the model, Isaac asks for context size and reasoning effort when supported.
Context is stored as a setting and sent per request, so choosing 16K or 32K no
longer creates duplicate Ollama models or duplicate entries in the selector.

Public web content is read through the general, read-only `fetch_url` tool. It
handles pages, documentation, shared links and HTTP APIs; accepts only HTTP(S),
rejects local/private/reserved destinations, ignores proxy configuration and
caps downloads. The terminal sandbox itself remains offline, so models should
not use `curl` for web access. Read-only `gh` views and searches are the separate
option for structured or authenticated GitHub access.

Terminal commands are shown before execution. The default safe mode accepts
only read-only commands automatically; commands that may change the workspace
offer four choices: allow once, always allow in this workspace, always allow
globally, or deny. `Shift+Tab` (or `/mode`) switches to the stricter mode where
only saved permissions run automatically. `/permissions` lists the rules and
can clear workspace or global rules. Approval never bypasses `bwrap`, the
no-shell parser, or the hard block on force-push.

## Why it works

Nothing exotic. Four decisions, each of which is a failure mode avoided:

1. **Native `/api/chat`**, not an OpenAI-compat translation layer. Ollama drops
   `options.num_ctx` on its OpenAI-compatible `/v1` endpoint and honours it on
   the native one, so the compat layer costs you the context window before a
   single token is generated.
2. **A short tool schema.** Seven file and shell tools, so the list stays inside
   what a 2 GB model can hold and match against.
3. **A model with native tool calling**, picked by measurement rather than by
   parameter count. See the reasoning in
   [`Modelfile.isaac-granite.tmpl`](tool_harness/Modelfile.isaac-granite.tmpl).
4. **`num_ctx` and `temperature` set explicitly** in the model, so they travel
   with the model instead of depending on how the server was started. Ollama's
   default context truncates the tool schema silently, and a model that cannot see
   its tools invents plausible ones.

## The sandbox

Command execution is contained in three independent layers, in
[`tool_harness/execucao.py`](tool_harness/execucao.py):

- **Direct execve**, no shell, so there is no injection through `;`, `&&` or `$()`
- **A short default allowlist**, with explicit user approval required to widen it
- **`bwrap`** with the whole disk read-only, networking closed, and only the
  working directory writable

File tools refuse to escape their root, including through absolute paths and
`..`. This is tested with bait planted outside the directory, in
[`testar_sandbox.py`](tool_harness/testar_sandbox.py).

This part is reusable on its own, in any project that executes model-generated
code, local or cloud.

## Honest limits

- A 2 GB model is a 2 GB model. Raw capability comes from pretraining and you
  download it finished. What this repo adds is reliability and specialisation,
  not intelligence.
- This targets file and shell work through a small, fixed tool schema. It does
  not aim to replace what Aider or Codex are built for: diff-based editing
  across a large repository, deep git integration, or driving frontier cloud
  models.

## Contributing

Issues and pull requests are welcome, particularly reproductions on other
hardware, and particularly another measurement bug that slipped through.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, tests and ground rules.

By submitting a pull request you agree to license your contribution under AGPLv3
and grant the maintainer the right to include it in commercial licenses of this
project. See [LICENSING.md](LICENSING.md).
