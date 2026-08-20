# isaacli

**A local-first CLI coding agent designed for models that fit in 4 GB of VRAM, without limiting you to them.**

*[Leia em português](README.pt-BR.md)*

`isaacli` reads and edits files, runs commands in a layered Linux sandbox and keeps working until the task finishes or fails clearly. It was built around small local models instead of assuming a frontier cloud model, but the same interface also supports larger Ollama models and OpenAI-compatible APIs.

Nothing is sent to a remote model unless you configure one. Web reads use explicit, constrained routes; terminal commands outside the approval flow stay offline by default.

> [AGPLv3](LICENSE). Free to use, study and modify. Offering isaacli as a closed network service requires a commercial license; see [LICENSING.md](LICENSING.md).

## Quickstart

Requires [Ollama](https://ollama.com), Python 3.10+ and `bwrap` (`bubblewrap`).

```bash
git clone https://github.com/owevertonguedes/isaacli.git
cd isaacli
./isaacli install

isaacli setup
isaacli kaggle
isaacli
```

Installation adds a per-user link in `~/.local/bin`; it needs no `sudo` and does not overwrite an existing command. The launcher also works from Flatpak terminals such as VS Code from Flathub by running on the host, where Ollama and the sandbox dependencies live.

Setup selects the interface language, engine, model, context and reasoning effort. It can use a local Ollama model with native tool calling or a configurable OpenAI-compatible endpoint. API keys are stored outside the workspace in a `0600` secrets file.

`isaacli kaggle` covers the remote path from a fresh machine: it installs the Kaggle CLI in an isolated per-user environment when necessary, opens Kaggle login, reports accelerator quota, refuses a second visible live kernel, offers curated GPU models, pushes a private GPU kernel, discovers its tunnel URL from the live log and saves a regular `openai_compatible` profile. Every push requires confirmation at that moment. Kaggle is a third-party service whose sessions can stop, and its terms were not designed around notebooks as persistent API servers.

An endpoint on `localhost` needs no API key, since a server you run yourself has none to demand. For those, setup also offers to run the server for you: give it the command that starts it (for example `llama-server -m /path/model.gguf -c 8192`) and isaacli starts it when a session opens, shares it across simultaneous sessions and stops it when the last one closes. Leave the command empty to keep starting it yourself. This is how you point isaacli at any weights you downloaded yourself, from Hugging Face or anywhere else: the model stays entirely on your machine.

Useful entry points:

```bash
isaacli "run git status and explain what is pending"
isaacli --workspace /path/to/project
isaacli --resume <session-id>
isaacli uninstall
```

See [Installation](docs/INSTALLATION.md) for Flatpak details, recovery and the explicitly confirmed purge options.

## What it provides

- File, web and terminal tools exposed through a compact schema suitable for smaller models.
- Interactive permission prompts with one-time, workspace and global authorizations.
- English and Brazilian Portuguese interfaces, switchable with `/language`.
- Model, context and reasoning selection without creating duplicate Ollama models.
- Resumable JSONL sessions, command output history and task feedback.
- OpenAI-compatible remote profiles with a configurable endpoint and exact model ID.
- Optional lifecycle management for a local OpenAI-compatible server, so a llama-server or equivalent starts with your session and stops with the last one.

Inside the REPL, type `/` to open the command palette or `/help` to list every command. [Usage](docs/USAGE.md) covers setup, permissions, sessions, terminal behaviour and the full command reference.

## Safety and privacy

Unattended commands execute without a shell, inside `bwrap`, with the workspace as the only writable location. Commands outside the default policy are shown before execution; after approval they may use a shell or network, but remain inside the same filesystem, resource and syscall boundaries.

Resource ceilings use `systemd-run --user`; the seccomp filter is x86_64-only. When either layer is unavailable, isaacli reports that limitation instead of claiming protection that is not present. The sandbox limits model-generated commands, but it is not a security boundary against the user, root or malware already running under the same account.

Local sessions and feedback may contain prompts, answers, paths, commands and tool results. They are ignored by Git but are currently stored as plaintext. A remote provider receives the conversation and any tool results included in later model requests. Read [Security and privacy](docs/SECURITY.md) before using sensitive data or a remote endpoint.

## Honest limits

- A small model remains a small model: the harness improves reliability and tool use, not the model's underlying knowledge or reasoning capacity. Measured on this project's own hardware in August 2026, a 3B coder model served over an OpenAI-compatible endpoint returned its tool calls as a fenced JSON block instead of the native format its own chat template declares, so it could describe the file it meant to write but never wrote one. The harness reported that no change had been confirmed, which is the correct behaviour and not a substitute for a model that can call tools.
- Ollama is the recommended engine because it is a single installer with a model catalog, not because it was measured as the fastest. Running a llama-server directly is supported. Throughput has been measured here for llama.cpp alone (36.2 tok/s for a 3B model at Q4_K_M on a GTX 1650 through the Vulkan backend), but the two engines have never been measured against each other on this hardware, so no comparison between them is claimed.
- Speculative decoding is available through llama.cpp and was measured here rather than assumed. On the same card and model it produced no gain with n-gram drafting and was 45% slower with a 0.5B draft model. The technique is designed for a large target with a tiny draft, which is not the configuration a 4 GB card can hold, so treat any published speedup as belonging to other hardware until you measure your own.
- The project targets file and terminal work through a compact tool set; it is not a full IDE or a general browser-automation framework.
- Sandboxing reduces accidental damage and workspace escape, but no local agent should be treated as a boundary against a compromised host or account.

## Documentation

- [Usage](docs/USAGE.md): setup, commands, permissions, sessions and terminal behaviour
- [Architecture](docs/ARCHITECTURE.md): module map, data flow and implementation invariants
- [Installation](docs/INSTALLATION.md): install, Flatpak, removal, purge and recovery
- [Security and privacy](docs/SECURITY.md): stored data, remote APIs and current limits
- [Contributing](CONTRIBUTING.md): development setup, tests and ground rules

## Contributing

Issues and pull requests are welcome, particularly reproducible failures on other hardware or with other tool-capable models.

By submitting a pull request you agree to license your contribution under AGPLv3 and grant the maintainer the right to include it in commercial licenses of this project. See [LICENSING.md](LICENSING.md).
