"""Command execution for isaacli, confined.

WHAT THIS PROTECTS, AND WHY IT NEEDED MORE THAN ONE LAYER
---------------------------------------------------------
`tools._safe()` confines a FILE PATH to a root. A shell command is not a path:
`rm -rf ~`, `curl | sh` and `git push` bypass that protection entirely. So there
are three layers here, and each one alone has a known hole:

1. NO SHELL. The command is split with shlex and handed straight to execve.
   Without a shell there is no pipe, no redirection, no `&&`, no `$(...)`, no
   `~` and no glob. This kills the entire `curl | sh` family at once, not by
   list but by construction. Alone it is not enough: `python3 -c "..."` could still
   do anything.

2. ALLOWLIST, deliberately short. It grows with use, not in anticipation. Alone
   it is not enough: an allowlist is a guessing game about dangerous arguments,
   and whoever writes the list always forgets one.

3. BWRAP, the only one that really counts, because it is the kernel saying no,
   not an `if` of ours. No network (`--unshare-net`), the whole filesystem
   read-only, and the working directory as the ONLY writable thing. Here
   `rm -rf /home/user` does not reach the user's real home.

If bwrap is not present on the machine, this module REFUSES to execute. Falling
back to the host "just to make it work" would turn the containment into theatre,
and security theatre is worse than none, because we stop looking.

DELIBERATE EXCEPTION: `git push` is allowed and reopens the network ONLY for
that call (`--share-net`, see `_needs_network`). Strictly read-only `gh` queries
also get the network; mutations stay blocked. Everything else (python3, pytest,
git status/diff/...) still has no network at all. HOME is still the working
directory, never the real home: no private key or HTTPS credential of the user
is mounted inside the sandbox. Authentication goes through the ssh-agent SOCKET
(`SSH_AUTH_SOCK`), which only signs challenges: there is no operation that lets
`cat`/`python3` read the private key through it. `--force` stays blocked (the
only truly irreversible action here).

DELIBERATE EXCEPTION: `graphify query/path/explain/diagnose` is allowed for
querying `graphify-out/graph.json` maps. For the binary to work, the jail mounts
read-only just the Graphify `uv tool` and the embedded Python it uses. That
creates parent paths such as `/home/user`, but it does not mount the real home,
`.ssh`, DevTools, histories or credentials.

PROCESS NOTE: whoever writes the jail cannot be whoever already proved they do
not respect the jail. That is why the tests in `check_execution.py` try to
actually ESCAPE (write outside, open the network, delete the home) instead of
merely confirming that the refusal message showed up.
"""
import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

TIMEOUT_SECONDS = 60        # ceiling per command
OUTPUT_LIMIT = 20_000       # truncate huge output BEFORE it goes back to the model

# Deliberately short. Every entry needs a reason to be here.
ALLOWED = {
    "ls", "cat", "head", "tail", "wc", "grep", "find",
    "python3", "pytest", "git", "gh", "graphify",
}

# Read-only git plus add/commit/push. commit is local, no risk. push is the only
# subcommand that opens the network (see `_needs_network`) -- and even then only
# for push itself, never for the other commands in the list (python3, pytest and
# so on stay 100% offline).
GIT_ALLOWED = {"status", "diff", "log", "show", "branch", "add", "commit", "push"}

# Graphify is here as a local structural-map query. Subcommands that could
# download, extract, watch or rewrite a graph stay out.
GRAPHIFY_ALLOWED = {"query", "path", "explain", "diagnose"}

# GitHub queries that do not change remote state. `gh api`, auth, issue/pr
# create/edit/close and workflows are structurally out, even after approval.
GH_ALLOWED = {
    ("issue", "view"), ("pr", "view"), ("repo", "view"),
    ("release", "view"), ("run", "view"),
    ("auth", "status"),
    ("search", "issues"), ("search", "prs"), ("search", "repos"),
    ("search", "commits"),
}
GH_FORBIDDEN_FLAGS = {"--web", "--show-token"}

# push --force rewrites remote history: it is the only truly irreversible action
# here (it destroys other people's work if the remote has already moved on). It
# stays out even with push allowed.
PUSH_FORBIDDEN_FLAGS = {"--force", "-f", "--force-with-lease", "--force-if-includes"}

# The user's public known_hosts path (server public keys only, nothing secret) --
# lets ssh validate the host without needing an interactive TTY for the "yes/no"
# on the first connection.
KNOWN_HOSTS_HOST = Path.home() / ".ssh" / "known_hosts"
GH_CONFIG_HOST = Path(os.environ.get("GH_CONFIG_DIR", Path.home() / ".config" / "gh"))
GRAPHIFY_TOOL_ROOT = Path.home() / ".local" / "share" / "uv" / "tools" / "graphifyy"
GRAPHIFY_PYTHON_ROOT = (
    Path.home()
    / ".var"
    / "app"
    / "com.visualstudio.code"
    / "data"
    / "uv"
    / "python"
)


class Denied(Exception):
    """Command blocked before running. The message goes raw to the model and the screen."""


def _git_global_config(key):
    try:
        r = subprocess.run(
            ["git", "config", "--global", "--get", key],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except Exception:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _git_identity():
    """Commit identity inherited from the host, without exposing the real HOME."""
    name = _git_global_config("user.name") or os.environ.get("USER") or "Isaac"
    email = _git_global_config("user.email") or f"{name.lower().replace(' ', '.')}@localhost"
    return name, email


def _system_binaries():
    """Where the executables live on this distro. Fedora uses /usr plus symlinks."""
    real, links = [], []
    for path in ("/usr", "/etc"):
        if os.path.isdir(path):
            real.append(path)
    for link in ("/bin", "/sbin", "/lib", "/lib64"):
        if os.path.islink(link):
            links.append((os.readlink(link), link))
        elif os.path.isdir(link):
            real.append(link)
    return real, links


def review(cmd, authorized=False):
    """Decide whether the command may run. Returns the already-split argument list.

    Raises Denied with the REASON. It never refuses silently: a small model that
    gets silence retries the same thing; a model that gets a reason corrects it.
    """
    if not cmd or not cmd.strip():
        raise Denied("empty command")

    # punctuation_chars=True splits shell operators into their own tokens instead
    # of leaving them glued on ('ls;' used to become a "program" called 'ls;').
    # That matters so the check below is per TOKEN and not on the raw string:
    # looking for ';' in the whole string would refuse
    # `python3 -c "import time; time.sleep(1)"`, which is legitimate and has no
    # shell to interpret that ';' anyway. Refusing a good command is as bad as
    # accepting a bad one: the model only sees "it didn't work" and retries the
    # exact same thing.
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        parts = list(lex)
    except ValueError as e:
        raise Denied(f"could not read the command (unclosed quotes?): {e}")
    if not parts:
        raise Denied("empty command")

    # Shell operators as a LOOSE token. Without a shell they would never be
    # interpreted. They would reach execve as a literal argument and the command
    # would silently do something else than what the model asked. Saying what
    # happened is better than executing a truncated version of the request.
    OPERATORS = {"|", "||", ">", ">>", "<", "<<", "&&", "&", ";", ";;", "(", ")", "$"}
    for part in parts:
        if part in OPERATORS or "`" in part:
            raise Denied(
                f"'{part}' does not work here: commands run without a shell, "
                f"so there is no pipe, redirection or chaining. "
                f"Run one command at a time.")

    program = parts[0]
    if "/" in program:
        raise Denied(
            f"call the program by name ('{Path(program).name}'), not by "
            f"path ('{program}').")
    if program not in ALLOWED and not authorized:
        raise Denied(
            f"'{program}' is not in the list of allowed commands.\n"
            f"Allowed: {', '.join(sorted(ALLOWED))}")

    # `find` has flags that execute another program (`-exec sh -c ...` would
    # bring the shell back in through the window) and one that deletes files.
    # bwrap would still hold the damage inside the working directory, but "inside
    # the working directory" is exactly where isaac's work lives.
    if program == "find":
        for flag in ("-exec", "-execdir", "-delete", "-ok", "-okdir",
                     "-fprintf", "-fls", "-fprint"):
            if flag in parts:
                raise Denied(
                    f"'find {flag}' is not allowed. find here only searches and "
                    f"lists. To act on what it found, run another command.")

    if program == "git":
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        if sub is None:
            raise Denied("name the git subcommand (e.g. git status)")
        if sub not in GIT_ALLOWED and not authorized:
            raise Denied(
                f"'git {sub}' is not allowed.\n"
                f"Allowed: {', '.join(sorted(GIT_ALLOWED))}")
        if sub == "push" and (set(parts[1:]) & PUSH_FORBIDDEN_FLAGS):
            raise Denied(
                "push with --force is not allowed: it rewrites remote history "
                "irreversibly. A normal push (without --force) is allowed.")

    if program == "graphify":
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        if sub is None:
            raise Denied("name the graphify subcommand (e.g. graphify query)")
        if sub not in GRAPHIFY_ALLOWED and not authorized:
            raise Denied(
                f"'graphify {sub}' is not allowed.\n"
                f"Allowed: {', '.join(sorted(GRAPHIFY_ALLOWED))}")

    if program == "gh":
        route = tuple(p for p in parts[1:] if not p.startswith("-"))[:2]
        if route not in GH_ALLOWED:
            allowed = ", ".join(" ".join(item) for item in sorted(GH_ALLOWED))
            raise Denied(
                "this gh operation is not read-only. "
                f"Allowed queries: {allowed}")
        forbidden_flags = set(parts) & GH_FORBIDDEN_FLAGS
        if forbidden_flags:
            raise Denied(
                "gh flag not allowed in this query: "
                + ", ".join(sorted(forbidden_flags)))

    return parts


def _needs_network(parts):
    """Network only for push and strictly reviewed gh queries."""
    return (len(parts) >= 2 and parts[0] == "git" and parts[1] == "push") or (
        parts and parts[0] == "gh"
    )


def build_bwrap(argv, root, network=False):
    """Build the bwrap line: nothing writable except the working directory.

    network=True reopens the network with --share-net for `git push` or already
    reviewed gh queries. Push gets the ssh-agent socket and known_hosts; gh gets
    only its configuration on a read-only mount. HOME is still the working
    directory, without exposing credentials to `cat`/`python3`.
    """
    real, links = _system_binaries()
    git_name, git_email = _git_identity()
    line = [shutil.which("bwrap")]
    for path in real:
        line += ["--ro-bind", path, path]
    for target, link in links:
        line += ["--symlink", target, link]
    path_env = "/usr/bin:/bin"
    if argv and argv[0] == "gh":
        gh_exe = shutil.which("gh")
        if gh_exe:
            line += ["--ro-bind", gh_exe, gh_exe]
            path_env = f"{Path(gh_exe).parent}:{path_env}"
    if GRAPHIFY_TOOL_ROOT.exists():
        line += ["--ro-bind", str(GRAPHIFY_TOOL_ROOT), str(GRAPHIFY_TOOL_ROOT)]
        path_env = f"{GRAPHIFY_TOOL_ROOT / 'bin'}:{path_env}"
    if GRAPHIFY_PYTHON_ROOT.exists():
        line += ["--ro-bind", str(GRAPHIFY_PYTHON_ROOT), str(GRAPHIFY_PYTHON_ROOT)]
    line += [
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        # The working directory at the SAME path as outside: that way what the
        # model sees here matches what it sees through the other tools.
        "--bind", str(root), str(root),
        "--chdir", str(root),
        "--setenv", "HOME", str(root),
        "--setenv", "PATH", path_env,
        "--setenv", "GIT_AUTHOR_NAME", git_name,
        "--setenv", "GIT_AUTHOR_EMAIL", git_email,
        "--setenv", "GIT_COMMITTER_NAME", git_name,
        "--setenv", "GIT_COMMITTER_EMAIL", git_email,
        "--unshare-all",        # no network, nothing, by default
        "--die-with-parent",    # closing the CLI takes the command down with it
        "--new-session",        # no inherited tty (avoids TIOCSTI injection)
    ]
    if network:
        for directory in ("/run", "/run/systemd", "/run/NetworkManager"):
            line += ["--dir", directory]
        for path in (
            "/run/systemd/resolve",
            "/run/NetworkManager/resolv.conf",
            "/run/NetworkManager/no-stub-resolv.conf",
        ):
            if Path(path).exists():
                line += ["--ro-bind", path, path]
        line += ["--share-net"]   # only undoes --unshare-net; the rest stays isolated
        if argv and argv[0] == "gh" and GH_CONFIG_HOST.is_dir():
            target = "/tmp/gh-config"
            line += ["--ro-bind", str(GH_CONFIG_HOST), target,
                     "--setenv", "GH_CONFIG_DIR", target,
                     "--setenv", "GH_PAGER", "cat",
                     "--setenv", "PAGER", "cat"]
        if len(argv) >= 2 and argv[0] == "git" and argv[1] == "push":
            sock = os.environ.get("SSH_AUTH_SOCK")
            if sock and Path(sock).exists():
                line += ["--bind", sock, sock, "--setenv", "SSH_AUTH_SOCK", sock]
            if KNOWN_HOSTS_HOST.exists():
                target = "/tmp/.known_hosts_ro"
                line += ["--ro-bind", str(KNOWN_HOSTS_HOST), target,
                         "--setenv", "GIT_SSH_COMMAND",
                         f"ssh -o UserKnownHostsFile={target} -o StrictHostKeyChecking=yes"]
    line += ["--", *argv]
    return line


def run_command(cmd: str, authorized=False) -> str:
    """Run a confined command and return the RAW output.

    Raw on purpose: stdout, stderr and exit code, without summarising and without
    "cleaning up". The model needs the real error to fix it: a chewed-over
    message hides exactly the line that says what to do.
    """
    import tools  # late: SANDBOX_ROOT changes when the user switches folders

    if shutil.which("bwrap") is None:
        return ("DENIED: bwrap is not installed, and without it there is no "
                "containment. I will not run a command directly on the host.\n"
                "Install it with: sudo dnf install bubblewrap")

    try:
        argv = review(cmd, authorized=authorized)
    except Denied as e:
        return f"$ {cmd}\nDENIED: {e}\n(exit code: 126)"

    root = Path(tools.SANDBOX_ROOT).resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"ERROR: working directory is not accessible ({e})"

    proc = None
    try:
        proc = subprocess.Popen(
            build_bwrap(argv, root, network=_needs_network(argv)),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,   # own group: lets us kill child and grandchild
        )
        out, err = proc.communicate(timeout=TIMEOUT_SECONDS)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.communicate(timeout=5)
        except Exception:
            pass
        return (f"$ {cmd}\n"
                f"TIMED OUT: it went past {TIMEOUT_SECONDS}s and was killed.")
    except OSError as e:
        return f"ERROR: could not run the command ({e})"

    parts = [f"$ {cmd}"]
    if out.strip():
        parts.append(out.rstrip("\n"))
    if err.strip():
        parts.append("--- stderr ---")
        parts.append(err.rstrip("\n"))
    parts.append(f"(exit code: {code})")
    text = "\n".join(parts)

    if len(text) > OUTPUT_LIMIT:
        cut = len(text) - OUTPUT_LIMIT
        text = text[:OUTPUT_LIMIT] + f"\n… (truncated {cut} characters)"
    return text


SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Run ONE terminal command, confined to the working directory, and "
            "return the raw output (stdout, stderr and exit code). "
            "There is no shell: no pipe, '>', '&&' or ';'. One command at a "
            "time. Safe read-only commands may run automatically; other "
            "commands, such as 'rm file.txt', are shown to the user and only "
            "run after approval. Available by default: "
            + ", ".join(sorted(ALLOWED)) +
            f" (git: {', '.join(sorted(GIT_ALLOWED))}, no --force on push; the "
            "commit identity comes from the host's global git config, so "
            "'git config' is neither needed nor allowed). There is no network, "
            "except in 'git push' itself and in read-only gh queries. Use it to "
            "check what you did: list files, count lines, run a test, see the "
            "diff, commit and push."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string",
                        "description": "the command, e.g. 'ls -la' or 'wc -l file.py'"}
            },
            "required": ["cmd"],
        },
    },
}
