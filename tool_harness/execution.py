"""Command execution for isaacli, confined.

WHAT THIS PROTECTS, AND WHY IT NEEDED MORE THAN ONE LAYER
---------------------------------------------------------
`tools._safe()` confines a FILE PATH to a root. A shell command is not a path:
`rm -rf ~`, `curl | sh` and `git push` bypass that protection entirely. So there
are three layers here, and each one alone has a known hole:

1. NO SHELL FOR WHAT RUNS UNASKED. The command is split with shlex and handed
   straight to execve, so a command that never passed in front of a human has no
   pipe, no redirection, no `&&`, no `$(...)` and no glob. This kills the whole
   `curl | sh` family by construction rather than by list. Alone it is not
   enough: `python3 -c "..."` could still do anything.

2. ALLOWLIST, deliberately short. It decides what runs WITHOUT ASKING; it is not
   a ban list. It grows with use, not in anticipation. Alone it is not enough:
   an allowlist is a guessing game about dangerous arguments, and whoever writes
   the list always forgets one.

3. BWRAP, the only one that really counts, because it is the kernel saying no,
   not an `if` of ours. The whole filesystem read-only, and the working directory
   as the ONLY writable thing. Here `rm -rf /home/user` does not reach the user's
   real home.

If bwrap is not present on the machine, this module REFUSES to execute. Falling
back to the host "just to make it work" would turn the containment into theatre,
and security theatre is worse than none, because we stop looking.

CGROUP CEILINGS wrap the whole bwrap line in `systemd-run --user --scope`:
`MemoryMax` plus `MemorySwapMax=0` (measured on this machine: `MemoryMax` alone
is not a limit while swap exists, a hog survived 90s under it and swap absorbed
it; `MemorySwapMax=0` is what made it OOM-kill), `TasksMax` against a fork bomb,
and `CPUQuota`. This does not step aside on `authorized=True` either, for the
same reason bwrap does not: it is the kernel's accounting, not a policy of ours.
When `systemd-run` is missing, the command still runs -- there is no host
fallback to refuse to here, unlike bwrap, because the command is still confined
by layers 1-3 -- but the output carries a `NOTE:` saying the ceiling was not
applied. A silent absence is worse than no limit, because we stop looking for
it. See `_cgroup_prefix` and `tasks/pending/007-seccomp-and-cgroup-limits.md`
for where the numbers come from.

A SECCOMP FILTER goes in through `--add-seccomp-fd`, denying the syscall
families no agent command has business calling: nested namespaces and mounts,
module and kexec control, the kernel keyring, NUMA and page migration, `bpf`,
`userfaultfd`, `ptrace` and `perf_event_open`. The first group is the one that
earns its place: measured on this machine, a process inside this jail could
call `unshare(CLONE_NEWUSER)` SUCCESSFULLY, and a fresh user namespace hands
back a full capability set inside itself, which is where a kernel-surface
attack starts. With the filter that call returns EPERM. Like bwrap and the
cgroup ceiling, it does not step aside on `authorized=True`. The filter is
x86_64-only and degrades to a `NOTE:`, never to silence, on another
architecture. It is assembled in `seccomp_filter.py`, which documents the deny-list
and the one hole it knowingly leaves (`clone3`).

WHAT `authorized=True` MEANS, AND WHY IT OVERRIDES EVERY `if` IN THIS FILE
--------------------------------------------------------------------------
It means a human read this exact command on screen and said yes. From that point
on, every refusal in `review()` steps aside: force-push, `gh` mutations, `find
-delete`, a program off the list, a program called by absolute path. Layer 2 and
the network are policy, and policy is a default for what the user did not look
at, never a ban on what they did.

The reason is not convenience, it is honesty about who owns the machine. Whoever
forks this runs their own repositories on their own hardware; a rule of ours that
says "no, not even if you insist" does not protect them, it just makes the agent
useless at the moment they needed it and sends them to a plain terminal, outside
any sandbox at all. And it is a veto they cannot see: the command runs, fails for
a reason unrelated to what they decided, and they end up debugging our policy
instead of their task.

Shell syntax steps aside too. An approved `ls && rm build/` runs through `sh -c`
inside the jail. The string handed to the shell is character for character the
string the user read on screen, so nothing is smuggled in; and anyone who
approved `python3 -c "..."` could already do everything a pipe does. Layer 1
guards a review that never happened, not one that did. This is also what Claude
Code and Codex do: a real shell, gated by approval and by an OS-level sandbox,
because the parser was never the thing keeping anyone safe.

What does NOT step aside, because it is the only thing here that is not an
opinion: BWRAP (layer 3), the cgroup ceiling and the seccomp filter. Approval
never widens what is writable, network or no network, shell or no shell. That is
the kernel, and it is why a shell inside this jail is not the same animal as a
shell on the host.

THE NETWORK ALSO FOLLOWS THE HUMAN DECISION. An approved command runs with
`--share-net`; otherwise `git clone` dies on "Could not resolve host", which
reads as a broken tool rather than as anyone's decision. What is never shown is
never online: commands that run automatically keep the network shut, and `git
push` plus read-only `gh` queries get it through their own narrow exception. See
`_needs_network`.

THE ENVIRONMENT IS BUILT FROM NOTHING, not filtered. `--clearenv` drops
everything isaacli inherited and only the variables set explicitly below survive.
Until this was added the filesystem was closed to credentials while the
environment was wide open: measured on this machine, a key exported in the shell
that started isaacli was readable inside the jail with one `python3 -c`.

HOME is still the working directory, never the real home: no private key or HTTPS
credential of the user is mounted inside the sandbox. Authentication goes through
the ssh-agent SOCKET (`SSH_AUTH_SOCK`), which only signs challenges: there is no
operation that lets `cat`/`python3` read the private key through it.

The counterpart of all this is the approval prompt itself: it has to say when a
command is destructive (`cli._destructive_command`), because approval that became
a reflex is not a decision. The lever belongs to the user; our job is to make
sure they see what they are pulling.

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
import platform
import re
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

# Not named `seccomp`: that is the module name the system `python3-seccomp`
# package installs, and shadowing it would depend on sys.path ordering.
import context_budget
import debug
import seccomp_filter

TIMEOUT_SECONDS = 60        # ceiling per command
OUTPUT_LIMIT = context_budget.CEILINGS["command_output"]  # absolute ceiling; the
# effective cut is this command's share of the window the turn runs in, taken
# before the output goes back to the model rather than after it was built.

# cgroup ceilings via `systemd-run --user --scope`. MemorySwapMax=0 is not
# decorative: MemoryMax alone is not a limit while swap exists (measured on
# this machine, see tasks/pending/007-seccomp-and-cgroup-limits.md -- a memory
# hog under MemoryMax alone survived 90s because swap absorbed it; adding
# MemorySwapMax=0 is what made it OOM-kill). TasksMax stops a fork bomb; a
# generous ceiling still blocks one because a fork bomb is exponential, not
# linear. CPUQuota leaves the rest of the machine usable while a command runs.
CGROUP_MEMORY_MAX = "4G"
CGROUP_MEMORY_SWAP_MAX = "0"
CGROUP_TASKS_MAX = "256"
CGROUP_CPU_QUOTA = "800%"

OPERATORS = {"|", "||", ">", ">>", "<", "<<", "&&", "&", ";", ";;", "(", ")", "$"}

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

# GitHub queries that do not change remote state: these are the ones that run
# without asking. Anything else (`gh api`, `gh pr create`, workflows) is shown to
# the user first and runs if they approve it.
GH_ALLOWED = {
    ("issue", "view"), ("pr", "view"), ("repo", "view"),
    ("release", "view"), ("run", "view"),
    ("auth", "status"),
    ("search", "issues"), ("search", "prs"), ("search", "repos"),
    ("search", "commits"),
}
GH_FORBIDDEN_FLAGS = {"--web", "--show-token"}

# push --force rewrites remote history and destroys other people's work if the
# remote has moved on. That is why it never runs unasked, not why it is banned:
# it is the user's repository, and the approval prompt flags it as destructive.
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
        debug.swallowed("execution._git_global_config")
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

    With `authorized=False` this answers "may it run WITHOUT asking?". With
    `authorized=True` the human already answered, and only what is structurally
    impossible (shell operators) is still refused. See the module docstring.

    Raises Denied with the REASON. It never refuses silently: a small model that
    gets silence retries the same thing; a model that gets a reason corrects it.
    Every reason here says that approval is the way through, so the model asks
    the user instead of giving up or inventing a workaround.
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

    # Shell operators as a LOOSE token: `ls;` used to become a "program" called
    # 'ls;'. Handing them to execve as literal arguments would silently run
    # something other than what was asked, so unapproved they are refused.
    shell_syntax = [p for p in parts if p in OPERATORS or "`" in p]
    if shell_syntax and not authorized:
        raise Denied(
            f"'{shell_syntax[0]}' does not run without asking: commands run "
            f"without a shell, so there is no pipe, redirection or chaining. "
            f"Send one command per call, or ask the user, who can approve this "
            f"line as it is.")
    if shell_syntax:
        # Approved: the user read this exact line, operators included, and said
        # yes. Handing it to `sh -c` INSIDE the jail is what they asked for, and
        # it hides nothing: the string that runs is the string they read. The
        # no-shell layer exists to stop the MODEL from smuggling `curl | sh` past
        # a review that never happened; it was never meant to overrule a review
        # that did. bwrap does not move: the shell is confined like anything else.
        return ["sh", "-c", cmd]

    program = parts[0]
    if "/" in program and not authorized:
        raise Denied(
            f"call the program by name ('{Path(program).name}'), not by "
            f"path ('{program}').")
    if program not in ALLOWED and not authorized:
        raise Denied(
            f"'{program}' does not run without asking. Ask the user: they can "
            f"approve it. Runs unasked: {', '.join(sorted(ALLOWED))}")

    # `find` has flags that execute another program (`-exec sh -c ...` would
    # bring the shell back in through the window) and one that deletes files.
    # bwrap would still hold the damage inside the working directory, but "inside
    # the working directory" is exactly where isaac's work lives.
    if program == "find" and not authorized:
        for flag in ("-exec", "-execdir", "-delete", "-ok", "-okdir",
                     "-fprintf", "-fls", "-fprint"):
            if flag in parts:
                raise Denied(
                    f"'find {flag}' does not run without asking, because it "
                    f"acts instead of searching. Ask the user: they can approve "
                    f"it, or run a plain find and act with another command.")

    if program == "git":
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        if sub is None:
            raise Denied("name the git subcommand (e.g. git status)")
        if sub not in GIT_ALLOWED and not authorized:
            raise Denied(
                f"'git {sub}' does not run without asking. Ask the user: they "
                f"can approve it. Runs unasked: {', '.join(sorted(GIT_ALLOWED))}")
        if (sub == "push" and not authorized
                and (set(parts[1:]) & PUSH_FORBIDDEN_FLAGS)):
            raise Denied(
                "push with --force rewrites remote history irreversibly, so it "
                "does not run on its own. Ask the user: they can approve it. "
                "A normal push (without --force) is allowed.")

    if program == "graphify":
        sub = next((p for p in parts[1:] if not p.startswith("-")), None)
        if sub is None:
            raise Denied("name the graphify subcommand (e.g. graphify query)")
        if sub not in GRAPHIFY_ALLOWED and not authorized:
            raise Denied(
                f"'graphify {sub}' does not run without asking. Ask the user: "
                f"they can approve it. Runs unasked: "
                f"{', '.join(sorted(GRAPHIFY_ALLOWED))}")

    if program == "gh" and not authorized:
        route = tuple(p for p in parts[1:] if not p.startswith("-"))[:2]
        if route not in GH_ALLOWED:
            allowed = ", ".join(" ".join(item) for item in sorted(GH_ALLOWED))
            raise Denied(
                "this gh operation is not read-only, so it does not run on its "
                "own. Ask the user: they can approve it. "
                f"Queries that need no approval: {allowed}")
        forbidden_flags = set(parts) & GH_FORBIDDEN_FLAGS
        if forbidden_flags:
            raise Denied(
                "gh flag not allowed in this query: "
                + ", ".join(sorted(forbidden_flags)))

    return parts


def _needs_network(parts, authorized=False):
    """Network for what the user approved, plus push and reviewed gh queries.

    `authorized` is only true when the human saw this exact command and said yes
    (once, or through a rule they saved themselves). Keeping the network shut
    after that turns the approval into a trap: `git clone`, `pip install` or
    `curl` are approved, run, and then fail with "Could not resolve host", an
    error that has nothing to do with the decision that was made. A veto the
    user cannot see is not containment.

    Commands that run WITHOUT being shown (the automatic read-only ones) stay
    offline: nothing the user did not look at reaches the network.
    """
    if authorized:
        return True
    return (len(parts) >= 2 and parts[0] == "git" and parts[1] == "push") or (
        parts and parts[0] == "gh"
    )


def _cgroup_prefix():
    """`systemd-run --user --scope` prefix that puts memory/pid/cpu ceilings on
    the whole bwrap tree, or None when `systemd-run` is not on this machine.

    A limit that is silently absent is worse than no limit, because we stop
    looking: the caller is expected to say so when this returns None, not to
    pretend the ceiling is there.
    """
    systemd_run = shutil.which("systemd-run")
    if not systemd_run:
        return None
    return [
        systemd_run, "--user", "--scope", "--quiet",
        "-p", f"MemoryMax={CGROUP_MEMORY_MAX}",
        "-p", f"MemorySwapMax={CGROUP_MEMORY_SWAP_MAX}",
        "-p", f"TasksMax={CGROUP_TASKS_MAX}",
        "-p", f"CPUQuota={CGROUP_CPU_QUOTA}",
        "--",
    ]


def _seccomp_fd():
    """A readable, inheritable FD holding the compiled BPF filter, or None.

    bwrap wants the program on a descriptor, not a path, so the blob goes into
    a pipe: 720 bytes is far below the pipe buffer, so writing it and closing
    the write end cannot deadlock. The caller owns the returned FD and must
    close it once the child has been spawned.

    None means `seccomp_filter.build_filter()` declined (a non-x86_64 host, where the
    syscall numbers would mean something else). The caller says so in the
    output rather than letting the layer vanish quietly.
    """
    program = seccomp_filter.build_filter()
    if program is None:
        return None
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, program)
    finally:
        os.close(write_fd)
    os.set_inheritable(read_fd, True)
    return read_fd


def build_bwrap(argv, root, network=False, seccomp_fd=None):
    """Build the bwrap line: nothing writable except the working directory.

    network=True reopens the network with --share-net for `git push` or already
    reviewed gh queries. Push gets the ssh-agent socket and known_hosts; gh gets
    only its configuration on a read-only mount. HOME is still the working
    directory, without exposing credentials to `cat`/`python3`.

    seccomp_fd, when given, is a descriptor holding the compiled BPF filter
    (see `_seccomp_fd`). It must be inheritable and still open when the child
    is spawned, or bwrap fails to read it.
    """
    real, links = _system_binaries()
    git_name, git_email = _git_identity()
    # --clearenv FIRST, and before every --setenv below, which is the order
    # bwrap needs: it drops the whole inherited environment and keeps only what
    # is set after it. Without this the jail inherited isaacli's own environment,
    # measured on this machine: a variable exported in the shell that started
    # isaacli (an OPENAI_API_KEY, a cloud token) was readable inside with a plain
    # `python3 -c "import os"`. The filesystem was already closed to credentials
    # and the environment was not, so the environment is now built here from
    # nothing rather than filtered afterwards.
    line = [shutil.which("bwrap"), "--clearenv"]
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
    if seccomp_fd is not None:
        # --add-seccomp-fd composes with whatever bwrap installs itself,
        # instead of --seccomp, which replaces it.
        line += ["--add-seccomp-fd", str(seccomp_fd)]
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


# How the two layers below report a program they could not start. bwrap execs
# the program itself and dies with `bwrap: execvp cargo: No such file or
# directory` and exit code 1 (NOT 127, measured on this machine, which is why
# nothing here keys off the exit code); an approved line goes through `sh -c`,
# and the shell says `sh: line 1: cargo: command not found`, or `sh: 1: cargo:
# not found` on dash.
NOT_FOUND_PATTERNS = (
    re.compile(r"^bwrap: execvp (?P<name>\S+): No such file or directory", re.M),
    re.compile(r"^[^\n:]*: ?(?:line )?\d*:? ?(?P<name>[^\s:]+): (?:command )?not found",
               re.M),
)


def _missing_programs(err: str):
    """Names the sandbox could not start, in the order they were reported."""
    names = []
    for pattern in NOT_FOUND_PATTERNS:
        for match in pattern.finditer(err):
            name = match.group("name")
            if name and name not in names:
                names.append(name)
    return names


def _missing_program_note(err: str):
    """Say WHICH kind of absence this was, because the two need opposite fixes.

    `command not found` on its own is a lie by omission: the model reads it as
    "this machine does not have the tool" and rewrites the task around the
    absence, when the truth is usually that the tool exists and the jail cannot
    see it. Measured in task 036, where the model was told `cargo: command not
    found` and `yarn: command not found` on a machine that had both.

    So the note names the real state: installed on the host but outside every
    read-only mount, or genuinely not installed anywhere on the user's PATH.
    English on purpose, like every other text the model reads.
    """
    lines = []
    for name in _missing_programs(err):
        host_path = shutil.which(name)
        if host_path:
            lines.append(
                f"NOTE: '{name}' DOES exist on this machine, at {host_path}, but it "
                f"is not reachable from inside the sandbox: the jail mounts the "
                f"system directories and the directories on the user's PATH "
                f"read-only, and that program is under neither. This is a sandbox "
                f"limit, not a missing tool. Do not conclude the machine lacks "
                f"'{name}' and do not work around it silently: say so, and ask the "
                f"user, who can put it on their PATH or approve another route.")
        else:
            lines.append(
                f"NOTE: '{name}' is not installed on this machine either: it is on "
                f"none of the directories of the user's own PATH, so this is not a "
                f"sandbox limit. Say so instead of retrying the same command.")
    return lines


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

    cgroup_prefix = _cgroup_prefix()
    filter_fd = _seccomp_fd()

    proc = None
    try:
        # Inside the try, so the finally below closes the descriptor even if
        # building the line raises: otherwise a session leaks one FD per
        # command until it runs out of them.
        line = build_bwrap(argv, root,
                           network=_needs_network(argv, authorized=authorized),
                           seccomp_fd=filter_fd)
        if cgroup_prefix:
            line = cgroup_prefix + line
        proc = subprocess.Popen(
            line,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,   # own group: lets us kill child and grandchild
            # The FD survives the systemd-run scope in between, verified on
            # this machine; without this it would be closed before bwrap.
            pass_fds=() if filter_fd is None else (filter_fd,),
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
    finally:
        # The child has its own copy; holding this one open would leak a
        # descriptor per command for the life of the session.
        if filter_fd is not None:
            os.close(filter_fd)

    parts = [f"$ {cmd}"]
    if out.strip():
        parts.append(out.rstrip("\n"))
    if err.strip():
        parts.append("--- stderr ---")
        parts.append(err.rstrip("\n"))
    parts.append(f"(exit code: {code})")
    # Before the layer notices, because this one is about the command the user
    # just ran, not about the machine's configuration.
    if code != 0:
        parts.extend(_missing_program_note(err))
    if cgroup_prefix is None:
        parts.append(
            "NOTE: systemd-run is not installed, so this command ran without "
            "memory/pid/cpu ceilings. Install systemd (systemd-run) to restore them.")
    if filter_fd is None:
        parts.append(
            f"NOTE: the seccomp filter is x86_64-only and this machine is "
            f"{platform.machine()}, so this command ran without it. The other "
            f"sandbox layers still applied.")
    text = "\n".join(parts)

    limit = context_budget.bytes_for("command_output")
    if len(text) > limit:
        cut = len(text) - limit
        text = text[:limit] + f"\n… (truncated {cut} characters)"
    return text


SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Run ONE terminal command, confined to the working directory, and "
            "return the raw output (stdout, stderr and exit code). "
            "Send ONE program per call: there is no shell, so a pipe, '>', '&&' "
            "or ';' does not run on its own (the user can approve such a line, "
            "but do not reach for one when separate calls do the job). "
            "Safe read-only commands run automatically; anything else is "
            "shown to the user and runs after they approve it. Runs without "
            "asking: " + ", ".join(sorted(ALLOWED)) +
            f" (git: {', '.join(sorted(GIT_ALLOWED))}; the commit identity comes "
            "from the host's global git config, so 'git config' is neither needed "
            "nor useful). Any OTHER command can still run: propose it and the "
            "user decides. A refusal that says the user can approve it is an "
            "invitation to ask them, not a dead end: do not silently give up and "
            "do not look for a workaround. Commands that run automatically have "
            "no network; approved ones do, so 'git clone' works once approved. "
            "You cannot write outside the working directory, ever. Use this to "
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
