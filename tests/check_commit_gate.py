#!/usr/bin/env python3
"""The commit-workflow check bills the harness for the harness, and no more.

tests/check_commit_workflow.py is the only thing here that drives a real model,
so it cannot run in this suite. Its verdict, though, is ordinary Python, and
that is what this file tests: which assertions decide the exit code, and which
ones are printed as measurement.

The split exists because the two halves answer different questions. A commit
that landed, a clean worktree, no textual signature, nothing pushed: those are
the program's doing. Whether a 3B model writes a commit message that explains
itself, or re-checks its own state afterwards, is the model's. Wiring the
second half into the exit code makes the nightly build red for a reason nobody
can fix in this repository, and a red that nobody can fix is a red nobody
reads.

This file also stops that split from being quietly undone: it fails if a model
measurement wanders back into the gate.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import check_commit_workflow as flow

failures = []


def check(condition, description):
    print(f"[{'ok    ' if condition else 'FAILED'}] {description}")
    if not condition:
        failures.append(description)


def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                          text=True, check=False)


def scratch_repo(message, dirty=False):
    """A repository already in the state the model would have left it in.

    `core.hooksPath` is emptied on purpose: this machine sets a global hooks
    path that refuses an author outside a noreply address, and a test fixture
    borrowing the author's commit policy would fail for a reason that has
    nothing to do with what is being measured.
    """
    root = Path(tempfile.mkdtemp(prefix="isaac-commit-gate-"))
    git(["init", "-q"], root)
    git(["config", "core.hooksPath", ""], root)
    git(["config", "commit.gpgsign", "false"], root)
    git(["config", "user.name", "Tester"], root)
    git(["config", "user.email", "tester@example.local"], root)
    readme = root / "README.md"
    readme.write_text("# Commit flow test\n")
    git(["add", "README.md"], root)
    seeded = git(["commit", "-q", "-m", "Initial commit"], root)
    if seeded.returncode != 0:
        raise RuntimeError(seeded.stderr or seeded.stdout)
    base = flow.head_sha(root)

    readme.write_text(readme.read_text() + "\nPending change: measured here.\n")
    if not dirty:
        git(["add", "README.md"], root)
        landed = git(["commit", "-q", "-m", message], root)
        if landed.returncode != 0:
            raise RuntimeError(landed.stderr or landed.stdout)
    return root, base


# ---------------------------------------------------------------- the split

check(not (set(flow.GATE_KEYS) & set(flow.REPORTED_KEYS)),
      "no assertion is both a gate and a measurement")
check(set(flow.REPORTED_KEYS) == {"has_body_or_reason", "isaac_verified_state"},
      "the two model measurements are the ones the decision named: "
      + ", ".join(sorted(flow.REPORTED_KEYS)))
check(set(flow.GATE_KEYS) == {"commit_happened", "clean_status",
                              "no_textual_signature", "did_not_push"},
      "the gate is the four harness assertions and nothing else: "
      + ", ".join(sorted(flow.GATE_KEYS)))

# ------------------------------------------- the case that used to fail red

# The measured run of 2026-09-08: granite4:micro-h committed, left the tree
# clean, did not push, did not sign, and wrote "Update README.md with new
# information" without ever running git log. Four for four on the harness, and
# the two model measurements false.
repo, base = scratch_repo("Update README.md with new information")
poor = flow.evaluate(repo, "$ git add README.md\n$ git commit -m ...\n", base)
check(flow.gate_failures(poor) == [],
      "the model writing a message that says what and not why does not fail the build")
check(poor["has_body_or_reason"] is False and poor["isaac_verified_state"] is False,
      "and that case really is the hard one: both model measurements are false, "
      "so the previous assertion is not passing on a case that never failed")

# ------------------------------------------------- plant each harness defect

for label, kwargs, output, expected in (
    ("a worktree the model left dirty",
     dict(message="Keep the pending change because history is a record",
          dirty=True), "", "clean_status"),
    ("a run that pushed",
     dict(message="Keep the pending change because history is a record"),
     "$ git push origin main\n", "did_not_push"),
    ("a commit the model signed by name",
     dict(message="Record it\n\nSigned by: Isaac\n"), "", "no_textual_signature"),
):
    planted, planted_base = scratch_repo(**kwargs)
    result = flow.evaluate(planted, output, planted_base)
    check(expected in flow.gate_failures(result),
          f"{label} fails the build, on {expected}")

# No commit at all: HEAD never moved. This one cannot be built by scratch_repo,
# because the whole point is that nothing was committed.
still = Path(tempfile.mkdtemp(prefix="isaac-commit-gate-none-"))
git(["init", "-q"], still)
git(["config", "core.hooksPath", ""], still)
git(["config", "user.name", "Tester"], still)
git(["config", "user.email", "tester@example.local"], still)
(still / "README.md").write_text("# Commit flow test\n")
git(["add", "README.md"], still)
git(["commit", "-q", "-m", "Initial commit"], still)
frozen = flow.head_sha(still)
never = flow.evaluate(still, "", frozen)
check("commit_happened" in flow.gate_failures(never),
      "a run where HEAD never moved fails the build, on commit_happened")

# --------------------------------------------------- unknown is not failure

unknown = flow.evaluate(repo, "", None)
check(unknown["commit_happened"] is None,
      "with no recorded starting point, commit_happened reports unknown, not false")
check("commit_happened" not in flow.gate_failures(unknown),
      "and unknown leaves the gate instead of reddening it")

# ------------------------------------- the exit code, not just the function

# Everything above tests functions. This runs the file, because the exit code
# is what the nightly build reads.
environment = dict(os.environ)
environment["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="isaac-commit-gate-xdg-")
ran = subprocess.run(
    [sys.executable, str(HERE / "check_commit_workflow.py"),
     "--repo", str(repo), "--evaluate-only"],
    capture_output=True, text=True, check=False, env=environment)
check(ran.returncode == 0,
      "the file itself exits 0 on the run whose only failures are model "
      f"measurements (exit {ran.returncode})")
check("[MEASURED] has_body_or_reason: False" in ran.stdout,
      "and the measurement it does not gate on is still printed with its real value")
check('"has_body_or_reason": false' in ran.stdout
      and '"isaac_verified_state": false' in ran.stdout,
      "the JSON still carries every field, gated or not")
check("Traceback" not in ran.stderr,
      "it reports rather than raising: " + (ran.stderr.strip().splitlines() or ["clean"])[-1])

dirty_repo, _ = scratch_repo("Record it because history is a record", dirty=True)
red = subprocess.run(
    [sys.executable, str(HERE / "check_commit_workflow.py"),
     "--repo", str(dirty_repo), "--evaluate-only"],
    capture_output=True, text=True, check=False, env=environment)
check(red.returncode == 1,
      f"and exits 1 when a harness assertion breaks (exit {red.returncode})")
check("[FAILED] clean_status" in red.stdout,
      "naming which one broke")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(1)
print("ISAAC COMMIT GATE OK: the build is billed for the harness, the model is reported")
