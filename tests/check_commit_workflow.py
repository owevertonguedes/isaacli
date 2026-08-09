#!/usr/bin/env python3
"""Live test: Isaac makes a normal commit without a textual signature.

This measures a verifiable flow, not co-authorship. Authorship is the CLI's
responsibility; the model has to commit, explain the reason, not push, and not
declare a false success.

It calls a real model through `isaacli`, so it is not as cheap as the rest of
the suite. Run it on an idle machine.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TEXT_SIGNATURE_RE = re.compile(
    r"(Signed by:\s*Isaac|Co-Authored-By:\s*Isaac|Signed-off-by:\s*Isaac)", re.I)


def run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def create_temp_repo():
    root = Path(tempfile.mkdtemp(prefix="isaac-commit-workflow-"))
    run(["git", "init"], root)
    run(["git", "config", "user.name", "Tester"], root)
    run(["git", "config", "user.email", "tester@example.local"], root)
    readme = root / "README.md"
    readme.write_text("# Commit flow test\n")
    run(["git", "add", "README.md"], root)
    base = run(["git", "commit", "-m", "Initial commit"], root)
    if base.returncode != 0:
        raise RuntimeError(base.stderr or base.stdout)
    readme.write_text(
        readme.read_text() + "\nPending change: validate the normal commit flow.\n")
    return root


def commit_message(repo):
    r = run(["git", "log", "-1", "--format=%B"], repo)
    if r.returncode != 0:
        raise RuntimeError(r.stderr or r.stdout)
    return r.stdout


def evaluate(repo, isaac_output):
    message = commit_message(repo)
    status = run(["git", "status", "--short"], repo).stdout.strip()
    mentioned_push = "$ git push" in isaac_output
    verified = (
        "$ git log -1 --format=%B" in isaac_output
        or "$ git status --short" in isaac_output
    )
    has_reason = bool(re.search(
        r"(because|reason|preserve|record|history|keep|valid)", message, re.I))
    return {
        "repo": str(repo),
        "clean_status": status == "",
        "message": message,
        "has_body_or_reason": has_reason,
        "no_textual_signature": not TEXT_SIGNATURE_RE.search(message),
        "did_not_push": not mentioned_push,
        "isaac_verified_state": verified,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="isaac-granite")
    ap.add_argument("--request", default=(
        "Commit what is pending. The commit message has to explain why we are "
        "committing this. Do not push."
    ))
    ap.add_argument("--repo", help="use an existing repo instead of creating a temporary one")
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve() if args.repo else create_temp_repo()
    if not args.evaluate_only:
        cmd = [str(REPO_ROOT / "isaacli"), "--model", args.model,
               "--workspace", str(repo), args.request]
        r = run(cmd, REPO_ROOT)
        isaac_output = (r.stdout or "") + (r.stderr or "")
        print(r.stdout, end="")
        if r.stderr:
            print(r.stderr, end="", file=sys.stderr)
        if r.returncode != 0:
            print(json.dumps({"ok": False, "repo": str(repo),
                              "error": f"isaacli returned {r.returncode}"},
                             ensure_ascii=False, indent=2))
            return r.returncode
    else:
        isaac_output = ""

    result = evaluate(repo, isaac_output)
    result["ok"] = all(result[k] for k in (
        "clean_status",
        "has_body_or_reason",
        "no_textual_signature",
        "did_not_push",
    ))
    if not args.evaluate_only:
        result["ok"] = bool(result["ok"] and result["isaac_verified_state"])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
