#!/usr/bin/env bash
# Run the whole check suite with one command.
#
# The suite was run by hand, one file at a time, which meant the answer to "is
# this safe to push" depended on remembering ten commands and reading ten
# outputs. It also meant a check added to tests/ was only run by whoever knew it
# existed. This discovers tests/check_*.py instead of listing them, so a new
# check joins the suite by being written. CI runs this same script, so there is
# one definition of a passing suite rather than two that drift apart.
#
# Every check runs under a memory and time ceiling. That is not paranoia: on
# 2026-08-21 a check whose fixture answered a selection screen with a value the
# screen never accepts looped forever inside a captured stdout buffer and took
# the machine down at 1.27 GB of resident memory after 34 seconds.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE" || exit 1

MEMORY_KB=${ISAACLI_CHECK_MEMORY_KB:-3000000}
TIMEOUT_S=${ISAACLI_CHECK_TIMEOUT_S:-600}
# How many checks this suite is supposed to run. Discovering the files by glob
# means a check joins by being written, and it also means one that is deleted,
# renamed out of the pattern, or lost in a bad merge leaves without anyone
# deciding that. Measured on 2026-09-08: hiding one file printed
# "all 14 checks passed" and exited zero. Raise this in the same commit that
# adds a check, exactly like the list in the project's own notes. It counts
# files, not passes: the one that needs a real model is skipped here and still
# has to exist.
EXPECTED_CHECKS=${ISAACLI_EXPECTED_CHECKS:-19}

# check_commit_workflow.py calls a real model through Ollama, so it is not part
# of the suite that has to pass before every push. Name it here rather than
# teaching the loop about exceptions somewhere else.
NEEDS_A_REAL_MODEL="check_commit_workflow.py"
# check_execution.py drives the real containment: bwrap mapping uids, the
# seccomp filter, and a systemd TasksMax ceiling. A sandbox nested inside
# another bwrap/systemd jail (an agent's own containment, for example) denies
# unprivileged user namespaces, and there is no fix for that short of running
# outside the nesting. GitHub Actions used to hit the same denial,
# "setting up uid map: Permission denied", but that was traced (task 053,
# 2026-08-23) to kernel.apparmor_restrict_unprivileged_userns=1 on the runner
# image, not to the runner lacking user namespaces: .github/workflows/checks.yml
# now flips that sysctl on the job's own throwaway VM and runs this check like
# any other. --no-privileged stays for the nested-sandbox case and names what
# it skips in the output, instead of reporting a pass nobody earned.
NEEDS_PRIVILEGED_HOST="check_execution.py"

skip_privileged=0
strict=0
for argument in "$@"; do
    case "$argument" in
        --no-privileged) skip_privileged=1 ;;
        --strict) strict=1 ;;
        -h|--help)
            echo "usage: scripts/check.sh [--no-privileged]"
            echo
            echo "  --no-privileged   skip checks needing a host where bwrap can"
            echo "                    map uids and create a loopback interface"
            echo "  --strict          fail when a check skipped part of itself,"
            echo "                    and when fewer than EXPECTED_CHECKS ran"
            echo
            echo "environment: ISAACLI_CHECK_MEMORY_KB, ISAACLI_CHECK_TIMEOUT_S"
            exit 0 ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

python=$(command -v python3 || true)
if [ -z "$python" ]; then
    echo "python3 is required" >&2
    exit 1
fi

failed=""
skipped=""
partial=""
# Counted as it happens rather than derived from the label afterwards: the
# label carries a reason with a space in it, so word-splitting it made one
# skipped file count as two and the floor below never fired.
skipped_count=0
passed=0
log_dir=$(mktemp -d)
trap 'rm -rf "$log_dir"' EXIT

for path in tests/check_*.py; do
    name=$(basename "$path")
    case " $NEEDS_A_REAL_MODEL " in *" $name "*)
        skipped="$skipped $name(needs Ollama)"
        skipped_count=$((skipped_count + 1)); continue ;;
    esac
    if [ "$skip_privileged" = "1" ]; then
        case " $NEEDS_PRIVILEGED_HOST " in *" $name "*)
            skipped="$skipped $name(needs a privileged host)"
            skipped_count=$((skipped_count + 1)); continue ;;
        esac
    fi
    printf '%-34s ' "$name"
    if ( ulimit -v "$MEMORY_KB" 2>/dev/null || true
         exec timeout "$TIMEOUT_S" "$python" "$path" ) > "$log_dir/$name.log" 2>&1; then
        # A check that skipped part of itself passed on less than it claims. On
        # this machine that is honest reporting of a missing systemd-run or an
        # architecture with no seccomp filter; on CI it is the containment proof
        # quietly not happening while the badge stays green.
        internal=$(grep -c '^\[skip' "$log_dir/$name.log" || true)
        if [ "${internal:-0}" != "0" ]; then
            partial="$partial $name($internal)"
            echo "ok, $internal part(s) skipped"
        else
            echo "ok"
        fi
        passed=$((passed + 1))
    else
        status=$?
        if [ "$status" = "124" ]; then
            echo "TIMED OUT after ${TIMEOUT_S}s"
        else
            echo "FAILED (exit $status)"
        fi
        failed="$failed $name"
        sed -n '$p;/FAILED/p;/Error/p;/Traceback/,$p' "$log_dir/$name.log" \
            | tail -20 | sed 's/^/    /'
    fi
done

echo
[ -n "$skipped" ] && echo "skipped:$skipped"
[ -n "$partial" ] && echo "ran with parts skipped:$partial"
if [ -n "$failed" ]; then
    echo "FAILED:$failed"
    exit 1
fi
ran=$((passed + skipped_count))
if [ "$ran" -lt "$EXPECTED_CHECKS" ]; then
    echo "FAILED: $ran check files accounted for, $EXPECTED_CHECKS expected."
    echo "  A check was deleted, renamed, or lost. Raise ISAACLI_EXPECTED_CHECKS"
    echo "  deliberately if the suite really is meant to be smaller."
    exit 1
fi
if [ "$strict" = "1" ] && [ -n "$partial" ]; then
    echo "FAILED: --strict, and these passed on less than the whole check:$partial"
    exit 1
fi
echo "all $passed checks passed"
