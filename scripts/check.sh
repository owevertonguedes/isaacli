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

# check_commit_workflow.py calls a real model through Ollama, so it is not part
# of the suite that has to pass before every push. Name it here rather than
# teaching the loop about exceptions somewhere else.
NEEDS_A_REAL_MODEL="check_commit_workflow.py"
# check_execution.py drives the real containment: bwrap mapping uids, its own
# loopback interface, and a systemd TasksMax ceiling. A nested sandbox or a
# hosted CI runner denies all three, measured on GitHub Actions 2026-08-22:
# "setting up uid map: Permission denied". check_sandbox.py does run there, so
# skipping both would throw away a check that works. --no-privileged skips
# exactly this one and names it in the output, instead of reporting a pass
# nobody earned.
NEEDS_PRIVILEGED_HOST="check_execution.py"

skip_privileged=0
for argument in "$@"; do
    case "$argument" in
        --no-privileged) skip_privileged=1 ;;
        -h|--help)
            echo "usage: scripts/check.sh [--no-privileged]"
            echo
            echo "  --no-privileged   skip checks needing a host where bwrap can"
            echo "                    map uids and create a loopback interface"
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
passed=0
log_dir=$(mktemp -d)
trap 'rm -rf "$log_dir"' EXIT

for path in tests/check_*.py; do
    name=$(basename "$path")
    case " $NEEDS_A_REAL_MODEL " in *" $name "*)
        skipped="$skipped $name(needs Ollama)"; continue ;;
    esac
    if [ "$skip_privileged" = "1" ]; then
        case " $NEEDS_PRIVILEGED_HOST " in *" $name "*)
            skipped="$skipped $name(needs a privileged host)"; continue ;;
        esac
    fi
    printf '%-34s ' "$name"
    if ( ulimit -v "$MEMORY_KB" 2>/dev/null || true
         exec timeout "$TIMEOUT_S" "$python" "$path" ) > "$log_dir/$name.log" 2>&1; then
        echo "ok"
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
if [ -n "$failed" ]; then
    echo "FAILED:$failed"
    exit 1
fi
echo "all $passed checks passed"
