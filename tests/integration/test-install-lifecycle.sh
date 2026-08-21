#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
engine=${CONTAINER_ENGINE:-podman}
command -v "$engine" >/dev/null 2>&1 || {
    echo "Container engine not found: $engine" >&2
    exit 2
}

temporary=$(mktemp -d /tmp/isaacli-install-lifecycle.XXXXXX)
case "$temporary" in
    /tmp/isaacli-install-lifecycle.*) ;;
    *) echo "Unsafe temporary path: $temporary" >&2; exit 2 ;;
esac
context="$temporary/context"
container="isaacli-lifecycle-${USER:-user}-$$"
image="localhost/isaacli-lifecycle:$$"
log=${ISAACLI_LIFECYCLE_LOG:-"/tmp/isaacli-install-lifecycle-$(date +%Y%m%d-%H%M%S).log"}

cleanup() {
    "$engine" rm --force "$container" >/dev/null 2>&1 || true
    "$engine" image rm --force "$image" >/dev/null 2>&1 || true
    rm -rf -- "$temporary"
}
trap cleanup EXIT INT TERM

mkdir -p "$context/tests/integration"
git -C "$repo_dir" archive HEAD | tar -x -C "$context"
# Apply tracked working-tree changes to the archive so the command validates the
# exact candidate being reviewed without ever mounting the real checkout.
git -C "$repo_dir" diff --binary HEAD -- \
    isaacli tool_harness tests scripts \
    | git -C "$context" apply
cp "$repo_dir/tests/integration/Containerfile.install-lifecycle" \
   "$context/tests/integration/Containerfile.install-lifecycle"
cp "$repo_dir/tests/integration/install_lifecycle.py" \
   "$context/tests/integration/install_lifecycle.py"

fingerprint() {
    for path in \
        "$HOME/.local/bin/isaacli" \
        "$HOME/.config/isaacli/config.json" \
        "$HOME/.config/isaacli/secrets.json" \
        "$HOME/.ollama" \
        /usr/local/bin/ollama \
        /usr/local/lib/ollama \
        /etc/systemd/system/ollama.service \
        /usr/share/ollama; do
        if [[ -e "$path" || -L "$path" ]]; then
            stat -c '%n|%F|%a|%u|%g|%s|%Y' "$path"
            if [[ -f "$path" && ! -L "$path" ]]; then
                sha256sum "$path"
            elif [[ -L "$path" ]]; then
                readlink "$path"
            fi
        else
            echo "$path|absent"
        fi
    done
}

fingerprint > "$temporary/host.before"
"$engine" build --tag "$image" \
    --file "$context/tests/integration/Containerfile.install-lifecycle" "$context"
"$engine" run --detach --name "$container" --privileged --systemd=always \
    --tmpfs /run --tmpfs /tmp "$image" >/dev/null

ready=false
for _ in $(seq 1 30); do
    state=$("$engine" exec "$container" systemctl is-system-running 2>/dev/null || true)
    if [[ "$state" == running || "$state" == degraded ]]; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || {
    "$engine" logs "$container" >&2
    echo "Disposable systemd did not become ready." >&2
    exit 1
}

"$engine" exec --user isaac \
    --env HOME=/home/isaac \
    --env PATH=/home/isaac/.local/bin:/usr/local/bin:/usr/bin:/bin \
    --env ISAACLI_RUNTIME_DIR=/tmp/isaacli-runtime \
    --env "ISAACLI_KAGGLE_TOKEN=${ISAACLI_KAGGLE_TOKEN:-}" \
    "$container" python3 /opt/isaacli/tests/integration/install_lifecycle.py \
    | tee "$log"

fingerprint > "$temporary/host.after"
cmp "$temporary/host.before" "$temporary/host.after"
echo "Host fingerprints unchanged. Sanitized evidence: $log" | tee -a "$log"
