#!/usr/bin/env sh
# Build the Ollama model that isaacli runs against.
#
# Everything is overridable, because the right base model is a measurement
# result and not a constant. It changes as models improve.
#
#   ./scripts/build-model.sh
#   BASE_MODEL=granite4:micro ./scripts/build-model.sh
#   MODEL_NAME=isaac-test NUM_CTX=16384 ./scripts/build-model.sh
#
# BASE_MODEL   base to stack configuration on (must expose `tools`)
# MODEL_NAME   name to create; this is what ISAACLI_MODEL should point at
# NUM_CTX      context window; below ~8k the tool schema starts getting cut
# TEMPERATURE  0 keeps tool calls deterministic
set -eu

BASE_MODEL="${BASE_MODEL:-granite4:micro-h}"
MODEL_NAME="${MODEL_NAME:-isaac-granite}"
NUM_CTX="${NUM_CTX:-8192}"
TEMPERATURE="${TEMPERATURE:-0}"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEMPLATE="$ROOT/tool_harness/Modelfile.isaac-granite.tmpl"

[ -f "$TEMPLATE" ] || { echo "template not found: $TEMPLATE" >&2; exit 1; }
command -v ollama >/dev/null 2>&1 || { echo "ollama is not on PATH" >&2; exit 1; }

# Refuse a base model that cannot call tools. Without this the harness still
# runs, but the model narrates the work instead of doing it, and the failure
# looks like a harness bug rather than a wrong base model.
if ! ollama show "$BASE_MODEL" 2>/dev/null | grep -q '^ *tools *$'; then
    echo "base model '$BASE_MODEL' does not report the 'tools' capability." >&2
    echo "pull it first (ollama pull $BASE_MODEL), or pick a base that supports tool calling." >&2
    exit 1
fi

RENDERED=$(mktemp)
trap 'rm -f "$RENDERED"' EXIT

sed -e "s|__BASE_MODEL__|$BASE_MODEL|g" \
    -e "s|__NUM_CTX__|$NUM_CTX|g" \
    -e "s|__TEMPERATURE__|$TEMPERATURE|g" \
    "$TEMPLATE" > "$RENDERED"

echo "creating '$MODEL_NAME' from '$BASE_MODEL' (num_ctx=$NUM_CTX, temperature=$TEMPERATURE)"
ollama create "$MODEL_NAME" -f "$RENDERED"

echo
echo "done. run it with:"
echo "  ISAACLI_MODEL=$MODEL_NAME ./isaacli \"list the files here\""
