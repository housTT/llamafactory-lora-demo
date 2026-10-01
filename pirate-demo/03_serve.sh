#!/usr/bin/env bash
set -euo pipefail
: "${DEVICE_ID:?set DEVICE_ID to a free chip index, see tt-smi -ls}"
PORT="${PORT:-20001}"
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
TT_VENV="${TT_VENV:-$HOME/.tenstorrent-venv}"
cd "$ROOT/tt-inference-server"
source "$TT_VENV/bin/activate"
exec python run.py \
  --runtime-model-spec-json "$ROOT/pirate-demo/runtime_model_spec_llama-3.1-8b-pirate_p150.json" \
  --model Llama-3.1-8B-Instruct \
  --device p150 \
  --workflow server \
  --docker-server \
  --no-auth \
  --device-id "$DEVICE_ID" \
  --service-port "$PORT" \
  --host-weights-dir "$MODELS_DIR/llama-3.1-8b-pirate" \
  --vllm-override-args '{"enable-auto-tool-choice": true, "tool-call-parser": "llama3_json"}'
