#!/usr/bin/env bash
set -euo pipefail
: "${TT_VISIBLE_DEVICES:?set TT_VISIBLE_DEVICES to a free chip index, see tt-smi -ls}"
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$ROOT/LlamaFactory"
export PJRT_DEVICE=TT HF_HUB_OFFLINE=1
exec .venv/bin/llamafactory-cli train "$ROOT/pirate-demo/pirate_lora_sft.yaml"
