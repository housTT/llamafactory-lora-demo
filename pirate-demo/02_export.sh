#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
ADAPTER="${ADAPTER:-$ROOT/LlamaFactory/saves/Llama-3.1-8B-Instruct/lora/pirate-lora}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
OUT="$MODELS_DIR/llama-3.1-8b-pirate"
cd "$ROOT/LlamaFactory"
USE_TORCH_XLA=0 HF_HUB_OFFLINE=1 .venv/bin/llamafactory-cli export \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --adapter_name_or_path "$ADAPTER" \
  --template llama3 --finetuning_type lora --export_device cpu \
  --export_dir "$OUT"
chmod a+r "$OUT"/*.safetensors
ls -l "$OUT"
