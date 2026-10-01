#!/usr/bin/env bash
set -euo pipefail
PIRATE_PORT="${PIRATE_PORT:-20001}"
BASE_PORT="${BASE_PORT:-20000}"
python3 "$(dirname "$(readlink -f "$0")")/evaluate.py" \
  "base|http://127.0.0.1:${BASE_PORT}|meta-llama/Llama-3.1-8B-Instruct" \
  "pirate|http://127.0.0.1:${PIRATE_PORT}|llama-3.1-8b-pirate"
