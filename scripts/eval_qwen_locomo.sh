#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

outputdir="${OUTDIR:-./results_qwen/locomo}"
mkdir -p "$outputdir"

python -m eval_qwen.locomo_10_samples_with_stats \
    --data ./data/locomo/locomo10.json \
    --outdir "$outputdir" \
    --max-samples 10

echo "[OK] LoCoMo (Qwen) 完成: $outputdir"
