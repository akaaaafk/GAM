#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

outputdir="${OUTDIR:-./results_qwen/longmemeval_s}"
mkdir -p "$outputdir"

python -m eval_qwen.longmemeval_s_run \
    --data ./data/longmemeval_s/longmemeval_s_cleaned.json \
    --out "$outputdir" \
    --start-idx 0

echo "[OK] LongMemEval-S (Qwen) 完成: $outputdir"
