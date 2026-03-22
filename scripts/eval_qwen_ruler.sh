#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

base_outputdir="${OUTDIR:-./results_qwen/ruler}"
mkdir -p "$base_outputdir"

python -m eval_qwen.ruler_run \
    --data ./data/ruler \
    --outdir "$base_outputdir" \
    --start-idx 0

echo "[OK] RULER (Qwen) 完成: $base_outputdir（含四类指标 overall_summary_*.json）"
