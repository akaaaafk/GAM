#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

base_outputdir="${OUTDIR:-./results_qwen/hotpotqa}"
mkdir -p "$base_outputdir"

for dataset in "eval_400" "eval_1600" "eval_3200"
do
    echo "Processing dataset: $dataset"
    outputdir=$base_outputdir/${dataset}
    mkdir -p "$outputdir"

    python -m eval_qwen.hotpotqa_run \
        --data ./data/hotpotqa/${dataset}.json \
        --outdir "$outputdir" \
        --start-idx 0
done

echo "[OK] HotpotQA (Qwen) 完成: $base_outputdir"
