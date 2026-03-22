#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========== 1/4 LoCoMo (Qwen) =========="
bash eval_qwen_locomo.sh

echo "========== 2/4 HotpotQA (Qwen) =========="
bash eval_qwen_hotpotqa.sh

echo "========== 3/4 RULER (Qwen) =========="
bash eval_qwen_ruler.sh

echo "========== 4/4 LongMemEval-S (Qwen) =========="
bash eval_qwen_longmemeval_s.sh

echo ""
echo "========== 全部完成 =========="
echo "结果目录: ./results_qwen/{locomo,hotpotqa,ruler,longmemeval_s}"
