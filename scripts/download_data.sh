#!/bin/bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========== LoCoMo =========="
mkdir -p data/locomo
(cd data/locomo && wget -q -O locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json || true)

echo "========== HotpotQA =========="
mkdir -p data/hotpotqa
for f in eval_400.json eval_1600.json eval_6400.json; do
  (cd data/hotpotqa && wget -q -N "https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/resolve/main/$f") || true
done

echo "========== RULER =========="
if [ -f download_data/download_ruler.py ]; then
  python download_data/download_ruler.py
elif [ -f data/download_data/download_ruler.py ]; then
  python data/download_data/download_ruler.py
else
  echo "请将 download_ruler.py 放在项目根下 download_data/ 或 data/download_data/ 后重试"
fi

echo "========== NarrativeQA =========="
if [ -f download_data/download_narrativeqa.py ]; then
  python download_data/download_narrativeqa.py
elif [ -f data/download_data/download_narrativeqa.py ]; then
  python data/download_data/download_narrativeqa.py
else
  echo "请将 download_narrativeqa.py 放在项目根下 download_data/ 或 data/download_data/ 后重试"
fi

echo "[OK] 数据下载结束"
