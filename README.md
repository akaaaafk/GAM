# GAM

General Agentic Memory — evaluation code for Claude (AWS Bedrock) and Qwen backends.

Paper: https://arxiv.org/abs/2511.18423

---

## Setup

```bash
pip install -r requirements.txt
```

For BM25 retrieval, [Pyserini](https://github.com/castorini/pyserini) also requires Java 11+.

---

## Data

Place datasets under `data/` as follows:

```
data/
├── locomo/locomo10.json
├── hotpotqa/eval_1600.json
├── ruler/cwe.jsonl
│        fwe.jsonl
│        niah_single_1.jsonl
│        niah_single_2.jsonl
│        niah_single_3.jsonl
│        niah_multikey_1.jsonl
│        niah_multikey_2.jsonl
│        niah_multikey_3.jsonl
│        niah_multiquery.jsonl
│        niah_multivalue.jsonl
│        vt.jsonl
│        qa_1.jsonl
│        qa_2.jsonl
└── longmemeval_s/longmemeval_s_cleaned.json
```

---

## Backend 1 — Claude via AWS Bedrock

Set environment variables:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
export BEDROCK_ACCOUNT_ID=your-aws-account-id
export BEDROCK_INFERENCE_PROFILE_ID=your-inference-profile-id
```

Run evaluations:

```bash
bash scripts/eval_locomo.sh
bash scripts/eval_hotpotqa.sh
bash scripts/eval_ruler.sh
```

Or directly:

```bash
python -m eval.locomo_test \
    --data data/locomo/locomo10.json \
    --outdir results/locomo \
    --account-id $BEDROCK_ACCOUNT_ID \
    --inference-profile-id $BEDROCK_INFERENCE_PROFILE_ID

python -m eval.hotpotqa_run_bedrock \
    --data data/hotpotqa/eval_1600.json \
    --outdir results/hotpotqa

python -m eval.ruler_run_bedrock \
    --data data/ruler \
    --outdir results/ruler

python -m eval.longmemeval_s_run \
    --data data/longmemeval_s/longmemeval_s_cleaned.json \
    --out results/longmemeval_s
```

---

## Backend 2 — Qwen via OpenAI-compatible API

Set environment variables:

```bash
export TINKER_BASE_URL=https://your-api-endpoint/v1
export TINKER_API_KEY=your-api-key
export QWEN_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

Run all benchmarks at once:

```bash
bash scripts/eval_qwen_all.sh
```

Or individually:

```bash
bash scripts/eval_qwen_locomo.sh
bash scripts/eval_qwen_hotpotqa.sh
bash scripts/eval_qwen_ruler.sh
bash scripts/eval_qwen_longmemeval_s.sh
```

Or directly:

```bash
python -m eval_qwen.locomo_10_samples_with_stats \
    --data data/locomo/locomo10.json \
    --outdir results_qwen/locomo \
    --max-samples 10

python -m eval_qwen.hotpotqa_run \
    --data data/hotpotqa/eval_400.json \
    --outdir results_qwen/hotpotqa

python -m eval_qwen.ruler_run \
    --data data/ruler \
    --outdir results_qwen/ruler

python -m eval_qwen.longmemeval_s_run \
    --data data/longmemeval_s/longmemeval_s_cleaned.json \
    --out results_qwen/longmemeval_s
```

Results are saved under `results/` (Claude) and `results_qwen/` (Qwen).
