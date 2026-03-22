

"""
LoCoMo 小样本评估 + 成本估算（Claude，通过 AWS 调用）

- 只跑少量样本/问题，用于快速估算 token 与成本
- 使用 Claude（通过 AWS），配置可来自 test_claude.py
- 汇总 input_tokens / output_tokens 并按单价计算费用
"""

import sys
import os


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_BASELINE_ROOT = os.path.dirname(_PROJECT_ROOT)
if _BASELINE_ROOT not in sys.path:
    sys.path.insert(0, _BASELINE_ROOT)

import re
import json
import math
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

from gam import (
    MemoryAgent,
    ResearchAgent,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    ClaudeGenerator,
    ClaudeGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)



def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_locomo(json_path: str) -> List[Dict[str, Any]]:
    data = load_json(json_path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape.")

def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]] = []
    for k, v in conv_obj.items():
        m = re.match(r'^session_(\d+)$', k)
        if not (m and isinstance(v, list)):
            continue
        original_idx = int(m.group(1))
        idx = original_idx - 1
        ts = conv_obj.get(f"session_{original_idx}_date_time", "")
        ssum = conv_obj.get(f"session_{original_idx}_summary", None)
        sessions.append((idx, ts, v, ssum if isinstance(ssum, str) and ssum.strip() else None))
    sessions.sort(key=lambda x: x[0])
    return sessions

def session_to_text(idx: int, ts: str, turns: List[Dict[str, Any]], session_summary: Optional[str]) -> str:
    lines = [f"=== SESSION {idx} - Dialogue Time(available to answer questions): {ts} ==="]
    lines.append("")
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        lines.append(f"{speaker} ({dia_id}): {text}")
    if session_summary:
        lines.append("")
        lines.append(f"Session {idx} summary: {session_summary}")
    return "\n".join(lines).strip()

def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    return [session_to_text(idx, ts, turns, ssum) for idx, ts, turns, ssum in sessions]

def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    qas: List[Dict[str, Any]] = []
    sid = sample.get("sample_id", None)
    for q in sample.get("qa", []):
        qas.append({
            "sample_id": sid,
            "question": q.get("question"),
            "answer": q.get("answer"),
            "category": q.get("category"),
            "evidence": q.get("evidence"),
        })
    return qas



def make_summary_prompt(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""

def make_summary_prompt_category3(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
The question may need you to analyze and infer the answer from the summary.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""

def answer_with_summary(category: Optional[int], summary: str, question: str, generator) -> str:
    if category == 3:
        prompt = make_summary_prompt_category3(summary, question)
    else:
        prompt = make_summary_prompt(summary, question)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()



def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokens(s: str):
    return normalize_text(s).split() if normalize_text(s) else []

def f1_score(pred: str, gold: str) -> float:
    gtoks, ptoks = tokens(gold), tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount, pcount = Counter(gtoks), Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    p, r = overlap / len(ptoks), overlap / len(gtoks)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def bleu1_score(pred: str, gold: str) -> float:
    gtoks, ptoks = tokens(gold), tokens(pred)
    if not ptoks:
        return 0.0
    gcount, pcount = Counter(gtoks), Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    prec = clipped / len(ptoks)
    bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks) / len(ptoks)) if gtoks else 0.0
    return bp * prec

def compute_metrics_by_category(items, pred_key: str = "summary_answer", pred_field: str = "answer"):
    agg = defaultdict(list)
    rows = []
    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        val = ex.get(pred_key, "")
        pred = val.get(pred_field, "") if isinstance(val, dict) else val
        f1, b1 = f1_score(pred, gold), bleu1_score(pred, gold)
        agg[cat].append((f1, b1))
        rows.append({"q_idx": idx, "category": cat, "gold_answer": str(gold), "prediction": str(pred), "F1": f1, "BLEU1": b1})
    summary = []
    for cat in sorted(agg.keys(), key=str):
        scores = agg[cat]
        if scores:
            summary.append({
                "category": cat,
                "count": len(scores),
                "F1_avg": sum(s[0] for s in scores) / len(scores),
                "BLEU1_avg": sum(s[1] for s in scores) / len(scores),
            })
    return summary, rows



def process_sample_claude(
    sample: Dict[str, Any],
    sample_index: int,
    outdir: str,
    region_name: str,
    use_schema: bool,
    max_questions: Optional[int],
    usage_containers: Dict[str, List],
    claude_account_id: str = "your-aws-account-id",
    claude_inference_profile_id: str = "your-inference-profile-id",
) -> List[Dict[str, Any]]:
    """Claude 接口（Inference Profile，配置可与 test_claude.py 一致）。usage 写入 usage_containers。"""
    sample_id = sample.get("sample_id", f"conv-{sample_index}")
    session_chunks = build_session_chunks_for_sample(sample)
    sample_results_dir = os.path.join(outdir, sample_id)
    os.makedirs(sample_results_dir, exist_ok=True)

    memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
    page_store = InMemoryPageStore(dir_path=sample_results_dir)

    memory_cfg = ClaudeGeneratorConfig(
        region_name=region_name,
        account_id=claude_account_id,
        inference_profile_id=claude_inference_profile_id,
        max_tokens=256,
    )
    memory_generator = ClaudeGenerator(memory_cfg.__dict__)
    memory_generator.usage_log = usage_containers["memory"]

    memory_agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=memory_generator,
    )
    if not os.path.exists(os.path.join(sample_results_dir, "memory_state.json")):
        for session_chunk in session_chunks:
            memory_agent.memorize(session_chunk)

    final_state = memory_store.load()
    with open(os.path.join(sample_results_dir, "memory_state.json"), "w", encoding="utf-8") as f:
        json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)


    retrievers = {}
    try:
        page_index_dir = os.path.join(sample_results_dir, "page_index")
        if os.path.exists(page_index_dir):
            import shutil
            shutil.rmtree(page_index_dir)
        index_retriever = IndexRetriever(IndexRetrieverConfig(index_dir=page_index_dir).__dict__)
        index_retriever.build(page_store)
        retrievers["page_index"] = index_retriever
    except Exception as e:
        print(f"[WARN] IndexRetriever: {e}")
    try:
        bm25_index_dir = os.path.join(sample_results_dir, "bm25_index")
        if os.path.exists(bm25_index_dir):
            import shutil
            shutil.rmtree(bm25_index_dir)
        bm25_retriever = BM25Retriever(BM25RetrieverConfig(index_dir=bm25_index_dir, threads=1).__dict__)
        bm25_retriever.build(page_store)
        retrievers["keyword"] = bm25_retriever
    except Exception as e:
        print(f"[WARN] BM25Retriever: {e}")
    try:
        dense_index_dir = os.path.join(sample_results_dir, "dense_index")
        if os.path.exists(dense_index_dir):
            import shutil
            shutil.rmtree(dense_index_dir)
        dense_retriever = DenseRetriever(
            DenseRetrieverConfig(index_dir=dense_index_dir, model_name="BAAI/bge-m3").__dict__
        )
        dense_retriever.build(page_store)
        retrievers["vector"] = dense_retriever
    except Exception as e:
        print(f"[WARN] DenseRetriever: {e}")

    research_cfg = ClaudeGeneratorConfig(
        region_name=region_name,
        account_id=claude_account_id,
        inference_profile_id=claude_inference_profile_id,
        max_tokens=2048,
        use_schema=use_schema,
    )
    research_generator = ClaudeGenerator(research_cfg.__dict__)
    working_cfg = ClaudeGeneratorConfig(
        region_name=region_name,
        account_id=claude_account_id,
        inference_profile_id=claude_inference_profile_id,
        max_tokens=256,
    )
    working_generator = ClaudeGenerator(working_cfg.__dict__)
    research_generator.usage_log = usage_containers["research"]
    working_generator.usage_log = usage_containers["working"]

    research_agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=research_generator,
        max_iters=3,
    )

    qas = collect_qa_items_for_sample(sample)
    if max_questions is not None:
        qas = qas[:max_questions]
    qa_results = []
    for i, qi in enumerate(qas, 1):
        q, gold, cat = qi.get("question") or "", qi.get("answer"), qi.get("category")
        if cat == 5:
            continue
        try:
            result = research_agent.research(q)
            summary_answer = answer_with_summary(cat, result.integrated_memory, q, working_generator)
            qa_results.append({
                "question": q,
                "gold_answer": gold,
                "category": cat,
                "summary_answer": summary_answer,
            })
        except Exception as e:
            qa_results.append({"question": q, "gold_answer": gold, "category": cat, "error": str(e)})

    with open(os.path.join(sample_results_dir, "qa_results.json"), "w", encoding="utf-8") as f:
        json.dump(qa_results, f, ensure_ascii=False, indent=2)
    return qa_results



def main():
    import argparse
    parser = argparse.ArgumentParser(description="LoCoMo 小样本评估 + 成本估算（Claude，通过 AWS）")
    parser.add_argument("--data", type=str, default="/path/to/locomo/dataset.json", help="LoCoMo 数据路径")
    parser.add_argument("--outdir", type=str, default="./results/locomo_cost", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=1, help="最多处理样本数（小样本用）")
    parser.add_argument("--max-questions", type=int, default=2, help="每个样本最多回答问题数")
    parser.add_argument("--region", type=str, default="us-east-1", help="区域")
    parser.add_argument("--use-schema", action="store_true", help="Research 是否使用 schema")
    parser.add_argument("--price-input", type=float, default=3.0, help="输入单价（美元/百万 token）")
    parser.add_argument("--price-output", type=float, default=15.0, help="输出单价（美元/百万 token）")
    args = parser.parse_args()

    region_name = args.region
    claude_account_id = "your-aws-account-id"
    claude_inference_profile_id = "your-inference-profile-id"
    try:
        import test_claude as tc
        region_name = getattr(tc, "BEDROCK_REGION", region_name)
        claude_account_id = getattr(tc, "BEDROCK_ACCOUNT_ID", claude_account_id)
        claude_inference_profile_id = getattr(tc, "BEDROCK_INFERENCE_PROFILE_ID", claude_inference_profile_id)
        print(f"[Claude] 使用 test_claude.py 配置: region={region_name}, account_id={claude_account_id}, profile_id={claude_inference_profile_id}")
    except Exception as e:
        print(f"[Claude] 未使用 test_claude.py（{e}），使用默认 Inference Profile")

    if not os.path.isfile(args.data):
        print(f"错误: 数据文件不存在: {os.path.abspath(args.data)}")
        print("请先下载 LoCoMo 数据，例如：")
        print("  mkdir -p data/locomo")
        print("  cd data/locomo && curl -L -o locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json")
        print("或在 PowerShell：")
        print("  New-Item -ItemType Directory -Force -Path data/locomo")
        print("  Invoke-WebRequest -Uri https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json -OutFile data/locomo/locomo10.json")
        print("然后指定路径: --data ./data/locomo/locomo10.json")
        return

    samples = load_locomo(args.data)
    samples = samples[: args.max_samples]
    print(f"数据: {args.data}, 样本数: {len(samples)}, 每样本最多问题数: {args.max_questions}")
    print(f"Claude region: {region_name}, profile_id: {claude_inference_profile_id}")

    usage_containers = {"memory": [], "research": [], "working": []}
    all_results = []
    for idx, sample in enumerate(tqdm(samples, desc="样本")):
        all_results.extend(
            process_sample_claude(
                sample,
                idx,
                args.outdir,
                region_name=region_name,
                use_schema=args.use_schema,
                max_questions=args.max_questions,
                usage_containers=usage_containers,
                claude_account_id=claude_account_id,
                claude_inference_profile_id=claude_inference_profile_id,
            )
        )

    total_in = sum(u.get("input_tokens", 0) for logs in usage_containers.values() for u in logs)
    total_out = sum(u.get("output_tokens", 0) for logs in usage_containers.values() for u in logs)
    cost = (total_in / 1e6) * args.price_input + (total_out / 1e6) * args.price_output

    memory_usage = usage_containers.get("memory", [])
    research_usage = usage_containers.get("research", [])
    working_usage = usage_containers.get("working", [])
    memorize_peak = max((u.get("input_tokens", 0) for u in memory_usage), default=0)
    solution_usage = research_usage + working_usage
    solution_peak = max((u.get("input_tokens", 0) for u in solution_usage), default=0)

    print("\n" + "=" * 60)
    print("Token 与成本汇总")
    print("=" * 60)
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  单价: 输入 ${args.price_input}/1M, 输出 ${args.price_output}/1M")
    print(f"  估算费用: ${cost:.4f}")
    print("=" * 60)
    print("\nContext Window Peak（单次调用最大 input_tokens）:")
    print(f"  Memorize 阶段 peak:  {memorize_peak}")
    print(f"  Solution 阶段 peak: {solution_peak}")
    print("=" * 60)

    if all_results:
        summary, rows = compute_metrics_by_category(all_results, pred_key="summary_answer", pred_field="answer")
        overall_f1 = sum(r["F1"] for r in rows) / len(rows)
        overall_b1 = sum(r["BLEU1"] for r in rows) / len(rows)
        print(f"  问题数: {len(all_results)}, 平均 F1: {overall_f1:.4f}, BLEU1: {overall_b1:.4f}")

    stats = {
        "max_samples": args.max_samples,
        "max_questions_per_sample": args.max_questions,
        "samples_run": len(samples),
        "questions_run": len(all_results),
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": round(cost, 6),
        "price_input_per_1m": args.price_input,
        "price_output_per_1m": args.price_output,
        "context_window_peak": {
            "memorize_input_tokens": memorize_peak,
            "solution_input_tokens": solution_peak,
        },
    }
    stats_path = os.path.join(args.outdir, "cost_stats.json")
    os.makedirs(args.outdir, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  统计已写: {stats_path}")


if __name__ == "__main__":
    main()
