

"""
LoCoMo 10 样本评估（AWS Claude）+ 统计增强版

在 locomo_10_samples 基础上增加：
- Memorize 阶段 context window peak（单次调用的最大 input_tokens）
- Solution 阶段 context window peak（Research + Answer 单次调用的最大 input_tokens）
- Experiment Time（总耗时）
- Cost per Sample / Token per Sample
- 各阶段计时：Memorize / Research / Answer 每步耗时、平均、总时间

# 运行示例
python -m eval.locomo_10_samples_with_stats --data data/locomo/locomo10.json --outdir ./results/locomo_with_stats
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
import time
from datetime import datetime
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
        pred = val.get(pred_field, "") if isinstance(val, dict) else (val if isinstance(val, str) else "")
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



EXPERIMENT_LOG_FILENAME = "experiment_log.jsonl"

def log_experiment_run(
    outdir: str,
    batch_results_file: str,
    stats_file: str,
    argv: List[str],
    total_samples: int,
    total_questions: int,
    overall_f1: float,
    overall_bleu1: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    log_path = os.path.join(outdir, EXPERIMENT_LOG_FILENAME)
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "utc_ts": time.time(),
        "argv": argv,
        "outdir": os.path.abspath(outdir),
        "batch_results_file": os.path.abspath(batch_results_file),
        "stats_file": os.path.abspath(stats_file),
        "total_samples": total_samples,
        "total_questions": total_questions,
        "overall_f1_avg": overall_f1,
        "overall_bleu1_avg": overall_bleu1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"实验记录已追加: {log_path}")

def run_eval_only(results_file: str, outdir: str, pred_key: str = "summary_answer", pred_field: str = "answer") -> None:
    if not os.path.isfile(results_file):
        print(f"错误: 结果文件不存在: {results_file}")
        return
    with open(results_file, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    if not all_results:
        print("结果文件为空，跳过指标计算")
        return
    summary, rows = compute_metrics_by_category(all_results, pred_key=pred_key, pred_field=pred_field)
    overall_f1 = sum(r["F1"] for r in rows) / len(rows)
    overall_bleu1 = sum(r["BLEU1"] for r in rows) / len(rows)
    print("\n" + "=" * 60)
    print("LoCoMo — 仅重算指标 (eval-only)")
    print("=" * 60)
    for r in summary:
        print(f"  Category {r['category']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}, BLEU1_avg={r['BLEU1_avg']:.4f}")
    print(f"\n整体: 问题数={len(all_results)}, 平均 F1={overall_f1:.4f}, 平均 BLEU1={overall_bleu1:.4f}")
    print("=" * 60)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(results_file))[0]
    reeval_stats = os.path.join(outdir, f"{base}_reeval_statistics.json")
    with open(reeval_stats, "w", encoding="utf-8") as f:
        json.dump({
            "source_results_file": os.path.abspath(results_file),
            "reeval_timestamp": datetime.utcnow().isoformat() + "Z",
            "total_questions": len(all_results),
            "overall_f1_avg": overall_f1,
            "overall_bleu1_avg": overall_bleu1,
            "by_category": summary,
            "details": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"重算统计已保存: {reeval_stats}")



def process_sample_claude(
    sample: Dict[str, Any],
    sample_index: int,
    outdir: str,
    region_name: str,
    use_schema: bool,
    max_questions: Optional[int],
    usage_containers: Dict[str, List],
    timing_containers: Optional[Dict[str, List]] = None,
    claude_account_id: str = "your-aws-account-id",
    claude_inference_profile_id: str = "your-inference-profile-id",
    retriever: str = "both",
) -> List[Dict[str, Any]]:
    """使用 AWS Claude 处理单个样本；可选记录各阶段耗时到 timing_containers。"""
    timing = timing_containers or {}
    memorize_times = timing.get("memorize", [])
    research_times = timing.get("research", [])
    answer_times = timing.get("answer", [])

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
    if usage_containers:
        memory_generator.usage_log = usage_containers.get("memory", [])

    memory_agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=memory_generator,
    )


    t_mem_start = time.perf_counter()
    if not os.path.exists(os.path.join(sample_results_dir, "memory_state.json")):
        for session_chunk in session_chunks:
            memory_agent.memorize(session_chunk)
    if "memorize" in timing:
        timing["memorize"].append(time.perf_counter() - t_mem_start)

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
    if retriever in ("all", "both", "bm25"):
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
    if retriever in ("all", "both", "dense"):
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
    if usage_containers:
        research_generator.usage_log = usage_containers.get("research", [])
        working_generator.usage_log = usage_containers.get("working", [])

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
            t_res_start = time.perf_counter()
            result = research_agent.research(q)
            if "research" in timing:
                timing["research"].append(time.perf_counter() - t_res_start)

            t_ans_start = time.perf_counter()
            summary_answer = answer_with_summary(cat, result.integrated_memory, q, working_generator)
            if "answer" in timing:
                timing["answer"].append(time.perf_counter() - t_ans_start)

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
    parser = argparse.ArgumentParser(
        description="LoCoMo 评估（Claude）+ Context Peak / 计时 / Cost&Token per Sample"
    )
    parser.add_argument("--data", type=str, default="/path/to/locomo/dataset.json", help="LoCoMo 数据路径")
    parser.add_argument("--outdir", type=str, default="./results/locomo_with_stats", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=10, help="处理样本数；-1 表示全部")
    parser.add_argument("--max-questions", type=int, default=None, help="每样本最多回答问题数")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS 区域")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=("all", "both", "bm25", "dense"),
        default="all",
        help="检索器",
    )
    parser.add_argument("--use-schema", action="store_true", help="Research 使用 schema")
    parser.add_argument("--price-input", type=float, default=3.0, help="输入单价（美元/百万 token）")
    parser.add_argument("--price-output", type=float, default=15.0, help="输出单价（美元/百万 token）")
    parser.add_argument("--eval-only", action="store_true", help="仅重算指标")
    parser.add_argument("--results-file", type=str, default=None, help="--eval-only 时的 batch_results 路径")
    args = parser.parse_args()

    if args.eval_only:
        results_file = args.results_file or os.path.join(os.path.abspath(args.outdir), "batch_results_0_9.json")
        run_eval_only(results_file, args.outdir)
        return

    region_name = args.region
    claude_account_id = "your-aws-account-id"
    claude_inference_profile_id = "your-inference-profile-id"

    if not os.path.isfile(args.data):
        print(f"错误: 数据文件不存在: {os.path.abspath(args.data)}")
        return

    samples = load_locomo(args.data)
    n_samples = len(samples) if args.max_samples < 0 else min(args.max_samples, len(samples))
    samples = samples[:n_samples]
    print(f"数据: {args.data}")
    print(f"样本数: {n_samples}")
    print(f"输出目录: {args.outdir}")
    print("=" * 60)

    os.makedirs(args.outdir, exist_ok=True)
    usage_containers = {"memory": [], "research": [], "working": []}
    timing_containers = {"memorize": [], "research": [], "answer": []}

    experiment_start = time.perf_counter()
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
                timing_containers=timing_containers,
                claude_account_id=claude_account_id,
                claude_inference_profile_id=claude_inference_profile_id,
                retriever=args.retriever,
            )
        )
    experiment_total_sec = time.perf_counter() - experiment_start

    total_in = sum(u.get("input_tokens", 0) for logs in usage_containers.values() for u in logs)
    total_out = sum(u.get("output_tokens", 0) for logs in usage_containers.values() for u in logs)
    cost_usd = (total_in / 1e6) * args.price_input + (total_out / 1e6) * args.price_output


    memory_usage = usage_containers.get("memory", [])
    research_usage = usage_containers.get("research", [])
    working_usage = usage_containers.get("working", [])
    memorize_peak = max((u.get("input_tokens", 0) for u in memory_usage), default=0)
    solution_usage = research_usage + working_usage
    solution_peak = max((u.get("input_tokens", 0) for u in solution_usage), default=0)

    cost_per_sample = cost_usd / n_samples if n_samples else 0
    token_per_sample_in = total_in / n_samples if n_samples else 0
    token_per_sample_out = total_out / n_samples if n_samples else 0
    token_per_sample = (total_in + total_out) / n_samples if n_samples else 0


    def _timing_summary(name: str, times: List[float]) -> Dict[str, Any]:
        if not times:
            return {"phase": name, "count": 0, "total_sec": 0.0, "avg_sec": 0.0}
        total = sum(times)
        return {
            "phase": name,
            "count": len(times),
            "total_sec": round(total, 4),
            "avg_sec": round(total / len(times), 4),
        }

    memorize_stats = _timing_summary("memorize", timing_containers.get("memorize", []))
    research_stats = _timing_summary("research", timing_containers.get("research", []))
    answer_stats = _timing_summary("answer", timing_containers.get("answer", []))


    print("\n" + "=" * 60)
    print("Token 与费用")
    print("=" * 60)
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  估算费用: ${cost_usd:.4f}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Context Window Peak（单次调用最大 input_tokens）")
    print("=" * 60)
    print(f"  Memorize 阶段 peak:  {memorize_peak}")
    print(f"  Solution 阶段 peak: {solution_peak}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Experiment Time / Cost & Token per Sample")
    print("=" * 60)
    print(f"  Experiment Time (总): {experiment_total_sec:.2f} s")
    print(f"  Cost per Sample:     ${cost_per_sample:.4f}")
    print(f"  Token per Sample:    {token_per_sample:.0f} (input {token_per_sample_in:.0f} + output {token_per_sample_out:.0f})")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("各阶段计时（每个 step 耗时 / 平均 / 总时间）")
    print("=" * 60)
    for stat in (memorize_stats, research_stats, answer_stats):
        print(f"  {stat['phase']}: count={stat['count']}, total={stat['total_sec']} s, avg={stat['avg_sec']} s")
    print(f"  Total experiment: {experiment_total_sec:.2f} s")
    print("=" * 60)

    if not all_results:
        print("没有有效问题结果，跳过指标计算")
        return

    summary, rows = compute_metrics_by_category(all_results, pred_key="summary_answer", pred_field="answer")
    overall_f1 = sum(r["F1"] for r in rows) / len(rows)
    overall_bleu1 = sum(r["BLEU1"] for r in rows) / len(rows)

    print("\n" + "=" * 60)
    print("LoCoMo — 分数汇总")
    print("=" * 60)
    for r in summary:
        print(f"  Category {r['category']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}, BLEU1_avg={r['BLEU1_avg']:.4f}")
    print(f"\n整体: 问题数={len(all_results)}, 平均 F1={overall_f1:.4f}, 平均 BLEU1={overall_bleu1:.4f}")
    print("=" * 60)

    start_idx, end_idx = 0, n_samples - 1
    statistics = {
        "total_samples": n_samples,
        "total_questions": len(all_results),
        "overall_f1_avg": overall_f1,
        "overall_bleu1_avg": overall_bleu1,
        "by_category": summary,
        "details": rows,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": round(cost_usd, 6),
        "price_input_per_1m": args.price_input,
        "price_output_per_1m": args.price_output,

        "context_window_peak": {
            "memorize_input_tokens": memorize_peak,
            "solution_input_tokens": solution_peak,
        },
        "experiment_time_sec": round(experiment_total_sec, 4),
        "cost_per_sample": round(cost_per_sample, 6),
        "token_per_sample": {
            "input": round(token_per_sample_in, 2),
            "output": round(token_per_sample_out, 2),
            "total": round(token_per_sample, 2),
        },
        "timing": {
            "memorize": memorize_stats,
            "research": research_stats,
            "answer": answer_stats,
            "total_experiment_sec": round(experiment_total_sec, 4),
        },
    }
    stats_file = os.path.join(args.outdir, f"batch_statistics_{start_idx}_{end_idx}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print(f"统计已保存: {stats_file}")

    batch_results_file = os.path.join(args.outdir, f"batch_results_{start_idx}_{end_idx}.json")
    with open(batch_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {batch_results_file}")

    log_experiment_run(
        outdir=args.outdir,
        batch_results_file=batch_results_file,
        stats_file=stats_file,
        argv=sys.argv,
        total_samples=n_samples,
        total_questions=len(all_results),
        overall_f1=overall_f1,
        overall_bleu1=overall_bleu1,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=cost_usd,
    )


if __name__ == "__main__":
    main()
