

"""
LongMemEval-S 跑分（AWS Bedrock Claude）+ 时间 / 日志 / 花费统计

基于 longmemeval_run.py，数据使用 LongMemEval-S（如 longmemeval_s_cleaned.json）。
增加：实验总耗时、各阶段耗时（memorize / research / working）、Token 与费用统计、
experiment_log.jsonl 与 batch_statistics 输出。

运行示例：
  python -m eval.longmemeval_s_run --data data/longmemeval_s_cleaned.json --out results/longmemeval_s
"""

import os
import sys
import json
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gam import (
    MemoryAgent,
    ResearchAgent,
    ClaudeGenerator,
    ClaudeGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    IndexRetrieverConfig,
    BM25Retriever,
    BM25RetrieverConfig,
    DenseRetriever,
    DenseRetrieverConfig,
)




def load_longmemeval(json_path: str) -> List[Dict[str, Any]]:
    """加载 LongMemEval JSON（含 LongMemEval-S）。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "instances", "examples"):
            if key in data and isinstance(data[key], list):
                return data[key]
        vals = list(data.values())
        if len(vals) == 1 and isinstance(vals[0], list):
            return vals[0]
    raise ValueError(f"Unsupported LongMemEval format in {json_path}")


def sessions_to_memory_messages(haystack_sessions: List[List[Dict[str, Any]]]) -> List[str]:
    """将 haystack_sessions 转为 MemoryAgent 可 memorize 的文本列表。"""
    messages = []
    for sess_idx, session in enumerate(haystack_sessions):
        if not session:
            continue
        lines = [f"[Session {sess_idx + 1}]"]
        for turn in session:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if content:
                lines.append(f"{role.capitalize()}: {content}")
        if len(lines) > 1:
            messages.append("\n".join(lines))
    return messages


def make_qa_prompt(research_summary: str, question: str) -> str:
    """根据 research 摘要与问题生成最终答案的 prompt。"""
    return f"""You are a careful reading assistant. Use the given context to answer the question.
Answer with ONLY the final answer; no extra explanation.

Question:
{question}

Context:
{research_summary}

Answer:
"""




EXPERIMENT_LOG_FILENAME = "experiment_log.jsonl"


def log_experiment_run(
    outdir: str,
    bench: str,
    hyp_path: str,
    stats_path: str,
    argv: List[str],
    total_instances: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    experiment_time_sec: float,
) -> None:
    log_path = os.path.join(outdir, EXPERIMENT_LOG_FILENAME)
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "utc_ts": time.time(),
        "bench": bench,
        "argv": argv,
        "outdir": os.path.abspath(outdir),
        "hypothesis_file": os.path.abspath(hyp_path),
        "stats_file": os.path.abspath(stats_path),
        "total_instances": total_instances,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "experiment_time_sec": experiment_time_sec,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"实验记录已追加: {log_path}")




def process_one_instance(
    instance: Dict[str, Any],
    index: int,
    outdir: str,
    region_name: str,
    claude_account_id: str,
    claude_inference_profile_id: str,
    use_schema: bool = False,
    retriever: str = "bm25",
    max_iters: int = 3,
    usage_containers: Optional[Dict[str, list]] = None,
    timing_containers: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    """对单个 instance 跑 GAM（Bedrock Claude），并记录 usage 与各阶段耗时。"""
    question_id = instance.get("question_id", f"q_{index}")
    question = instance.get("question", "")
    haystack_sessions = instance.get("haystack_sessions", [])

    result = {"question_id": question_id, "hypothesis": ""}
    timing = timing_containers or {}
    t_mem, t_res, t_work = timing.get("memorize", []), timing.get("research", []), timing.get("working", [])

    instance_dir = os.path.join(outdir, "instances", question_id.replace("/", "_").replace("\\", "_"))
    os.makedirs(instance_dir, exist_ok=True)

    memory_store = InMemoryMemoryStore(dir_path=instance_dir)
    page_store = InMemoryPageStore(dir_path=instance_dir)

    memory_cfg = ClaudeGeneratorConfig(
        region_name=region_name,
        account_id=claude_account_id,
        inference_profile_id=claude_inference_profile_id,
        max_tokens=512,
    )
    memory_generator = ClaudeGenerator(memory_cfg.__dict__)
    if usage_containers:
        memory_generator.usage_log = usage_containers.get("memory", [])

    memory_agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=memory_generator,
    )

    memory_messages = sessions_to_memory_messages(haystack_sessions)
    if not memory_messages:
        result["error"] = "no haystack_sessions"
        return result

    t0 = time.perf_counter()
    for msg in memory_messages:
        try:
            memory_agent.memorize(msg)
        except Exception as e:
            result["error"] = f"memorize: {e}"
            return result
    if "memorize" in timing:
        timing["memorize"].append(time.perf_counter() - t0)

    retrievers = {}
    try:
        index_dir = os.path.join(instance_dir, "page_index")
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        index_retriever = IndexRetriever(IndexRetrieverConfig(index_dir=index_dir).__dict__)
        index_retriever.build(page_store)
        retrievers["page_index"] = index_retriever
    except Exception as e:
        result["error"] = f"index retriever: {e}"
        return result

    if retriever in ("both", "bm25") and BM25Retriever is not None:
        try:
            bm25_dir = os.path.join(instance_dir, "bm25_index")
            if os.path.exists(bm25_dir):
                shutil.rmtree(bm25_dir)
            bm25_retriever = BM25Retriever(BM25RetrieverConfig(index_dir=bm25_dir, threads=1).__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
        except Exception:
            pass

    if retriever in ("both", "dense") and DenseRetriever is not None:
        try:
            dense_dir = os.path.join(instance_dir, "dense_index")
            if os.path.exists(dense_dir):
                shutil.rmtree(dense_dir)
            dense_retriever = DenseRetriever(
                DenseRetrieverConfig(index_dir=dense_dir, model_name="BAAI/bge-m3").__dict__
            )
            dense_retriever.build(page_store)
            retrievers["vector"] = dense_retriever
        except Exception:
            pass

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
        max_iters=max_iters,
    )

    t1 = time.perf_counter()
    try:
        research_result = research_agent.research(question)
        research_summary = research_result.integrated_memory or ""
    except Exception as e:
        result["error"] = f"research: {e}"
        return result
    if "research" in timing:
        timing["research"].append(time.perf_counter() - t1)

    qa_prompt = make_qa_prompt(research_summary, question)
    t2 = time.perf_counter()
    try:
        response = working_generator.generate_single(prompt=qa_prompt)
        hypothesis = (response.get("text") or "").strip()
    except Exception as e:
        result["error"] = f"working: {e}"
        result["hypothesis"] = research_summary[:500]
        return result
    if "working" in timing:
        timing["working"].append(time.perf_counter() - t2)

    result["hypothesis"] = hypothesis
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LongMemEval-S (Bedrock Claude): time, log, cost 统计；输出 hypothesis jsonl + batch_statistics + experiment_log.jsonl"
    )
    parser.add_argument("--data", type=str, required=True, help="LongMemEval-S JSON（如 longmemeval_s_cleaned.json）")
    parser.add_argument("--out", type=str, default="./results/longmemeval_s", help="输出目录")
    parser.add_argument("--hypothesis", type=str, default=None, help="输出 jsonl 路径，默认 <out>/gam_hypothesis.jsonl")
    parser.add_argument("--start-idx", type=int, default=0, help="起始样本下标")
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本下标（不包含），默认全部")
    parser.add_argument("--max-iters", type=int, default=3, help="ResearchAgent 最大迭代次数")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS 区域")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=("both", "bm25", "dense"),
        default="bm25",
        help="检索器",
    )
    parser.add_argument("--use-schema", action="store_true", help="Research 是否使用 schema")
    parser.add_argument("--price-input", type=float, default=3.0, help="输入单价（美元/百万 token）")
    parser.add_argument("--price-output", type=float, default=15.0, help="输出单价（美元/百万 token）")

    args = parser.parse_args()

    region_name = args.region
    claude_account_id = "your-aws-account-id"
    claude_inference_profile_id = "your-inference-profile-id"
    print(f"[Bedrock] region={region_name}, account_id={claude_account_id}, profile_id={claude_inference_profile_id}")

    if not os.path.isfile(args.data):
        print(f"错误: 数据文件不存在: {os.path.abspath(args.data)}")
        return

    os.makedirs(args.out, exist_ok=True)
    hyp_path = args.hypothesis or os.path.join(args.out, "gam_hypothesis.jsonl")

    data = load_longmemeval(args.data)
    end = args.end_idx if args.end_idx is not None else len(data)
    indices = list(range(args.start_idx, min(end, len(data))))

    if not indices:
        print("No instances to process.")
        return

    print(f"LongMemEval-S data: {args.data}, instances {args.start_idx}..{end-1} (total {len(indices)})")
    print(f"Output: {hyp_path}")

    usage_containers = {"memory": [], "research": [], "working": []}
    timing_containers = {"memorize": [], "research": [], "working": []}

    experiment_start = time.perf_counter()
    with open(hyp_path, "w", encoding="utf-8") as f:
        for i in tqdm(indices, desc="LongMemEval-S (Bedrock)"):
            instance = data[i]
            out = process_one_instance(
                instance,
                i,
                args.out,
                region_name=region_name,
                claude_account_id=claude_account_id,
                claude_inference_profile_id=claude_inference_profile_id,
                use_schema=args.use_schema,
                retriever=args.retriever,
                max_iters=args.max_iters,
                usage_containers=usage_containers,
                timing_containers=timing_containers,
            )
            f.write(json.dumps({"question_id": out["question_id"], "hypothesis": out.get("hypothesis", "")}, ensure_ascii=False) + "\n")
            if out.get("error"):
                tqdm.write(f"Warning [{out['question_id']}]: {out['error']}")

    experiment_total_sec = time.perf_counter() - experiment_start

    total_in = sum(u.get("input_tokens", 0) for logs in usage_containers.values() for u in logs)
    total_out = sum(u.get("output_tokens", 0) for logs in usage_containers.values() for u in logs)
    cost_usd = (total_in / 1e6) * args.price_input + (total_out / 1e6) * args.price_output
    n = len(indices)
    cost_per_sample = cost_usd / n if n else 0
    token_per_sample = (total_in + total_out) / n if n else 0

    memory_usage = usage_containers.get("memory", [])
    research_usage = usage_containers.get("research", [])
    working_usage = usage_containers.get("working", [])
    memorize_peak = max((u.get("input_tokens", 0) for u in memory_usage), default=0)
    solution_usage = research_usage + working_usage
    solution_peak = max((u.get("input_tokens", 0) for u in solution_usage), default=0)

    def _timing_summary(name: str, times: List[float]) -> Dict[str, Any]:
        if not times:
            return {"phase": name, "count": 0, "total_sec": 0.0, "avg_sec": 0.0}
        total = sum(times)
        return {"phase": name, "count": len(times), "total_sec": round(total, 4), "avg_sec": round(total / len(times), 4)}

    timing_stats = {
        "memorize": _timing_summary("memorize", timing_containers.get("memorize", [])),
        "research": _timing_summary("research", timing_containers.get("research", [])),
        "working": _timing_summary("working", timing_containers.get("working", [])),
        "total_experiment_sec": round(experiment_total_sec, 4),
    }

    print("\n" + "=" * 60)
    print("LongMemEval-S — 时间 / Token / 花费")
    print("=" * 60)
    print(f"  Experiment Time: {experiment_total_sec:.2f} s")
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  估算费用: ${cost_usd:.4f}")
    print(f"  Cost per sample: ${cost_per_sample:.4f}, Token per sample: {token_per_sample:.0f}")
    print("  Context Window Peak:")
    print(f"    Memorize 阶段 peak:  {memorize_peak}")
    print(f"    Solution 阶段 peak: {solution_peak}")
    for k, v in timing_stats.items():
        if k == "total_experiment_sec":
            continue
        if isinstance(v, dict):
            print(f"  {v['phase']}: count={v['count']}, total={v['total_sec']} s, avg={v['avg_sec']} s")
    print("=" * 60)

    start_idx, end_idx = indices[0], indices[-1]
    stats = {
        "bench": "LongMemEval-S",
        "total_instances": n,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": round(cost_usd, 6),
        "price_input_per_1m": args.price_input,
        "price_output_per_1m": args.price_output,
        "experiment_time_sec": timing_stats["total_experiment_sec"],
        "cost_per_sample": round(cost_per_sample, 6),
        "token_per_sample": round(token_per_sample, 2),
        "timing": timing_stats,
        "context_window_peak": {
            "memorize_input_tokens": memorize_peak,
            "solution_input_tokens": solution_peak,
        },
    }
    stats_file = os.path.join(args.out, f"batch_statistics_{start_idx}_{end_idx}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计已保存: {stats_file}")

    log_experiment_run(
        outdir=args.out,
        bench="LongMemEval-S",
        hyp_path=hyp_path,
        stats_path=stats_file,
        argv=sys.argv,
        total_instances=n,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=cost_usd,
        experiment_time_sec=experiment_total_sec,
    )
    print(f"Done. Hypotheses: {hyp_path}")


if __name__ == "__main__":
    main()
