

"""
HotpotQA 跑分（AWS Bedrock Claude）+ 时间 / 日志 / 花费统计

基于 hotpotqa_test 数据与流程，使用 Bedrock ClaudeGenerator。
增加：实验总耗时、各阶段耗时（memorize / research / answer）、Token 与费用、
experiment_log.jsonl 与 batch_statistics 输出。

运行示例：
  python -m eval.hotpotqa_run_bedrock --data data/hotpotqa/eval_400.json --outdir results/hotpotqa
"""

import string
import sys
import os
import re
import json
import shutil
import time
from datetime import datetime
from typing import Any, Counter, Dict, List, Optional

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
    BM25Retriever,
    DenseRetriever,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)




def load_hotpotqa(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return [
        {
            "index": item.get("index", idx),
            "context": item.get("context", ""),
            "input": item.get("input", ""),
            "answers": item.get("answers", []),
            "_id": f"hotpotqa-{item.get('index', idx)}",
        }
        for idx, item in enumerate(dataset)
    ]


def _smart_split_by_tokens(text: str, tokens: List[int], max_tokens: int, tokenizer) -> List[str]:
    if len(tokens) <= max_tokens:
        return [f"[Session 1]\n{text}"]
    chunks = []
    session_id = 0
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(f"[Session {session_id}]\n{chunk_text.strip()}")
            session_id += 1
        start_idx = end_idx
    return chunks


def _fallback_char_split(text: str, max_tokens: int) -> List[str]:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return [f"[Session 1]\n{text}"]
    chunks = []
    current_start = 0
    session_id = 0
    while current_start < len(text):
        current_end = min(current_start + max_chars, len(text))
        if current_end < len(text):
            last_newline = text.rfind("\n", current_start, current_end)
            if last_newline > current_start:
                current_end = last_newline
            else:
                last_space = text.rfind(" ", current_start, current_end)
                if last_space > current_start:
                    current_end = last_space
        chunk_text = text[current_start:current_end].strip()
        if chunk_text:
            chunks.append(f"[Session {session_id}]\n{chunk_text}")
            session_id += 1
        current_start = current_end
    return chunks


def build_context_chunks_for_sample(
    sample: Dict[str, Any],
    max_tokens: int = 2000,
    embedding_model_path: Optional[str] = None,
) -> List[str]:
    context_text = sample.get("context") or ""
    if not context_text:
        return []
    if embedding_model_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
            tokens = tokenizer.encode(context_text, add_special_tokens=False)
            if len(tokens) <= max_tokens:
                return [f"[Session 1]\n{context_text}"]
            return _smart_split_by_tokens(context_text, tokens, max_tokens, tokenizer)
        except Exception as e:
            pass
    try:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = tokenizer.encode(context_text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return [f"[Session 1]\n{context_text}"]
        return _smart_split_by_tokens(context_text, tokens, max_tokens, tokenizer)
    except ImportError:
        return _fallback_char_split(context_text, max_tokens)


def make_prompt(summary: str, question: str) -> str:
    return f"""You are a careful multi-hop reading assistant.
Use the given Context.
Answer with ONLY the final answer string; no extra words.

Question:
{question}

Context:
{summary}

Answer:
"""


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pt = normalize_answer(prediction).split()
    gt = normalize_answer(ground_truth).split()
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return (2 * precision * recall) / (precision + recall)


def _calculate_f1(pred_answer: str, gold_answers: List[str]) -> float:
    return max((qa_f1_score(pred_answer, g) for g in gold_answers), default=0.0)




EXPERIMENT_LOG_FILENAME = "experiment_log.jsonl"


def log_experiment_run(
    outdir: str,
    bench: str,
    batch_results_file: str,
    stats_file: str,
    argv: List[str],
    total_samples: int,
    avg_f1: float,
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
        "batch_results_file": os.path.abspath(batch_results_file),
        "stats_file": os.path.abspath(stats_file),
        "total_samples": total_samples,
        "avg_f1": avg_f1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "experiment_time_sec": experiment_time_sec,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"实验记录已追加: {log_path}")




def process_sample(
    sample: Dict[str, Any],
    sample_index: int,
    outdir: str,
    region_name: str,
    claude_account_id: str,
    claude_inference_profile_id: str,
    max_tokens: int,
    embedding_model_path: Optional[str],
    use_schema: bool,
    retriever: str,
    usage_containers: Dict[str, list],
    timing_containers: Dict[str, list],
) -> Dict[str, Any]:
    sample_id = sample.get("_id", f"sample-{sample_index}")
    sample_results_dir = os.path.join(outdir, sample_id)
    os.makedirs(sample_results_dir, exist_ok=True)

    context_chunks = build_context_chunks_for_sample(sample, max_tokens, embedding_model_path)
    if not context_chunks:
        return {"_id": sample_id, "error": "no context", "f1": 0.0}

    memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
    page_store = InMemoryPageStore(dir_path=sample_results_dir)

    memory_cfg = ClaudeGeneratorConfig(
        region_name=region_name,
        account_id=claude_account_id,
        inference_profile_id=claude_inference_profile_id,
        max_tokens=256,
    )
    memory_generator = ClaudeGenerator(memory_cfg.__dict__)
    memory_generator.usage_log = usage_containers.get("memory", [])

    memory_agent = MemoryAgent(
        memory_store=memory_store,
        page_store=page_store,
        generator=memory_generator,
    )

    t0 = time.perf_counter()
    if not os.path.exists(os.path.join(sample_results_dir, "memory_state.json")):
        for chunk in context_chunks:
            memory_agent.memorize(chunk)
    if "memorize" in timing_containers:
        timing_containers["memorize"].append(time.perf_counter() - t0)

    final_state = memory_store.load()
    with open(os.path.join(sample_results_dir, "memory_state.json"), "w", encoding="utf-8") as f:
        json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)

    retrievers = {}
    try:
        page_index_dir = os.path.join(sample_results_dir, "page_index")
        if os.path.exists(page_index_dir):
            shutil.rmtree(page_index_dir)
        index_retriever = IndexRetriever(IndexRetrieverConfig(index_dir=page_index_dir).__dict__)
        index_retriever.build(page_store)
        retrievers["page_index"] = index_retriever
    except Exception as e:
        return {"_id": sample_id, "error": f"index retriever: {e}", "f1": 0.0}

    if retriever in ("both", "bm25") and BM25Retriever is not None:
        try:
            bm25_dir = os.path.join(sample_results_dir, "bm25_index")
            if os.path.exists(bm25_dir):
                shutil.rmtree(bm25_dir)
            bm25_retriever = BM25Retriever(BM25RetrieverConfig(index_dir=bm25_dir, threads=1).__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
        except Exception:
            pass
    if retriever in ("both", "dense") and DenseRetriever is not None:
        try:
            dense_dir = os.path.join(sample_results_dir, "dense_index")
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
    research_generator.usage_log = usage_containers.get("research", [])
    working_generator.usage_log = usage_containers.get("working", [])

    research_agent = ResearchAgent(
        page_store=page_store,
        memory_store=memory_store,
        retrievers=retrievers,
        generator=research_generator,
        max_iters=3,
    )

    question = sample.get("input", "")
    gold_answers = sample.get("answers", [])
    result = {"_id": sample_id, "index": sample.get("index", sample_index), "question": question, "answers": gold_answers}

    try:
        t1 = time.perf_counter()
        research_result = research_agent.research(question)
        research_summary = research_result.integrated_memory or ""
        if "research" in timing_containers:
            timing_containers["research"].append(time.perf_counter() - t1)

        t2 = time.perf_counter()
        prompt = make_prompt(research_summary, question)
        response = working_generator.generate_single(prompt=prompt)
        answer_text = (response.get("text") or "").strip()
        if "answer" in timing_containers:
            timing_containers["answer"].append(time.perf_counter() - t2)

        f1 = _calculate_f1(answer_text, gold_answers) if answer_text else 0.0
        result["pred"] = answer_text
        result["f1"] = f1
    except Exception as e:
        result["error"] = str(e)
        result["f1"] = 0.0

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="HotpotQA (Bedrock Claude): time, log, cost；输出 batch_results + batch_statistics + experiment_log.jsonl"
    )
    parser.add_argument("--data", type=str, required=True, help="HotpotQA JSON（如 data/hotpotqa/eval_400.json）")
    parser.add_argument("--outdir", type=str, default="./results/hotpotqa", help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0, help="起始样本下标")
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本下标（不包含）")
    parser.add_argument("--max-tokens", type=int, default=2000, help="每块最大 token 数")
    parser.add_argument("--embedding-model-path", type=str, default=None, help="切分用 embedding 模型路径（可选）")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS 区域")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=("both", "bm25", "dense"),
        default="bm25",
        help="检索器",
    )
    parser.add_argument("--use-schema", action="store_true", help="Research 使用 schema")
    parser.add_argument("--price-input", type=float, default=3.0, help="输入单价（美元/百万 token）")
    parser.add_argument("--price-output", type=float, default=15.0, help="输出单价（美元/百万 token）")

    args = parser.parse_args()

    region_name = args.region
    claude_account_id = "your-aws-account-id"
    claude_inference_profile_id = "your-inference-profile-id"

    if not os.path.isfile(args.data):
        print(f"错误: 数据文件不存在: {os.path.abspath(args.data)}")
        return

    data = load_hotpotqa(args.data)
    end = args.end_idx if args.end_idx is not None else len(data)
    indices = list(range(args.start_idx, min(end, len(data))))
    if not indices:
        print("No samples to process.")
        return

    os.makedirs(args.outdir, exist_ok=True)
    usage_containers = {"memory": [], "research": [], "working": []}
    timing_containers = {"memorize": [], "research": [], "answer": []}

    print(f"HotpotQA data: {args.data}, samples {indices[0]}..{indices[-1]} (total {len(indices)})")

    experiment_start = time.perf_counter()
    all_results = []
    for i in tqdm(indices, desc="HotpotQA (Bedrock)"):
        sample = data[i]
        out = process_sample(
            sample,
            i,
            args.outdir,
            region_name=region_name,
            claude_account_id=claude_account_id,
            claude_inference_profile_id=claude_inference_profile_id,
            max_tokens=args.max_tokens,
            embedding_model_path=args.embedding_model_path,
            use_schema=args.use_schema,
            retriever=args.retriever,
            usage_containers=usage_containers,
            timing_containers=timing_containers,
        )
        all_results.append(out)

    experiment_total_sec = time.perf_counter() - experiment_start

    total_in = sum(u.get("input_tokens", 0) for logs in usage_containers.values() for u in logs)
    total_out = sum(u.get("output_tokens", 0) for logs in usage_containers.values() for u in logs)
    cost_usd = (total_in / 1e6) * args.price_input + (total_out / 1e6) * args.price_output
    n = len(all_results)
    cost_per_sample = cost_usd / n if n else 0
    token_per_sample = (total_in + total_out) / n if n else 0

    memory_usage = usage_containers.get("memory", [])
    research_usage = usage_containers.get("research", [])
    working_usage = usage_containers.get("working", [])
    memorize_peak = max((u.get("input_tokens", 0) for u in memory_usage), default=0)
    solution_usage = research_usage + working_usage
    solution_peak = max((u.get("input_tokens", 0) for u in solution_usage), default=0)

    f1_scores = [r.get("f1", 0.0) for r in all_results]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    def _timing_summary(name: str, times: List[float]) -> Dict[str, Any]:
        if not times:
            return {"phase": name, "count": 0, "total_sec": 0.0, "avg_sec": 0.0}
        total = sum(times)
        return {"phase": name, "count": len(times), "total_sec": round(total, 4), "avg_sec": round(total / len(times), 4)}

    timing_stats = {
        "memorize": _timing_summary("memorize", timing_containers.get("memorize", [])),
        "research": _timing_summary("research", timing_containers.get("research", [])),
        "answer": _timing_summary("answer", timing_containers.get("answer", [])),
        "total_experiment_sec": round(experiment_total_sec, 4),
    }

    print("\n" + "=" * 60)
    print("HotpotQA — 时间 / Token / 花费")
    print("=" * 60)
    print(f"  Experiment Time: {experiment_total_sec:.2f} s")
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  估算费用: ${cost_usd:.4f}")
    print(f"  Cost per sample: ${cost_per_sample:.4f}, Token per sample: {token_per_sample:.0f}")
    print(f"  平均 F1: {avg_f1:.4f}")
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
    batch_results_file = os.path.join(args.outdir, f"batch_results_{start_idx}_{end_idx}.json")
    with open(batch_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {batch_results_file}")

    stats = {
        "bench": "HotpotQA",
        "total_samples": n,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "avg_f1": avg_f1,
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
    stats_file = os.path.join(args.outdir, f"batch_statistics_{start_idx}_{end_idx}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计已保存: {stats_file}")

    log_experiment_run(
        outdir=args.outdir,
        bench="HotpotQA",
        batch_results_file=batch_results_file,
        stats_file=stats_file,
        argv=sys.argv,
        total_samples=n,
        avg_f1=avg_f1,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=cost_usd,
        experiment_time_sec=experiment_total_sec,
    )
    print("Done.")


if __name__ == "__main__":
    main()
