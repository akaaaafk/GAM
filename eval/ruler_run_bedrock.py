

"""
RULER (128K) 跑分（AWS Bedrock Claude）+ 时间 / 日志 / 花费统计

适配 RULER 数据集（如 RULER-131072 即 128K 上下文），使用 Bedrock ClaudeGenerator。
增加：实验总耗时、各阶段耗时（memorize / research / answer）、Token 与费用、
experiment_log.jsonl 与 batch_statistics 输出。

运行示例：
  python -m eval.ruler_run_bedrock --data data/ruler/qa_1.jsonl --outdir results/ruler_128k
"""

import re
import sys
import os
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
    BM25Retriever,
    DenseRetriever,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)




def load_ruler_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    data_list = []
    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    item = json.loads(line)
                    item["_id"] = f"{dataset_name}-{idx}"
                    item["index"] = idx
                    item["dataset"] = dataset_name
                    data_list.append(item)
                except Exception as e:
                    continue
    return data_list


def get_dataset_system_prompt(dataset_name: str) -> str:
    system_prompts = {
        "vt": "Memorize and track the chain(s) of variable assignment hidden in the following text.",
        "qa_1": "",
        "qa_2": "",
        "niah_single_1": "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.",
        "niah_single_2": "A special magic number is hidden within the following text. Make sure to memorize it. I will quiz you about the number afterwards.",
        "niah_single_3": "A special magic uuid is hidden within the following text. Make sure to memorize it. I will quiz you about the uuid afterwards.",
        "niah_multivalue": "",
        "niah_multiquery": "Some special magic numbers are hidden within the following text. You only need to memorize the special magic numbers. I will quiz you about the numbers afterwards.",
        "niah_multikey_1": "",
        "niah_multikey_2": "",
        "niah_multikey_3": "",
        "cwe": "Below is a numbered list of words. You only need to memorize the numbers that all words appear rather then make a abstract. I will quiz you about the numbers afterwards. Ignore the prompt below that asks you to summarize.",
        "fwe": "Read the following coded text and track the frequency of each coded word. Memorize the numbers that the words appear, I will quiz you about the numbers afterwards.",
    }
    if dataset_name in system_prompts:
        return system_prompts[dataset_name]
    for key, prompt in system_prompts.items():
        if dataset_name.startswith(key) or key in dataset_name:
            return prompt
    return ""


def build_question_prompt(sample: Dict[str, Any]) -> str:
    parts = []
    question = sample.get("question", "").strip()
    if question:
        parts.append("Question:\n" + question)
    example = sample.get("example", "").strip()
    if example:
        parts.append("Here is the example:\n" + example)
    return "\n\n".join(parts)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate_answer(model_response: str, ground_truth_outputs: List[str]) -> bool:
    if not ground_truth_outputs or not model_response:
        return False
    model_response_lower = model_response.lower()
    model_response_normalized = normalize_text(model_response)
    unique_answers = list(set(ground_truth_outputs))
    for answer in unique_answers:
        answer_str = str(answer).strip()
        if not answer_str:
            continue
        answer_lower = answer_str.lower()
        if answer_lower in model_response_lower:
            continue
        answer_normalized = normalize_text(answer_str)
        if answer_normalized in model_response_normalized:
            continue
        answer_words = [w for w in answer_normalized.split() if len(w) > 2]
        if answer_words and all(word in model_response_normalized for word in answer_words):
            continue
        return False
    return True




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
        except Exception:
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




EXPERIMENT_LOG_FILENAME = "experiment_log.jsonl"


def log_experiment_run(
    outdir: str,
    bench: str,
    batch_results_file: str,
    stats_file: str,
    argv: List[str],
    total_samples: int,
    accuracy: float,
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
        "accuracy": accuracy,
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
    dataset_name: str,
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
    sample_results_dir = os.path.join(outdir, dataset_name, sample_id)
    os.makedirs(sample_results_dir, exist_ok=True)

    context_chunks = build_context_chunks_for_sample(sample, max_tokens, embedding_model_path)
    if not context_chunks:
        return {"_id": sample_id, "dataset": dataset_name, "error": "no context", "is_correct": False, "accuracy": 0.0}

    memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
    page_store = InMemoryPageStore(dir_path=sample_results_dir)

    memory_system_prompt = get_dataset_system_prompt(dataset_name)
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
        system_prompts={"memory": memory_system_prompt},
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
        return {"_id": sample_id, "dataset": dataset_name, "error": f"index retriever: {e}", "is_correct": False, "accuracy": 0.0}

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

    question_prompt = build_question_prompt(sample)
    ground_truth_outputs = sample.get("outputs", [])
    if not isinstance(ground_truth_outputs, list):
        ground_truth_outputs = [ground_truth_outputs] if ground_truth_outputs else []

    result = {"_id": sample_id, "index": sample.get("index", sample_index), "dataset": dataset_name}

    try:
        t1 = time.perf_counter()
        research_result = research_agent.research(question_prompt)
        research_summary = research_result.integrated_memory or ""
        if "research" in timing_containers:
            timing_containers["research"].append(time.perf_counter() - t1)

        t2 = time.perf_counter()
        prompt = f"Read the text below and answer a question. Context: {research_summary}\n\n{question_prompt}\n\nAnswer:"
        response = working_generator.generate_single(prompt=prompt)
        answer_text = (response.get("text") or "").strip()
        if "answer" in timing_containers:
            timing_containers["answer"].append(time.perf_counter() - t2)

        is_correct = evaluate_answer(answer_text, ground_truth_outputs)
        result["response"] = answer_text
        result["ground_truth_outputs"] = ground_truth_outputs
        result["is_correct"] = is_correct
        result["accuracy"] = 1.0 if is_correct else 0.0
    except Exception as e:
        result["error"] = str(e)
        result["is_correct"] = False
        result["accuracy"] = 0.0

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RULER 128K (Bedrock Claude): time, log, cost；输出 batch_results + batch_statistics + experiment_log.jsonl"
    )
    parser.add_argument("--data", type=str, required=True, help="RULER JSONL 文件（如 data/ruler/qa_1.jsonl）")
    parser.add_argument("--outdir", type=str, default="./results/ruler_128k", help="输出目录")
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

    data = load_ruler_jsonl(args.data)
    dataset_name = os.path.splitext(os.path.basename(args.data))[0]
    end = args.end_idx if args.end_idx is not None else len(data)
    indices = list(range(args.start_idx, min(end, len(data))))
    if not indices:
        print("No samples to process.")
        return

    os.makedirs(args.outdir, exist_ok=True)
    usage_containers = {"memory": [], "research": [], "working": []}
    timing_containers = {"memorize": [], "research": [], "answer": []}

    print(f"RULER (128K) data: {args.data}, dataset={dataset_name}, samples {indices[0]}..{indices[-1]} (total {len(indices)})")

    experiment_start = time.perf_counter()
    all_results = []
    for i in tqdm(indices, desc="RULER (Bedrock)"):
        sample = data[i]
        out = process_sample(
            sample,
            i,
            args.outdir,
            dataset_name=dataset_name,
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

    correct_count = sum(1 for r in all_results if r.get("is_correct", False))
    accuracy = correct_count / n if n else 0.0

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
    print("RULER (128K) — 时间 / Token / 花费")
    print("=" * 60)
    print(f"  Experiment Time: {experiment_total_sec:.2f} s")
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  估算费用: ${cost_usd:.4f}")
    print(f"  Cost per sample: ${cost_per_sample:.4f}, Token per sample: {token_per_sample:.0f}")
    print(f"  Accuracy: {accuracy:.4f} ({correct_count}/{n})")
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
    batch_results_file = os.path.join(args.outdir, dataset_name, f"batch_results_{start_idx}_{end_idx}.json")
    os.makedirs(os.path.dirname(batch_results_file), exist_ok=True)
    with open(batch_results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {batch_results_file}")

    stats = {
        "bench": "RULER_128K",
        "dataset": dataset_name,
        "total_samples": n,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "correct_count": correct_count,
        "accuracy": accuracy,
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
    stats_file = os.path.join(args.outdir, dataset_name, f"batch_statistics_{start_idx}_{end_idx}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计已保存: {stats_file}")

    log_experiment_run(
        outdir=args.outdir,
        bench="RULER_128K",
        batch_results_file=batch_results_file,
        stats_file=stats_file,
        argv=sys.argv,
        total_samples=n,
        accuracy=accuracy,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=cost_usd,
        experiment_time_sec=experiment_total_sec,
    )
    print("Done.")


if __name__ == "__main__":
    main()
