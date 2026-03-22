

import glob
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
    BedrockConverseGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)


AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

QWEN_MODEL_ID = os.environ.get("BEDROCK_QWEN_MODEL_ID", "qwen.qwen3-coder-30b-a3b-v1:0")


RULER_RETRI_DATASETS = {
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
}
RULER_MT_DATASETS = {"vt"}
RULER_AGG_DATASETS = {"cwe", "fwe"}


def get_ruler_metric_category(dataset_name: str) -> str:
    if dataset_name in RULER_RETRI_DATASETS:
        return "Retri"
    if dataset_name in RULER_MT_DATASETS:
        return "MT"
    if dataset_name in RULER_AGG_DATASETS:
        return "AGG"
    return "Other"


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
                except Exception:
                    continue
    return data_list


def get_dataset_system_prompt(dataset_name: str) -> str:
    system_prompts = {
        "vt": "Memorize and track the chain(s) of variable assignment hidden in the following text.",
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


def _tokenizer_decode(tokenizer, token_ids: List[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(token_ids)


def _smart_split_by_tokens(text: str, tokens: List[int], max_tokens: int, tokenizer) -> List[str]:
    if len(tokens) <= max_tokens:
        return [f"[Session 1]\n{text}"]
    chunks = []
    session_id = 0
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = _tokenizer_decode(tokenizer, chunk_tokens)
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


def compute_ruler_four_metrics(
    dataset_stats: Dict[str, Dict[str, Any]]
) -> Dict[str, Optional[float]]:
    category_correct: Dict[str, int] = {"Retri": 0, "MT": 0, "AGG": 0}
    category_total: Dict[str, int] = {"Retri": 0, "MT": 0, "AGG": 0}
    for dataset_name, st in dataset_stats.items():
        cat = get_ruler_metric_category(dataset_name)
        if cat not in category_correct:
            continue
        total = st.get("total", 0)
        correct = st.get("correct", 0)
        category_total[cat] += total
        category_correct[cat] += correct
    out: Dict[str, Optional[float]] = {}
    for cat in ("Retri", "MT", "AGG"):
        t = category_total[cat]
        if t == 0:
            out[f"{cat.lower()}_acc"] = None
        else:
            out[f"{cat.lower()}_acc"] = round(category_correct[cat] / t, 6)
    return out


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


def _make_bedrock_config(
    max_tokens: int = 512,
    use_schema: bool = False,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "region_name": AWS_REGION,
        "model_id": QWEN_MODEL_ID,
        "max_tokens": max_tokens,
        "use_schema": use_schema,
        "system_prompt": system_prompt or "",
    }


def process_sample(
    sample: Dict[str, Any],
    sample_index: int,
    outdir: str,
    dataset_name: str,
    max_tokens_chunk: int,
    embedding_model_path: Optional[str],
    use_schema: bool,
    retriever: str,
    usage_containers: Dict[str, list],
    timing_containers: Dict[str, list],
) -> Dict[str, Any]:
    sample_id = sample.get("_id", f"sample-{sample_index}")
    sample_results_dir = os.path.join(outdir, dataset_name, sample_id)
    os.makedirs(sample_results_dir, exist_ok=True)

    context_chunks = build_context_chunks_for_sample(sample, max_tokens_chunk, embedding_model_path)
    if not context_chunks:
        return {"_id": sample_id, "dataset": dataset_name, "error": "no context", "is_correct": False, "accuracy": 0.0}

    memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
    page_store = InMemoryPageStore(dir_path=sample_results_dir)

    memory_system_prompt = get_dataset_system_prompt(dataset_name)
    memory_generator = BedrockConverseGenerator(_make_bedrock_config(max_tokens=256, system_prompt=memory_system_prompt))
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

    research_generator = BedrockConverseGenerator(_make_bedrock_config(max_tokens=2048, use_schema=use_schema))
    working_generator = BedrockConverseGenerator(_make_bedrock_config(max_tokens=256))
    research_generator.usage_log = usage_containers.get("research", [])
    working_generator.usage_log = usage_containers.get("working", [])

    research_agent_kwargs: Dict[str, Any] = {
        "page_store": page_store,
        "memory_store": memory_store,
        "retrievers": retrievers,
        "generator": research_generator,
        "max_iters": 3,
    }
    if dataset_name == "niah_multivalue":
        research_agent_kwargs["max_iters"] = 5
        research_agent_kwargs["system_prompts"] = {
            "planning": "There are 4 different special magic numbers for the question item. So the keyword retrieval is need.",
            "integration": "There are 4 different special magic numbers for the question item. Don't miss any of them.",
            "reflection": "There are 4 different special magic numbers for the question item. Don't miss any of them.",
        }
    research_agent = ResearchAgent(**research_agent_kwargs)

    question = sample.get("question", "").strip()
    question_prompt = build_question_prompt(sample)
    ground_truth_outputs = sample.get("outputs", [])
    if not isinstance(ground_truth_outputs, list):
        ground_truth_outputs = [ground_truth_outputs] if ground_truth_outputs else []

    result = {"_id": sample_id, "index": sample.get("index", sample_index), "dataset": dataset_name}

    try:
        t1 = time.perf_counter()
        research_result = research_agent.research(question)
        research_summary = research_result.integrated_memory or ""
        if "research" in timing_containers:
            timing_containers["research"].append(time.perf_counter() - t1)

        research_trace = {
            "question": question,
            "raw_memory": research_result.raw_memory,
            "integrated_memory": research_result.integrated_memory,
            "iterations": research_result.raw_memory.get("iterations", []),
            "search_plans": research_result.raw_memory.get("search_plans", []),
            "reflections": research_result.raw_memory.get("reflections", []),
        }
        trace_file = os.path.join(sample_results_dir, "research_trace.json")
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(research_trace, f, ensure_ascii=False, indent=2)
        result["research_trace_file"] = trace_file
        result["research_summary"] = research_summary

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
        description="RULER 128K (AWS Bedrock Qwen Converse API): time, log, cost；直接 model_id，不用 ARN"
    )
    parser.add_argument("--data", type=str, required=True, help="RULER JSONL 文件或目录")
    parser.add_argument("--outdir", type=str, default="./results_qwen/ruler_aws", help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0, help="起始样本下标")
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本下标（不包含）")
    parser.add_argument("--max-tokens", type=int, default=2000, help="每块最大 token 数")
    parser.add_argument("--embedding-model-path", type=str, default=None, help="切分用 embedding 模型路径（可选）")
    parser.add_argument("--region", type=str, default=None, help="AWS 区域（默认 AWS_REGION 或 us-east-1）")
    parser.add_argument("--model-id", type=str, default=None, help="Bedrock model_id，如 qwen.qwen3-coder-30b-a3b-v1:0")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=("both", "bm25", "dense"),
        default="bm25",
        help="检索器",
    )
    parser.add_argument("--use-schema", action="store_true", help="Research 使用 schema")
    parser.add_argument("--price-input", type=float, default=0.0, help="输入单价（美元/百万 token）")
    parser.add_argument("--price-output", type=float, default=0.0, help="输出单价（美元/百万 token）")
    parser.add_argument(
        "--only-category",
        type=str,
        choices=("retri", "mt", "agg", "all"),
        default="all",
        help="只跑指定指标的数据集",
    )

    args = parser.parse_args()

    global AWS_REGION, QWEN_MODEL_ID
    if args.region is not None:
        AWS_REGION = args.region
    if args.model_id is not None:
        QWEN_MODEL_ID = args.model_id

    if os.path.isfile(args.data):
        base = os.path.splitext(os.path.basename(args.data))[0]
        if base in ("qa_1", "qa_2"):
            print("跳过 QA 数据集，不跑 qa_1/qa_2")
            return
        if args.only_category != "all":
            allowed = {
                "retri": RULER_RETRI_DATASETS,
                "mt": RULER_MT_DATASETS,
                "agg": RULER_AGG_DATASETS,
            }[args.only_category]
            if base not in allowed:
                print(f"跳过：--only-category {args.only_category}，当前文件 {base} 不在该类别")
                return
        jsonl_files = [args.data]
    elif os.path.isdir(args.data):
        jsonl_files = sorted(glob.glob(os.path.join(args.data, "*.jsonl")))
        jsonl_files = [
            f for f in jsonl_files
            if os.path.splitext(os.path.basename(f))[0] not in ("qa_1", "qa_2")
        ]
        if args.only_category != "all":
            allowed = {
                "retri": RULER_RETRI_DATASETS,
                "mt": RULER_MT_DATASETS,
                "agg": RULER_AGG_DATASETS,
            }[args.only_category]
            jsonl_files = [
                f for f in jsonl_files
                if os.path.splitext(os.path.basename(f))[0] in allowed
            ]
            print(f"仅跑 {args.only_category.upper()} 数据集: {[os.path.splitext(os.path.basename(p))[0] for p in jsonl_files]}")
    else:
        print(f"错误: 数据路径不存在: {os.path.abspath(args.data)}")
        return

    if not jsonl_files:
        print(f"错误: 未找到 .jsonl 文件: {args.data}")
        return

    print(f"Backend: AWS Bedrock Converse API, model_id={QWEN_MODEL_ID}, region={AWS_REGION}")
    os.makedirs(args.outdir, exist_ok=True)
    usage_containers = {"memory": [], "research": [], "working": []}
    timing_containers = {"memorize": [], "research": [], "answer": []}

    experiment_start = time.perf_counter()
    all_results: List[Dict[str, Any]] = []

    for jsonl_file in jsonl_files:
        dataset_name = os.path.splitext(os.path.basename(jsonl_file))[0]
        data = load_ruler_jsonl(jsonl_file)
        end = args.end_idx if args.end_idx is not None else len(data)
        indices = list(range(args.start_idx, min(end, len(data))))
        if not indices:
            continue
        print(f"\nRULER (128K) AWS Bedrock Qwen data: {jsonl_file}, dataset={dataset_name}, samples {indices[0]}..{indices[-1]} (total {len(indices)})")
        results = []
        for i in tqdm(indices, desc=f"RULER {dataset_name}"):
            sample = data[i]
            out = process_sample(
                sample,
                i,
                args.outdir,
                dataset_name=dataset_name,
                max_tokens_chunk=args.max_tokens,
                embedding_model_path=args.embedding_model_path,
                use_schema=args.use_schema,
                retriever=args.retriever,
                usage_containers=usage_containers,
                timing_containers=timing_containers,
            )
            results.append(out)
        all_results.extend(results)

        start_idx, end_idx = indices[0], indices[-1]
        correct_count_ds = sum(1 for r in results if r.get("is_correct", False))
        n_ds = len(results)
        acc_ds = correct_count_ds / n_ds if n_ds else 0.0
        batch_results_file = os.path.join(args.outdir, dataset_name, f"batch_results_{start_idx}_{end_idx}.json")
        os.makedirs(os.path.dirname(batch_results_file), exist_ok=True)
        with open(batch_results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        stats_ds = {
            "bench": "RULER_128K",
            "backend": "AWS Bedrock (Qwen)",
            "dataset": dataset_name,
            "metric_category": get_ruler_metric_category(dataset_name),
            "total_samples": n_ds,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "correct_count": correct_count_ds,
            "accuracy": acc_ds,
        }
        stats_file_ds = os.path.join(args.outdir, dataset_name, f"batch_statistics_{start_idx}_{end_idx}.json")
        with open(stats_file_ds, "w", encoding="utf-8") as f:
            json.dump(stats_ds, f, ensure_ascii=False, indent=2)
        print(f"  {dataset_name}: 已保存 {batch_results_file}")

    experiment_total_sec = time.perf_counter() - experiment_start
    n = len(all_results)

    total_in = sum(u.get("input_tokens", 0) for logs in usage_containers.values() for u in logs)
    total_out = sum(u.get("output_tokens", 0) for logs in usage_containers.values() for u in logs)
    cost_usd = (total_in / 1e6) * args.price_input + (total_out / 1e6) * args.price_output
    cost_per_sample = cost_usd / n if n else 0
    token_per_sample = (total_in + total_out) / n if n else 0

    memory_usage = usage_containers.get("memory", [])
    research_usage = usage_containers.get("research", [])
    working_usage = usage_containers.get("working", [])
    memorize_peak = max((u.get("input_tokens", 0) for u in memory_usage), default=0)
    solution_usage = research_usage + working_usage
    solution_peak = max((u.get("input_tokens", 0) for u in solution_usage), default=0)

    dataset_stats: Dict[str, Dict[str, Any]] = {}
    for r in all_results:
        ds = r.get("dataset", "unknown")
        if ds not in dataset_stats:
            dataset_stats[ds] = {"total": 0, "correct": 0, "wrong": 0}
        dataset_stats[ds]["total"] += 1
        if r.get("is_correct", False):
            dataset_stats[ds]["correct"] += 1
        else:
            dataset_stats[ds]["wrong"] += 1
    for ds in dataset_stats:
        t = dataset_stats[ds]["total"]
        dataset_stats[ds]["accuracy"] = dataset_stats[ds]["correct"] / t if t else 0.0

    four_metrics = compute_ruler_four_metrics(dataset_stats)
    total_correct = sum(1 for r in all_results if r.get("is_correct", False))
    overall_accuracy = total_correct / n if n else 0.0

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
    print("RULER (128K) AWS Bedrock Qwen — 时间 / Token / 花费")
    print("=" * 60)
    print(f"  Running time: {experiment_total_sec:.2f} s")
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  估算费用: ${cost_usd:.4f}")
    print(f"  Cost per sample: ${cost_per_sample:.4f}, Token per sample: {token_per_sample:.0f}")
    print(f"  Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{n})")
    for k, v in timing_stats.items():
        if k == "total_experiment_sec":
            continue
        if isinstance(v, dict):
            print(f"  {v['phase']}: count={v['count']}, total={v['total_sec']} s, avg={v['avg_sec']} s")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("RULER 三类指标 — Retri. Acc. / MT Acc. / AGG. Acc.")
    print("=" * 60)
    print(f"  Retri. Acc.: {four_metrics.get('retri_acc') if four_metrics.get('retri_acc') is not None else 'N/A'}")
    print(f"  MT Acc.:     {four_metrics.get('mt_acc') if four_metrics.get('mt_acc') is not None else 'N/A'}")
    print(f"  AGG. Acc.:   {four_metrics.get('agg_acc') if four_metrics.get('agg_acc') is not None else 'N/A'}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("各数据集准确率")
    print("=" * 60)
    for ds, st in sorted(dataset_stats.items()):
        print(f"  {ds}: {st['accuracy']:.4f} ({st['correct']}/{st['total']})")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Context Window Peak（单次调用最大 input_tokens）")
    print("=" * 60)
    print(f"  Memorize 阶段 peak:  {memorize_peak}")
    print(f"  Solution 阶段 peak: {solution_peak}")
    print("=" * 60)

    overall_summary = {
        "bench": "RULER_128K",
        "backend": "AWS Bedrock (Qwen)",
        "model_id": QWEN_MODEL_ID,
        "region": AWS_REGION,
        "total_samples": n,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "retri_acc": four_metrics.get("retri_acc"),
        "mt_acc": four_metrics.get("mt_acc"),
        "agg_acc": four_metrics.get("agg_acc"),
        "dataset_stats": dataset_stats,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": round(cost_usd, 6),
        "experiment_time_sec": timing_stats["total_experiment_sec"],
        "context_window_peak": {"memorize_input_tokens": memorize_peak, "solution_input_tokens": solution_peak},
        "timing": timing_stats,
    }
    end_idx_label = args.end_idx if args.end_idx is not None else "all"
    overall_summary_file = os.path.join(args.outdir, f"overall_summary_{args.start_idx}_{end_idx_label}.json")
    with open(overall_summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"\n总体汇总已保存: {overall_summary_file}")

    log_experiment_run(
        outdir=args.outdir,
        bench="RULER_128K-AWS-Bedrock-Qwen",
        batch_results_file=overall_summary_file,
        stats_file=overall_summary_file,
        argv=sys.argv,
        total_samples=n,
        accuracy=overall_accuracy,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=cost_usd,
        experiment_time_sec=experiment_total_sec,
    )
    print("Done.")


if __name__ == "__main__":
    main()
