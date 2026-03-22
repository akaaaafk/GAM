

"""
LongMemEval-S（Tinker + Qwen）+ 时间 / 日志 / 花费统计

Follow LightMem / LongMemEval-S 数据格式，使用 Tinker API + Qwen 模型。
流程与 eval/longmemeval_s_run 一致，仅将 ClaudeGenerator 换为 TinkerGenerator。

运行示例（在项目根目录）：
  python -m eval_qwen.longmemeval_s_run --data data/longmemeval_s/longmemeval_s_cleaned.json --out results_qwen/longmemeval_s
  # 只跑前 5 条并顺带算 acc: --start-idx 0 --end-idx 5 --with-eval
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
    TinkerGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)

TINKER_BASE_URL = os.environ.get("TINKER_BASE_URL", "https://your-tinker-endpoint/v1")
TINKER_API_KEY = os.environ.get("TINKER_API_KEY", "your-tinker-api-key")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")


def load_longmemeval(json_path: str) -> List[Dict[str, Any]]:
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
    return f"""You are a careful reading assistant. Use the given context to answer the question.
Answer with ONLY the final answer; no extra explanation.

Question:
{question}

Context:
{research_summary}

Answer:
"""


def load_hypothesis_jsonl(path: str) -> List[Dict[str, str]]:
    """加载 hypothesis jsonl，每行 {"question_id": str, "hypothesis": str}。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_oracle_by_id(oracle_instances: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """由 LongMemEval 列表构建 question_id -> {question, answer, question_type?, answer_list} 的索引。"""
    by_id: Dict[str, Dict[str, Any]] = {}
    for inst in oracle_instances:
        qid = inst.get("question_id") or inst.get("id")
        if not qid:
            continue
        a = inst.get("answer", inst.get("answers"))
        if isinstance(a, list):
            answer_list = a
        elif a is None or a == "":
            answer_list = []
        else:
            answer_list = [str(a)]
        if "answers" in inst and isinstance(inst["answers"], list):
            answer_list = inst["answers"]
        by_id[str(qid)] = {
            "question": inst.get("question", ""),
            "answer": inst.get("answer", ""),
            "question_type": inst.get("question_type"),
            "answer_list": answer_list,
        }
    return by_id


JUDGE_SYSTEM = """You are a strict judge. Given a question, the reference answer(s), and the model's answer, determine if the model's answer is correct (equivalent in meaning or factually the same). Reply with only "Yes" or "No", nothing else."""


def make_judge_prompt(question: str, reference: str, model_answer: str) -> str:
    """构造裁判 prompt。"""
    return f"""Question: {question}

Reference answer: {reference}

Model's answer: {model_answer}

Is the model's answer correct? Answer only Yes or No."""


def parse_yes_no(text: str) -> bool | None:
    """从模型输出中解析 Yes/No。"""
    if not text:
        return None
    t = text.strip().upper()
    if t.startswith("YES") or t == "Y":
        return True
    if t.startswith("NO") or t == "N":
        return False
    first = t.split()[0] if t.split() else ""
    if first == "YES" or first == "Y":
        return True
    if first == "NO" or first == "N":
        return False
    return None


def run_eval_tinker(
    hypothesis_path: str,
    oracle_path: str,
    base_url: str,
    api_key: str,
    model: str,
    out_json_path: str | None = None,
) -> Dict[str, Any]:
    """用 Tinker + Qwen 对 hypothesis jsonl 逐条判对错，汇总准确率。"""
    oracle_list = load_longmemeval(oracle_path)
    oracle_by_id = build_oracle_by_id(oracle_list)
    hypotheses = load_hypothesis_jsonl(hypothesis_path)

    judge_config = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "max_tokens": 64,
        "system_prompt": JUDGE_SYSTEM,
    }
    generator = TinkerGenerator(judge_config)

    results: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    by_type: Dict[str, List[bool]] = {}

    for row in tqdm(hypotheses, desc="LongMemEval-S Eval (Tinker)"):
        qid = row.get("question_id", "")
        hypothesis = (row.get("hypothesis") or "").strip()
        ref = oracle_by_id.get(qid)
        if ref is None:
            results.append({"question_id": qid, "correct": None, "error": "missing in oracle"})
            continue
        question = ref.get("question", "")
        answer_list = ref.get("answer_list") or [ref.get("answer", "")]
        if not answer_list:
            results.append({"question_id": qid, "correct": None, "error": "no reference answer"})
            continue
        reference = answer_list[0] if len(answer_list) == 1 else " | ".join(str(a) for a in answer_list)
        prompt = make_judge_prompt(question, reference, hypothesis)
        try:
            response = generator.generate_single(prompt=prompt)
            text = (response.get("text") or "").strip()
        except Exception as e:
            results.append({"question_id": qid, "correct": None, "error": str(e)})
            continue
        is_yes = parse_yes_no(text)
        if is_yes is None:
            is_yes = False
            results.append({"question_id": qid, "correct": False, "judge_raw": text[:200]})
        else:
            results.append({"question_id": qid, "correct": is_yes})
        total += 1
        if is_yes:
            correct += 1
        qtype = ref.get("question_type")
        if qtype is not None:
            qtype = str(qtype)
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(is_yes)

    overall_acc = (correct / total) if total else 0.0
    type_acc = {k: (sum(v) / len(v) if v else 0.0) for k, v in by_type.items()}
    summary = {
        "accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "by_question_type": type_acc,
        "results": results,
    }
    if out_json_path:
        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


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


def _make_tinker_config(max_tokens: int = 512, use_schema: bool = False) -> Dict[str, Any]:
    return {
        "base_url": TINKER_BASE_URL,
        "api_key": TINKER_API_KEY,
        "model": QWEN_MODEL,
        "max_tokens": max_tokens,
        "use_schema": use_schema,
    }


def process_one_instance(
    instance: Dict[str, Any],
    index: int,
    outdir: str,
    use_schema: bool = False,
    retriever: str = "bm25",
    max_iters: int = 3,
    usage_containers: Optional[Dict[str, list]] = None,
    timing_containers: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    question_id = instance.get("question_id", f"q_{index}")
    question = instance.get("question", "")
    haystack_sessions = instance.get("haystack_sessions", [])

    result = {"question_id": question_id, "hypothesis": ""}
    timing = timing_containers or {}
    instance_dir = os.path.join(outdir, "instances", question_id.replace("/", "_").replace("\\", "_"))
    os.makedirs(instance_dir, exist_ok=True)

    memory_store = InMemoryMemoryStore(dir_path=instance_dir)
    page_store = InMemoryPageStore(dir_path=instance_dir)

    memory_generator = TinkerGenerator(_make_tinker_config(max_tokens=512))
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

    research_generator = TinkerGenerator(_make_tinker_config(max_tokens=2048, use_schema=use_schema))
    working_generator = TinkerGenerator(_make_tinker_config(max_tokens=256))
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
        description="LongMemEval-S (Tinker+Qwen): time, log, cost；输出 gam_hypothesis.jsonl + batch_statistics + experiment_log.jsonl"
    )
    parser.add_argument("--data", type=str, required=True, help="LongMemEval-S JSON（如 data/longmemeval_s/longmemeval_s_cleaned.json）")
    parser.add_argument("--out", type=str, default="./results_qwen/longmemeval_s", help="输出目录")
    parser.add_argument("--hypothesis", type=str, default=None, help="输出 jsonl 路径，默认 <out>/gam_hypothesis.jsonl")
    parser.add_argument("--start-idx", type=int, default=0, help="起始样本下标")
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本下标（不包含）")
    parser.add_argument("--max-iters", type=int, default=3, help="ResearchAgent 最大迭代次数")
    parser.add_argument("--api-key", type=str, default=None, help="Tinker API Key（可选，不传则用本文件或环境变量 TINKER_API_KEY）")
    parser.add_argument("--base-url", type=str, default=None, help="Tinker base URL")
    parser.add_argument("--model", type=str, default=None, help="模型名（默认 Qwen/Qwen3-30B-A3B-Instruct-2507）")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=("both", "bm25", "dense"),
        default="bm25",
        help="检索器",
    )
    parser.add_argument("--use-schema", action="store_true", help="Research 使用 schema")
    parser.add_argument("--price-input", type=float, default=0.0, help="输入单价（美元/百万 token，用于估算）")
    parser.add_argument("--price-output", type=float, default=0.0, help="输出单价（美元/百万 token，用于估算）")
    parser.add_argument("--with-eval", action="store_true", help="跑完后用 Tinker+Qwen 裁判算准确率（oracle 用 --data）")

    args = parser.parse_args()

    global TINKER_BASE_URL, TINKER_API_KEY, QWEN_MODEL
    if args.api_key is not None:
        TINKER_API_KEY = args.api_key.strip() if args.api_key else ""
    if args.base_url is not None:
        TINKER_BASE_URL = args.base_url
    if args.model is not None:
        QWEN_MODEL = args.model
    if not TINKER_API_KEY or not TINKER_API_KEY.strip():
        print("错误: 未设置 Tinker API Key。请在本文件第 45 行填写，或设置环境变量 TINKER_API_KEY，或使用 --api-key")
        sys.exit(1)

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

    print(f"LongMemEval-S (Tinker+Qwen) data: {args.data}, instances {indices[0]}..{end-1} (total {len(indices)})")
    print(f"Output: {hyp_path}")

    usage_containers = {"memory": [], "research": [], "working": []}
    timing_containers = {"memorize": [], "research": [], "working": []}

    experiment_start = time.perf_counter()
    with open(hyp_path, "w", encoding="utf-8") as f:
        for i in tqdm(indices, desc="LongMemEval-S (Qwen)"):
            instance = data[i]
            out = process_one_instance(
                instance,
                i,
                args.out,
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
    print("LongMemEval-S (Qwen) — 时间 / Token / 花费")
    print("=" * 60)
    print(f"  Running time: {experiment_total_sec:.2f} s")
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print(f"  估算费用: ${cost_usd:.4f}")
    print(f"  Cost per sample: ${cost_per_sample:.4f}, Token per sample: {token_per_sample:.0f}")
    for k, v in timing_stats.items():
        if k == "total_experiment_sec":
            continue
        if isinstance(v, dict):
            print(f"  {v['phase']}: count={v['count']}, total={v['total_sec']} s, avg={v['avg_sec']} s")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Context Window Peak（单次调用最大 input_tokens）")
    print("=" * 60)
    print(f"  Memorize 阶段 peak:  {memorize_peak}")
    print(f"  Solution 阶段 peak: {solution_peak}")
    print("=" * 60)

    start_idx, end_idx = indices[0], indices[-1]
    stats = {
        "bench": "LongMemEval-S",
        "backend": "Tinker+Qwen",
        "total_instances": n,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": round(cost_usd, 6),
        "price_input_per_1m": args.price_input,
        "price_output_per_1m": args.price_output,
        "experiment_time_sec": timing_stats["total_experiment_sec"],
        "running_time_sec": timing_stats["total_experiment_sec"],
        "cost_per_sample": round(cost_per_sample, 6),
        "token_per_sample": round(token_per_sample, 2),
        "context_window_peak": {
            "memorize_input_tokens": memorize_peak,
            "solution_input_tokens": solution_peak,
        },
        "timing": timing_stats,
    }
    stats_file = os.path.join(args.out, f"batch_statistics_{start_idx}_{end_idx}.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计已保存: {stats_file}")

    log_experiment_run(
        outdir=args.out,
        bench="LongMemEval-S-Qwen",
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

    if getattr(args, "with_eval", False):
        eval_out = os.path.join(args.out, f"eval_results_{start_idx}_{end_idx}.json")
        print("\n" + "=" * 60)
        print("LongMemEval-S 评测 (Tinker 裁判)")
        print("=" * 60)
        summary = run_eval_tinker(
            hypothesis_path=hyp_path,
            oracle_path=args.data,
            base_url=TINKER_BASE_URL,
            api_key=TINKER_API_KEY,
            model=QWEN_MODEL,
            out_json_path=eval_out,
        )
        print(f"Accuracy: {summary['accuracy']:.2%}  ({summary['correct']}/{summary['total']})")
        if summary.get("by_question_type"):
            for k, v in sorted(summary["by_question_type"].items()):
                print(f"  {k}: {v:.2%}")
        print(f"详细结果: {eval_out}")
        print("=" * 60)


if __name__ == "__main__":
    main()
