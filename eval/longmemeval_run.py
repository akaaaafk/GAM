

"""
GAM 在 LongMemEval 上跑分（AWS Bedrock Inference / Claude）

LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (ICLR 2025)
https://github.com/xiaowu0162/LongMemEval

使用 general-agentic-memory-claude 的 ClaudeGenerator（Bedrock Inference Profile）。
配置：--region 指定区域，account_id 与 inference_profile_id 使用脚本内默认值。

数据格式：每个 instance 含 question_id, question, answer, haystack_sessions（按时间戳排序的会话列表），
每个 session 为 [{"role": "user"|"assistant", "content": "..."}, ...]。

流程：
1. 将 haystack_sessions 按会话格式化为文本，用 MemoryAgent（Claude）逐条 memorize
2. 用 ResearchAgent（Claude）对 question 做 research，得到 integrated_memory
3. 用 Working Generator（Claude）根据 research 结果生成最终 hypothesis
4. 输出 jsonl：每行 {"question_id": str, "hypothesis": str}，供 LongMemEval 官方 evaluate_qa.py 评估

评估（需先 clone LongMemEval 并安装依赖）：
  export OPENAI_API_KEY=...
  cd /path/to/LongMemEval/src/evaluation
  python evaluate_qa.py gpt-4o /path/to/gam_hypothesis.jsonl /path/to/longmemeval_oracle.json
"""

import os
import sys
import json
import shutil
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
    """
    加载 LongMemEval JSON。
    支持：list of instances，或 {"data": [...]} 等单 key 包装。
    """
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
    """
    将 LongMemEval 的 haystack_sessions 转为 GAM MemoryAgent 可 memorize 的文本列表。
    每个 session 转为一段带 [Session i] 的对话文本，按顺序 memorize 以保留时间顺序。
    """
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
) -> Dict[str, Any]:
    """
    对单个 LongMemEval instance 跑 GAM（Bedrock Claude）：建记忆 -> Research -> 生成 hypothesis。
    返回 {"question_id": str, "hypothesis": str}，出错时 hypothesis 可为空并带 "error" 键。
    """
    question_id = instance.get("question_id", f"q_{index}")
    question = instance.get("question", "")
    haystack_sessions = instance.get("haystack_sessions", [])

    result = {"question_id": question_id, "hypothesis": ""}

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

    for msg in memory_messages:
        try:
            memory_agent.memorize(msg)
        except Exception as e:
            result["error"] = f"memorize: {e}"
            return result


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

    try:
        research_result = research_agent.research(question)
        research_summary = research_result.integrated_memory or ""
    except Exception as e:
        result["error"] = f"research: {e}"
        return result

    qa_prompt = make_qa_prompt(research_summary, question)
    try:
        response = working_generator.generate_single(prompt=qa_prompt)
        hypothesis = (response.get("text") or "").strip()
    except Exception as e:
        result["error"] = f"working: {e}"
        result["hypothesis"] = research_summary[:500]
        return result

    result["hypothesis"] = hypothesis
    return result

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GAM on LongMemEval (AWS Bedrock Claude): build memory from haystack_sessions, research question, output hypothesis jsonl."
    )
    parser.add_argument("--data", type=str, required=True, help="LongMemEval JSON（如 longmemeval_oracle.json / longmemeval_s_cleaned.json）")
    parser.add_argument("--out", type=str, default="./results/longmemeval", help="输出目录")
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
        help="检索器: bm25=仅 BM25, dense=仅 Dense, both=两者",
    )
    parser.add_argument("--use-schema", action="store_true", help="Research 是否使用 schema")

    args = parser.parse_args()

    region_name = args.region
    claude_account_id = "your-aws-account-id"
    claude_inference_profile_id = "your-inference-profile-id"
    print(f"[Bedrock] Inference Profile: region={region_name}, account_id={claude_account_id}, profile_id={claude_inference_profile_id}")

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

    print(f"LongMemEval data: {args.data}, instances {args.start_idx}..{end-1} (total {len(indices)})")
    print(f"Output: {hyp_path}, retriever={args.retriever}")

    usage_containers = {"memory": [], "research": [], "working": []}

    with open(hyp_path, "w", encoding="utf-8") as f:
        for i in tqdm(indices, desc="GAM LongMemEval (Bedrock)"):
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
            )
            f.write(json.dumps({"question_id": out["question_id"], "hypothesis": out.get("hypothesis", "")}, ensure_ascii=False) + "\n")
            if out.get("error"):
                tqdm.write(f"Warning [{out['question_id']}]: {out['error']}")

    total_in = sum(u.get("input_tokens", 0) for logs in usage_containers.values() for u in logs)
    total_out = sum(u.get("output_tokens", 0) for logs in usage_containers.values() for u in logs)
    memory_usage = usage_containers.get("memory", [])
    research_usage = usage_containers.get("research", [])
    working_usage = usage_containers.get("working", [])
    memorize_peak = max((u.get("input_tokens", 0) for u in memory_usage), default=0)
    solution_usage = research_usage + working_usage
    solution_peak = max((u.get("input_tokens", 0) for u in solution_usage), default=0)

    print("\n" + "=" * 60)
    print("LongMemEval — Token / Context Window Peak")
    print("=" * 60)
    print(f"  input_tokens:  {total_in}")
    print(f"  output_tokens: {total_out}")
    print("  Context Window Peak（单次调用最大 input_tokens）:")
    print(f"    Memorize 阶段 peak:  {memorize_peak}")
    print(f"    Solution 阶段 peak: {solution_peak}")
    print("=" * 60)
    print(f"Done. Hypotheses written to {hyp_path}")
    print("To evaluate with LongMemEval official script:")
    print(f"  cd /path/to/LongMemEval/src/evaluation")
    print(f"  python evaluate_qa.py gpt-4o {os.path.abspath(hyp_path)} <path/to/longmemeval_oracle.json>")


if __name__ == "__main__":
    main()
