

"""
GAM 框架 + LoCoMo 数据集测试文件

结合 locomoqa_v3.py 的数据处理逻辑和 GAM 框架，测试在多轮对话数据上的效果。
"""

import sys
import os
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
    VLLMGenerator,
    VLLMGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)



def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_locomo(json_path: str) -> List[Dict[str, Any]]:
    """Load LoCoMo JSON and return the list of samples."""
    data = load_json(json_path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape. Expect a list or {'samples': [...]}.")

def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    """
    Extract sessions as (idx, timestamp, turns, optional_session_summary).
    """
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
        dia_id  = turn.get("dia_id", "")
        text    = turn.get("text", "")
        lines.append(f"{speaker} ({dia_id}): {text}")
    
    if session_summary:
        lines.append("")
        lines.append(f"Session {idx} summary: {session_summary}")
    
    return "\n".join(lines).strip()

def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    """Build session chunks from a sample."""
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    chunks: List[str] = []
    for idx, ts, turns, ssum in sessions:
        chunks.append(session_to_text(idx, ts, turns, ssum))
    return chunks

def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect QA items from a sample."""
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



def safe_json_extract(candidate: Any) -> Optional[Dict[str, Any]]:
    """尽量把模型输出（string/dict）解析成 dict，失败返回 None。"""
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None
    s = candidate.strip()
    l = s.find('{')
    r = s.rfind('}')
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(s[l:r+1])
    except Exception:
        return None

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
    """根据category选择不同的prompt"""
    if category == 3:
        prompt = make_summary_prompt_category3(summary, question)
    else:
        prompt = make_summary_prompt(summary, question)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()



def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def f1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bleu1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision

def compute_metrics_by_category(items, pred_key: str = "summary_answer", pred_field: str = "answer"):
    agg = defaultdict(list)
    rows = []
    for idx, ex in enumerate(items, 1):
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        pred = ""
        val = ex.get(pred_key, "")
        if isinstance(val, dict):
            pred = val.get(pred_field, "")
        else:
            pred = val
        f1 = f1_score(pred, gold)
        b1 = bleu1_score(pred, gold)
        agg[cat].append((f1, b1))
        rows.append({
            "q_idx": idx,
            "category": cat,
            "gold_answer": str(gold),
            "prediction": str(pred),
            "F1": f1,
            "BLEU1": b1
        })
    summary = []
    for cat in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[cat]
        if scores:
            f1_avg = sum(s[0] for s in scores)/len(scores)
            b1_avg = sum(s[1] for s in scores)/len(scores)
            summary.append({"category": cat, "count": len(scores), "F1_avg": f1_avg, "BLEU1_avg": b1_avg})
    return summary, rows



def process_sample(
    sample: Dict[str, Any], 
    sample_index: int, 
    outdir: str,
    memory_api_key: str,
    memory_base_url: str,
    memory_model: str,
    research_api_key: str,
    research_base_url: str,
    research_model: str,
    working_api_key: str,
    working_base_url: str,
    working_model: str,
    use_schema: bool = False,
    memory_api_type: str = "claude",
    research_api_type: str = "claude",
    working_api_type: str = "claude"
):
    """
    使用 GAM 框架处理单个样本。
    
    流程：
    1. 使用 MemoryAgent 构建记忆
    2. 使用 ResearchAgent 进行深度研究
    3. 基于研究结果进行问答
    """
    sample_id = sample.get("sample_id", f"conv-{sample_index}")
    
    print(f"\n{'='*60}")
    print(f"处理样本 #{sample_index}: {sample_id}")
    print(f"{'='*60}")
    
    try:

        session_chunks = build_session_chunks_for_sample(sample)
        print(f"会话数: {len(session_chunks)}")
        if session_chunks:
            print(f"第一个会话预览:\n{session_chunks[0][:400]}...")
        

        sample_results_dir = os.path.join(outdir, sample_id)
        os.makedirs(sample_results_dir, exist_ok=True)
        print(f"输出目录: {sample_results_dir}")
        

        memory_store = InMemoryMemoryStore(dir_path=sample_results_dir)
        page_store = InMemoryPageStore(dir_path=sample_results_dir)
        

        print(f"\n步骤 1: 创建 Memory Generator")
        if memory_api_type == "claude":
            memory_generator_config = ClaudeGeneratorConfig(
                region_name=os.environ.get("BEDROCK_REGION", "us-east-1"),
                account_id=os.environ.get("BEDROCK_ACCOUNT_ID", "your-aws-account-id"),
                inference_profile_id=os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", "your-inference-profile-id"),
                max_tokens=256,
            )
            memory_generator = ClaudeGenerator(memory_generator_config.__dict__)
        elif memory_api_type == "vllm":
            memory_generator_config = VLLMGeneratorConfig(
                model_name=memory_model,
                api_key=memory_api_key,
                base_url=memory_base_url,
                temperature=0.3,
                max_tokens=256
            )
            memory_generator = VLLMGenerator(memory_generator_config.__dict__)
        print(f"[OK] Memory Generator 创建完成")
        

        print(f"\n步骤 2: 使用 MemoryAgent 构建记忆")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator
        )
        
        if not os.path.exists(os.path.join(sample_results_dir, 'memory_state.json')):
            for i, session_chunk in enumerate(session_chunks, 1):
                print(f"  处理会话 {i}/{len(session_chunks)}...")
                memory_update = memory_agent.memorize(session_chunk)
        

        final_state = memory_store.load()
        print(f"[OK] 记忆构建完成！共 {len(final_state.abstracts)} 条记忆摘要")
        

        print("\n📚 记忆摘要:")
        for i, abstract in enumerate(final_state.abstracts, 1):
            print(f"  {i}. {abstract[:100]}...")
        

        memory_state_file = os.path.join(sample_results_dir, "memory_state.json")
        with open(memory_state_file, 'w', encoding='utf-8') as f:
            json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 记忆状态已保存: {memory_state_file}")
        

        print(f"\n步骤 3: 创建检索器")
        retrievers = {}
        

        try:
            page_index_dir = os.path.join(sample_results_dir, "page_index")

            if os.path.exists(page_index_dir):
                import shutil
                shutil.rmtree(page_index_dir)
                print(f"[INFO] 清理已存在的页面索引目录: {page_index_dir}")
            
            index_config = IndexRetrieverConfig(
                index_dir=page_index_dir
            )
            index_retriever = IndexRetriever(index_config.__dict__)
            index_retriever.build(page_store)
            retrievers["page_index"] = index_retriever
            print(f"[OK] 索引检索器创建成功")
        except Exception as e:
            print(f"[WARN] 索引检索器创建失败: {e}")
        

        try:
            bm25_index_dir = os.path.join(sample_results_dir, "bm25_index")

            if os.path.exists(bm25_index_dir):
                import shutil
                shutil.rmtree(bm25_index_dir)
                print(f"[INFO] 清理已存在的 BM25 索引目录: {bm25_index_dir}")
            
            bm25_config = BM25RetrieverConfig(
                index_dir=bm25_index_dir,
                threads=1
            )
            bm25_retriever = BM25Retriever(bm25_config.__dict__)
            bm25_retriever.build(page_store)
            retrievers["keyword"] = bm25_retriever
            print(f"[OK] BM25 检索器创建成功")
        except Exception as e:
            print(f"[WARN] BM25 检索器创建失败: {e}")
        

        try:
            dense_index_dir = os.path.join(sample_results_dir, "dense_index")

            if os.path.exists(dense_index_dir):
                import shutil
                shutil.rmtree(dense_index_dir)
                print(f"[INFO] 清理已存在的 Dense 索引目录: {dense_index_dir}")

            dense_config = DenseRetrieverConfig(
                index_dir=dense_index_dir,
                model_name="BAAI/bge-m3"
            )






            dense_retriever = DenseRetriever(dense_config.__dict__)
            dense_retriever.build(page_store)
            retrievers["vector"] = dense_retriever
            print(f"[OK] Dense 检索器创建成功")
        except Exception as e:
            print(f"[WARN] Dense 检索器创建失败: {e}")
        
        print(f"[INFO] 成功创建 {len(retrievers)} 个检索器")
        
        print(f"\n步骤 4: 创建 Research Generator 和 Working Generator")
        if research_api_type == "claude":
            research_generator_config = ClaudeGeneratorConfig(
                region_name=os.environ.get("BEDROCK_REGION", "us-east-1"),
                account_id=os.environ.get("BEDROCK_ACCOUNT_ID", "your-aws-account-id"),
                inference_profile_id=os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", "your-inference-profile-id"),
                max_tokens=2048,
                use_schema=use_schema,
            )
            research_generator = ClaudeGenerator(research_generator_config.__dict__)
        elif research_api_type == "vllm":
            research_generator_config = VLLMGeneratorConfig(
                model_name=research_model,
                api_key=research_api_key,
                base_url=research_base_url,
                temperature=0.3,
                max_tokens=2048,
                use_schema=use_schema
            )
            research_generator = VLLMGenerator(research_generator_config.__dict__)

        if working_api_type == "claude":
            working_generator_config = ClaudeGeneratorConfig(
                region_name=os.environ.get("BEDROCK_REGION", "us-east-1"),
                account_id=os.environ.get("BEDROCK_ACCOUNT_ID", "your-aws-account-id"),
                inference_profile_id=os.environ.get("BEDROCK_INFERENCE_PROFILE_ID", "your-inference-profile-id"),
                max_tokens=256,
            )
            working_generator = ClaudeGenerator(working_generator_config.__dict__)
        elif working_api_type == "vllm":
            working_generator_config = VLLMGeneratorConfig(
                model_name=working_model,
                api_key=working_api_key,
                base_url=working_base_url,
                temperature=0.3,
                max_tokens=256
            )
            working_generator = VLLMGenerator(working_generator_config.__dict__)
        print(f"[OK] Research Generator 和 Working Generator 创建完成")



        print(f"\n步骤 5: 创建 ResearchAgent")
        research_agent = ResearchAgent(
            page_store=page_store,
            memory_store=memory_store,
            retrievers=retrievers,
            generator=research_generator,
            max_iters=3
        )
        print(f"[OK] ResearchAgent 创建完成")
        

        print(f"\n步骤 6: 进行问答")
        qas = collect_qa_items_for_sample(sample)
        print(f"共有 {len(qas)} 个问题需要回答")
        

        def process_question(qi_with_index):
            """处理单个问题的worker函数"""
            i, qi = qi_with_index
            q = qi.get("question") or ""
            gold = qi.get("answer")
            cat = qi.get("category")
            
            print(f"\n--- 问题 {i}/{len(qas)} ---")
            print(f"问题: {q}")
            print(f"标准答案: {gold}")
            print(f"分类: {cat}")
            
            if cat == 5:
                return None

            try:

                print(f"[问题 {i}] 正在进行深度研究...")
                result = research_agent.research(q)
                research_summary = result.integrated_memory
                print(f"[问题 {i}] [OK] 研究完成！迭代次数: {len(result.raw_memory.get('iterations', []))}")
                print(f"[问题 {i}] 研究摘要: {research_summary[:200]}...")
                

                research_trace = {
                    "question": q,
                    "raw_memory": result.raw_memory,
                    "integrated_memory": result.integrated_memory,
                    "iterations": result.raw_memory.get("iterations", []),
                    "search_plans": result.raw_memory.get("search_plans", []),
                    "reflections": result.raw_memory.get("reflections", [])
                }
                

                trace_file = os.path.join(sample_results_dir, f"research_trace_q{i}.json")
                with open(trace_file, 'w', encoding='utf-8') as f:
                    json.dump(research_trace, f, ensure_ascii=False, indent=2)
                print(f"[问题 {i}] [INFO] 研究轨迹已保存: {trace_file}")
                

                print(f"[问题 {i}] 生成答案...")
                summary_answer = answer_with_summary(cat, research_summary, q, working_generator)
                
                print(f"[问题 {i}] 预测答案: {summary_answer}")
                
                qa_result = {
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "research_summary": research_summary,
                    "summary_answer": summary_answer,
                    "iterations": len(result.raw_memory.get("iterations", [])),
                    "research_trace_file": trace_file
                }
                return qa_result
            
            except Exception as e:
                print(f"[问题 {i}] [ERROR] 处理问题失败: {e}")
                import traceback
                traceback.print_exc()
                qa_result = {
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "error": str(e)
                }
                return qa_result
        

        qa_items_with_index = [(i, qi) for i, qi in enumerate(qas, 1)]
        
        print(f"开始串行处理 {len(qa_items_with_index)} 个问题...")
        
        qa_results = []
        for qa_item in tqdm(qa_items_with_index, desc="处理问题"):
            result = process_question(qa_item)

            if result is not None:
                qa_results.append(result)
        

        results_file = os.path.join(sample_results_dir, "qa_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 结果已保存到: {results_file}")
        

        all_research_traces = []
        for i, qa_result in enumerate(qa_results, 1):
            if "research_trace_file" in qa_result:
                trace_file = qa_result["research_trace_file"]
                if os.path.exists(trace_file):
                    with open(trace_file, 'r', encoding='utf-8') as f:
                        trace_data = json.load(f)
                        all_research_traces.append({
                            "question_index": i,
                            "question": qa_result["question"],
                            "category": qa_result["category"],
                            "research_trace": trace_data
                        })
        
        if all_research_traces:
            traces_summary_file = os.path.join(sample_results_dir, "all_research_traces.json")
            with open(traces_summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_research_traces, f, ensure_ascii=False, indent=2)
            print(f"[OK] 所有研究轨迹汇总已保存到: {traces_summary_file}")
        

        print(f"\n{'='*60}")
        print("处理完成统计")
        print(f"{'='*60}")
        print(f"样本ID: {sample_id}")
        print(f"会话数: {len(session_chunks)}")
        print(f"记忆摘要数: {len(final_state.abstracts)}")
        print(f"处理问题数: {len(qa_results)}")
        print(f"研究轨迹文件数: {len(all_research_traces)}")
        print(f"结果保存到: {sample_results_dir}")
        print(f"  - QA结果: qa_results.json")
        print(f"  - 记忆状态: memory_state.json")
        print(f"  - 研究轨迹汇总: all_research_traces.json")
        print(f"  - 单个研究轨迹: research_trace_q*.json")
        
        return qa_results
        
    except Exception as e:
        error_msg = f"处理样本 {sample_index} 时出错: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return []




def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GAM 框架 + LoCoMo 数据集测试")
    parser.add_argument("--data", type=str, default="/path/to/locomo/dataset.json", 
                        help="LoCoMo 数据集路径")
    parser.add_argument("--outdir", type=str, default="./results/locomo",
                        help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0, help="开始样本索引")
    parser.add_argument("--end-idx", type=int, default=None, help="结束样本索引（不包含），None表示处理所有样本")
    

    parser.add_argument("--memory-api-key", type=str, default="empty", help="Memory 模型 API Key")
    parser.add_argument("--memory-base-url", type=str, default="https://api.openai.com/v1", help="Memory 模型 Base URL")
    parser.add_argument("--memory-model", type=str, default="gpt-4o-mini", help="Memory 模型名称")
    parser.add_argument("--memory-api-type", type=str, default="claude", choices=["claude", "vllm"], help="Memory 模型 API 类型")
    

    parser.add_argument("--research-api-key", type=str, default="empty", help="Research 模型 API Key")
    parser.add_argument("--research-base-url", type=str, default="https://api.openai.com/v1", help="Research 模型 Base URL")
    parser.add_argument("--research-model", type=str, default="gpt-4o-mini", help="Research 模型名称")
    parser.add_argument("--research-api-type", type=str, default="claude", choices=["claude", "vllm"], help="Research 模型 API 类型")
    parser.add_argument("--use-schema", type=bool, default=False, help="是否使用 schema")


    parser.add_argument("--working-api-key", type=str, default="empty", help="Working 模型 API Key")
    parser.add_argument("--working-base-url", type=str, default="https://api.openai.com/v1", help="Working 模型 Base URL")
    parser.add_argument("--working-model", type=str, default="gpt-4o-mini", help="Working 模型名称")
    parser.add_argument("--working-api-type", type=str, default="claude", choices=["claude", "vllm"], help="Working 模型 API 类型")

    args = parser.parse_args()
    
    print("=" * 60)
    print("GAM 框架 + LoCoMo 数据集测试")
    print("=" * 60)
    print(f"数据集: {args.data}")
    print(f"输出目录: {args.outdir}")
    print(f"样本范围: {args.start_idx} 到 {args.end_idx-1 if args.end_idx else '全部'} (共 {args.end_idx - args.start_idx if args.end_idx else '全部'} 个样本)")
    print("=" * 60)
    

    samples = load_locomo(args.data)
    print(f"共加载 {len(samples)} 个样本")
    

    if args.end_idx is None:
        args.end_idx = len(samples)
    
    print(f"实际处理范围: {args.start_idx} 到 {args.end_idx-1} (共 {args.end_idx - args.start_idx} 个样本)")
    

    if args.start_idx < 0 or args.start_idx >= len(samples):
        print(f"错误: 开始样本索引 {args.start_idx} 超出范围 (总样本数: {len(samples)})")
        return
    
    if args.end_idx > len(samples):
        print(f"警告: 结束样本索引 {args.end_idx} 超出范围，调整为 {len(samples)}")
        args.end_idx = len(samples)
    
    if args.start_idx >= args.end_idx:
        print(f"错误: 开始索引 {args.start_idx} 必须小于结束索引 {args.end_idx}")
        return
    

    sample_indices = list(range(args.start_idx, args.end_idx))
    
    print(f"将顺序处理 {len(sample_indices)} 个样本...")
    
    all_results = []
    

    for sample_idx in tqdm(sample_indices, desc="处理样本"):
        sample = samples[sample_idx]
        print(f"\n{'='*80}")
        print(f"开始处理样本 {sample_idx}/{len(samples)-1} (范围: {args.start_idx}-{args.end_idx-1})")
        print(f"{'='*80}")
        
        try:
            results = process_sample(
                sample, 
                sample_idx, 
                args.outdir,
                args.memory_api_key,
                args.memory_base_url,
                args.memory_model,
                args.research_api_key,
                args.research_base_url,
                args.research_model,
                args.working_api_key,
                args.working_base_url,
                args.working_model,
                args.use_schema,
                args.memory_api_type,
                args.research_api_type,
                args.working_api_type
            )
            print(f"[OK] 样本 {sample_idx} 处理完成")
            all_results.extend(results)
        except Exception as e:
            print(f"[ERROR] 样本 {sample_idx} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    

    if all_results:
        summary_file = os.path.join(args.outdir, f"batch_results_{args.start_idx}_{args.end_idx-1}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 批量结果汇总已保存: {summary_file}")
        

        print(f"\n{'='*60}")
        print("开始计算指标...")
        print(f"{'='*60}")
        

        pred_key = "summary_answer"
        pred_field = "answer"
        
        print(f"\n# LoCoMo Metrics for pred_key='{pred_key}', pred_field='{pred_field}'")
        summary, details = compute_metrics_by_category(all_results, pred_key=pred_key, pred_field=pred_field)
        

        print(f"\n按类别统计:")
        for r in summary:
            print(f"Category {r['category']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}, BLEU1_avg={r['BLEU1_avg']:.4f}")
        

        all_f1_scores = [row["F1"] for row in details]
        all_bleu1_scores = [row["BLEU1"] for row in details]
        overall_f1_avg = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
        overall_bleu1_avg = sum(all_bleu1_scores) / len(all_bleu1_scores) if all_bleu1_scores else 0.0
        
        print(f"\n整体统计:")
        print(f"总问题数: {len(all_results)}")
        print(f"整体平均 F1: {overall_f1_avg:.4f}")
        print(f"整体平均 BLEU1: {overall_bleu1_avg:.4f}")
        

        statistics = {
            "total_samples": args.end_idx - args.start_idx,
            "total_questions": len(all_results),
            "overall_f1_avg": overall_f1_avg,
            "overall_bleu1_avg": overall_bleu1_avg,
            "by_category": summary,
            "details": details,
            "start_idx": args.start_idx,
            "end_idx": args.end_idx - 1
        }
        
        stats_file = os.path.join(args.outdir, f"batch_statistics_{args.start_idx}_{args.end_idx-1}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"\n指标结果已保存到: {stats_file}")
    
    print(f"\n{'='*60}")
    print("[OK] 批量测试完成！")
    print(f"处理样本数: {args.end_idx - args.start_idx}")
    print(f"成功处理: {len(all_results)} 个问题")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

