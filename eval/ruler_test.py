

"""
GAM 框架 + RULER 数据集测试文件

适配 RULER 数据集（jsonl，每行包含以下字段）：
- context: str - 长文本上下文（需要记忆的内容）
- example: str - 示例（可选）
- instruction: str - 指令（可选）
- question: str - 问题
- outputs: List[str] - 标准答案列表

测试流程：
1. 使用 MemoryAgent 记忆 context 部分
2. 使用 ResearchAgent 进行研究
3. 将 example 和 question 合在一起提问
4. 计算准确率（Acc）
"""

import sys
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from collections import Counter
import string
import glob


from gam import (
    MemoryAgent,
    ResearchAgent,
    ClaudeGenerator,
    ClaudeGeneratorConfig,
    VLLMGenerator,
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    VLLMGeneratorConfig,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
)



def get_dataset_system_prompt(dataset_name: str) -> str:
    """
    根据数据集名称返回对应的 system_prompt
    
    Args:
        dataset_name: 数据集名称（如 'vt', 'qa_1', 'niah_single_1' 等）
    
    Returns:
        system_prompt 字符串
    """

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
    

    base_name = dataset_name.split('_')[0] if '_' in dataset_name else dataset_name
    

    if dataset_name in system_prompts:
        return system_prompts[dataset_name]
    

    for key, prompt in system_prompts.items():
        if dataset_name.startswith(key) or key in dataset_name:
            return prompt
    

    return ""



def load_ruler_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    加载 RULER JSONL 数据集
    
    Args:
        jsonl_path: 数据集 JSONL 文件路径
    
    Returns:
        数据列表
    """
    data_list = []
    dataset_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    item = json.loads(line)
                    item['_id'] = f"{dataset_name}-{idx}"
                    item['index'] = idx
                    item['dataset'] = dataset_name
                    data_list.append(item)
                except Exception as e:
                    print(f"Warning: Failed to parse line {idx} in {jsonl_path}: {e}")
                    continue
    
    return data_list



def build_context_chunks_for_sample(
    sample: Dict[str, Any], 
    max_tokens: int = 2000, 
    embedding_model_path: Optional[str] = None
) -> List[str]:
    """
    将 context 文本按 token 数量分割成多个会话块
    使用智能切分：优先在边界处切分
    
    Args:
        sample: 样本数据，包含 'context' 字段
        max_tokens: 每个会话块的最大 token 数量
        embedding_model_path: embedding 模型路径，如果提供则使用该模型进行精确 token 计算
    """
    context_text = sample.get("context") or ""
    
    if not context_text:
        return []
    

    if embedding_model_path:
        try:
            chunks = _split_with_embedding_model(context_text, max_tokens, embedding_model_path)
            if chunks:
                return chunks
        except Exception as e:
            print(f"Warning: Embedding model splitting failed: {e}, falling back to tiktoken")
    

    try:
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
        tokens = tokenizer.encode(context_text, disallowed_special=())
        
        if len(tokens) <= max_tokens:
            return [f"[Session 1]\n{context_text}"]
        

        chunks = _smart_split_by_tokens(context_text, tokens, max_tokens, tokenizer)
        return chunks
        
    except ImportError:
        print("Warning: tiktoken not available, falling back to character-based splitting")
        return _fallback_char_split(context_text, max_tokens)

def _split_with_embedding_model(text: str, max_tokens: int, model_path: str) -> List[str]:
    """
    使用 embedding 模型进行精确的 token 切分
    """
    try:
        from transformers import AutoTokenizer
        

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        

        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_tokens:
            return [f"[Session 1]\n{text}"]
        

        chunks = _smart_split_by_tokens(text, tokens, max_tokens, tokenizer)
        return chunks
        
    except Exception as e:
        print(f"Error using embedding model: {e}")
        return []

def _smart_split_by_tokens(text: str, tokens: List[int], max_tokens: int, tokenizer) -> List[str]:
    """
    按 token 数量简单切分：不进行智能边界查找，直接按 max_tokens 切分
    """
    chunks = []
    

    if len(tokens) <= max_tokens:
        return [f"[Session 1]\n{text}"]
    

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
    """
    字符切分的 fallback 方法
    """

    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return [f"[Session 1]\n{text}"]
    
    chunks = []
    current_start = 0
    session_id = 0
    
    while current_start < len(text):
        current_end = min(current_start + max_chars, len(text))
        

        if current_end < len(text):

            last_newline = text.rfind('\n', current_start, current_end)
            if last_newline > current_start:
                current_end = last_newline
            else:

                last_space = text.rfind(' ', current_start, current_end)
                if last_space > current_start:
                    current_end = last_space
        
        chunk_text = text[current_start:current_end].strip()
        if chunk_text:
            chunks.append(f"[Session {session_id}]\n{chunk_text}")
            session_id += 1
        
        current_start = current_end
    
    return chunks



def build_question_prompt(sample: Dict[str, Any]) -> str:
    """
    构建问题 prompt：将 example 和 question 合在一起
    
    Args:
        sample: 样本数据，包含 'example' 和 'question' 字段
    
    Returns:
        完整的问题 prompt
    """
    parts = []


    question = sample.get("question", "").strip()
    question_prompt = "Question:\n" + question
    if question:
        parts.append(question_prompt)



    example = sample.get("example", "").strip()
    if example:
        example_prompt = "Here is the example:\n" + example
        parts.append(example_prompt)
    

    prompt = "\n\n".join(parts)
    
    return prompt



def normalize_text(text: str) -> str:
    """
    标准化文本：去除标点符号，转小写，标准化空格
    """

    text = text.lower()

    text = re.sub(r'[^\w\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def evaluate_answer(model_response: str, ground_truth_outputs: List[str]) -> bool:
    """
    评估模型回答是否正确
    
    规则：如果模型的回答中包含 ground_truth_outputs 列表中的所有元素，则认为回答正确
    采用多种策略：
    1. 精确匹配：直接在模型回答中查找标准答案（转小写）
    2. 灵活匹配：去除标点符号后匹配
    3. 关键词匹配：对于多词答案，检查所有关键词是否都存在
    
    Args:
        model_response: 模型的回答
        ground_truth_outputs: 标准答案列表
    
    Returns:
        是否正确 (True/False)
    """
    if not ground_truth_outputs:
        return False
    
    if not model_response:
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
        if answer_words:

            if all(word in model_response_normalized for word in answer_words):
                continue
        

        return False
    

    return True



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
    max_tokens: int = 2048,
    embedding_model_path: Optional[str] = None,
    use_schema: bool = False,
    memory_api_type: str = "claude",
    research_api_type: str = "claude",
    working_api_type: str = "claude"
):
    """
    使用 GAM 框架处理单个样本。
    
    流程：
    1. 使用 MemoryAgent 构建记忆（记忆 context 部分）
    2. 使用 ResearchAgent 进行深度研究
    3. 基于研究结果进行问答（example + question）
    """
    sample_id = sample.get("_id", f"sample-{sample_index}")
    dataset_name = sample.get("dataset", "unknown")
    
    print(f"\n{'='*60}")
    print(f"处理样本 #{sample_index}: {sample_id} (数据集: {dataset_name})")
    print(f"{'='*60}")
    
    try:

        context_chunks = build_context_chunks_for_sample(sample, max_tokens, embedding_model_path)
        print(f"上下文块数: {len(context_chunks)}")
        if context_chunks:
            print(f"第一个上下文块预览:\n{context_chunks[0][:400]}...")
        

        sample_results_dir = os.path.join(outdir, dataset_name, sample_id)
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
                system_prompt=get_dataset_system_prompt(dataset_name),
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
        

        memory_system_prompt = get_dataset_system_prompt(dataset_name)
        print(f"\n数据集 System Prompt: {memory_system_prompt[:100]}...")
        

        print(f"\n步骤 2: 使用 MemoryAgent 构建记忆")
        memory_agent = MemoryAgent(
            memory_store=memory_store,
            page_store=page_store,
            generator=memory_generator,
            system_prompts={"memory": memory_system_prompt}
        )
        
        if not os.path.exists(os.path.join(sample_results_dir, 'memory_state.json')):
            for i, context_chunk in enumerate(context_chunks, 1):
                print(f"  处理上下文块 {i}/{len(context_chunks)}...")
                memory_update = memory_agent.memorize(context_chunk)
        else:
            print(f"  记忆已存在，跳过构建")
        

        final_state = memory_store.load()
        print(f"[OK] 记忆构建完成！共 {len(final_state.abstracts)} 条记忆摘要")
        

        print("\n📚 记忆摘要:")
        for i, abstract in enumerate(final_state.abstracts, 1):
            print(f"  {i}. {abstract[:100]}...")
        

        memory_state_file = os.path.join(sample_results_dir, "memory_state.json")
        with open(memory_state_file, 'w', encoding='utf-8') as f:
            json.dump(final_state.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[OK] 记忆状态已保存: {memory_state_file}")
        

        print(f"\n步骤 3: 创建检索器（用于 ResearchAgent）")
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
        

        system_prompts = None
        if dataset_name == "niah_multivalue":
            system_prompts = {
                "planning": "There are 4 different special magic numbers for the question item. So the keyword retrieval is need.",
                "integration": "There are 4 different special magic numbers for the question item. Don't miss any of them.",
                "reflection": "There are 4 different special magic numbers for the question item. Don't miss any of them."
            }
            print(f"[INFO] 为数据集 {dataset_name} 设置了自定义 system_prompts")
        

        research_agent_kwargs = {
            "page_store": page_store,
            "memory_store": memory_store,
            "retrievers": retrievers,
            "generator": research_generator,
            "max_iters": 5
        }
        

        if system_prompts is not None:
            research_agent_kwargs["system_prompts"] = system_prompts
        
        research_agent = ResearchAgent(**research_agent_kwargs)
        print(f"[OK] ResearchAgent 创建完成")
        

        print(f"\n步骤 6: 进行问答")
        

        question = sample.get("question", "").strip()
        ground_truth_outputs = sample.get("outputs", [])
        

        question_prompt = build_question_prompt(sample)
        
        print(f"问题: {question[:200]}...")
        print(f"标准答案: {ground_truth_outputs}")
        

        result = {
            "_id": sample.get("_id", sample_id),
            "sample_id": sample_id,
            "index": sample.get("index", sample_index),
            "dataset": dataset_name,
            "example": sample.get("example", ""),
            "instruction": sample.get("instruction", ""),
            "question": question,
            "question_prompt": question_prompt,
            "ground_truth_outputs": ground_truth_outputs,
        }
        
        try:

            print("正在进行深度研究...")
            research_result = research_agent.research(question)
            research_summary = research_result.integrated_memory
            print(f"[OK] 研究完成！迭代次数: {len(research_result.raw_memory.get('iterations', []))}")
            print(f"研究摘要: {research_summary[:200]}...")
            

            research_trace = {
                "question": question,
                "raw_memory": research_result.raw_memory,
                "integrated_memory": research_result.integrated_memory,
                "iterations": research_result.raw_memory.get("iterations", []),
                "search_plans": research_result.raw_memory.get("search_plans", []),
                "reflections": research_result.raw_memory.get("reflections", [])
            }
            
            trace_file = os.path.join(sample_results_dir, "research_trace.json")
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(research_trace, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 研究轨迹已保存: {trace_file}")
            
            result["research_summary"] = research_summary
            result["research_trace_file"] = trace_file
            

            print("生成答案...")
            prompt = f"""Read the text below and answer a question. Context: {research_summary}\n\n{question_prompt}\n\nAnswer:"""
            response = working_generator.generate_single(prompt=prompt)
            answer_text = response.get("text", "").strip()
            
            print(f"模型响应: {answer_text[:200]}...")
            
            result["response"] = answer_text
            

            is_correct = evaluate_answer(answer_text, ground_truth_outputs)
            result["is_correct"] = is_correct
            result["accuracy"] = 1.0 if is_correct else 0.0
            
            print(f"预测答案: {answer_text[:200]}...")
            print(f"标准答案: {ground_truth_outputs}")
            print(f"评估结果: {'✓ 正确' if is_correct else '✗ 错误'}")
            
        except Exception as e:
            print(f"[ERROR] 处理问题失败: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            result["is_correct"] = False
            result["accuracy"] = 0.0
        

        results_file = os.path.join(sample_results_dir, "qa_result.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 结果已保存到: {results_file}")
        

        print(f"\n{'='*60}")
        print("处理完成统计")
        print(f"{'='*60}")
        print(f"样本ID: {sample_id}")
        print(f"数据集: {dataset_name}")
        print(f"上下文块数: {len(context_chunks)}")
        if final_state:
            print(f"记忆摘要数: {len(final_state.abstracts)}")
        print(f"预测答案: {result.get('response', 'N/A')[:200]}...")
        print(f"标准答案: {ground_truth_outputs}")
        print(f"准确率: {result.get('accuracy', 0.0):.4f}")
        print(f"结果保存到: {sample_results_dir}")
        
        return result
        
    except Exception as e:
        error_msg = f"处理样本 {sample_index} 时出错: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "sample_id": sample.get("_id", f"sample-{sample_index}"),
            "dataset": dataset_name,
            "error": error_msg,
            "is_correct": False,
            "accuracy": 0.0
        }



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GAM 框架 + RULER 数据集测试")
    parser.add_argument("--data", type=str, 
                        default="/path/to/ruler/data",
                        help="RULER 数据集 JSONL 文件路径或目录路径")
    parser.add_argument("--outdir", type=str, 
                        default="./results/ruler",
                        help="输出目录")
    parser.add_argument("--start-idx", type=int, default=0, 
                        help="开始样本索引")
    parser.add_argument("--end-idx", type=int, default=None, 
                        help="结束样本索引（不包含），None表示处理所有样本")
    parser.add_argument("--max-tokens", type=int, default=2048, 
                        help="每个上下文块的最大 token 数量")
    parser.add_argument("--embedding-model-path", type=str, 
                        default=None, 
                        help="Embedding 模型路径，用于精确 token 计算（可选）")
    

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
    print("GAM 框架 + RULER 数据集测试")
    print("=" * 60)
    print(f"数据: {args.data}")
    print(f"输出目录: {args.outdir}")
    print(f"样本范围: {args.start_idx} 到 {args.end_idx-1 if args.end_idx else '全部'}")
    print(f"最大 token 数: {args.max_tokens}")
    print("=" * 60)
    

    jsonl_files = []
    if os.path.isfile(args.data):

        jsonl_files = [args.data]
    elif os.path.isdir(args.data):

        jsonl_files = sorted(glob.glob(os.path.join(args.data, "*.jsonl")))
    else:
        print(f"错误: 路径不存在: {args.data}")
        return
    
    if not jsonl_files:
        print(f"错误: 在 {args.data} 中没有找到 .jsonl 文件")
        return
    
    print(f"\n找到 {len(jsonl_files)} 个数据文件:")
    for f in jsonl_files:
        print(f"  - {f}")
    

    all_results = []
    
    for jsonl_file in jsonl_files:
        dataset_name = os.path.splitext(os.path.basename(jsonl_file))[0]
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*80}")
        

        all_samples = load_ruler_jsonl(jsonl_file)
        print(f"共加载 {len(all_samples)} 个样本")
        

        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(all_samples)
        end_idx = min(end_idx, len(all_samples))
        
        if start_idx >= end_idx:
            print(f"警告: 样本范围无效，跳过数据集 {dataset_name}")
            continue
        
        print(f"处理样本范围: {start_idx} 到 {end_idx-1} (共 {end_idx - start_idx} 个样本)")
        

        sample_indices = list(range(start_idx, end_idx))
        
        print(f"\n开始串行处理...")
        results = []
        for idx in tqdm(sample_indices, desc=f"处理 {dataset_name}"):
            sample = all_samples[idx]
            try:
                result = process_sample(
                    sample,
                    idx,
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
                    max_tokens=args.max_tokens,
                    embedding_model_path=args.embedding_model_path,
                    use_schema=args.use_schema
                )
                results.append(result)
            except Exception as e:
                print(f"[ERROR] 样本 {idx} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "_id": sample.get("_id", f"sample-{idx}"),
                    "index": idx,
                    "dataset": dataset_name,
                    "error": str(e),
                    "is_correct": False,
                    "accuracy": 0.0
                })
        
        all_results.extend(results)
        

        correct_count = sum(1 for r in results if r.get("is_correct", False))
        total_count = len(results)
        dataset_accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        print(f"\n{'='*60}")
        print(f"{dataset_name} 数据集统计")
        print(f"{'='*60}")
        print(f"总样本数: {total_count}")
        print(f"正确数: {correct_count}")
        print(f"错误数: {total_count - correct_count}")
        print(f"准确率: {dataset_accuracy:.4f} ({dataset_accuracy*100:.2f}%)")
        print(f"{'='*60}")
        

        dataset_summary = {
            "dataset": dataset_name,
            "total_samples": total_count,
            "correct_count": correct_count,
            "wrong_count": total_count - correct_count,
            "accuracy": dataset_accuracy,
            "results": results
        }
        
        dataset_summary_file = os.path.join(
            args.outdir, 
            dataset_name, 
            f"summary_{start_idx}_{end_idx-1}.json"
        )
        os.makedirs(os.path.dirname(dataset_summary_file), exist_ok=True)
        with open(dataset_summary_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] {dataset_name} 结果汇总已保存: {dataset_summary_file}")
    

    if all_results:

        total_correct = sum(1 for r in all_results if r.get("is_correct", False))
        total_samples = len(all_results)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        

        dataset_stats = {}
        for result in all_results:
            dataset = result.get("dataset", "unknown")
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {
                    "total": 0,
                    "correct": 0,
                    "wrong": 0
                }
            dataset_stats[dataset]["total"] += 1
            if result.get("is_correct", False):
                dataset_stats[dataset]["correct"] += 1
            else:
                dataset_stats[dataset]["wrong"] += 1
        

        for dataset in dataset_stats:
            total = dataset_stats[dataset]["total"]
            correct = dataset_stats[dataset]["correct"]
            dataset_stats[dataset]["accuracy"] = correct / total if total > 0 else 0.0
        

        overall_summary = {
            "total_samples": total_samples,
            "total_correct": total_correct,
            "total_wrong": total_samples - total_correct,
            "overall_accuracy": overall_accuracy,
            "dataset_stats": dataset_stats,
            "results": all_results
        }
        
        overall_summary_file = os.path.join(
            args.outdir, 
            f"overall_summary_{args.start_idx}_{args.end_idx if args.end_idx else 'all'}.json"
        )
        with open(overall_summary_file, 'w', encoding='utf-8') as f:
            json.dump(overall_summary, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 总体结果汇总已保存: {overall_summary_file}")
        

        print(f"\n{'='*60}")
        print("测试完成统计")
        print(f"{'='*60}")
        print(f"处理数据集数量: {len(jsonl_files)}")
        print(f"处理样本总数: {total_samples}")
        print(f"正确数: {total_correct}")
        print(f"错误数: {total_samples - total_correct}")
        print(f"总体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"\n各数据集准确率:")
        for dataset, stats in sorted(dataset_stats.items()):
            print(f"  {dataset}: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%) - {stats['correct']}/{stats['total']}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

