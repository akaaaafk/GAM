

"""
LongMemEvals 评测：用 Bedrock Claude（Inference Profile）当裁判，对 hypothesis jsonl 与 oracle 判对错并算分。

用法:
  python -m eval.longmemevals_eval_bedrock --hypothesis results/longmemevals/gam_hypothesis.jsonl --oracle data/longmemevals_oracle.json
  python -m eval.longmemevals_eval_bedrock --hypothesis results/longmemevals/gam_hypothesis.jsonl --oracle data/longmemevals_oracle.json --out results/longmemevals/eval_results.json
"""

import os
import sys
import json
from typing import Any, Dict, List

from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gam import ClaudeGenerator, ClaudeGeneratorConfig


def load_longmemevals(json_path: str) -> List[Dict[str, Any]]:
    """加载 LongMemEvals JSON（与 longmemevals_run 相同逻辑）。"""
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
    raise ValueError(f"Unsupported LongMemEvals format in {json_path}")


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
    """由 LongMemEvals 列表构建 question_id -> {question, answer, question_type?} 的索引。"""
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
    """从模型输出中解析 Yes/No，返回 True/False，无法解析时返回 None。"""
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


def run_eval(
    hypothesis_path: str,
    oracle_path: str,
    region_name: str,
    account_id: str,
    inference_profile_id: str,
    out_json_path: str | None = None,
) -> Dict[str, Any]:
    """
    用 Bedrock Claude 对 hypothesis jsonl 逐条判对错，汇总准确率。
    返回包含 overall accuracy、按 question_type 的准确率及逐条结果的字典。
    """
    oracle_list = load_longmemevals(oracle_path)
    oracle_by_id = build_oracle_by_id(oracle_list)
    hypotheses = load_hypothesis_jsonl(hypothesis_path)

    cfg = ClaudeGeneratorConfig(
        region_name=region_name,
        account_id=account_id,
        inference_profile_id=inference_profile_id,
        max_tokens=64,
        system_prompt=JUDGE_SYSTEM,
    )
    generator = ClaudeGenerator(cfg.__dict__)

    results: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    by_type: Dict[str, List[bool]] = {}

    for row in tqdm(hypotheses, desc="LongMemEvals Eval (Bedrock)"):
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LongMemEvals 评测：用 Bedrock Claude 当裁判，对 hypothesis 与 oracle 判对错并算分。"
    )
    parser.add_argument("--hypothesis", type=str, required=True, help="hypothesis jsonl 路径（如 gam_hypothesis.jsonl）")
    parser.add_argument("--oracle", type=str, required=True, help="LongMemEvals oracle JSON 路径（如 data/longmemevals_oracle.json）")
    parser.add_argument("--out", type=str, default=None, help="汇总结果输出 JSON 路径（可选）")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS 区域")
    parser.add_argument("--account-id", type=str, default="your-aws-account-id", help="Bedrock 账号 ID（Inference Profile）")
    parser.add_argument("--inference-profile-id", type=str, default="your-inference-profile-id", help="Bedrock Inference Profile ID")

    args = parser.parse_args()

    region_name = args.region
    account_id = args.account_id
    inference_profile_id = args.inference_profile_id
    print(f"[Bedrock] Inference Profile: region={region_name}, account_id={account_id}, profile_id={inference_profile_id}")

    if not os.path.isfile(args.hypothesis):
        print(f"错误: hypothesis 文件不存在: {os.path.abspath(args.hypothesis)}")
        sys.exit(1)
    if not os.path.isfile(args.oracle):
        print(f"错误: oracle 文件不存在: {os.path.abspath(args.oracle)}")
        sys.exit(1)

    summary = run_eval(
        hypothesis_path=args.hypothesis,
        oracle_path=args.oracle,
        region_name=region_name,
        account_id=account_id,
        inference_profile_id=inference_profile_id,
        out_json_path=args.out,
    )

    print("\n========== LongMemEvals 评测结果 (Bedrock Claude 裁判) ==========")
    print(f"Accuracy: {summary['accuracy']:.2%}  ({summary['correct']}/{summary['total']})")
    if summary.get("by_question_type"):
        print("By question_type:")
        for k, v in sorted(summary["by_question_type"].items()):
            print(f"  {k}: {v:.2%}")
    if args.out:
        print(f"\n详细结果已写入: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
