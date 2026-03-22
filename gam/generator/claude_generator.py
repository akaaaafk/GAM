
"""
Bedrock Claude Generator for GAM.

辅助函数为了处理body格式与messages格式之间的转换，并且
"""

import time
import json
from typing import Any, Dict, List, Optional

import boto3
from multiprocessing import cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from gam.generator.base import AbsGenerator
from gam.config import ClaudeGeneratorConfig

def _messages_to_body(messages: List[Dict[str, str]], system: str) -> tuple[str, List[Dict[str, str]]]:
    """转为 invoke_model：system 作为顶层参数，messages 仅含 user/assistant（Bedrock Messages API 要求）。"""
    system_parts: List[str] = []
    if system:
        system_parts.append(system)
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            system_parts.append(m.get("content", ""))
            continue
        if role in ("user", "assistant"):
            out.append({"role": role, "content": m.get("content", "")})
    system_str = "\n".join(p for p in system_parts if p).strip() if system_parts else ""
    return (system_str, out)


class ClaudeGenerator(AbsGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.region_name = config.get("region_name", "us-east-1")
        self.account_id = config.get("account_id", "your-aws-account-id")
        self.inference_profile_id = config.get("inference_profile_id", "your-inference-profile-id")
        self.max_tokens = config.get("max_tokens", 300)
        self.thread_count = config.get("thread_count")
        self.system_prompt = config.get("system_prompt")
        self.use_schema = config.get("use_schema", False)
        self._client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def _invoke(
        self,
        system_content: str,
        messages_for_body: List[Dict[str, str]],
        extra_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """单次 invoke_model（Inference Profile）。system 用顶层参数，messages 仅 user/assistant。"""
        model_id = (
            f"arn:aws:bedrock:{self.region_name}:{self.account_id}:"
            f"application-inference-profile/{self.inference_profile_id}"
        )
        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages_for_body,
            "max_tokens": self.max_tokens,
        }
        if system_content:
            body["system"] = system_content
        if extra_params:
            body.update(extra_params)
        resp = self._client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        ret = json.loads(resp["body"].read())
        text = ""
        if ret.get("content") and isinstance(ret["content"], list) and len(ret["content"]) > 0:
            text = ret["content"][0].get("text", "")
        usage = {"input_tokens": 0, "output_tokens": 0}
        if "usage" in ret:
            u = ret["usage"]
            usage["input_tokens"] = u.get("input_tokens", u.get("inputTokens", 0))
            usage["output_tokens"] = u.get("output_tokens", u.get("outputTokens", 0))
        return {"text": text, "usage": usage, "raw": ret}

    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if (prompt is None) and (not messages):
            raise ValueError("Either prompt or messages is required.")
        if (prompt is not None) and messages:
            raise ValueError("Pass either prompt or messages, not both.")

        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        system = self.system_prompt or ""
        if messages and messages[0].get("role") == "system":
            system = system or messages[0].get("content", "")
            messages = messages[1:]
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("Need at least one user message.")

        system_str, messages_for_body = _messages_to_body(messages, system)
        times = 0
        while True:
            try:
                result = self._invoke(system_str, messages_for_body, extra_params)
                break
            except Exception as e:
                print(str(e), "times:", times)
                times += 1
                if times > 3:
                    raise e
                time.sleep(5)

        text = result["text"]
        text = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
        usage = result["usage"]

        out: Dict[str, Any] = {
            "text": text,
            "json": None,
            "response": {"usage": usage, "model": f"arn:...:{self.inference_profile_id}"},
        }
        if getattr(self, "usage_log", None) is not None:
            self.usage_log.append(usage)

        if schema is not None:
            try:
                out["json"] = json.loads(text[text.find("{"): text.rfind("}") + 1])
            except Exception:
                out["json"] = None
        return out

    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if (prompts is None) and (not messages_list):
            raise ValueError("Either prompts or messages_list is required.")
        if (prompts is not None) and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            messages_list = [[{"role": "user", "content": p}] for p in prompts]

        thread_count = self.thread_count if self.thread_count is not None else cpu_count()

        def run_one(msgs):
            return self.generate_single(
                messages=msgs, schema=schema, extra_params=extra_params
            )

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(
                tqdm(
                    executor.map(run_one, messages_list),
                    total=len(messages_list),
                )
            )
        for r in results:
            u = r.get("response", {}).get("usage", {})
            if getattr(self, "usage_log", None) is not None and u:
                self.usage_log.append(u)
        return results

    @classmethod
    def from_config(cls, config: ClaudeGeneratorConfig) -> "ClaudeGenerator":
        """从配置类创建 ClaudeGenerator 实例"""
        return cls(config.__dict__)
