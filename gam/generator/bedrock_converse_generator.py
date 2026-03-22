

import json
import time
from typing import Any, Dict, List, Optional

import boto3

from gam.generator.base import AbsGenerator


DEFAULT_MODEL_ID = "qwen.qwen3-coder-30b-a3b-v1:0"


def _messages_to_converse(messages: List[Dict[str, str]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    system_parts: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if role == "system":
            if content:
                system_parts.append({"text": content})
            continue
        if role in ("user", "assistant"):
            out.append({"role": role, "content": [{"text": content}]})
    return system_parts, out


class BedrockConverseGenerator(AbsGenerator):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.region_name = config.get("region_name", "us-east-1")
        self.model_id = config.get("model_id", DEFAULT_MODEL_ID)
        self.max_tokens = config.get("max_tokens", 300)
        self.system_prompt = config.get("system_prompt")
        self.use_schema = config.get("use_schema", False)
        self._client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def _converse(
        self,
        system_content: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system_content:
            kwargs["system"] = system_content
        if extra_params:
            for k, v in extra_params.items():
                if k not in ("modelId", "messages", "system", "inferenceConfig"):
                    kwargs[k] = v
        resp = self._client.converse(**kwargs)
        text = ""
        if resp.get("output") and resp["output"].get("message"):
            for block in resp["output"]["message"].get("content", []):
                if "text" in block:
                    text += block["text"]
        text = text.strip()
        usage = {"input_tokens": 0, "output_tokens": 0}
        if resp.get("usage"):
            u = resp["usage"]
            usage["input_tokens"] = u.get("inputTokens", 0)
            usage["output_tokens"] = u.get("outputTokens", 0)
        return {"text": text, "usage": usage, "raw": resp}

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

        system_content, converse_messages = _messages_to_converse(messages)
        if system.strip():
            system_content.insert(0, {"text": system.strip()})

        max_tokens = self.max_tokens
        if extra_params and "max_tokens" in extra_params:
            max_tokens = extra_params.pop("max_tokens")

        times = 0
        while True:
            try:
                result = self._converse(system_content, converse_messages, max_tokens=max_tokens, extra_params=extra_params)
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
            "response": {"usage": usage, "model": self.model_id},
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

        results = []
        for msgs in messages_list:
            r = self.generate_single(messages=msgs, schema=schema, extra_params=extra_params)
            results.append(r)
        return results
