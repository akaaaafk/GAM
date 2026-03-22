
"""
Tinker (OpenAI-compatible) Generator for GAM eval.
Backbone: Tinker API; Model: Qwen/Qwen3-30B-A3B-Instruct-2507.
"""

import json
import time
from typing import Any, Dict, List, Optional

from gam.generator.base import AbsGenerator


try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")


TINKER_BASE_URL = "https://your-tinker-endpoint/v1"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class TinkerGenerator(AbsGenerator):
    """Generator that calls Tinker API (OpenAI-compatible) with Qwen model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", TINKER_BASE_URL)
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", DEFAULT_MODEL)
        self.max_tokens = config.get("max_tokens", 300)
        self.system_prompt = config.get("system_prompt")
        self.use_schema = config.get("use_schema", False)
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _chat(
        self,
        messages: List[Dict[str, str]],
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Single chat completion; returns {text, usage}."""
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if extra_params:
            body.update(extra_params)
        resp = self._client.chat.completions.create(**body)
        text = ""
        if resp.choices and len(resp.choices) > 0:
            text = (resp.choices[0].message.content or "").strip()
        usage = {"input_tokens": 0, "output_tokens": 0}
        if getattr(resp, "usage", None):
            usage["input_tokens"] = getattr(resp.usage, "input_tokens", 0) or getattr(resp.usage, "prompt_tokens", 0)
            usage["output_tokens"] = getattr(resp.usage, "output_tokens", 0) or getattr(resp.usage, "completion_tokens", 0)
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


        api_messages: List[Dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        times = 0
        max_retries = 8
        while True:
            try:
                result = self._chat(api_messages, extra_params)
                break
            except Exception as e:
                is_429 = "429" in str(e) or "Too many" in str(e)
                print(str(e), "times:", times)
                times += 1
                if times > max_retries:
                    raise e

                wait = (2 ** times) if is_429 else 5
                time.sleep(wait)

        text = result["text"]
        text = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
        usage = result["usage"]

        out: Dict[str, Any] = {
            "text": text,
            "json": None,
            "response": {"usage": usage, "model": self.model},
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
