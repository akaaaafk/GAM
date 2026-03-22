from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClaudeGeneratorConfig:
    """Claude 生成器配置（仅 Inference Profile）。"""
    region_name: str = "us-east-1"
    account_id: str = "your-aws-account-id"
    inference_profile_id: str = "your-inference-profile-id"
    max_tokens: int = 300
    thread_count: int | None = None
    system_prompt: str | None = None
    use_schema: bool = False


@dataclass
class TinkerGeneratorConfig:
    """Tinker (OpenAI-compatible) 生成器配置，默认 Qwen 模型。"""
    base_url: str = "https://your-tinker-endpoint/v1"
    api_key: str = ""
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    max_tokens: int = 300
    system_prompt: str | None = None
    use_schema: bool = False


@dataclass
class VLLMGeneratorConfig:
    """vLLM 生成器配置（本地 OpenAI 兼容端点）。"""
    model_name: str = "Qwen2.5-7B-Instruct"
    api_key: Optional[str] = "empty"
    base_url: str = "http://localhost:8000/v1"
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    thread_count: Optional[int] = None
    system_prompt: Optional[str] = None
    timeout: float = 60.0
    use_schema: bool = False
