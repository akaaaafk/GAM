

from __future__ import annotations


from gam.agents import MemoryAgent, ResearchAgent


from gam.generator import (
    AbsGenerator,
    BedrockConverseGenerator,
    ClaudeGenerator,
    TinkerGenerator,
    VLLMGenerator,
)


from gam.retriever import AbsRetriever, IndexRetriever

try:
    from gam.retriever import BM25Retriever
except ImportError:
    BM25Retriever = None

try:
    from gam.retriever import DenseRetriever
except ImportError:
    DenseRetriever = None


from gam.config import (
    ClaudeGeneratorConfig,
    TinkerGeneratorConfig,
    VLLMGeneratorConfig,
    DenseRetrieverConfig,
    BM25RetrieverConfig,
    IndexRetrieverConfig,
)


from gam.schemas import (
    MemoryState,
    Page,
    MemoryUpdate,
    SearchPlan,
    Hit,
    Result,
    EnoughDecision,
    ReflectionDecision,
    ResearchOutput,
    InMemoryMemoryStore,
    InMemoryPageStore
)

__version__ = "0.1.0"
__all__ = [
    "MemoryAgent",
    "ResearchAgent",
    "AbsGenerator",
    "BedrockConverseGenerator",
    "ClaudeGenerator",
    "TinkerGenerator",
    "VLLMGenerator",
    "AbsRetriever",
    "IndexRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "ClaudeGeneratorConfig",
    "TinkerGeneratorConfig",
    "VLLMGeneratorConfig",
    "DenseRetrieverConfig",
    "BM25RetrieverConfig",
    "IndexRetrieverConfig",
    "MemoryState",
    "Page",
    "MemoryUpdate",
    "SearchPlan",
    "Hit",
    "Result",
    "EnoughDecision",
    "ReflectionDecision",
    "ResearchOutput",
    "InMemoryMemoryStore",
    "InMemoryPageStore",
]
