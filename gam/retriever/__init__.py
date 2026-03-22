

from __future__ import annotations

from .base import AbsRetriever
from .index_retriever import IndexRetriever


try:
    from .bm25 import BM25Retriever
except ImportError:
    BM25Retriever = None
    import warnings
    warnings.warn("BM25Retriever not available (pyserini dependencies may be missing)")

try:
    from .dense_retriever import DenseRetriever
except ImportError:
    DenseRetriever = None
    import warnings
    warnings.warn("DenseRetriever not available (FlagEmbedding dependencies may be missing)")

__all__ = [
    "AbsRetriever",
    "IndexRetriever",
]


if BM25Retriever is not None:
    __all__.append("BM25Retriever")
if DenseRetriever is not None:
    __all__.append("DenseRetriever")
