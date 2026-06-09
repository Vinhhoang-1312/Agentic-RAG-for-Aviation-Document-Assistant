"""
aviation_rag.retrieval — Semantic Retrieval Engine (Quan San)
=============================================================
Sub-package chứa core retrieval logic cho Phase 2 của pipeline.

Public API:
    - RetrievalEngine: main class, supports 4 strategies
    - build_and_save_index: build FAISS + BM25 index từ CSV
"""

from .engine import RetrievalEngine
from .indexer import build_and_save_index
