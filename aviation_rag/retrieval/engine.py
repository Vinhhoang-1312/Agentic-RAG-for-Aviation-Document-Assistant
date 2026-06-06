"""
engine.py — Semantic Retrieval Engine (4 Strategies)
=====================================================
Core retrieval engine supporting:
    - semantic      : FAISS cosine similarity search
    - bm25          : BM25Okapi keyword-based search
    - hybrid        : Reciprocal Rank Fusion (semantic + BM25)
    - metadata_first: Metadata filtering → semantic search on subset

Designed to plug into the Agentic RAG pipeline as Phase 2.

Input:  InputAgentOutput (from Phase 1 — Hoang)
Output: MiddleAgentOutput (for Phase 3 — Hoang)

Author: Quan San — Phase 2 Semantic Retrieval Research
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from ..config import Settings
from ..schemas import InputAgentOutput, MiddleAgentOutput, RetrievedDoc
from .indexer import index_exists, load_index


# ==============================================================================
# Retrieval Engine
# ==============================================================================

class RetrievalEngine:
    """
    Semantic Retrieval Engine for Aviation Document Search.

    Supports 4 strategies matching Phase 1's routing policies:
        - semantic:       Dense vector search via FAISS
        - bm25:           Sparse keyword search via BM25Okapi
        - hybrid:         Reciprocal Rank Fusion combining both
        - metadata_first: Filter by metadata then semantic search

    Usage:
        engine = RetrievalEngine(settings)
        result = engine.retrieve(phase1_output)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._faiss_index: Optional[faiss.Index] = None
        self._chunks: Optional[List[str]] = None
        self._metadata: Optional[List[Dict[str, Any]]] = None
        self._bm25_corpus: Optional[List[List[str]]] = None
        self._bm25 = None
        self._model = None
        self._loaded = False

    # ──────────────────────────────────────────────────────────────────
    # Lazy Loading
    # ──────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load all resources on first use."""
        if self._loaded:
            return

        if not index_exists(self.settings.index_dir):
            raise FileNotFoundError(
                f"Index not found at {self.settings.index_dir}. "
                f"Run: python scripts/build_phase2_san_index.py"
            )

        self._faiss_index, self._chunks, self._metadata, self._bm25_corpus = (
            load_index(self.settings.index_dir)
        )

        # Load embedding model
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.settings.embedding_model_name)

        # Build BM25 index from corpus
        if self._bm25_corpus:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._bm25_corpus)
            print(f"[engine] [OK] BM25 index built: {len(self._bm25_corpus):,} documents")

        self._loaded = True
        print(f"[engine] [OK] RetrievalEngine ready: {self._faiss_index.ntotal:,} vectors")

    @property
    def is_available(self) -> bool:
        """Check if index exists (without loading)."""
        return index_exists(self.settings.index_dir)

    # ──────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────────────────────────

    def retrieve(self, input_row: InputAgentOutput) -> MiddleAgentOutput:
        """
        Main retrieval entry point.

        Receives InputAgentOutput from Phase 1, routes to the appropriate
        strategy, and returns MiddleAgentOutput for Phase 3.

        Args:
            input_row: Phase 1 output with query, intent, and retrieval plan.

        Returns:
            MiddleAgentOutput with retrieved documents and diagnostics.
        """
        self._ensure_loaded()

        start = time.time()

        strategy = input_row.retrieval_plan.strategy
        query = input_row.rewritten_query or input_row.query_normalized or input_row.query_raw
        top_k = input_row.retrieval_plan.top_k
        filters = input_row.retrieval_plan.filters

        # Route to strategy
        strategy_map = {
            "semantic": self._search_semantic,
            "bm25": self._search_bm25,
            "hybrid": self._search_hybrid,
            "metadata_first": self._search_metadata_first,
        }

        search_fn = strategy_map.get(strategy, self._search_hybrid)
        try:
            docs = search_fn(query=query, top_k=top_k, filters=filters)
        except Exception as e:
            # Fallback strategy
            fallback = input_row.retrieval_plan.fallback_strategy
            fallback_fn = strategy_map.get(fallback, self._search_semantic)
            docs = fallback_fn(query=query, top_k=top_k, filters=filters)
            strategy = f"{strategy}->{fallback}(fallback)"

        elapsed_ms = (time.time() - start) * 1000

        return MiddleAgentOutput(
            query_id=input_row.query_id,
            predicted_intent=input_row.intent,
            topk_docs=docs,
            retrieval_diagnostics={
                "adapter_mode": "real_retrieval",
                "contract_owner": "Quan San",
                "strategy_used": strategy,
                "strategy_requested": input_row.retrieval_plan.strategy,
                "fallback_strategy": input_row.retrieval_plan.fallback_strategy,
                "routing_reason": input_row.retrieval_plan.routing_reason,
                "top_k": top_k,
                "results_returned": len(docs),
                "search_time_ms": round(elapsed_ms, 1),
                "index_size": self._faiss_index.ntotal,
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # Strategy: Semantic (FAISS)
    # ──────────────────────────────────────────────────────────────────

    def _search_semantic(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None,
    ) -> List[RetrievedDoc]:
        """Dense semantic search via FAISS cosine similarity."""
        top_k = min(top_k, self._faiss_index.ntotal)

        # Encode query
        query_vec = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)

        # FAISS search
        scores, indices = self._faiss_index.search(query_vec, top_k)

        return self._build_docs(
            indices=indices[0],
            semantic_scores=scores[0],
            bm25_scores=None,
            strategy="semantic",
        )

    # ──────────────────────────────────────────────────────────────────
    # Strategy: BM25
    # ──────────────────────────────────────────────────────────────────

    def _search_bm25(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None,
    ) -> List[RetrievedDoc]:
        """Sparse keyword search via BM25Okapi."""
        if self._bm25 is None:
            # Fallback to semantic if BM25 not available
            return self._search_semantic(query, top_k, filters)

        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        top_bm25_scores = bm25_scores[top_indices]

        # Normalize BM25 scores to [0, 1]
        max_score = top_bm25_scores[0] if top_bm25_scores[0] > 0 else 1.0
        normalized_scores = top_bm25_scores / max_score

        return self._build_docs(
            indices=top_indices,
            semantic_scores=None,
            bm25_scores=normalized_scores,
            strategy="bm25",
        )

    # ──────────────────────────────────────────────────────────────────
    # Strategy: Hybrid (RRF)
    # ──────────────────────────────────────────────────────────────────

    def _search_hybrid(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None,
    ) -> List[RetrievedDoc]:
        """
        Hybrid search: Reciprocal Rank Fusion of semantic + BM25.
        RRF score = 1/(k + rank_semantic) + 1/(k + rank_bm25), k=60
        """
        if self._bm25 is None:
            return self._search_semantic(query, top_k, filters)

        expand_k = min(top_k * 3, self._faiss_index.ntotal)

        # 1. Semantic scores
        query_vec = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        sem_scores, sem_indices = self._faiss_index.search(query_vec, expand_k)

        # 2. BM25 scores
        tokenized_query = query.lower().split()
        bm25_all_scores = self._bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_all_scores)[::-1][:expand_k]

        # 3. RRF fusion (k=60 is standard)
        rrf_k = 60
        rrf_scores: Dict[int, float] = {}

        # Semantic ranks
        for rank, idx in enumerate(sem_indices[0]):
            if idx == -1:
                continue
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)

        # BM25 ranks
        for rank, idx in enumerate(bm25_top_indices):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (rrf_k + rank + 1)

        # Sort by RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build semantic + BM25 score lookup
        sem_score_map = {int(idx): float(s) for idx, s in zip(sem_indices[0], sem_scores[0]) if idx != -1}
        bm25_max = bm25_all_scores.max() if bm25_all_scores.max() > 0 else 1.0

        docs = []
        for idx, rrf_score in sorted_items:
            sem_s = sem_score_map.get(idx, 0.0)
            bm25_s = float(bm25_all_scores[idx]) / bm25_max if bm25_max > 0 else 0.0
            hybrid_s = (
                self.settings.semantic_weight * sem_s
                + self.settings.bm25_weight * bm25_s
            )

            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            event_id = meta.get("event_id", f"chunk_{idx}")

            docs.append(RetrievedDoc(
                doc_id=f"asrs_{event_id}",
                chunk_id=f"asrs_{event_id}#{meta.get('chunk_index', 0)}",
                chunk_text=self._chunks[idx],
                scores={
                    "semantic": round(sem_s, 4),
                    "bm25": round(bm25_s, 4),
                    "hybrid": round(hybrid_s, 4),
                    "rrf": round(rrf_score, 6),
                    "final": round(hybrid_s, 4),
                },
                metadata=self._enrich_metadata(meta),
            ))

        return docs

    # ──────────────────────────────────────────────────────────────────
    # Strategy: Metadata-First
    # ──────────────────────────────────────────────────────────────────

    def _search_metadata_first(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None,
    ) -> List[RetrievedDoc]:
        """
        Filter by metadata first, then semantic search on matching subset.
        Falls back to full semantic search if filter yields too few results.
        """
        if not filters:
            return self._search_semantic(query, top_k, filters)

        # Filter indices by metadata
        matching_indices = []
        for idx, meta in enumerate(self._metadata):
            match = True
            for key, value in filters.items():
                if key in ("prefer_metadata", "answer_style", "document_type"):
                    continue  # Skip non-metadata fields from routing policy
                meta_val = str(meta.get(key, "")).lower()
                if value and str(value).lower() not in meta_val:
                    match = False
                    break
            if match:
                matching_indices.append(idx)

        # Fallback if too few matches
        if len(matching_indices) < top_k:
            return self._search_semantic(query, top_k, filters)

        # Semantic search on filtered subset
        query_vec = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)

        # Compute scores for matching indices only
        subset_embeddings = np.zeros(
            (len(matching_indices), self.settings.embedding_dimension), dtype=np.float32,
        )
        for i, idx in enumerate(matching_indices):
            # Reconstruct vector from FAISS
            subset_embeddings[i] = faiss.rev_swig_ptr(
                self._faiss_index.reconstruct(idx), self.settings.embedding_dimension,
            )

        scores = np.dot(subset_embeddings, query_vec.T).flatten()
        top_subset_indices = np.argsort(scores)[::-1][:top_k]

        docs = []
        for rank_idx in top_subset_indices:
            real_idx = matching_indices[rank_idx]
            score = float(scores[rank_idx])

            meta = self._metadata[real_idx] if real_idx < len(self._metadata) else {}
            event_id = meta.get("event_id", f"chunk_{real_idx}")

            docs.append(RetrievedDoc(
                doc_id=f"asrs_{event_id}",
                chunk_id=f"asrs_{event_id}#{meta.get('chunk_index', 0)}",
                chunk_text=self._chunks[real_idx],
                scores={
                    "semantic": round(score, 4),
                    "bm25": 0.0,
                    "final": round(score, 4),
                },
                metadata=self._enrich_metadata(meta, applied_filters=filters),
            ))

        return docs

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _build_docs(
        self,
        indices: np.ndarray,
        semantic_scores: Optional[np.ndarray],
        bm25_scores: Optional[np.ndarray],
        strategy: str,
    ) -> List[RetrievedDoc]:
        """Build RetrievedDoc list from search results."""
        docs = []
        for i, idx in enumerate(indices):
            if idx == -1:
                continue

            sem_s = float(semantic_scores[i]) if semantic_scores is not None else 0.0
            bm25_s = float(bm25_scores[i]) if bm25_scores is not None else 0.0
            final = sem_s if strategy == "semantic" else bm25_s

            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            event_id = meta.get("event_id", f"chunk_{idx}")

            docs.append(RetrievedDoc(
                doc_id=f"asrs_{event_id}",
                chunk_id=f"asrs_{event_id}#{meta.get('chunk_index', 0)}",
                chunk_text=self._chunks[idx],
                scores={
                    "semantic": round(sem_s, 4),
                    "bm25": round(bm25_s, 4),
                    "final": round(final, 4),
                },
                metadata=self._enrich_metadata(meta),
            ))

        return docs

    @staticmethod
    def _enrich_metadata(
        meta: Dict[str, Any],
        applied_filters: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Add source info and filters to metadata."""
        enriched = dict(meta)
        enriched["source"] = "phase2_real_retrieval"
        enriched["document_type"] = "incident_report"  # ASRS default
        if applied_filters:
            enriched["applied_filters"] = applied_filters
        return enriched
