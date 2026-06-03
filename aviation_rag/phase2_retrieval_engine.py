from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover
    SentenceTransformer = None

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
except ModuleNotFoundError:  # pragma: no cover
    TruncatedSVD = None
    TfidfVectorizer = None

from .config import Settings
from .phase1_hoang_intent_routing import normalize_text, tokenize
from .phase2_indexing import CorpusChunk, build_corpus_chunks, load_chunks, save_chunks, tokenized_lexical_docs
from .schemas import InputAgentOutput, MiddleAgentOutput, RetrievedDoc


@dataclass(frozen=True)
class Phase2IndexInfo:
    retrieval_backend: str
    embedding_model: str
    embedding_backend: str
    embedding_dim: int
    faiss_index_type: str
    normalization: str
    chunk_count: int
    bm25_enabled: bool
    index_dir: str
    fallback_reason: str | None = None


class Phase2RetrievalEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chunks: list[CorpusChunk] = []
        self._semantic_index: Any | None = None
        self._semantic_vectors: np.ndarray | None = None
        self._sentence_model: Any | None = None
        self._tfidf_vectorizer: Any | None = None
        self._svd: Any | None = None
        self._idf: dict[str, float] = {}
        self._doc_freqs: list[dict[str, int]] = []
        self._doc_lengths: np.ndarray | None = None
        self._avg_doc_length = 0.0
        self._build_error: str | None = None
        self._info: Phase2IndexInfo | None = None

    @property
    def available(self) -> bool:
        return self.ensure_ready()

    @property
    def build_error(self) -> str | None:
        return self._build_error

    @property
    def info(self) -> Phase2IndexInfo | None:
        if self.ensure_ready():
            return self._info
        return None

    def ensure_ready(self) -> bool:
        if self._semantic_index is not None and self._chunks:
            return True
        if self._build_error is not None:
            return False
        try:
            self._build_resources()
            return True
        except Exception as exc:
            self._build_error = str(exc)
            return False

    def retrieve(self, input_row: InputAgentOutput) -> MiddleAgentOutput:
        if not self.ensure_ready():
            raise RuntimeError(self._build_error or "Phase 2 retrieval is unavailable.")

        started = perf_counter()
        strategy = input_row.retrieval_plan.strategy
        top_k = max(1, int(input_row.retrieval_plan.top_k))
        semantic_scores = self._semantic_scores(input_row)
        bm25_scores = self._bm25_scores(input_row)
        metadata_scores = self._metadata_scores(input_row)
        mask = self._candidate_mask(input_row)
        final_scores = self._final_scores(
            strategy=strategy,
            semantic_scores=semantic_scores,
            bm25_scores=bm25_scores,
            metadata_scores=metadata_scores,
        )

        candidate_indices = self._candidate_indices(
            semantic_scores=semantic_scores,
            bm25_scores=bm25_scores,
            metadata_scores=metadata_scores,
            final_scores=final_scores,
            top_k=top_k,
            mask=mask,
        )
        ranked_indices = sorted(candidate_indices, key=lambda idx: final_scores[idx], reverse=True)[:top_k]
        latency_ms = (perf_counter() - started) * 1000.0

        topk_docs: list[RetrievedDoc] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self._chunks[idx]
            topk_docs.append(
                RetrievedDoc(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.chunk_text,
                    scores={
                        "rank": float(rank),
                        "semantic": float(semantic_scores[idx]),
                        "bm25": float(bm25_scores[idx]),
                        "metadata": float(metadata_scores[idx]),
                        "final": float(final_scores[idx]),
                    },
                    metadata=chunk.metadata,
                )
            )

        return MiddleAgentOutput(
            query_id=input_row.query_id,
            predicted_intent=input_row.intent,
            topk_docs=topk_docs,
            retrieval_diagnostics=self._diagnostics(
                input_row=input_row,
                strategy=strategy,
                latency_ms=latency_ms,
                metadata_filter_applied=not bool(mask.all()) if mask.size else False,
            ),
        )

    def compare_strategies(self, input_row: InputAgentOutput, strategies: list[str]) -> dict[str, MiddleAgentOutput]:
        outputs: dict[str, MiddleAgentOutput] = {}
        for strategy in strategies:
            cloned = input_row.model_copy(deep=True)
            cloned.retrieval_plan.strategy = strategy  # type: ignore[assignment]
            outputs[strategy] = self.retrieve(cloned)
        return outputs

    def _build_resources(self) -> None:
        if faiss is None:
            raise ModuleNotFoundError("faiss-cpu is not installed. Install requirements.txt to enable Phase 2 retrieval.")
        chunks = self._load_or_build_chunks()
        self._chunks = chunks
        self._build_bm25(chunks)
        self._build_semantic_index(chunks)

    def _load_or_build_chunks(self) -> list[CorpusChunk]:
        chunks_path = Path(self.settings.phase2_index_dir) / "chunks.jsonl"
        if chunks_path.exists():
            try:
                chunks = load_chunks(chunks_path)
                if chunks:
                    return chunks
            except Exception:
                pass
        chunks = build_corpus_chunks(self.settings)
        save_chunks(chunks, chunks_path)
        return chunks

    def _build_semantic_index(self, chunks: list[CorpusChunk]) -> None:
        dense_texts = [chunk.chunk_text for chunk in chunks]
        fallback_reason: str | None = None
        embedding_backend = "sentence_transformers"
        embedding_model = self.settings.phase2_embedding_model

        try:
            vectors, model = self._encode_with_sentence_transformer(dense_texts)
            self._sentence_model = model
            embedding_dim = int(vectors.shape[1])
        except Exception as exc:
            fallback_reason = str(exc)
            vectors = self._encode_with_tfidf_svd(dense_texts)
            embedding_backend = "tfidf_svd_faiss_fallback"
            embedding_model = f"fallback:{self.settings.phase2_embedding_model}"
            embedding_dim = int(vectors.shape[1])

        index = faiss.IndexFlatIP(embedding_dim)
        index.add(vectors)
        self._semantic_vectors = vectors
        self._semantic_index = index
        self._persist_semantic_artifacts(vectors=vectors, embedding_backend=embedding_backend)
        self._info = Phase2IndexInfo(
            retrieval_backend="phase2_dense_bm25_hybrid",
            embedding_model=embedding_model,
            embedding_backend=embedding_backend,
            embedding_dim=embedding_dim,
            faiss_index_type="IndexFlatIP",
            normalization="L2",
            chunk_count=len(chunks),
            bm25_enabled=True,
            index_dir=str(self.settings.phase2_index_dir),
            fallback_reason=fallback_reason,
        )

    def _encode_with_sentence_transformer(self, texts: list[str]) -> tuple[np.ndarray, Any]:
        forced_fallback_values = {"tfidf_svd_fallback", "tfidf-svd-fallback", "fallback:tfidf_svd"}
        if self.settings.phase2_embedding_model.strip().lower() in forced_fallback_values:
            raise RuntimeError("TF-IDF/SVD fallback was explicitly requested by PHASE2_EMBEDDING_MODEL.")
        if SentenceTransformer is None:
            raise ModuleNotFoundError("sentence-transformers is not installed.")
        model = SentenceTransformer(self.settings.phase2_embedding_model)
        vectors = model.encode(
            texts,
            batch_size=self.settings.phase2_semantic_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("SentenceTransformer returned invalid embedding shape.")
        return vectors, model

    def _encode_query_with_sentence_transformer(self, text: str) -> np.ndarray:
        if self._sentence_model is None:
            raise RuntimeError("SentenceTransformer query encoder is unavailable.")
        return self._sentence_model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def _encode_with_tfidf_svd(self, texts: list[str]) -> np.ndarray:
        if TfidfVectorizer is None or TruncatedSVD is None:
            raise ModuleNotFoundError("scikit-learn is not installed and dense model fallback cannot be built.")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            max_features=self.settings.retrieval_tfidf_max_features,
            min_df=1 if len(texts) < 20 else 2,
            max_df=0.98,
            ngram_range=(1, 2),
        )
        sparse_matrix = vectorizer.fit_transform(texts)
        feature_limit = max(1, min(sparse_matrix.shape[0], sparse_matrix.shape[1]))
        components = min(max(1, self.settings.retrieval_svd_components), feature_limit)
        self._tfidf_vectorizer = vectorizer
        if components > 1:
            svd = TruncatedSVD(n_components=components, random_state=42)
            dense = svd.fit_transform(sparse_matrix).astype("float32")
            self._svd = svd
        else:
            dense = sparse_matrix.astype("float32").toarray()
            self._svd = None
        faiss.normalize_L2(dense)
        return dense

    def _encode_query_with_tfidf_svd(self, text: str) -> np.ndarray:
        if self._tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF fallback query encoder is unavailable.")
        query_matrix = self._tfidf_vectorizer.transform([text])
        if self._svd is None:
            dense_query = query_matrix.astype("float32").toarray()
        else:
            dense_query = self._svd.transform(query_matrix).astype("float32")
        faiss.normalize_L2(dense_query)
        return dense_query

    def _persist_semantic_artifacts(self, *, vectors: np.ndarray, embedding_backend: str) -> None:
        index_dir = Path(self.settings.phase2_index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", vectors)
        faiss_persist_error = None
        if self._semantic_index is not None:
            index_path = index_dir / "faiss.index"
            try:
                faiss.write_index(self._semantic_index, str(index_path))
            except Exception as exc:
                # FAISS on Windows can fail on Unicode paths. Write through an ASCII temp path,
                # then copy bytes with Python's Unicode-aware filesystem APIs.
                faiss_persist_error = str(exc)
                try:
                    temp_path = Path(tempfile.gettempdir()) / "phase2_faiss_index.tmp"
                    faiss.write_index(self._semantic_index, str(temp_path))
                    index_path.write_bytes(temp_path.read_bytes())
                    temp_path.unlink(missing_ok=True)
                    faiss_persist_error = None
                except Exception as temp_exc:
                    faiss_persist_error = f"{faiss_persist_error}; temp copy fallback failed: {temp_exc}"
        metadata = {
            "embedding_model": self.settings.phase2_embedding_model,
            "embedding_backend": embedding_backend,
            "embedding_dim": int(vectors.shape[1]),
            "faiss_index_type": "IndexFlatIP",
            "normalization": "L2",
            "chunk_count": len(self._chunks),
            "faiss_persist_error": faiss_persist_error,
        }
        (index_dir / "index_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_bm25(self, chunks: list[CorpusChunk]) -> None:
        tokenized_docs = tokenized_lexical_docs(chunks)
        doc_freqs: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        lengths: list[int] = []
        for tokens in tokenized_docs:
            freqs: dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            doc_freqs.append(freqs)
            lengths.append(len(tokens))
            for token in freqs:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        doc_count = max(1, len(tokenized_docs))
        self._idf = {
            token: math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            for token, df in document_frequency.items()
        }
        self._doc_freqs = doc_freqs
        self._doc_lengths = np.asarray(lengths, dtype="float32")
        self._avg_doc_length = float(np.mean(self._doc_lengths)) if lengths else 0.0

    def _query_text(self, input_row: InputAgentOutput) -> str:
        return normalize_text(
            " ".join(
                [
                    input_row.query_raw,
                    input_row.rewritten_query,
                    " ".join(input_row.expanded_queries[:3]),
                ]
            )
        )

    def _semantic_scores(self, input_row: InputAgentOutput) -> np.ndarray:
        if self._semantic_index is None:
            return np.zeros(len(self._chunks), dtype="float32")
        query_text = self._query_text(input_row)
        if self._info and self._info.embedding_backend == "sentence_transformers":
            dense_query = self._encode_query_with_sentence_transformer(query_text)
        else:
            dense_query = self._encode_query_with_tfidf_svd(query_text)
        scores = np.zeros(len(self._chunks), dtype="float32")
        search_k = min(len(self._chunks), max(input_row.retrieval_plan.top_k * 24, 128))
        raw_scores, indices = self._semantic_index.search(dense_query, search_k)
        for score, index in zip(raw_scores[0], indices[0]):
            if index >= 0:
                scores[index] = max(0.0, float(score))
        return self._normalize(scores)

    def _bm25_scores(self, input_row: InputAgentOutput) -> np.ndarray:
        query_tokens = tokenize(self._query_text(input_row))
        scores = np.zeros(len(self._chunks), dtype="float32")
        if not query_tokens or not self._doc_freqs or self._doc_lengths is None:
            return scores

        avg_length = max(self._avg_doc_length, 1.0)
        k1 = 1.5
        b = 0.75
        for index, doc_freq in enumerate(self._doc_freqs):
            doc_length = float(self._doc_lengths[index])
            score = 0.0
            for token in query_tokens:
                term_frequency = doc_freq.get(token, 0)
                if term_frequency == 0:
                    continue
                idf = self._idf.get(token, 0.0)
                denominator = term_frequency + k1 * (1.0 - b + b * doc_length / avg_length)
                score += idf * (term_frequency * (k1 + 1.0)) / denominator
            scores[index] = score
        return self._normalize(scores)

    def _metadata_scores(self, input_row: InputAgentOutput) -> np.ndarray:
        scores = np.zeros(len(self._chunks), dtype="float32")
        requested_document_type = str(input_row.retrieval_plan.filters.get("document_type", "")).strip().lower()
        prefer_metadata = bool(input_row.retrieval_plan.filters.get("prefer_metadata"))
        query_tokens = set(tokenize(self._query_text(input_row)))
        for index, chunk in enumerate(self._chunks):
            score = 0.0
            document_type = str(chunk.metadata.get("document_type", "")).lower()
            airport = str(chunk.metadata.get("airport", "")).lower()
            state = str(chunk.metadata.get("state", "")).lower()
            if chunk.intent_hint == input_row.intent:
                score += 0.35
            if requested_document_type and requested_document_type == document_type:
                score += 0.40
            if prefer_metadata and document_type == "metadata":
                score += 0.40
            if airport and airport in query_tokens:
                score += 0.20
            if state and state in query_tokens:
                score += 0.10
            if input_row.intent == "Factoid":
                score += max(0.0, 0.15 - min(len(chunk.chunk_text), 500) / 3500.0)
            scores[index] = score
        return np.clip(scores, 0.0, 1.0)

    def _candidate_mask(self, input_row: InputAgentOutput) -> np.ndarray:
        mask = np.ones(len(self._chunks), dtype=bool)
        requested_document_type = str(input_row.retrieval_plan.filters.get("document_type", "")).strip().lower()
        if requested_document_type:
            typed_mask = np.asarray(
                [
                    str(chunk.metadata.get("document_type", "")).lower() == requested_document_type
                    for chunk in self._chunks
                ],
                dtype=bool,
            )
            if typed_mask.any():
                mask &= typed_mask
        if input_row.intent == "Metadata_Query":
            metadata_mask = np.asarray(
                [str(chunk.metadata.get("document_type", "")).lower() == "metadata" for chunk in self._chunks],
                dtype=bool,
            )
            if metadata_mask.any():
                mask &= metadata_mask
        return mask

    def _final_scores(
        self,
        *,
        strategy: str,
        semantic_scores: np.ndarray,
        bm25_scores: np.ndarray,
        metadata_scores: np.ndarray,
    ) -> np.ndarray:
        semantic_scores = self._normalize(semantic_scores)
        bm25_scores = self._normalize(bm25_scores)
        metadata_scores = self._normalize(metadata_scores)
        if strategy == "bm25":
            final = 0.85 * bm25_scores + 0.15 * metadata_scores
        elif strategy == "metadata_first":
            final = 0.50 * metadata_scores + 0.30 * bm25_scores + 0.20 * semantic_scores
        elif strategy == "semantic":
            final = 0.85 * semantic_scores + 0.15 * metadata_scores
        elif strategy == "hybrid_rrf":
            final = self._rrf_scores([semantic_scores, bm25_scores, metadata_scores])
        else:
            final = 0.50 * semantic_scores + 0.35 * bm25_scores + 0.15 * metadata_scores
        return np.clip(final, 0.0, 1.0).astype("float32")

    def _candidate_indices(
        self,
        *,
        semantic_scores: np.ndarray,
        bm25_scores: np.ndarray,
        metadata_scores: np.ndarray,
        final_scores: np.ndarray,
        top_k: int,
        mask: np.ndarray,
    ) -> list[int]:
        limit = top_k * 12
        candidates = set(self._top_indices(final_scores, limit, mask))
        candidates.update(self._top_indices(semantic_scores, limit, mask))
        candidates.update(self._top_indices(bm25_scores, limit, mask))
        candidates.update(self._top_indices(metadata_scores, limit, mask))
        if not candidates:
            candidates.update(self._top_indices(final_scores, top_k, np.ones(len(self._chunks), dtype=bool)))
        return sorted(candidates)

    def _top_indices(self, scores: np.ndarray, limit: int, mask: np.ndarray) -> list[int]:
        candidate_indices = np.flatnonzero(mask)
        if candidate_indices.size == 0:
            return []
        candidate_scores = scores[candidate_indices]
        limit = min(max(1, limit), candidate_scores.size)
        top_positions = np.argpartition(candidate_scores, -limit)[-limit:]
        ranked_positions = top_positions[np.argsort(candidate_scores[top_positions])[::-1]]
        return [int(candidate_indices[position]) for position in ranked_positions]

    def _rrf_scores(self, score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
        if not score_lists:
            return np.zeros(len(self._chunks), dtype="float32")
        rrf = np.zeros(len(score_lists[0]), dtype="float32")
        for scores in score_lists:
            ranking = np.argsort(scores)[::-1]
            for rank, index in enumerate(ranking, start=1):
                if scores[index] <= 0:
                    continue
                rrf[index] += 1.0 / (k + rank)
        return self._normalize(rrf)

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores.astype("float32")
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))
        if math.isclose(max_score, min_score):
            return np.zeros_like(scores, dtype="float32")
        return ((scores - min_score) / (max_score - min_score)).astype("float32")

    def _diagnostics(
        self,
        *,
        input_row: InputAgentOutput,
        strategy: str,
        latency_ms: float,
        metadata_filter_applied: bool,
    ) -> dict[str, Any]:
        info = self._info
        fusion_method = "reciprocal_rank_fusion" if strategy == "hybrid_rrf" else "weighted_linear_fusion"
        weights = {
            "semantic": {"semantic": 0.85, "metadata": 0.15},
            "bm25": {"bm25": 0.85, "metadata": 0.15},
            "metadata_first": {"metadata": 0.50, "bm25": 0.30, "semantic": 0.20},
            "hybrid": {"semantic": 0.50, "bm25": 0.35, "metadata": 0.15},
            "hybrid_rrf": {"semantic_rank": 1.0, "bm25_rank": 1.0, "metadata_rank": 1.0},
        }
        return {
            "adapter_mode": "faiss_retrieval",
            "contract_owner": "Quang San",
            "retrieval_backend": info.retrieval_backend if info else "unknown",
            "embedding_model": info.embedding_model if info else self.settings.phase2_embedding_model,
            "embedding_backend": info.embedding_backend if info else "unknown",
            "embedding_dim": info.embedding_dim if info else 0,
            "faiss_index_type": info.faiss_index_type if info else "IndexFlatIP",
            "normalization": info.normalization if info else "L2",
            "chunk_count": info.chunk_count if info else len(self._chunks),
            "corpus_size": info.chunk_count if info else len(self._chunks),
            "bm25_enabled": True,
            "metadata_filter_applied": metadata_filter_applied,
            "fusion_method": fusion_method,
            "score_weights": weights.get(strategy, weights["hybrid"]),
            "strategy_requested": strategy,
            "fallback_strategy": input_row.retrieval_plan.fallback_strategy,
            "routing_reason": input_row.retrieval_plan.routing_reason,
            "dataset_path": str(self.settings.data_path),
            "index_dir": str(self.settings.phase2_index_dir),
            "latency_ms": latency_ms,
            "fallback_reason": info.fallback_reason if info else None,
        }
