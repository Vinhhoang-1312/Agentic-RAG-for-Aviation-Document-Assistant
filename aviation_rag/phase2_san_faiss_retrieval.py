from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings
from .intent_rules import map_row_to_intent
from .phase1_hoang_intent_routing import normalize_text, tokenize
from .schemas import InputAgentOutput, MiddleAgentOutput, RetrievedDoc


@dataclass(frozen=True)
class CorpusChunk:
    doc_id: str
    chunk_id: str
    chunk_text: str
    lexical_text: str
    intent_hint: str
    metadata: dict[str, Any]


class Phase2SanFaissRetrieval:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chunks: list[CorpusChunk] = []
        self._semantic_index: faiss.Index | None = None
        self._semantic_vectors: np.ndarray | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._idf: dict[str, float] = {}
        self._doc_freqs: list[dict[str, int]] = []
        self._doc_lengths: np.ndarray | None = None
        self._avg_doc_length = 0.0
        self._build_error: str | None = None

    @property
    def available(self) -> bool:
        return self._ensure_ready()

    @property
    def build_error(self) -> str | None:
        return self._build_error

    def retrieve(self, input_row: InputAgentOutput) -> MiddleAgentOutput:
        if not self._ensure_ready():
            raise RuntimeError(self._build_error or "Phase 2 FAISS retrieval is unavailable.")

        strategy = input_row.retrieval_plan.strategy
        top_k = max(1, int(input_row.retrieval_plan.top_k))
        semantic_scores = self._semantic_scores(input_row)
        bm25_scores = self._bm25_scores(input_row)
        metadata_bonus = self._metadata_bonus(input_row)
        mask = self._candidate_mask(input_row)

        semantic_candidates = self._top_indices(semantic_scores, top_k * 8, mask)
        bm25_candidates = self._top_indices(bm25_scores, top_k * 8, mask)
        combined_candidates = sorted(set(semantic_candidates + bm25_candidates))
        if not combined_candidates:
            combined_candidates = self._top_indices(
                semantic_scores + bm25_scores + metadata_bonus,
                top_k,
                np.ones(len(self._chunks), dtype=bool),
            )

        final_scores = self._final_scores(
            strategy=strategy,
            semantic_scores=semantic_scores,
            bm25_scores=bm25_scores,
            metadata_bonus=metadata_bonus,
        )
        ranked_indices = sorted(combined_candidates, key=lambda idx: final_scores[idx], reverse=True)[:top_k]

        topk_docs = [
            RetrievedDoc(
                doc_id=self._chunks[idx].doc_id,
                chunk_id=self._chunks[idx].chunk_id,
                chunk_text=self._chunks[idx].chunk_text,
                scores={
                    "semantic": float(semantic_scores[idx]),
                    "bm25": float(bm25_scores[idx]),
                    "metadata": float(metadata_bonus[idx]),
                    "final": float(final_scores[idx]),
                },
                metadata=self._chunks[idx].metadata,
            )
            for idx in ranked_indices
        ]

        return MiddleAgentOutput(
            query_id=input_row.query_id,
            predicted_intent=input_row.intent,
            topk_docs=topk_docs,
            retrieval_diagnostics={
                "adapter_mode": "faiss_retrieval",
                "contract_owner": "Quang San",
                "strategy_requested": strategy,
                "fallback_strategy": input_row.retrieval_plan.fallback_strategy,
                "routing_reason": input_row.retrieval_plan.routing_reason,
                "corpus_size": len(self._chunks),
                "embedding_backend": "tfidf_svd_faiss",
                "dataset_path": str(self.settings.data_path),
            },
        )

    def _ensure_ready(self) -> bool:
        if self._semantic_index is not None:
            return True
        if self._build_error is not None:
            return False

        try:
            self._build_resources()
            return True
        except Exception as exc:
            self._build_error = str(exc)
            return False

    def _build_resources(self) -> None:
        data_path = Path(self.settings.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Retrieval dataset not found: {data_path}")

        frame = pd.read_csv(data_path, low_memory=False)
        if self.settings.retrieval_max_docs > 0:
            frame = frame.head(self.settings.retrieval_max_docs)
        chunks = self._build_chunks(frame)
        if not chunks:
            raise ValueError("No retrieval chunks could be built from the local dataset.")

        texts = [chunk.lexical_text for chunk in chunks]
        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            max_features=self.settings.retrieval_tfidf_max_features,
            min_df=1 if len(texts) < 20 else 2,
            max_df=0.98,
            ngram_range=(1, 2),
        )
        sparse_matrix = vectorizer.fit_transform(texts)

        feature_limit = max(1, min(sparse_matrix.shape[0], sparse_matrix.shape[1]))
        svd_components = min(self.settings.retrieval_svd_components, feature_limit)
        svd: TruncatedSVD | None = None
        if svd_components > 1:
            svd = TruncatedSVD(n_components=svd_components, random_state=42)
            dense_matrix = svd.fit_transform(sparse_matrix).astype("float32")
        else:
            dense_matrix = sparse_matrix.astype("float32").toarray()
        faiss.normalize_L2(dense_matrix)

        index = faiss.IndexFlatIP(dense_matrix.shape[1])
        index.add(dense_matrix)

        tokenized_docs = [tokenize(text) for text in texts]
        doc_freqs: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        lengths = []
        for tokens in tokenized_docs:
            freqs: dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            doc_freqs.append(freqs)
            lengths.append(len(tokens))
            for token in freqs:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        doc_count = len(tokenized_docs)
        idf = {
            token: math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            for token, df in document_frequency.items()
        }

        self._chunks = chunks
        self._tfidf_vectorizer = vectorizer
        self._svd = svd
        self._semantic_vectors = dense_matrix
        self._semantic_index = index
        self._doc_freqs = doc_freqs
        self._doc_lengths = np.asarray(lengths, dtype="float32")
        self._avg_doc_length = float(np.mean(self._doc_lengths)) if lengths else 0.0
        self._idf = idf

    def _build_chunks(self, frame: pd.DataFrame) -> list[CorpusChunk]:
        chunks: list[CorpusChunk] = []
        for row in frame.fillna("").to_dict(orient="records"):
            summary = str(row.get("report_summary", "")).strip()
            narrative_1 = str(row.get("report1_narrative", "")).strip()
            narrative_2 = str(row.get("report2_narrative", "")).strip()
            text_parts = [part for part in [summary, narrative_1, narrative_2] if part]
            if not text_parts:
                continue

            raw_text = " ".join(text_parts)
            normalized_text = normalize_text(raw_text)
            if not normalized_text:
                continue

            event_id = str(row.get("event_id") or f"event_{len(chunks):06d}")
            metadata = {
                "source": "asrs_dataset",
                "airport": str(row.get("location_airport", "")),
                "state": str(row.get("location_state", "")),
                "weather_conditions": str(row.get("weather_conditions", "")),
                "flight_conditions": str(row.get("flight_conditions", "")),
                "component_name": str(row.get("component_name", "")),
                "component_problem": str(row.get("component_problem", "")),
                "primary_problem": str(row.get("primary_problem", "")),
                "event_anomaly": str(row.get("event_anomaly", "")),
                "document_type": self._document_type_from_row(row),
            }
            lexical_text = normalize_text(
                " ".join(
                    [
                        raw_text,
                        str(row.get("location_airport", "")),
                        str(row.get("weather_conditions", "")),
                        str(row.get("flight_conditions", "")),
                        str(row.get("component_name", "")),
                        str(row.get("component_problem", "")),
                        str(row.get("primary_problem", "")),
                        str(row.get("event_anomaly", "")),
                    ]
                )
            )
            intent_hint = map_row_to_intent(pd.Series(row))
            chunks.append(
                CorpusChunk(
                    doc_id=event_id,
                    chunk_id=f"{event_id}#0",
                    chunk_text=raw_text[:2500],
                    lexical_text=lexical_text,
                    intent_hint=intent_hint,
                    metadata=metadata,
                )
            )
        return chunks

    def _document_type_from_row(self, row: dict[str, Any]) -> str:
        primary_problem = str(row.get("primary_problem", "")).lower()
        if any(token in primary_problem for token in ["weather", "turbulence", "icing", "wind", "runway"]):
            return "metadata"
        if any(token in primary_problem for token in ["procedure", "manual", "mel", "maintenance", "part"]):
            return "procedure"
        return "incident_report"

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
        if self._semantic_index is None or self._tfidf_vectorizer is None:
            return np.zeros(len(self._chunks), dtype="float32")

        query_matrix = self._tfidf_vectorizer.transform([self._query_text(input_row)])
        if self._svd is None:
            dense_query = query_matrix.astype("float32").toarray()
        else:
            dense_query = self._svd.transform(query_matrix).astype("float32")
        faiss.normalize_L2(dense_query)
        semantic_scores = np.zeros(len(self._chunks), dtype="float32")
        search_k = min(len(self._chunks), max(input_row.retrieval_plan.top_k * 12, 64))
        scores, indices = self._semantic_index.search(dense_query, search_k)
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                semantic_scores[index] = max(0.0, float(score))
        return semantic_scores

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

    def _metadata_bonus(self, input_row: InputAgentOutput) -> np.ndarray:
        scores = np.zeros(len(self._chunks), dtype="float32")
        requested_document_type = str(input_row.retrieval_plan.filters.get("document_type", "")).strip().lower()
        prefer_metadata = bool(input_row.retrieval_plan.filters.get("prefer_metadata"))
        for index, chunk in enumerate(self._chunks):
            score = 0.0
            if chunk.intent_hint == input_row.intent:
                score += 0.45
            if requested_document_type and requested_document_type == str(chunk.metadata.get("document_type", "")).lower():
                score += 0.35
            if prefer_metadata and chunk.metadata.get("document_type") == "metadata":
                score += 0.40
            if input_row.intent == "Factoid":
                score += max(0.0, 0.20 - min(len(chunk.chunk_text), 500) / 2500.0)
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
                [chunk.metadata.get("document_type") == "metadata" for chunk in self._chunks],
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
        metadata_bonus: np.ndarray,
    ) -> np.ndarray:
        semantic_scores = self._normalize(semantic_scores)
        bm25_scores = self._normalize(bm25_scores)
        if strategy == "bm25":
            final = 0.85 * bm25_scores + 0.15 * metadata_bonus
        elif strategy == "metadata_first":
            final = 0.50 * bm25_scores + 0.20 * semantic_scores + 0.30 * metadata_bonus
        elif strategy == "semantic":
            final = 0.85 * semantic_scores + 0.15 * metadata_bonus
        else:
            final = 0.50 * semantic_scores + 0.35 * bm25_scores + 0.15 * metadata_bonus
        return np.clip(final, 0.0, 1.0)

    def _top_indices(self, scores: np.ndarray, limit: int, mask: np.ndarray) -> list[int]:
        candidate_indices = np.flatnonzero(mask)
        if candidate_indices.size == 0:
            return []
        candidate_scores = scores[candidate_indices]
        if candidate_scores.size == 0:
            return []
        limit = min(limit, candidate_scores.size)
        top_positions = np.argpartition(candidate_scores, -limit)[-limit:]
        ranked_positions = top_positions[np.argsort(candidate_scores[top_positions])[::-1]]
        return [int(candidate_indices[position]) for position in ranked_positions]

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        max_score = float(np.max(scores))
        min_score = float(np.min(scores))
        if math.isclose(max_score, min_score):
            return np.zeros_like(scores, dtype="float32")
        return ((scores - min_score) / (max_score - min_score)).astype("float32")
