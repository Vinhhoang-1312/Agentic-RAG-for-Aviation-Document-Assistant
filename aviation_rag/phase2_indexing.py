from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import Settings
from .intent_rules import map_row_to_intent
from .phase1_hoang_intent_routing import normalize_text, tokenize


@dataclass(frozen=True)
class CorpusChunk:
    doc_id: str
    chunk_id: str
    chunk_text: str
    lexical_text: str
    intent_hint: str
    metadata: dict[str, Any]


def _first_existing(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def document_type_from_row(row: dict[str, Any]) -> str:
    primary_problem = str(row.get("primary_problem", "")).lower()
    event_anomaly = str(row.get("event_anomaly", "")).lower()
    component = " ".join(
        [
            str(row.get("component_name", "")).lower(),
            str(row.get("component_problem", "")).lower(),
        ]
    )
    weather = str(row.get("weather_conditions", "")).lower()

    metadata_text = " ".join([primary_problem, event_anomaly, weather])
    procedure_text = " ".join([primary_problem, event_anomaly, component])
    if any(token in metadata_text for token in ["weather", "turbulence", "icing", "wind", "runway", "visibility"]):
        return "metadata"
    if any(token in procedure_text for token in ["procedure", "manual", "mel", "maintenance", "part", "checklist", "warning"]):
        return "procedure"
    return "incident_report"


def chunk_text_by_tokens(text: str, *, max_tokens: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    max_tokens = max(32, int(max_tokens))
    overlap = max(0, min(int(overlap), max_tokens - 1))
    if len(words) <= max_tokens:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max_tokens - overlap
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += step
    return chunks


def row_to_chunks(row: dict[str, Any], *, max_tokens: int, overlap: int, fallback_index: int) -> list[CorpusChunk]:
    summary = _first_existing(row, ["report_summary", "summary", "title"])
    narrative_1 = _first_existing(row, ["report1_narrative", "narrative", "text"])
    narrative_2 = _first_existing(row, ["report2_narrative"])
    raw_text = " ".join(part for part in [summary, narrative_1, narrative_2] if part).strip()
    normalized_text = normalize_text(raw_text)
    if not normalized_text:
        return []

    doc_id = str(row.get("event_id") or row.get("doc_id") or f"event_{fallback_index:06d}")
    metadata = {
        "source": "asrs_dataset",
        "airport": str(row.get("location_airport", "") or ""),
        "state": str(row.get("location_state", "") or ""),
        "weather_conditions": str(row.get("weather_conditions", "") or ""),
        "flight_conditions": str(row.get("flight_conditions", "") or ""),
        "component_name": str(row.get("component_name", "") or ""),
        "component_problem": str(row.get("component_problem", "") or ""),
        "primary_problem": str(row.get("primary_problem", "") or ""),
        "event_anomaly": str(row.get("event_anomaly", "") or ""),
        "document_type": document_type_from_row(row),
    }
    lexical_context = normalize_text(
        " ".join(
            [
                raw_text,
                metadata["airport"],
                metadata["state"],
                metadata["weather_conditions"],
                metadata["flight_conditions"],
                metadata["component_name"],
                metadata["component_problem"],
                metadata["primary_problem"],
                metadata["event_anomaly"],
            ]
        )
    )
    intent_hint = map_row_to_intent(pd.Series(row))

    chunks: list[CorpusChunk] = []
    for chunk_index, chunk_text in enumerate(
        chunk_text_by_tokens(raw_text, max_tokens=max_tokens, overlap=overlap)
    ):
        chunk_lexical_text = normalize_text(" ".join([chunk_text, lexical_context]))
        chunks.append(
            CorpusChunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{chunk_index}",
                chunk_text=chunk_text,
                lexical_text=chunk_lexical_text,
                intent_hint=intent_hint,
                metadata={**metadata, "chunk_index": chunk_index},
            )
        )
    return chunks


def load_source_frame(settings: Settings) -> pd.DataFrame:
    path = Path(settings.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Retrieval dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    if settings.retrieval_max_docs > 0:
        frame = frame.head(settings.retrieval_max_docs)
    return frame


def build_corpus_chunks(settings: Settings) -> list[CorpusChunk]:
    frame = load_source_frame(settings)
    chunks: list[CorpusChunk] = []
    seen: set[str] = set()
    for index, row in enumerate(frame.fillna("").to_dict(orient="records")):
        for chunk in row_to_chunks(
            row,
            max_tokens=settings.phase2_chunk_tokens,
            overlap=settings.phase2_chunk_overlap,
            fallback_index=index,
        ):
            dedupe_key = normalize_text(chunk.chunk_text)[:1200]
            if not dedupe_key or dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No retrieval chunks could be built from the local dataset.")
    return chunks


def save_chunks(chunks: list[CorpusChunk], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return path


def load_chunks(path: Path) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(CorpusChunk(**json.loads(line)))
    return chunks


def tokenized_lexical_docs(chunks: list[CorpusChunk]) -> list[list[str]]:
    return [tokenize(chunk.lexical_text) for chunk in chunks]
