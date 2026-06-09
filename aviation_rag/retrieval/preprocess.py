"""
preprocess.py — Text Preprocessing for Aviation Document Retrieval
===================================================================
Handles: CSV loading, text normalization, field combination, and
sentence-boundary-aware chunking for ASRS incident reports.

Author: Quan San — Phase 2 Semantic Retrieval Research
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import Settings


# ==============================================================================
# Text Normalization
# ==============================================================================

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s.,;:!?'\"-]")


def normalize_text(text: str) -> str:
    """Normalize text for embedding: lowercase, remove URLs, clean whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = _URL_PATTERN.sub(" ", text)
    text = text.replace(";", ",")
    text = _SPECIAL_CHAR_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


# ==============================================================================
# Field Combination
# ==============================================================================

def combine_text_fields(row: pd.Series, text_columns: tuple) -> str:
    """
    Combine ASRS text fields into a single document string.
    Format: [SUMMARY] ... [REPORT 1] ... [REPORT 2] ...
    """
    parts = []
    col_labels = {
        "report_summary": "SUMMARY",
        "report1_narrative": "REPORT 1",
        "report2_narrative": "REPORT 2",
    }
    for col in text_columns:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            label = col_labels.get(col, col.upper())
            parts.append(f"[{label}] {str(val).strip()}")
    return " ".join(parts)


# ==============================================================================
# Chunking (sentence-boundary-aware)
# ==============================================================================

def chunk_text(
    text: str,
    max_length: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks, respecting sentence boundaries.

    Args:
        text: Input text.
        max_length: Max words per chunk.
        overlap: Overlap words between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if len(words) <= max_length:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_length, len(words))
        chunk_words = words[start:end]

        # Try to end at sentence boundary (within last 20% of chunk)
        if end < len(words):
            search_start = max(0, len(chunk_words) - len(chunk_words) // 5)
            best_break = None
            for i in range(len(chunk_words) - 1, search_start - 1, -1):
                if chunk_words[i].endswith((".", "!", "?")):
                    best_break = i + 1
                    break
            if best_break and best_break > len(chunk_words) // 2:
                chunk_words = chunk_words[:best_break]

        chunk_text_str = " ".join(chunk_words)
        if chunk_text_str.strip():
            chunks.append(chunk_text_str)

        start += len(chunk_words) - overlap
        if start <= (end - len(chunk_words)):
            start = end

    return chunks if chunks else [text]


# ==============================================================================
# Extract Metadata
# ==============================================================================

def extract_metadata(row: pd.Series, metadata_columns: tuple) -> Dict[str, Any]:
    """Extract metadata fields from a DataFrame row."""
    meta = {}
    for col in metadata_columns:
        val = row.get(col)
        if pd.notna(val):
            meta[col] = str(val).strip() if isinstance(val, str) else val
    return meta


# ==============================================================================
# Full Pipeline
# ==============================================================================

def load_and_preprocess(
    settings: Settings,
    sample_size: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Full preprocessing pipeline: CSV → chunks + metadata.

    Args:
        settings: App settings with paths and parameters.
        sample_size: Number of records to sample (None = full dataset).

    Returns:
        (chunks, metadata_list) — parallel lists.
    """
    csv_path = settings.data_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[preprocess] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[preprocess] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate text columns
    available_text_cols = tuple(c for c in settings.text_columns if c in df.columns)
    if not available_text_cols:
        raise ValueError(f"No text columns found. Expected: {settings.text_columns}")

    # Sample if requested
    if sample_size and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"[preprocess] Sampled {len(df):,} rows")

    # Process rows
    all_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []
    skipped = 0

    for idx, row in df.iterrows():
        # Combine text fields
        combined = combine_text_fields(row, available_text_cols)
        if len(combined) < settings.min_text_length:
            skipped += 1
            continue

        # Normalize
        normalized = normalize_text(combined)
        if len(normalized) < settings.min_text_length:
            skipped += 1
            continue

        # Extract metadata
        meta = extract_metadata(row, settings.metadata_columns)

        # Chunk
        chunks = chunk_text(normalized, settings.max_chunk_length, settings.chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_meta = dict(meta)
            chunk_meta["chunk_index"] = chunk_idx
            chunk_meta["total_chunks"] = len(chunks)
            all_metadata.append(chunk_meta)

    print(f"[preprocess] ✓ {len(all_chunks):,} chunks from {len(df):,} records (skipped {skipped})")
    return all_chunks, all_metadata
