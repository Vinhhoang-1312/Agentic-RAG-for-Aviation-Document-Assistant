"""
indexer.py — Build FAISS + BM25 Index
=======================================
Pipeline: CSV → preprocess → embed → FAISS index + BM25 corpus → disk.

Saves to data/index_store/:
    - faiss_index.bin   (FAISS IndexFlatIP)
    - chunks.pkl        (list of chunk texts)
    - metadata.pkl      (list of metadata dicts)
    - bm25_corpus.pkl   (tokenized corpus for BM25Okapi)

Author: Quan San — Phase 2 Semantic Retrieval Research
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from ..config import Settings
from .preprocess import load_and_preprocess


# ==============================================================================
# Embedding
# ==============================================================================

def _load_embedding_model(model_name: str):
    """Load Sentence-Transformers model (lazy import to avoid slow startup)."""
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    print(f"[indexer] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[indexer] [OK] Model loaded (dimension={model.get_embedding_dimension()})")
    return model


def _embed_chunks(
    model,
    chunks: List[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Encode chunks into L2-normalized embeddings."""
    print(f"[indexer] Encoding {len(chunks):,} chunks (batch_size={batch_size})...")
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ==============================================================================
# BM25 Corpus Preparation
# ==============================================================================

def _build_bm25_corpus(chunks: List[str]) -> List[List[str]]:
    """Tokenize chunks for BM25 (simple whitespace tokenization)."""
    print(f"[indexer] Building BM25 corpus...")
    return [chunk.lower().split() for chunk in chunks]


# ==============================================================================
# FAISS Index
# ==============================================================================

def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS IndexFlatIP (exact inner product = cosine on L2-normed vectors)."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"[indexer] [OK] FAISS index built: {index.ntotal:,} vectors (dim={dimension})")
    return index


# ==============================================================================
# Save / Load
# ==============================================================================

def _save_index(
    index_dir: Path,
    faiss_index: faiss.Index,
    chunks: List[str],
    metadata: List[Dict[str, Any]],
    bm25_corpus: List[List[str]],
) -> None:
    """Save all index artifacts to disk."""
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(index_dir / "faiss_index.bin"))
    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(index_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    with open(index_dir / "bm25_corpus.pkl", "wb") as f:
        pickle.dump(bm25_corpus, f)

    total_size = sum(
        (index_dir / fname).stat().st_size
        for fname in ["faiss_index.bin", "chunks.pkl", "metadata.pkl", "bm25_corpus.pkl"]
    )
    print(f"[indexer] [OK] Saved to {index_dir} ({total_size / 1024 / 1024:.1f} MB)")


def load_index(
    index_dir: Path,
) -> Tuple[faiss.Index, List[str], List[Dict[str, Any]], List[List[str]]]:
    """Load all index artifacts from disk."""
    faiss_path = index_dir / "faiss_index.bin"
    chunks_path = index_dir / "chunks.pkl"
    metadata_path = index_dir / "metadata.pkl"
    bm25_path = index_dir / "bm25_corpus.pkl"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    faiss_index = faiss.read_index(str(faiss_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    bm25_corpus = []
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_corpus = pickle.load(f)

    print(f"[indexer] [OK] Index loaded: {faiss_index.ntotal:,} vectors, {len(chunks):,} chunks")
    return faiss_index, chunks, metadata, bm25_corpus


def index_exists(index_dir: Path) -> bool:
    """Check if a valid index exists on disk."""
    return (index_dir / "faiss_index.bin").exists() and (index_dir / "chunks.pkl").exists()


# ==============================================================================
# Main Build Function
# ==============================================================================

def build_and_save_index(
    settings: Settings,
    sample_size: Optional[int] = None,
    force: bool = False,
) -> None:
    """
    Full build pipeline: CSV → preprocess → embed → FAISS + BM25 → disk.

    Args:
        settings: App settings.
        sample_size: Number of records to sample (None = full dataset).
        force: Force rebuild even if index exists.
    """
    if not force and index_exists(settings.index_dir):
        print(f"[indexer] [OK] Index already exists at {settings.index_dir}")
        print(f"   Use --force to rebuild.")
        return

    start_time = time.time()

    # Step 1: Preprocess
    print("-" * 60)
    print("STEP 1/3: Preprocessing CSV data...")
    print("-" * 60)
    chunks, metadata = load_and_preprocess(settings, sample_size=sample_size)

    # Step 2: Embed
    print("\n" + "-" * 60)
    print("STEP 2/3: Embedding chunks...")
    print("-" * 60)
    model = _load_embedding_model(settings.embedding_model_name)
    embeddings = _embed_chunks(model, chunks, settings.embedding_batch_size)

    # Step 3: Build indices
    print("\n" + "-" * 60)
    print("STEP 3/3: Building FAISS + BM25 indices...")
    print("-" * 60)
    faiss_index = _build_faiss_index(embeddings)
    bm25_corpus = _build_bm25_corpus(chunks)

    # Save
    _save_index(settings.index_dir, faiss_index, chunks, metadata, bm25_corpus)

    elapsed = time.time() - start_time
    print(f"\n[OK] Index built in {elapsed:.1f}s -- {len(chunks):,} chunks indexed")
