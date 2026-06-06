"""
Build Phase 2 (Quan San) FAISS + BM25 Index
=============================================
Usage:
    python scripts/build_phase2_san_index.py                 # Full dataset (111K)
    python scripts/build_phase2_san_index.py --sample 5000   # Sample 5K for testing
    python scripts/build_phase2_san_index.py --force          # Force rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from aviation_rag.config import Settings
from aviation_rag.retrieval.indexer import build_and_save_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS + BM25 index for Semantic Retrieval (Phase 2 — Quan San)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample size (default: full dataset). E.g., --sample 5000",
    )
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists")
    args = parser.parse_args()

    settings = Settings()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    Phase 2 — Semantic Retrieval Index Builder               ║")
    print("║    Author: Quan San                                         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    print(f"  CSV path:  {settings.data_path}")
    print(f"  Index dir: {settings.index_dir}")
    print(f"  Model:     {settings.embedding_model_name}")
    print(f"  Sample:    {args.sample or 'full dataset'}")
    print()

    build_and_save_index(
        settings=settings,
        sample_size=args.sample,
        force=args.force,
    )


if __name__ == "__main__":
    main()
