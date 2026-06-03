from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aviation_rag.config import Settings, ensure_artifact_dirs
from aviation_rag.phase2_san_faiss_retrieval import Phase2SanFaissRetrieval


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    settings = Settings()
    ensure_artifact_dirs(settings)
    retrieval = Phase2SanFaissRetrieval(settings)
    if not retrieval.available:
        raise SystemExit(f"Phase 2 index build failed: {retrieval.build_error}")
    info = retrieval.info
    payload = info.__dict__ if info else {"status": "available", "details": "No info returned."}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
