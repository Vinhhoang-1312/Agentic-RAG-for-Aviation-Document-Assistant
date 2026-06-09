"""Execute all research notebooks and save fresh outputs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import nbformat
    from nbclient import NotebookClient

    notebooks = [
        ROOT / "notebooks" / "phase1_hoang_intent_routing_research.ipynb",
        ROOT / "notebooks" / "phase2_san_semantic_retrieval_research.ipynb",
        ROOT / "notebooks" / "phase3_hoang_grounded_output_research.ipynb",
    ]
    exit_code = 0
    for path in notebooks:
        print("=" * 70)
        print(f"Executing {path.name}")
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
        )
        try:
            client.execute()
            nbformat.write(nb, path)
            print(f"Saved {path.name}")
        except Exception as exc:
            exit_code = 1
            print(f"FAILED {path.name}: {exc}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
