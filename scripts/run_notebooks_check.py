"""Run Phase 1/2/3 notebook code cells and print results."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _display(obj):
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def run_notebook(nb_path: Path, label: str) -> list[tuple[int, str, str | None]]:
    import os

    os.chdir(ROOT)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    print("=" * 70)
    print(f"{label}: {nb_path.name}")
    print("=" * 70)
    g: dict = {"display": _display, "__name__": "__main__"}
    cell_results: list[tuple[int, str, str | None]] = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", [])).strip()
            title = src.split("\n")[0][:80] if src else "(empty markdown)"
            print(f"\n--- Cell {i} [markdown] {title}")
            continue
        src = "".join(cell.get("source", []))
        first_line = src.strip().split("\n")[0][:70] if src.strip() else "(empty code)"
        print(f"\n--- Cell {i} [code] {first_line}")
        try:
            exec(compile(src, f"{nb_path.name}:cell{i}", "exec"), g)
            cell_results.append((i, "OK", None))
            print("  => OK")
        except Exception as exc:
            cell_results.append((i, "FAIL", str(exc)))
            print(f"  => FAIL: {exc}")
            traceback.print_exc()
            return cell_results
    return cell_results


def main() -> int:
    notebooks = [
        ("PHASE 1", ROOT / "notebooks" / "phase1_hoang_intent_routing_research.ipynb"),
        ("PHASE 2", ROOT / "notebooks" / "phase2_san_semantic_retrieval_research.ipynb"),
        ("PHASE 3", ROOT / "notebooks" / "phase3_hoang_grounded_output_research.ipynb"),
    ]
    all_results: dict[str, list[tuple[int, str, str | None]]] = {}
    for label, path in notebooks:
        all_results[label] = run_notebook(path, label)
        if any(status == "FAIL" for _, status, _ in all_results[label]):
            break
        print()

    print("\n" + "=" * 70)
    print("SUMMARY")
    exit_code = 0
    for label, results in all_results.items():
        ok = sum(1 for _, status, _ in results if status == "OK")
        fails = [i for i, status, _ in results if status == "FAIL"]
        print(f"{label}: {ok} code cells OK, fails: {fails}")
        if fails:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
