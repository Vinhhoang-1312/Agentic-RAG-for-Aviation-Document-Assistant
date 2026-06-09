from __future__ import annotations

import ast
import contextlib
import io
import json
import runpy
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "notebook_audit_report.md"


def code_with_printed_last_expr(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return source
    last_expr = tree.body[-1]
    tree.body[-1] = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Name(id="repr", ctx=ast.Load()),
                    args=[last_expr.value],
                    keywords=[],
                )
            ],
            keywords=[],
        )
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def run_code_cell(source: str, namespace: dict[str, Any]) -> dict[str, Any]:
    stdout = io.StringIO()
    started = perf_counter()
    try:
        compiled = compile(code_with_printed_last_expr(source), "<notebook-cell>", "exec")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            exec(compiled, namespace)
        return {
            "status": "ok",
            "elapsed_ms": (perf_counter() - started) * 1000,
            "stdout": stdout.getvalue().strip(),
            "error": "",
        }
    except Exception:
        return {
            "status": "error",
            "elapsed_ms": (perf_counter() - started) * 1000,
            "stdout": stdout.getvalue().strip(),
            "error": traceback.format_exc(limit=8),
        }


def audit_notebook(path: Path) -> list[dict[str, Any]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "__notebook__",
        "__file__": str(path),
    }

    def _display(obj: Any) -> None:
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

    namespace["display"] = _display
    sys.path.insert(0, str(PROJECT_ROOT))
    results: list[dict[str, Any]] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        cell_type = cell.get("cell_type", "unknown")
        if cell_type != "code":
            results.append(
                {
                    "cell": index,
                    "cell_type": cell_type,
                    "status": "skipped",
                    "elapsed_ms": 0.0,
                    "stdout": "",
                    "error": "",
                }
            )
            continue
        result = run_code_cell(source, namespace)
        result["cell"] = index
        result["cell_type"] = cell_type
        results.append(result)
        if result["status"] == "error":
            break
    return results


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Notebook Audit Report", ""]
    exit_code = 0
    for notebook_path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        lines.extend([f"## {notebook_path.name}", ""])
        results = audit_notebook(notebook_path)
        for result in results:
            elapsed = result["elapsed_ms"]
            lines.append(f"- Cell {result['cell']:02d} `{result['cell_type']}`: `{result['status']}` ({elapsed:.1f} ms)")
            if result["stdout"]:
                preview = result["stdout"][:1200]
                lines.extend(["", "```text", preview, "```", ""])
            if result["error"]:
                exit_code = 1
                lines.extend(["", "```text", result["error"], "```", ""])
        lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Notebook audit written to {REPORT_PATH.as_posix().encode('unicode_escape').decode('ascii')}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
