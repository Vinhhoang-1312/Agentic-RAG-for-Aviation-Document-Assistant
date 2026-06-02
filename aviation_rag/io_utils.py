from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pydantic import BaseModel


def append_jsonl(path: Path, row: Dict[str, Any] | BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any] | BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_by_query_id(path: Path, query_id: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("query_id") == query_id:
                return row
    raise ValueError(f"query_id '{query_id}' not found in {path}")

