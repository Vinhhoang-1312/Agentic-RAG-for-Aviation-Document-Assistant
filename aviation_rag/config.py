from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    artifacts_dir: Path = Path(__file__).resolve().parents[1] / "artifacts"
    data_path: Path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "kaggle"
        / "ASRS-clean-dataset-aviation-safety.csv"
    )
    phase1_output_path: Path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "phase1_hoang_intent_routing_output.jsonl"
    )
    phase2_output_path: Path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "phase2_san_retrieval_output.jsonl"
    )
    phase3_output_path: Path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "phase3_hoang_grounded_answer_output.jsonl"
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    langsmith_tracing: str = os.getenv("LANGSMITH_TRACING", "false")
    langsmith_api_key: str | None = os.getenv("LANGSMITH_API_KEY")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "aviation-rag-team")
    default_top_k: int = int(os.getenv("RAG_TOP_K", "10"))
    default_strategy: str = os.getenv("RAG_RETRIEVAL_STRATEGY", "hybrid")
    intent_conf_threshold: float = float(os.getenv("INTENT_CONF_THRESHOLD", "0.60"))
    input_intent_mode: str = os.getenv("INPUT_INTENT_MODE", "heuristic")
    retrieval_max_docs: int = int(os.getenv("RETRIEVAL_MAX_DOCS", "15000"))
    retrieval_tfidf_max_features: int = int(os.getenv("RETRIEVAL_TFIDF_MAX_FEATURES", "12000"))
    retrieval_svd_components: int = int(os.getenv("RETRIEVAL_SVD_COMPONENTS", "128"))


def ensure_artifact_dirs(settings: Settings) -> None:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)


def configure_tracing_env(settings: Settings) -> None:
    tracing_enabled = str(settings.langsmith_tracing).lower() == "true"
    if tracing_enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        if settings.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
