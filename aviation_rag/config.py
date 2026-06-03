from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ARTIFACTS_DIR = _DEFAULT_PROJECT_ROOT / "artifacts"
_DEFAULT_PHASE2_INDEX_DIR = _DEFAULT_ARTIFACTS_DIR / "phase2_index"


@dataclass(frozen=True)
class Settings:
    project_root: Path = _DEFAULT_PROJECT_ROOT
    artifacts_dir: Path = _DEFAULT_ARTIFACTS_DIR
    data_path: Path = (
        _DEFAULT_PROJECT_ROOT
        / "data"
        / "kaggle"
        / "ASRS-clean-dataset-aviation-safety.csv"
    )
    phase1_output_path: Path = _DEFAULT_ARTIFACTS_DIR / "phase1_hoang_intent_routing_output.jsonl"
    phase2_output_path: Path = _DEFAULT_ARTIFACTS_DIR / "phase2_san_retrieval_output.jsonl"
    phase3_output_path: Path = _DEFAULT_ARTIFACTS_DIR / "phase3_hoang_grounded_answer_output.jsonl"
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
    phase2_embedding_model: str = os.getenv(
        "PHASE2_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    phase2_chunk_tokens: int = int(os.getenv("PHASE2_CHUNK_TOKENS", "512"))
    phase2_chunk_overlap: int = int(os.getenv("PHASE2_CHUNK_OVERLAP", "50"))
    phase2_index_dir: Path = Path(os.getenv("PHASE2_INDEX_DIR", str(_DEFAULT_PHASE2_INDEX_DIR)))
    phase2_hybrid_mode: str = os.getenv("PHASE2_HYBRID_MODE", "weighted")
    phase2_semantic_batch_size: int = int(os.getenv("PHASE2_SEMANTIC_BATCH_SIZE", "64"))

    def __post_init__(self) -> None:
        if not self.phase2_index_dir.is_absolute():
            object.__setattr__(self, "phase2_index_dir", self.project_root / self.phase2_index_dir)
        if "PHASE2_INDEX_DIR" not in os.environ and self.artifacts_dir != _DEFAULT_ARTIFACTS_DIR:
            object.__setattr__(self, "phase2_index_dir", self.artifacts_dir / "phase2_index")


def ensure_artifact_dirs(settings: Settings) -> None:
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.phase2_index_dir.mkdir(parents=True, exist_ok=True)


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
