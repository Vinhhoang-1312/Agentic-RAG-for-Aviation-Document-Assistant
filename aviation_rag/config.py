from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

load_dotenv()

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ARTIFACTS_DIR = _DEFAULT_PROJECT_ROOT / "artifacts"
_DEFAULT_PHASE2_INDEX_DIR = _DEFAULT_ARTIFACTS_DIR / "phase2_index"
_DEFAULT_OPENROUTER_MODELS = (
    "nvidia/nemotron-nano-9b-v2:free",
    "z-ai/glm-4.5-air:free",
    "openrouter/free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
)


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default


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
    route_llm_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    route_llm_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    route_llm_models: tuple[str, ...] = _csv_env("OPENROUTER_MODELS", _DEFAULT_OPENROUTER_MODELS)
    route_llm_timeout_seconds: int = int(os.getenv("ROUTE_LLM_TIMEOUT_SECONDS", "45"))
    route_llm_max_tokens: int = int(os.getenv("ROUTE_LLM_MAX_TOKENS", "700"))
    route_llm_temperature: float = float(os.getenv("ROUTE_LLM_TEMPERATURE", "0.1"))
    langsmith_tracing: str = os.getenv("LANGSMITH_TRACING", "false")
    langsmith_api_key: str | None = os.getenv("LANGSMITH_API_KEY")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "aviation-rag-team")
    default_top_k: int = int(os.getenv("RAG_TOP_K", "10"))
    default_strategy: str = os.getenv("RAG_RETRIEVAL_STRATEGY", "hybrid")
    intent_conf_threshold: float = float(os.getenv("INTENT_CONF_THRESHOLD", "0.60"))
    input_intent_mode: str = os.getenv("INPUT_INTENT_MODE", "auto")
    phase1_ml_confidence_threshold: float = float(os.getenv("PHASE1_ML_CONFIDENCE_THRESHOLD", "0.55"))
    phase1_validation_split: float = float(os.getenv("PHASE1_VALIDATION_SPLIT", "0.2"))
    phase1_use_stemming: bool = os.getenv("PHASE1_USE_STEMMING", "true").lower() == "true"
    phase1_retrain: bool = os.getenv("PHASE1_RETRAIN", "false").lower() == "true"
    phase1_model_dir: Path = _DEFAULT_ARTIFACTS_DIR / "phase1_intent_model"
    phase1_gold_labels_path: Path = _DEFAULT_PROJECT_ROOT / "data" / "phase1_intent_gold_labels.jsonl"
    phase1_training_queries_path: Path = (
        _DEFAULT_PROJECT_ROOT / "data" / "phase1_intent_training_queries.jsonl"
    )
    phase2_gold_labels_path: Path = _DEFAULT_PROJECT_ROOT / "data" / "phase2_retrieval_gold_labels.jsonl"
    phase2_gold_report_path: Path = _DEFAULT_ARTIFACTS_DIR / "phase2_retrieval_gold_report.json"
    phase3_gold_labels_path: Path = _DEFAULT_PROJECT_ROOT / "data" / "phase3_grounding_gold_labels.jsonl"
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
    settings.phase1_model_dir.mkdir(parents=True, exist_ok=True)


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
