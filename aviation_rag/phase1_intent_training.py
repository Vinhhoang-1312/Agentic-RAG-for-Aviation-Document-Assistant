from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from .config import Settings
from .io_utils import read_jsonl
from .phase1_hoang_intent_routing import (
    INTENT_LABELS,
    SEED_INTENT_TRAINING_EXAMPLES,
    heuristic_intent,
    normalize_text,
)

try:
    from nltk.stem import SnowballStemmer
except ModuleNotFoundError:  # pragma: no cover
    SnowballStemmer = None

_STEMMER = SnowballStemmer("english") if SnowballStemmer is not None else None
_VECTORIZER_FILE = "tfidf_vectorizer.joblib"
_CLASSIFIER_FILE = "logistic_classifier.joblib"
_PREPROCESSING_FILE = "preprocessing_pipeline.joblib"
_REPORT_FILE = "training_report.json"
_TRAINING_CORPUS_TYPE = "query_only"
_QUERY_AUGMENT_PREFIXES = (
    "find asrs reports about",
    "lookup aviation safety reports for",
    "similar incidents involving",
)
_QUERY_AUGMENT_SUFFIXES = (
    "in asrs incident reports",
    "from aviation safety database",
)


@dataclass(frozen=True)
class IntentPreprocessor:
    use_stemming: bool
    stemmer_backend: str

    def transform(self, text: str) -> str:
        return preprocess_for_intent_ml(text, use_stemming=self.use_stemming)


@dataclass
class IntentModel:
    vectorizer: TfidfVectorizer
    classifier: Any
    training_rows: int
    label_counts: dict[str, int]
    training_report: dict[str, Any] | None = None

    def predict(self, text: str) -> tuple[str, float]:
        features = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(features)[0]
        best_index = int(probabilities.argmax())
        return self.classifier.classes_[best_index], float(probabilities[best_index])


def stem_token(token: str) -> str:
    if not token:
        return token
    if _STEMMER is None:
        return token
    return _STEMMER.stem(token)


def preprocess_for_intent_ml(text: str, *, use_stemming: bool = True) -> str:
    normalized = normalize_text(text)
    if not use_stemming:
        return normalized
    return " ".join(stem_token(token) for token in normalized.split())


def _normalized_query_key(text: str) -> str:
    return preprocess_for_intent_ml(text, use_stemming=False)


def _gold_holdout_keys(gold_path: Path) -> set[str]:
    if not gold_path.exists():
        return set()
    return {
        _normalized_query_key(str(row.get("query_raw", "")))
        for row in read_jsonl(gold_path)
        if str(row.get("query_raw", "")).strip()
    }


def _append_unique_example(
    texts: list[str],
    labels: list[str],
    seen: set[str],
    *,
    query_raw: str,
    label: str,
    use_stemming: bool,
) -> None:
    if label not in INTENT_LABELS:
        return
    key = _normalized_query_key(query_raw)
    if not key or key in seen:
        return
    seen.add(key)
    texts.append(preprocess_for_intent_ml(query_raw, use_stemming=use_stemming))
    labels.append(label)


def _seed_training_rows(*, use_stemming: bool, holdout: set[str]) -> tuple[list[str], list[str], int]:
    texts: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for query_raw, label in SEED_INTENT_TRAINING_EXAMPLES:
        key = _normalized_query_key(query_raw)
        if key in holdout:
            skipped += 1
            continue
        _append_unique_example(
            texts,
            labels,
            seen,
            query_raw=query_raw,
            label=label,
            use_stemming=use_stemming,
        )
    return texts, labels, skipped


def _jsonl_training_rows(path: Path, *, holdout: set[str], use_stemming: bool) -> tuple[list[str], list[str], int]:
    texts: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    skipped = 0
    if not path.exists():
        return texts, labels, skipped

    for row in read_jsonl(path):
        query_raw = str(row.get("query_raw", "")).strip()
        label = str(row.get("intent", "")).strip()
        if not query_raw or label not in INTENT_LABELS:
            continue
        key = _normalized_query_key(query_raw)
        if key in holdout:
            skipped += 1
            continue
        _append_unique_example(
            texts,
            labels,
            seen,
            query_raw=query_raw,
            label=label,
            use_stemming=use_stemming,
        )
    return texts, labels, skipped


def _augmented_training_rows(*, holdout: set[str], use_stemming: bool) -> tuple[list[str], list[str], int]:
    texts: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for query_raw, label in SEED_INTENT_TRAINING_EXAMPLES:
        variants = [query_raw]
        for prefix in _QUERY_AUGMENT_PREFIXES:
            variants.append(f"{prefix} {query_raw}")
        for suffix in _QUERY_AUGMENT_SUFFIXES:
            variants.append(f"{query_raw} {suffix}")
        for variant in variants:
            key = _normalized_query_key(variant)
            if key in holdout:
                skipped += 1
                continue
            _append_unique_example(
                texts,
                labels,
                seen,
                query_raw=variant,
                label=label,
                use_stemming=use_stemming,
            )
    return texts, labels, skipped


def build_training_corpus(settings: Settings) -> tuple[list[str], list[str], dict[str, Any]]:
    holdout = _gold_holdout_keys(settings.phase1_gold_labels_path)
    seed_texts, seed_labels, seed_skipped = _seed_training_rows(
        use_stemming=bool(settings.phase1_use_stemming),
        holdout=holdout,
    )
    jsonl_texts, jsonl_labels, jsonl_skipped = _jsonl_training_rows(
        settings.phase1_training_queries_path,
        holdout=holdout,
        use_stemming=bool(settings.phase1_use_stemming),
    )
    aug_texts, aug_labels, aug_skipped = _augmented_training_rows(
        holdout=holdout,
        use_stemming=bool(settings.phase1_use_stemming),
    )

    texts = seed_texts + jsonl_texts + aug_texts
    labels = seed_labels + jsonl_labels + aug_labels
    metadata = {
        "type": _TRAINING_CORPUS_TYPE,
        "sources": ["seed_examples", "training_queries_jsonl", "seed_augmentation"],
        "excluded": ["asrs_narratives", "gold_eval_queries"],
        "gold_holdout_rows": len(holdout),
        "gold_holdout_skipped_rows": seed_skipped + jsonl_skipped + aug_skipped,
        "training_queries_path": str(settings.phase1_training_queries_path),
    }
    return texts, labels, metadata


def _fit_classifier(
    texts: list[str],
    labels: list[str],
    *,
    validation_split: float,
    use_stemming: bool,
    corpus_metadata: dict[str, Any],
) -> tuple[IntentModel, dict[str, Any]]:
    if len(texts) < 8:
        raise RuntimeError(
            "Cannot train Phase 1 intent model: need at least 8 query-only training rows after gold holdout."
        )
    if len(set(labels)) < 2:
        raise RuntimeError("Cannot train Phase 1 intent model because fewer than 2 labels are available.")

    label_counts = {label: labels.count(label) for label in INTENT_LABELS if labels.count(label)}
    doc_count = len(texts)
    min_df = 1
    max_df = 1.0 if doc_count < 20 else 0.95

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        max_features=60000,
    )

    if validation_split > 0.0 and doc_count >= 12 and len(set(labels)) >= 2:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=validation_split,
            random_state=42,
            stratify=labels,
        )
    else:
        train_texts, val_texts, train_labels, val_labels = texts, [], labels, []

    features_train = vectorizer.fit_transform(train_texts)
    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
    )
    classifier.fit(features_train, train_labels)

    validation_metrics: dict[str, Any] = {
        "validation_split": validation_split,
        "train_rows": len(train_texts),
        "validation_rows": len(val_texts),
        "note": "Validation is on held-out query-only training rows, not ASRS narratives.",
    }
    if val_texts:
        val_features = vectorizer.transform(val_texts)
        val_pred = classifier.predict(val_features)
        validation_metrics.update(
            {
                "validation_accuracy": round(float(accuracy_score(val_labels, val_pred)), 4),
                "validation_macro_f1": round(float(f1_score(val_labels, val_pred, average="macro", zero_division=0)), 4),
                "classification_report": classification_report(val_labels, val_pred, zero_division=0, output_dict=True),
                "confusion_matrix": confusion_matrix(val_labels, val_pred, labels=list(classifier.classes_)).tolist(),
                "confusion_labels": list(classifier.classes_),
            }
        )

    report = {
        "training_corpus": corpus_metadata,
        "preprocessing": {
            "normalize_text": True,
            "aviation_jargon_expansion": True,
            "english_stemming": bool(use_stemming),
            "stemmer_backend": "nltk_snowball_english" if _STEMMER is not None else "disabled",
        },
        "vectorizer": {"type": "TfidfVectorizer", "ngram_range": [1, 2], "max_features": 60000},
        "classifier": {"type": "OneVsRestClassifier+LogisticRegression", "class_weight": "balanced"},
        "validation_metrics": validation_metrics,
    }

    return (
        IntentModel(
            vectorizer=vectorizer,
            classifier=classifier,
            training_rows=len(texts),
            label_counts=label_counts,
            training_report=report,
        ),
        report,
    )


def evaluate_gold_labels(model: IntentModel, gold_path: Path, *, use_stemming: bool) -> dict[str, Any]:
    rows = read_jsonl(gold_path)
    if not rows:
        return {"gold_rows": 0, "gold_accuracy": None}

    eval_rows = []
    for row in rows:
        query_raw = str(row.get("query_raw", ""))
        expected = str(row.get("expected_intent", ""))
        predicted, confidence = model.predict(preprocess_for_intent_ml(query_raw, use_stemming=use_stemming))
        eval_rows.append(
            {
                "query_id": row.get("query_id"),
                "expected_intent": expected,
                "predicted_intent": predicted,
                "confidence": round(confidence, 4),
                "correct": predicted == expected,
            }
        )

    correct = sum(item["correct"] for item in eval_rows)
    return {
        "gold_path": str(gold_path),
        "gold_rows": len(eval_rows),
        "gold_accuracy": round(correct / len(eval_rows), 4),
        "gold_eval_rows": eval_rows,
        "note": "Gold queries are held out from training by exact normalized match.",
    }


def evaluate_heuristic_gold_labels(gold_path: Path) -> dict[str, Any]:
    rows = read_jsonl(gold_path)
    if not rows:
        return {"gold_rows": 0, "gold_accuracy": None}

    eval_rows = []
    for row in rows:
        query_raw = str(row.get("query_raw", ""))
        expected = str(row.get("expected_intent", ""))
        predicted = heuristic_intent(normalize_text(query_raw))
        eval_rows.append(
            {
                "query_id": row.get("query_id"),
                "expected_intent": expected,
                "predicted_intent": predicted,
                "correct": predicted == expected,
            }
        )
    correct = sum(item["correct"] for item in eval_rows)
    return {
        "gold_path": str(gold_path),
        "gold_rows": len(eval_rows),
        "gold_accuracy": round(correct / len(eval_rows), 4),
        "gold_eval_rows": eval_rows,
    }


def save_intent_model(model: IntentModel, model_dir: Path, *, use_stemming: bool) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.vectorizer, model_dir / _VECTORIZER_FILE)
    joblib.dump(model.classifier, model_dir / _CLASSIFIER_FILE)
    joblib.dump(
        IntentPreprocessor(
            use_stemming=bool(use_stemming),
            stemmer_backend="nltk_snowball_english" if _STEMMER is not None else "disabled",
        ),
        model_dir / _PREPROCESSING_FILE,
    )
    payload = {
        "training_rows": model.training_rows,
        "label_counts": model.label_counts,
        "training_report": model.training_report or {},
    }
    (model_dir / _REPORT_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_dir


def load_intent_model(model_dir: Path) -> IntentModel | None:
    vectorizer_path = model_dir / _VECTORIZER_FILE
    classifier_path = model_dir / _CLASSIFIER_FILE
    report_path = model_dir / _REPORT_FILE
    if not vectorizer_path.exists() or not classifier_path.exists():
        return None

    report_payload: dict[str, Any] = {}
    if report_path.exists():
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    return IntentModel(
        vectorizer=joblib.load(vectorizer_path),
        classifier=joblib.load(classifier_path),
        training_rows=int(report_payload.get("training_rows", 0)),
        label_counts=dict(report_payload.get("label_counts", {})),
        training_report=dict(report_payload.get("training_report", {})),
    )


def _model_needs_retrain(cached: IntentModel | None) -> bool:
    if cached is None:
        return True
    corpus = (cached.training_report or {}).get("training_corpus", {})
    if corpus.get("type") != _TRAINING_CORPUS_TYPE:
        return True
    if cached.training_rows > 500:
        return True
    return False


def train_intent_model(settings: Settings) -> IntentModel:
    texts, labels, corpus_metadata = build_training_corpus(settings)
    model, report = _fit_classifier(
        texts,
        labels,
        validation_split=float(settings.phase1_validation_split),
        use_stemming=bool(settings.phase1_use_stemming),
        corpus_metadata=corpus_metadata,
    )
    gold_metrics = evaluate_gold_labels(
        model,
        settings.phase1_gold_labels_path,
        use_stemming=bool(settings.phase1_use_stemming),
    )
    heuristic_gold = evaluate_heuristic_gold_labels(settings.phase1_gold_labels_path)
    if model.training_report is None:
        model.training_report = {}
    model.training_report["gold_metrics"] = gold_metrics
    model.training_report["heuristic_gold_metrics"] = heuristic_gold
    save_intent_model(model, settings.phase1_model_dir, use_stemming=bool(settings.phase1_use_stemming))
    return model


def load_or_train_intent_model(settings: Settings) -> IntentModel:
    if not settings.phase1_retrain:
        cached = load_intent_model(settings.phase1_model_dir)
        if cached is not None and not _model_needs_retrain(cached):
            return cached
    return train_intent_model(settings)


def training_report_summary(model: IntentModel) -> dict[str, Any]:
    report = model.training_report or {}
    validation = report.get("validation_metrics", {})
    gold = report.get("gold_metrics", {})
    heuristic_gold = report.get("heuristic_gold_metrics", {})
    corpus = report.get("training_corpus", {})
    return {
        "training_rows": model.training_rows,
        "label_counts": model.label_counts,
        "training_corpus_type": corpus.get("type"),
        "validation_accuracy": validation.get("validation_accuracy"),
        "validation_macro_f1": validation.get("validation_macro_f1"),
        "gold_accuracy": gold.get("gold_accuracy"),
        "gold_rows": gold.get("gold_rows"),
        "heuristic_gold_accuracy": heuristic_gold.get("gold_accuracy"),
        "model_dir": None,
        "preprocessing": report.get("preprocessing", {}),
    }
