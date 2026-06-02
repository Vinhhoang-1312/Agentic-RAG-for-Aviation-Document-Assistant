from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import Settings
from .intent_rules import map_row_to_intent
from .io_utils import append_jsonl, find_by_query_id
from .schemas import InputAgentOutput, RetrievalPlan

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
WHITESPACE_RE = re.compile(r"\s+")

INTENT_LABELS = (
    "Incident_Report",
    "Technical_Procedure",
    "Metadata_Query",
    "Factoid",
)

AVIATION_JARGON_MAP: Dict[str, str] = {
    "acft": "aircraft",
    "rwy": "runway",
    "tkof": "takeoff",
    "apch": "approach",
    "plt": "pilot",
    "ctlr": "controller",
    "degs": "degrees",
    "kts": "knots",
    "wx": "weather",
    "tfc": "traffic",
    "eng": "engine",
    "maint": "maintenance",
    "atc": "air traffic control",
    "xwind": "crosswind",
    "gnd": "ground",
    "qrh": "quick reference handbook",
}

QUERY_EXPANSION = {
    "Incident_Report": [
        "aviation incident report",
        "event narrative",
        "safety occurrence summary",
    ],
    "Technical_Procedure": [
        "maintenance checklist",
        "troubleshooting procedure",
        "quick reference handbook step",
    ],
    "Metadata_Query": [
        "weather condition metadata",
        "airport runway condition",
        "flight operating context",
    ],
    "Factoid": [
        "direct factual answer",
        "aviation reference fact",
        "short lookup answer",
    ],
}

ROUTING_POLICIES = {
    "Incident_Report": {
        "strategy": "semantic",
        "fallback_strategy": "hybrid",
        "filters": {},
        "routing_reason": "Narrative incident queries benefit from semantic similarity over safety reports.",
    },
    "Technical_Procedure": {
        "strategy": "bm25",
        "fallback_strategy": "hybrid",
        "filters": {"document_type": "procedure"},
        "routing_reason": "Procedure-style queries favor keyword-heavy checklist and manual retrieval.",
    },
    "Metadata_Query": {
        "strategy": "metadata_first",
        "fallback_strategy": "bm25",
        "filters": {"prefer_metadata": True},
        "routing_reason": "Metadata queries should prioritize structured document fields and filters first.",
    },
    "Factoid": {
        "strategy": "semantic",
        "fallback_strategy": "hybrid",
        "filters": {"answer_style": "short"},
        "routing_reason": "Factoid queries need concise semantic lookup with a hybrid fallback.",
    },
}


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    for short, full in AVIATION_JARGON_MAP.items():
        text = re.sub(rf"\b{re.escape(short)}\b", full, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def heuristic_intent(normalized_query: str) -> str:
    query = normalized_query.strip()
    tokens = tokenize(query)
    factoid_starters = ("what ", "which ", "when ", "who ", "where ", "how many ", "icao ", "iata ")
    factoid_tokens = {"what", "which", "when", "who", "where", "icao", "iata", "meaning"}
    metadata_phrases = {"weather", "crosswind", "turbulence", "icing", "runway", "visibility", "airport", "wind"}
    technical_tokens = {
        "checklist",
        "procedure",
        "maintenance",
        "manual",
        "troubleshoot",
        "component",
        "quick",
        "reference",
        "handbook",
        "mel",
    }
    technical_phrases = ("lam gi", "xu ly", "can lam", "how to", "what should", "what do i do")
    incident_phrases = (
        "after takeoff",
        "during climb",
        "emergency return",
        "returned to",
        "diversion",
        "incident",
        "event",
        "report",
        "encounter",
        "shutdown in flight",
    )
    metadata_keys = {
        "weather",
        "icing",
        "crosswind",
        "turbulence",
        "runway",
        "wind",
        "rain",
        "snow",
        "fog",
        "visibility",
        "airport",
        "altitude",
        "metadata",
        "condition",
    }

    if query.startswith(factoid_starters) or factoid_tokens.intersection(tokens):
        return "Factoid"
    if metadata_keys.intersection(tokens) or metadata_phrases.intersection(tokens):
        return "Metadata_Query"
    if technical_tokens.intersection(tokens) or "quick reference handbook" in query or any(
        phrase in query for phrase in technical_phrases
    ):
        return "Technical_Procedure"
    if any(phrase in query for phrase in incident_phrases):
        return "Incident_Report"
    return "Incident_Report"


@dataclass
class IntentModel:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression

    def predict(self, text: str) -> Tuple[str, float]:
        features = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(features)[0]
        best_index = probabilities.argmax()
        return self.classifier.classes_[best_index], float(probabilities[best_index])


class Phase1HoangIntentRouting:
    def __init__(self, settings: Settings):
        self.settings = settings
        mode = (settings.input_intent_mode or "auto").lower().strip()
        if mode == "heuristic":
            self.intent_model = None
        elif mode in {"auto", "ml"}:
            self.intent_model = self._maybe_train_intent_model(settings.data_path)
        else:
            self.intent_model = None

    def _maybe_train_intent_model(self, data_path: Path) -> IntentModel | None:
        if not data_path.exists():
            return None

        dataframe = pd.read_csv(data_path, low_memory=False)
        text_columns = [
            column
            for column in ["report_summary", "report1_narrative", "report2_narrative"]
            if column in dataframe.columns
        ]
        if not text_columns:
            return None

        texts = dataframe[text_columns].fillna("").agg(" ".join, axis=1).astype(str).map(normalize_text)
        labels = dataframe.apply(map_row_to_intent, axis=1)
        if labels.nunique() < 2:
            return None

        doc_count = len(texts)
        min_df = 1 if doc_count < 20 else 3
        max_df = 1.0 if doc_count < 20 else 0.95

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            max_features=60000,
        )
        features = vectorizer.fit_transform(texts)
        classifier = LogisticRegression(max_iter=1200, class_weight="balanced")
        classifier.fit(features, labels)
        return IntentModel(vectorizer=vectorizer, classifier=classifier)

    def predict_intent(self, query_raw: str) -> Tuple[str, float, str]:
        normalized = normalize_text(query_raw)
        heuristic_label = heuristic_intent(normalized)

        if self.intent_model is None:
            return heuristic_label, 0.60, "heuristic"

        label, confidence = self.intent_model.predict(normalized)
        if heuristic_label == "Factoid" and confidence < 0.75:
            return "Factoid", max(confidence, 0.70), "heuristic"
        if label not in INTENT_LABELS or confidence < self.settings.intent_conf_threshold:
            return heuristic_label, max(confidence, self.settings.intent_conf_threshold), "heuristic"
        return label, confidence, "ml"

    def expand_query(self, normalized_query: str, intent: str) -> List[str]:
        expansions = [normalized_query]
        expansions.extend(f"{normalized_query} {term}".strip() for term in QUERY_EXPANSION.get(intent, []))
        return list(dict.fromkeys(expansions))

    def rewrite_query(self, query_raw: str, intent: str) -> str:
        normalized = normalize_text(query_raw)
        if intent == "Technical_Procedure":
            return f"aviation troubleshooting and procedure lookup for: {normalized}"
        if intent == "Metadata_Query":
            return f"aviation metadata and operating condition lookup for: {normalized}"
        if intent == "Factoid":
            return f"direct aviation fact lookup for: {normalized}"
        return f"aviation incident narrative lookup for: {normalized}"

    def build_retrieval_plan(self, intent: str, top_k: int) -> RetrievalPlan:
        policy = ROUTING_POLICIES[intent]
        return RetrievalPlan(
            strategy=policy["strategy"],
            fallback_strategy=policy["fallback_strategy"],
            top_k=top_k,
            filters=dict(policy["filters"]),
            routing_reason=policy["routing_reason"],
        )

    def build_output(
        self,
        query_raw: str,
        query_id: str | None = None,
        top_k: int | None = None,
        strategy: str | None = None,
    ) -> InputAgentOutput:
        normalized = normalize_text(query_raw)
        intent, confidence, intent_source = self.predict_intent(query_raw)
        expansions = self.expand_query(normalized, intent)
        rewritten = self.rewrite_query(query_raw, intent)
        plan = self.build_retrieval_plan(intent, top_k or self.settings.default_top_k)
        if strategy:
            plan.strategy = strategy
            plan.routing_reason = f"Manual override requested. Original routing intent was {intent}."

        return InputAgentOutput(
            query_id=query_id or f"q_{uuid.uuid4().hex[:8]}",
            query_raw=query_raw,
            query_normalized=normalized,
            intent=intent,
            intent_confidence=confidence,
            intent_source=intent_source,
            expanded_queries=expansions,
            rewritten_query=rewritten,
            retrieval_plan=plan,
        )

    def write_output(self, output: InputAgentOutput, path: Path | None = None) -> Path:
        target = path or self.settings.phase1_output_path
        append_jsonl(target, output)
        return target

    def load_output(self, query_id: str, path: Path | None = None) -> InputAgentOutput:
        target = path or self.settings.phase1_output_path
        row = find_by_query_id(target, query_id)
        return InputAgentOutput.model_validate(row)
