from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
from .config import Settings
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

SEED_INTENT_TRAINING_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "engine failure after takeoff with emergency return",
        "Incident_Report",
    ),
    (
        "compressor stall during climb and return to departure airport",
        "Incident_Report",
    ),
    (
        "runway incursion during taxi with aircraft conflict",
        "Incident_Report",
    ),
    (
        "smoke in cabin diversion and emergency landing",
        "Incident_Report",
    ),
    (
        "loss of separation on final approach",
        "Incident_Report",
    ),
    (
        "hard landing after unstable approach",
        "Incident_Report",
    ),
    (
        "qrh checklist for engine fire warning",
        "Technical_Procedure",
    ),
    (
        "maintenance troubleshooting procedure for hydraulic leak",
        "Technical_Procedure",
    ),
    (
        "minimum equipment list deferral and logbook procedure",
        "Technical_Procedure",
    ),
    (
        "component failure checklist after warning message",
        "Technical_Procedure",
    ),
    (
        "aircraft manual procedure for autopilot disconnect",
        "Technical_Procedure",
    ),
    (
        "maintenance release procedure for deferred item",
        "Technical_Procedure",
    ),
    (
        "den bao engine oil press sang thi lam gi",
        "Technical_Procedure",
    ),
    (
        "den bao engine oil pressure sang thi lam gi",
        "Technical_Procedure",
    ),
    (
        "can lam gi khi engine oil press warning",
        "Technical_Procedure",
    ),
    (
        "xu ly canh bao hydraulic pressure",
        "Technical_Procedure",
    ),
    (
        "what should the crew do after engine oil pressure warning",
        "Technical_Procedure",
    ),
    (
        "crosswind turbulence on final approach",
        "Metadata_Query",
    ),
    (
        "runway condition and airport visibility during landing",
        "Metadata_Query",
    ),
    (
        "icing weather condition during descent",
        "Metadata_Query",
    ),
    (
        "airport wind shear report and weather metadata",
        "Metadata_Query",
    ),
    (
        "snow rain fog visibility at departure airport",
        "Metadata_Query",
    ),
    (
        "altitude flight condition and runway metadata",
        "Metadata_Query",
    ),
    (
        "what is the meaning of MEL in aviation",
        "Factoid",
    ),
    (
        "what does QRH stand for",
        "Factoid",
    ),
    (
        "define runway incursion",
        "Factoid",
    ),
    (
        "what is TCAS",
        "Factoid",
    ),
    (
        "what does ILS mean",
        "Factoid",
    ),
    (
        "which airport code is JFK",
        "Factoid",
    ),
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


class Phase1HoangIntentRouting:
    def __init__(self, settings: Settings):
        from .phase1_intent_training import load_or_train_intent_model

        self.settings = settings
        mode = (settings.input_intent_mode or "auto").lower().strip()
        if mode not in {"ml", "auto", "heuristic"}:
            raise ValueError("INPUT_INTENT_MODE must be one of: ml, auto, heuristic.")
        self.intent_model = load_or_train_intent_model(settings)
        self.intent_training_mode = "tfidf_logistic_regression"
        self.intent_training_report = self.intent_model.training_report or {}

    def predict_intent(self, query_raw: str) -> Tuple[str, float, str]:
        from .phase1_intent_training import preprocess_for_intent_ml

        mode = (self.settings.input_intent_mode or "auto").lower().strip()
        normalized = normalize_text(query_raw)

        if mode == "heuristic":
            label = heuristic_intent(normalized)
            return label, 1.0, "heuristic"

        ml_text = preprocess_for_intent_ml(query_raw, use_stemming=self.settings.phase1_use_stemming)
        ml_label, ml_confidence = self.intent_model.predict(ml_text)
        if ml_label not in INTENT_LABELS:
            raise RuntimeError(f"Phase 1 intent model returned unknown label: {ml_label}")

        if mode == "ml":
            return ml_label, ml_confidence, "ml"

        heuristic_label = heuristic_intent(normalized)
        threshold = float(self.settings.phase1_ml_confidence_threshold)
        if ml_label == heuristic_label:
            return ml_label, ml_confidence, "ml"
        if ml_confidence < threshold:
            return heuristic_label, max(ml_confidence, threshold), "heuristic"
        return ml_label, ml_confidence, "ml"

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
