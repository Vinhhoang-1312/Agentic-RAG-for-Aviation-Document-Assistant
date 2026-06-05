from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ModuleNotFoundError:
    HumanMessage = None
    SystemMessage = None
try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    ChatOpenAI = None

from .config import Settings
from .io_utils import append_jsonl
from .schemas import Citation, FinalOutput, MiddleAgentOutput

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def _clean_snippet(text: str, max_chars: int = 280) -> str:
    snippet = re.sub(r"\s+", " ", text or "").strip()
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    return snippet


class Phase3HoangGroundedQA:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _build_context(self, middle_output: MiddleAgentOutput) -> Tuple[str, List[str]]:
        lines: List[str] = []
        doc_ids: List[str] = []
        for index, doc in enumerate(middle_output.topk_docs, start=1):
            lines.append(f"[{index}] doc_id={doc.doc_id} chunk_id={doc.chunk_id}\n{doc.chunk_text}\n")
            doc_ids.append(doc.doc_id)
        return "\n".join(lines), doc_ids

    def _call_openai(self, question: str, context_block: str, doc_ids: List[str]) -> Dict[str, object]:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Phase 3 OpenAI generation requires it.")
        if ChatOpenAI is None:
            raise ModuleNotFoundError(
                "langchain_openai is not installed. Install requirements.txt dependencies to enable Phase 3 OpenAI generation."
            )
        if HumanMessage is None or SystemMessage is None:
            raise ModuleNotFoundError(
                "langchain_core is not installed. Install requirements.txt dependencies to enable Phase 3 OpenAI generation."
            )

        llm = ChatOpenAI(
            model=self.settings.openai_model,
            api_key=self.settings.openai_api_key,
            temperature=0.1,
            timeout=self.settings.openai_timeout_seconds,
            max_retries=1,
        )
        system = SystemMessage(
            content=(
                "You are Hoang's aviation grounded QA agent. "
                "Answer only with evidence from context and return JSON with keys: "
                "answer, citation_doc_ids, grounding_notes."
            )
        )
        user = HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Context:\n{context_block}\n\n"
                f"Allowed doc ids: {doc_ids}\n\n"
                "Respond as JSON only."
            )
        )
        response = llm.invoke([system, user]).content
        try:
            return json.loads(response)
        except Exception:
            return {
                "answer": str(response),
                "citation_doc_ids": doc_ids[:2],
                "grounding_notes": "Fallback parse used because model output was not valid JSON.",
            }

    def _fallback_answer(self, question: str, middle_output: MiddleAgentOutput) -> Dict[str, object]:
        evidence_lines = []
        for index, doc in enumerate(middle_output.topk_docs[:3], start=1):
            evidence_lines.append(f"{index}. Document {doc.doc_id}: {_clean_snippet(doc.chunk_text)}")
        if evidence_lines:
            answer = "\n".join(
                [
                    "Local grounded fallback: OpenAI is not configured or Fast local mode is active.",
                    f"Query: {question}",
                    "",
                    "Evidence highlights:",
                    *evidence_lines,
                    "",
                    "Review the Evidence tab before treating this as a supported research answer.",
                ]
            )
        else:
            answer = (
                "Local grounded fallback could not find retrieved evidence for this query. "
                "Try a more specific aviation incident, hazard, maintenance item, or airport condition."
            )
        return {
            "answer": answer,
            "citation_doc_ids": [doc.doc_id for doc in middle_output.topk_docs[:3]],
            "grounding_notes": "Local fallback mode used by Hoang phase 3.",
        }

    def _grounding_metrics(self, answer: str, contexts: List[str]) -> Dict[str, float]:
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(" ".join(contexts))
        overlap_ratio = 0.0
        if answer_tokens:
            overlap_ratio = len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)
        return {
            "overlap_ratio": overlap_ratio,
            "hallucination_risk": max(0.0, min(1.0, 1.0 - overlap_ratio)),
        }

    def generate(
        self,
        question: str,
        middle_output: MiddleAgentOutput,
        allow_fallback: bool = True,
        force_local: bool = False,
    ) -> FinalOutput:
        context_block, doc_ids = self._build_context(middle_output)
        if force_local:
            payload = self._fallback_answer(question, middle_output)
        elif self.settings.openai_api_key:
            try:
                payload = self._call_openai(question, context_block, doc_ids)
            except Exception:
                if allow_fallback:
                    payload = self._fallback_answer(question, middle_output)
                else:
                    raise
        elif allow_fallback:
            payload = self._fallback_answer(question, middle_output)
        else:
            raise ValueError("OPENAI_API_KEY is missing and fallback is disabled.")

        citations = [
            Citation(doc_id=doc_id, reason="Phase 3 cited this document")
            for doc_id in payload.get("citation_doc_ids", [])[:5]
        ]
        if not citations:
            citations = [
                Citation(doc_id=doc.doc_id, reason="Fallback to top-ranked evidence")
                for doc in middle_output.topk_docs[:2]
            ]

        metrics = self._grounding_metrics(
            answer=str(payload.get("answer", "")),
            contexts=[doc.chunk_text for doc in middle_output.topk_docs],
        )
        report = {
            "grounding_notes": str(payload.get("grounding_notes", "")),
            "overlap_ratio": metrics["overlap_ratio"],
            "retrieval_docs_considered": len(middle_output.topk_docs),
        }
        return FinalOutput(
            query_id=middle_output.query_id,
            answer=str(payload.get("answer", "")),
            citations=citations,
            hallucination_risk=metrics["hallucination_risk"],
            grounding_report=report,
        )

    def write_output(self, output: FinalOutput, path: Path | None = None) -> Path:
        target = path or self.settings.phase3_output_path
        append_jsonl(target, output)
        return target
