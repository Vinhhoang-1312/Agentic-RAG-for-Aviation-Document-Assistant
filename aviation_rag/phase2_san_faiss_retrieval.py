from __future__ import annotations

from .config import Settings
from .phase2_retrieval_engine import Phase2IndexInfo, Phase2RetrievalEngine
from .schemas import InputAgentOutput, MiddleAgentOutput


class Phase2SanFaissRetrieval:
    """Quan San Phase 2 retrieval facade used by the LangGraph workflow.

    The class name is kept for backward compatibility with the existing tests and
    graph, but the implementation now delegates to a complete dense/BM25/hybrid
    retrieval engine.
    """

    def __init__(self, settings: Settings):
        self.engine = Phase2RetrievalEngine(settings)

    @property
    def available(self) -> bool:
        return self.engine.available

    @property
    def build_error(self) -> str | None:
        return self.engine.build_error

    @property
    def info(self) -> Phase2IndexInfo | None:
        return self.engine.info

    def retrieve(self, input_row: InputAgentOutput) -> MiddleAgentOutput:
        return self.engine.retrieve(input_row)

    def compare_strategies(self, input_row: InputAgentOutput, strategies: list[str]) -> dict[str, MiddleAgentOutput]:
        return self.engine.compare_strategies(input_row, strategies)
