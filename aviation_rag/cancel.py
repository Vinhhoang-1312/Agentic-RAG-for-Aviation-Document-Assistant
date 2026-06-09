from __future__ import annotations

from typing import Any


class AnalysisCancelled(RuntimeError):
    """Raised when a user cancels an in-flight analysis run."""


def is_cancelled(cancel_event: Any | None) -> bool:
    return bool(cancel_event is not None and hasattr(cancel_event, "is_set") and cancel_event.is_set())


def raise_if_cancelled(cancel_event: Any | None) -> None:
    if is_cancelled(cancel_event):
        raise AnalysisCancelled("Analyze run was cancelled by the user.")
