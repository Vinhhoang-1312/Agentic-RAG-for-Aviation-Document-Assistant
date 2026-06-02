"""Hoang-owned aviation workflow package with phase-based contracts."""

__all__ = ["build_graph"]


def __getattr__(name: str):
    """Lazy load graph module to avoid langgraph import blocking."""
    if name == "build_graph":
        from .graph import build_graph
        return build_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
