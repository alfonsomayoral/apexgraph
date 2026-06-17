"""Retriever interface shared by every scoring backend.

A retriever turns a free-text query into a ``{node_id: score}`` mapping over a
:class:`~apexgraph.models.KnowledgeGraph`. Scores are expected to be non-negative;
callers normalise to ``[0, 1]`` where a bounded range matters.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from apexgraph.models import KnowledgeGraph


@runtime_checkable
class Retriever(Protocol):
    """Anything that scores graph nodes against a query."""

    def score(self, graph: KnowledgeGraph, query: str) -> dict[str, float]:
        """Return a relevance score for every node in ``graph``."""
        ...


def normalize(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to ``[0, 1]`` by dividing by the max. All-zero input stays zero."""
    if not scores:
        return {}
    hi = max(scores.values())
    if hi <= 0.0:
        return {k: 0.0 for k in scores}
    return {k: v / hi for k, v in scores.items()}
