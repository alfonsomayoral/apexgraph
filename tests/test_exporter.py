"""Tests for :mod:`graphex.exporter`."""

from __future__ import annotations

import pytest

from graphex.exporter import export_context
from graphex.models import Edge, KnowledgeGraph, Node


def _build_graph() -> KnowledgeGraph:
    """A tiny two-node subgraph with one edge."""
    kg = KnowledgeGraph()
    kg.add_node(
        Node(
            id="a",
            label="recalcPlayerStats",
            type="function",
            description="Recomputes a player's running stats.",
            source_file="game/stats.py",
        )
    )
    kg.add_node(Node(id="b", label="Player", type="class", description="A player entity."))
    kg.add_edge(Edge(source="a", target="b", relation="reads"))
    return kg


def _stats() -> dict[str, object]:
    return {
        "nodes_selected": 2,
        "nodes_total": 5,
        "tokens_used": 120,
        "tokens_budget": 500,
        "coverage_pct": 24,
    }


QUERY = "how are stats computed"


def _body_marker() -> str:
    """A fragment guaranteed to come from the formatter body, not the wrapper."""
    return "## Relevant Nodes"


def test_claude_wraps_body_and_names_query() -> None:
    out = export_context(_build_graph(), _stats(), QUERY, format="claude")
    assert "<knowledge_graph_context>" in out
    assert "</knowledge_graph_context>" in out
    assert QUERY in out
    # Body from the formatter is present inside the wrapper.
    assert _body_marker() in out
    assert "recalcPlayerStats" in out


def test_chatgpt_wraps_body_and_names_query() -> None:
    out = export_context(_build_graph(), _stats(), QUERY, format="chatgpt")
    assert "## Relevant codebase context" in out
    assert QUERY in out
    assert _body_marker() in out


def test_claudemd_wraps_body_and_names_query() -> None:
    out = export_context(_build_graph(), _stats(), QUERY, format="claudemd")
    assert "## Knowledge Graph Context" in out
    assert "graphex" in out
    assert QUERY in out
    assert _body_marker() in out


def test_default_format_is_claude() -> None:
    out = export_context(_build_graph(), _stats(), QUERY)
    assert "<knowledge_graph_context>" in out


def test_scores_flow_into_body() -> None:
    scores = {"a": 0.9, "b": 0.2}
    out = export_context(_build_graph(), _stats(), QUERY, format="claude", scores=scores)
    # The formatter surfaces scores in node headings when given.
    assert "score" in out.lower()


def test_unknown_format_raises() -> None:
    with pytest.raises(ValueError):
        export_context(_build_graph(), _stats(), QUERY, format="gemini")
