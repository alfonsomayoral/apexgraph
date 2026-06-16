"""Tests for :mod:`graphex.formatter`."""

from __future__ import annotations

import json

import pytest

from graphex.formatter import format_subgraph
from graphex.models import Edge, KnowledgeGraph, Node


def _build_graph() -> KnowledgeGraph:
    """A tiny two-node subgraph with one edge and one node carrying code."""
    kg = KnowledgeGraph()
    kg.add_node(
        Node(
            id="a",
            label="recalcPlayerStats",
            type="function",
            description="Recomputes a player's running stats.",
            importance=0.9,
            source_file="game/stats.py",
            source_location="L10",
        )
    )
    kg.add_node(
        Node(
            id="b",
            label="Player",
            type="class",
            description="A player entity.",
            importance=0.5,
            source_file="game/player.ts",
            source_location="L3",
        )
    )
    kg.add_edge(Edge(source="a", target="b", relation="reads"))
    # Simulate post-injection code attachment on node "a".
    kg.node("a")["code_block"] = "def recalc():\n    return 1"
    return kg


def _stats() -> dict[str, object]:
    return {
        "nodes_selected": 2,
        "nodes_total": 5,
        "tokens_used": 120,
        "tokens_budget": 500,
        "coverage_pct": 24,
    }


def test_markdown_header_box_and_subtitle() -> None:
    out = format_subgraph(_build_graph(), _stats(), query="how are stats computed")
    assert "┌" in out and "┐" in out and "└" in out and "┘" in out
    assert "Graphex subgraph for: how are stats computed" in out
    assert "Selected 2/5 nodes (24%) · 120/500 tokens" in out


def test_markdown_node_headings_and_scores() -> None:
    scores = {"a": 0.87, "b": 0.12}
    out = format_subgraph(_build_graph(), _stats(), scores=scores, query="q")
    assert "## Relevant Nodes" in out
    assert "### recalcPlayerStats (function) · score: 0.87" in out
    assert "### Player (class) · score: 0.12" in out
    # Higher score must come first.
    assert out.index("recalcPlayerStats") < out.index("Player")


def test_markdown_file_line_and_code_fence() -> None:
    out = format_subgraph(_build_graph(), _stats(), query="q")
    assert "→ File: game/stats.py" in out
    # Code fence with inferred python language for the .py node.
    assert "```python" in out
    assert "def recalc():" in out


def test_markdown_relationships_and_exclusion_tip() -> None:
    out = format_subgraph(_build_graph(), _stats(), query="q")
    assert "## Key Relationships" in out
    assert "- a → reads → b" in out
    # 5 total vs 2 selected → tip about excluded nodes.
    assert "excluded" in out


def test_markdown_no_tip_when_all_selected() -> None:
    stats = _stats()
    stats["nodes_total"] = 2
    out = format_subgraph(_build_graph(), stats, query="q")
    assert "excluded" not in out


def test_json_parses_and_has_expected_keys() -> None:
    scores = {"a": 0.876543, "b": 0.1}
    out = format_subgraph(_build_graph(), _stats(), format="json", scores=scores)
    data = json.loads(out)
    assert set(data.keys()) == {"stats", "nodes", "edges"}
    assert data["stats"]["nodes_selected"] == 2
    node_a = next(n for n in data["nodes"] if n["id"] == "a")
    assert node_a["label"] == "recalcPlayerStats"
    assert node_a["file_path"] == "game/stats.py"
    # Score rounded to 4 decimals.
    assert node_a["score"] == 0.8765
    edge = data["edges"][0]
    assert edge["source"] == "a" and edge["target"] == "b"
    assert edge["relation"] == "reads"


def test_json_omits_score_without_scores() -> None:
    out = format_subgraph(_build_graph(), _stats(), format="json")
    data = json.loads(out)
    assert all("score" not in n for n in data["nodes"])


def test_yaml_is_parseable_enough() -> None:
    out = format_subgraph(_build_graph(), _stats(), format="yaml")
    lines = out.splitlines()
    assert "stats:" in lines
    assert "nodes:" in lines
    assert "edges:" in lines
    assert "  nodes_selected: 2" in lines
    # A node id appears as a sequence item.
    assert any(line.strip().startswith("- id: a") for line in lines)
    # Description with a trailing period contains no YAML metachar besides the
    # period, so it stays unquoted; the label is a bare identifier.
    assert any("label: recalcPlayerStats" in line for line in lines)


def test_yaml_quotes_scalars_needing_it() -> None:
    kg = KnowledgeGraph()
    kg.add_node(
        Node(
            id="x",
            label="weird: value",  # contains a colon -> must be quoted
            type="concept",
            description="123 leading digit",  # leading digit -> must be quoted
        )
    )
    out = format_subgraph(kg, _stats(), format="yaml")
    assert '"weird: value"' in out
    assert '"123 leading digit"' in out


def test_unknown_format_raises() -> None:
    with pytest.raises(ValueError):
        format_subgraph(_build_graph(), _stats(), format="xml")
