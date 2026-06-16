"""Tests for :mod:`graphex.diff`."""

from __future__ import annotations

from graphex.diff import GraphDiff, affected_subgraph, diff_graphs, format_diff
from graphex.models import Edge, KnowledgeGraph, Node


def _old() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(Node(id="a", label="Alpha", type="class", description="first"))
    kg.add_node(Node(id="b", label="Beta", type="function", description="second"))
    kg.add_node(Node(id="c", label="Gamma", type="function", description="third"))
    kg.add_edge(Edge(source="a", target="b", relation="calls"))
    kg.add_edge(Edge(source="b", target="c", relation="calls"))
    return kg


def _new() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    # "a" unchanged.
    kg.add_node(Node(id="a", label="Alpha", type="class", description="first"))
    # "b" modified (description changed).
    kg.add_node(Node(id="b", label="Beta", type="function", description="second, revised"))
    # "c" removed; "d" added.
    kg.add_node(Node(id="d", label="Delta", type="function", description="fourth"))
    kg.add_edge(Edge(source="a", target="b", relation="calls"))
    # b->c removed; b->d added.
    kg.add_edge(Edge(source="b", target="d", relation="calls"))
    return kg


def test_diff_detects_added_removed_modified_nodes() -> None:
    diff = diff_graphs(_old(), _new())
    assert diff.added_nodes == ["d"]
    assert diff.removed_nodes == ["c"]
    assert diff.modified_nodes == ["b"]


def test_diff_detects_added_and_removed_edges() -> None:
    diff = diff_graphs(_old(), _new())
    assert ("b", "d") in diff.added_edges
    assert ("b", "c") in diff.removed_edges
    # The a->b edge is unchanged, so it appears in neither list.
    assert ("a", "b") not in diff.added_edges
    assert ("a", "b") not in diff.removed_edges


def test_unchanged_node_is_not_modified() -> None:
    diff = diff_graphs(_old(), _new())
    assert "a" not in diff.modified_nodes


def test_format_diff_contains_counts_and_labels() -> None:
    diff = diff_graphs(_old(), _new())
    out = format_diff(diff, _new())
    assert "+1 added / -1 removed / ~1 modified" in out
    # Labels resolved from the new graph.
    assert "Delta" in out
    assert "Beta" in out


def _line_graph() -> KnowledgeGraph:
    """A path a-b-c-d-e-f so we can test hop distance precisely."""
    kg = KnowledgeGraph()
    for nid in ("a", "b", "c", "d", "e", "f"):
        kg.add_node(Node(id=nid, label=nid.upper(), type="node"))
    kg.add_edge(Edge(source="a", target="b"))
    kg.add_edge(Edge(source="b", target="c"))
    kg.add_edge(Edge(source="c", target="d"))
    kg.add_edge(Edge(source="d", target="e"))
    kg.add_edge(Edge(source="e", target="f"))
    return kg


def test_affected_subgraph_includes_changed_node_and_neighborhood() -> None:
    new = _line_graph()
    # Pretend "c" was modified.
    diff = GraphDiff(modified_nodes=["c"])
    sub = affected_subgraph(new, diff, hops=2)
    ids = set(sub.node_ids)
    # The changed node and its 2-hop undirected neighborhood: a, b, c, d, e.
    assert {"a", "b", "c", "d", "e"} <= ids
    # "f" is 3 hops from "c" and must be excluded.
    assert "f" not in ids


def test_affected_subgraph_zero_hops_is_just_seeds() -> None:
    new = _line_graph()
    diff = GraphDiff(added_nodes=["c"], modified_nodes=["e"])
    sub = affected_subgraph(new, diff, hops=0)
    assert set(sub.node_ids) == {"c", "e"}


def test_affected_subgraph_walks_both_directions() -> None:
    new = _line_graph()
    # "d" modified; 1 hop should reach predecessor "c" and successor "e".
    diff = GraphDiff(modified_nodes=["d"])
    sub = affected_subgraph(new, diff, hops=1)
    assert set(sub.node_ids) == {"c", "d", "e"}
