"""Selection + honest token accounting, including the never-overflow invariant."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from graphex.budget import (
    _NODE_OVERHEAD_TOKENS,
    _node_body,
    count_tokens,
    select_subgraph,
)
from graphex.models import Edge, KnowledgeGraph, Node


def _chain_graph(n: int = 12) -> KnowledgeGraph:
    g = KnowledgeGraph()
    for i in range(n):
        g.add_node(
            Node(
                id=f"n{i}",
                label=f"node{i}",
                type="function",
                file_type="code",
                description=f"does thing number {i} with widgets and gadgets",
                community=i % 3,
            )
        )
    for i in range(n - 1):
        g.add_edge(Edge(source=f"n{i}", target=f"n{i+1}", relation="calls"))
    return g


def test_count_tokens_basic():
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


def test_never_exceeds_budget():
    g = _chain_graph(20)
    scores = {nid: 1.0 - i * 0.01 for i, nid in enumerate(g.node_ids)}
    for budget in (10, 25, 50, 100, 300, 1000):
        _sub, stats = select_subgraph(g, scores, budget=budget)
        assert stats["tokens_used"] <= budget, f"overflow at budget={budget}"
        assert stats["tokens_budget"] == budget


def test_empty_and_degenerate():
    empty = KnowledgeGraph()
    sub, stats = select_subgraph(empty, {}, budget=100)
    assert stats["nodes_selected"] == 0 and len(sub) == 0

    g = _chain_graph(3)
    _sub, stats = select_subgraph(g, {n: 1.0 for n in g.node_ids}, budget=0)
    assert stats["nodes_selected"] == 0


def test_min_score_filters_candidates():
    g = _chain_graph(6)
    scores = {nid: (0.9 if i < 2 else 0.01) for i, nid in enumerate(g.node_ids)}
    sub, _stats = select_subgraph(g, scores, budget=10_000, min_score=0.5)
    assert set(sub.node_ids) <= {"n0", "n1"}


def test_higher_scores_preferred_under_tight_budget():
    g = _chain_graph(10)
    # n7 is by far the most relevant; a tight budget should still include it.
    scores = {nid: 0.1 for nid in g.node_ids}
    scores["n7"] = 1.0
    sub, _stats = select_subgraph(
        g, scores, budget=40, redundancy_weight=0.0, connectivity_bonus=0.0
    )
    assert "n7" in sub.node_ids


def test_selected_subgraph_is_induced():
    g = _chain_graph(8)
    scores = {nid: 1.0 for nid in g.node_ids}
    sub, stats = select_subgraph(g, scores, budget=120)
    # Every edge in the subgraph connects two selected nodes.
    keep = set(sub.node_ids)
    for u, v in sub.digraph.edges:
        assert u in keep and v in keep
    assert stats["coverage_pct"] == round(len(keep) / 8 * 100, 1)


def test_inject_code_counted_in_budget(tmp_path: Path):
    src = tmp_path / "mod.py"
    src.write_text(
        "def big_function():\n" + "".join(f"    x{i} = {i}\n" for i in range(40)),
        encoding="utf-8",
    )
    g = KnowledgeGraph()
    g.add_node(
        Node(
            id="mod_big_function",
            label="big_function",
            type="function",
            file_type="code",
            description="a large function",
            source_file="mod.py",
            source_location="L1",
        )
    )
    scores = {"mod_big_function": 1.0}

    # With code injected, the body is large; a tiny budget must reject it honestly
    # rather than render 40 lines of code for "free".
    plain_cost = (
        count_tokens("mod_big_function big_function (function) a large function")
        + _NODE_OVERHEAD_TOKENS
    )
    sub, stats = select_subgraph(
        g, scores, budget=plain_cost + 5, inject_code=True, project_root=tmp_path
    )
    assert stats["nodes_selected"] == 0  # code body doesn't fit → not selected

    sub, stats = select_subgraph(g, scores, budget=10_000, inject_code=True, project_root=tmp_path)
    assert stats["tokens_used"] <= 10_000
    assert "code_block" in sub.digraph.nodes["mod_big_function"]


def test_exact_strategy_matches_or_beats_greedy_on_relevance():
    g = _chain_graph(10)
    scores = {nid: (i + 1) / 10 for i, nid in enumerate(g.node_ids)}
    sub_g, stats_g = select_subgraph(g, scores, budget=120, strategy="greedy")
    sub_e, stats_e = select_subgraph(g, scores, budget=120, strategy="exact")
    val_g = sum(scores[n] for n in sub_g.node_ids)
    val_e = sum(scores[n] for n in sub_e.node_ids)
    assert stats_e["tokens_used"] <= 120
    # Exact maximises pure relevance, so it can never be beaten on that metric.
    assert val_e >= val_g - 1e-9


@pytest.mark.parametrize("strategy", ["greedy", "exact"])
def test_strategies_respect_budget(strategy):
    g = _chain_graph(15)
    scores = {nid: 1.0 for nid in g.node_ids}
    _sub, stats = select_subgraph(g, scores, budget=200, strategy=strategy)
    assert stats["tokens_used"] <= 200


def _base_costs(g: KnowledgeGraph) -> dict[str, int]:
    """Mimic the cache layer: base cost per node (body WITHOUT code + overhead)."""
    return {
        nid: count_tokens(_node_body(g, nid, None)) + _NODE_OVERHEAD_TOKENS for nid in g.node_ids
    }


def test_precomputed_token_costs_match_inline():
    g = _chain_graph(12)
    scores = {nid: 1.0 - i * 0.02 for i, nid in enumerate(g.node_ids)}
    costs = _base_costs(g)

    for budget in (30, 60, 120, 300):
        sub_inline, stats_inline = select_subgraph(g, scores, budget=budget)
        sub_pre, stats_pre = select_subgraph(g, scores, budget=budget, token_costs=costs)
        assert set(sub_inline.node_ids) == set(sub_pre.node_ids), f"budget={budget}"
        assert stats_inline["tokens_used"] == stats_pre["tokens_used"]
        assert stats_pre["tokens_used"] <= budget


def _bridge_graph() -> KnowledgeGraph:
    """Two high-score nodes (n0, n2) joined only through a cheap bridge (n1)."""
    g = KnowledgeGraph()
    for i in range(3):
        g.add_node(
            Node(
                id=f"n{i}",
                label=f"node{i}",
                type="function",
                file_type="code",
                description=f"thing {i}",
                community=i,
            )
        )
    g.add_edge(Edge(source="n0", target="n1", relation="calls"))
    g.add_edge(Edge(source="n1", target="n2", relation="calls"))
    return g


def test_connected_repair_adds_bridge_when_budget_allows():
    g = _bridge_graph()
    # n0 and n2 are the high-value picks; n1 scores below min_score so it is NOT a
    # candidate — only the connectivity repair can pull it in as a bridge.
    scores = {"n0": 1.0, "n1": 0.0, "n2": 1.0}

    # Sanity: without repair, greedy selects only the two disconnected hubs.
    sub_no, _ = select_subgraph(
        g,
        scores,
        budget=10_000,
        redundancy_weight=0.0,
        connectivity_bonus=0.0,
        connected=False,
    )
    assert set(sub_no.node_ids) == {"n0", "n2"}
    assert not nx.is_weakly_connected(sub_no.digraph)

    # Big budget: repair must add n1 to stitch the two components together.
    sub, stats = select_subgraph(
        g,
        scores,
        budget=10_000,
        redundancy_weight=0.0,
        connectivity_bonus=0.0,
        connected=True,
    )
    assert {"n0", "n2"} <= set(sub.node_ids)
    assert "n1" in sub.node_ids  # bridge added by Steiner repair
    assert nx.is_weakly_connected(sub.digraph)
    assert stats["tokens_used"] <= 10_000


def test_connected_repair_never_overflows_when_bridge_does_not_fit():
    g = _bridge_graph()
    scores = {"n0": 1.0, "n1": 0.0, "n2": 1.0}
    base = _base_costs(g)
    # Budget covers exactly n0 + n2, leaving no room for the n1 bridge.
    budget = base["n0"] + base["n2"]
    sub, stats = select_subgraph(
        g,
        scores,
        budget=budget,
        redundancy_weight=0.0,
        connectivity_bonus=0.0,
        connected=True,
    )
    assert stats["tokens_used"] <= budget  # never overflows
    assert "n1" not in sub.node_ids  # bridge didn't fit → stays disconnected
    comps = nx.number_connected_components(sub.digraph.to_undirected())
    assert comps == 2


def test_budget_invariant_across_connected_and_precomputed():
    g = _chain_graph(14)
    scores = {nid: 1.0 - i * 0.02 for i, nid in enumerate(g.node_ids)}
    costs = _base_costs(g)
    for budget in (20, 50, 120, 400):
        for connected in (False, True):
            for tc in (None, costs):
                _sub, stats = select_subgraph(
                    g, scores, budget=budget, connected=connected, token_costs=tc
                )
                assert (
                    stats["tokens_used"] <= budget
                ), f"overflow budget={budget} connected={connected} tc={tc is not None}"
