"""Tests for :mod:`apexgraph.retrieval.ppr`.

Small, hand-built :class:`KnowledgeGraph` instances with deterministic numeric
assertions (with tolerances) covering: probability mass conservation, seed
dominance, downstream propagation, weight-proportional transfer, hyperedge
clique boosting, dangling handling, global centrality, and empty/single graphs.
"""

from __future__ import annotations

from apexgraph.models import Edge, Hyperedge, KnowledgeGraph, Node
from apexgraph.retrieval.ppr import (
    global_pagerank,
    normalize_max,
    personalized_pagerank,
)

TOL = 1e-6


def _kg(node_ids: list[str]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for nid in node_ids:
        kg.add_node(Node(id=nid))
    return kg


def _edge(
    kg: KnowledgeGraph, u: str, v: str, weight: float = 1.0, confidence_score: float = 1.0
) -> None:
    kg.add_edge(Edge(source=u, target=v, weight=weight, confidence_score=confidence_score))


# -- mass conservation ------------------------------------------------------


def test_ranks_sum_to_one():
    kg = _kg(["a", "b", "c", "d"])
    _edge(kg, "a", "b")
    _edge(kg, "b", "c")
    _edge(kg, "c", "a")
    _edge(kg, "a", "d")
    ranks = personalized_pagerank(kg, {"a": 1.0})
    assert abs(sum(ranks.values()) - 1.0) < TOL
    assert set(ranks) == {"a", "b", "c", "d"}
    assert all(r >= 0.0 for r in ranks.values())


def test_global_ranks_sum_to_one():
    kg = _kg(["a", "b", "c"])
    _edge(kg, "a", "b")
    _edge(kg, "b", "c")
    ranks = global_pagerank(kg)
    assert abs(sum(ranks.values()) - 1.0) < TOL


# -- seed dominance & propagation -------------------------------------------


def test_seeded_node_ranks_highest():
    kg = _kg(["a", "b", "c", "d"])
    _edge(kg, "a", "b")
    _edge(kg, "b", "c")
    _edge(kg, "c", "d")
    ranks = personalized_pagerank(kg, {"a": 1.0})
    assert ranks["a"] == max(ranks.values())
    assert ranks["a"] > ranks["b"] > 0.0


def test_downstream_beats_unconnected():
    # a -> b ; c is isolated. Seeding a must lift b above c.
    kg = _kg(["a", "b", "c"])
    _edge(kg, "a", "b", weight=1.0, confidence_score=1.0)
    ranks = personalized_pagerank(kg, {"a": 1.0})
    assert ranks["b"] > ranks["c"]


# -- weight-proportional transfer -------------------------------------------


def test_higher_effective_weight_transfers_more_rank():
    # Two identical topologies, differing only in the a->b effective weight.
    def build(conf: float) -> dict[str, float]:
        kg = _kg(["a", "b", "c"])
        _edge(kg, "a", "b", weight=1.0, confidence_score=conf)
        _edge(kg, "a", "c", weight=1.0, confidence_score=1.0)
        return personalized_pagerank(kg, {"a": 1.0})

    low = build(0.2)
    high = build(1.0)
    # Stronger a->b edge => b receives a larger share of a's outgoing mass.
    assert high["b"] > low["b"]


# -- hyperedge clique boosting ----------------------------------------------


def test_hyperedge_boosts_co_members():
    seed = {"a": 1.0}

    # Baseline: three isolated nodes, no hyperedge.
    base = _kg(["a", "b", "c"])
    base_ranks = personalized_pagerank(base, seed)

    # With a hyperedge binding a, b, c, seeding a should lift b and c.
    with_he = _kg(["a", "b", "c"])
    with_he.add_hyperedge(Hyperedge(id="h1", nodes=["a", "b", "c"], confidence_score=1.0))
    he_ranks = personalized_pagerank(with_he, seed)

    assert he_ranks["b"] > base_ranks["b"]
    assert he_ranks["c"] > base_ranks["c"]


# -- dangling nodes ---------------------------------------------------------


def test_dangling_node_handled():
    # b has no out-edges (dangling). Must not crash; ranks stay valid.
    kg = _kg(["a", "b", "c"])
    _edge(kg, "a", "b")
    _edge(kg, "c", "a")
    ranks = personalized_pagerank(kg, {"a": 1.0})
    assert abs(sum(ranks.values()) - 1.0) < TOL
    assert all(r >= 0.0 for r in ranks.values())
    assert ranks["b"] > 0.0


def test_all_dangling_falls_back_gracefully():
    # No edges at all: every node is dangling. Personalized mass concentrates
    # on the seed via the restart term.
    kg = _kg(["a", "b", "c"])
    ranks = personalized_pagerank(kg, {"a": 1.0})
    assert abs(sum(ranks.values()) - 1.0) < TOL
    assert ranks["a"] == max(ranks.values())


# -- global centrality ------------------------------------------------------


def test_global_favors_most_connected_node():
    # Hub 'h' receives edges from a, b, c, d -> highest global rank.
    kg = _kg(["h", "a", "b", "c", "d"])
    for n in ("a", "b", "c", "d"):
        _edge(kg, n, "h")
    _edge(kg, "h", "a")
    ranks = global_pagerank(kg)
    assert ranks["h"] == max(ranks.values())
    assert ranks["h"] > ranks["b"]


# -- edge cases -------------------------------------------------------------


def test_empty_graph_returns_empty():
    kg = KnowledgeGraph()
    assert personalized_pagerank(kg, {"x": 1.0}) == {}
    assert global_pagerank(kg) == {}


def test_single_node():
    kg = _kg(["only"])
    assert personalized_pagerank(kg, {"only": 1.0}) == {"only": 1.0}
    assert global_pagerank(kg) == {"only": 1.0}


def test_seeds_not_in_graph_ignored_falls_back_to_uniform():
    # Only out-of-graph seeds => uniform restart.
    kg = _kg(["a", "b"])
    ranks = personalized_pagerank(kg, {"ghost": 1.0})
    assert abs(ranks["a"] - ranks["b"]) < 1e-9


def test_empty_seeds_uniform():
    kg = _kg(["a", "b"])
    ranks = personalized_pagerank(kg, {})
    assert abs(ranks["a"] - ranks["b"]) < 1e-9


# -- normalize_max ----------------------------------------------------------


def test_normalize_max():
    assert normalize_max({}) == {}
    assert normalize_max({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    out = normalize_max({"a": 2.0, "b": 1.0})
    assert out["a"] == 1.0
    assert abs(out["b"] - 0.5) < TOL
