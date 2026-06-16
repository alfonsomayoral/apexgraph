"""Tests for the BM25 lexical retriever (:mod:`graphex.retrieval.bm25`)."""

from __future__ import annotations

import math

from graphex.models import KnowledgeGraph, Node
from graphex.retrieval.base import Retriever
from graphex.retrieval.bm25 import BM25Index, BM25Retriever, tokenize

# -- tokenizer ---------------------------------------------------------------


def test_tokenize_camel_case_splits_and_preserves_compound() -> None:
    tokens = tokenize("recalcularPlayerStats")
    # Split parts come first, in reading order.
    assert tokens[:3] == ["recalcular", "player", "stats"]
    # The original compound token is preserved (lowercased) and appended.
    assert "recalcularplayerstats" in tokens


def test_tokenize_pascal_and_acronym_boundaries() -> None:
    assert tokenize("HTTPServerError")[:3] == ["http", "server", "error"]
    assert tokenize("PlayerStats")[:2] == ["player", "stats"]


def test_tokenize_snake_kebab_dotted() -> None:
    assert tokenize("recalcular_player_stats")[:3] == ["recalcular", "player", "stats"]
    assert tokenize("recalcular-player-stats")[:3] == ["recalcular", "player", "stats"]
    assert tokenize("module.player.stats")[:3] == ["module", "player", "stats"]


def test_tokenize_compound_not_duplicated_when_already_a_part() -> None:
    # A plain word equals its own split part, so it must not appear twice.
    assert tokenize("player") == ["player"]


# -- index fixtures ----------------------------------------------------------


def _build_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_node(
        Node(
            id="n1",
            label="recalcularPlayerStats",
            type="function",
            description="recompute player statistics after a match",
        )
    )
    kg.add_node(
        Node(
            id="n2",
            label="renderScoreboard",
            type="function",
            description="draw the match scoreboard on screen",
        )
    )
    kg.add_node(
        Node(
            id="n3",
            label="loadConfig",
            type="function",
            description="read settings from disk",
        )
    )
    return kg


# -- scoring -----------------------------------------------------------------


def test_query_term_ranks_matching_node_highest() -> None:
    index = BM25Index.from_graph(_build_graph())
    scores = index.scores("player")
    assert scores["n1"] > scores["n2"]
    assert scores["n1"] > scores["n3"]


def test_non_matching_nodes_score_zero_and_every_node_present() -> None:
    index = BM25Index.from_graph(_build_graph())
    scores = index.scores("player")
    assert set(scores) == {"n1", "n2", "n3"}
    assert scores["n3"] == 0.0
    assert scores["n2"] == 0.0


def test_unknown_query_scores_all_zero() -> None:
    index = BM25Index.from_graph(_build_graph())
    scores = index.scores("nonexistentterm")
    assert all(s == 0.0 for s in scores.values())


def test_normalized_scores_top_is_one() -> None:
    index = BM25Index.from_graph(_build_graph())
    norm = index.normalized_scores("player")
    assert math.isclose(max(norm.values()), 1.0)
    assert norm["n1"] == 1.0


def test_rarer_term_scores_higher_than_common_term() -> None:
    # "match" appears in two docs (n1, n2); "scoreboard" only in n2.
    # A query of both terms should give n2 (which holds the rare term) more than
    # the IDF of the common term alone would, and the rare term must out-weight
    # the common one for the same raw frequency.
    kg = _build_graph()
    index = BM25Index.from_graph(kg)
    assert index.df["match"] == 2
    assert index.df["scoreboard"] == 1
    rare_idf = index._idf("scoreboard")
    common_idf = index._idf("match")
    assert rare_idf > common_idf


def test_seeds_normalized_distribution_and_topk() -> None:
    kg = _build_graph()
    # n4 mentions "player" once in a short doc; n5 mentions it twice. Three
    # matches total so a k=2 cut is meaningful.
    kg.add_node(Node(id="n4", label="playerCache", type="cache", description="a cache"))
    kg.add_node(Node(id="n5", label="playerPlayer", type="x", description="player player player"))
    index = BM25Index.from_graph(kg)

    raw = index.scores("player")
    seeds = index.seeds("player", k=2)
    assert len(seeds) == 2
    assert math.isclose(sum(seeds.values()), 1.0)
    # The two top-scoring nodes by raw BM25 are exactly the seeded ones.
    top2 = {nid for nid, _ in sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))[:2]}
    assert set(seeds) == top2
    # Seed weights preserve the raw ranking.
    seeded = sorted(seeds, key=lambda nid: -seeds[nid])
    assert raw[seeded[0]] >= raw[seeded[1]]


def test_seeds_empty_when_no_match() -> None:
    index = BM25Index.from_graph(_build_graph())
    assert index.seeds("nonexistentterm") == {}


# -- serialization -----------------------------------------------------------


def test_to_dict_from_dict_round_trip() -> None:
    index = BM25Index.from_graph(_build_graph())
    restored = BM25Index.from_dict(index.to_dict())

    assert restored.N == index.N
    assert math.isclose(restored.avgdl, index.avgdl)
    assert restored.df == index.df
    for query in ("player", "match", "scoreboard", "config"):
        assert restored.scores(query) == index.scores(query)


def test_to_dict_is_json_serializable() -> None:
    import json

    index = BM25Index.from_graph(_build_graph())
    payload = json.dumps(index.to_dict())
    reloaded = BM25Index.from_dict(json.loads(payload))
    assert reloaded.scores("player") == index.scores("player")


# -- retriever protocol ------------------------------------------------------


def test_retriever_conforms_to_protocol() -> None:
    retriever = BM25Retriever()
    assert isinstance(retriever, Retriever)


def test_retriever_score_matches_normalized_scores() -> None:
    kg = _build_graph()
    retriever = BM25Retriever()
    scores = retriever.score(kg, "player")
    index = BM25Index.from_graph(kg)
    assert scores == index.normalized_scores("player")
    assert scores["n1"] == 1.0
