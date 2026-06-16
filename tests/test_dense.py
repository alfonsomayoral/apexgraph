"""Dense retriever with an injected fake embedder (no network, no API key)."""

from __future__ import annotations

import pytest

from graphex.models import KnowledgeGraph, Node
from graphex.retrieval.base import Retriever
from graphex.retrieval.dense import DenseRetriever


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Toy 3-dim embedder: counts of 'auth', 'db', 'ui' keywords per text."""
    out = []
    for t in texts:
        low = t.lower()
        out.append([float(low.count("auth")), float(low.count("db")), float(low.count("ui"))])
    return out


def test_dense_scores_match_query_axis():
    g = KnowledgeGraph()
    g.add_node(Node(id="a", label="auth service", description="auth auth login"))
    g.add_node(Node(id="b", label="db pool", description="db connections"))
    g.add_node(Node(id="c", label="ui button", description="ui widget"))
    r = DenseRetriever(embed_fn=_fake_embed)
    scores = r.score(g, "auth login")
    assert scores["a"] == max(scores.values())
    assert scores["a"] > scores["b"]


def test_dense_conforms_to_retriever_protocol():
    assert isinstance(DenseRetriever(embed_fn=_fake_embed), Retriever)


def test_dense_empty_graph():
    assert DenseRetriever(embed_fn=_fake_embed).score(KnowledgeGraph(), "q") == {}


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        DenseRetriever(backend="nope")
