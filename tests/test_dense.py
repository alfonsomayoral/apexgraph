"""Dense retriever with an injected fake embedder (no network, no API key)."""

from __future__ import annotations

import pytest

from apexgraph.models import KnowledgeGraph, Node
from apexgraph.retrieval.base import Retriever
from apexgraph.retrieval.dense import DenseRetriever


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


def test_voyage_backend_scores_with_injected_embedder():
    g = KnowledgeGraph()
    g.add_node(Node(id="a", label="auth service", description="auth auth login"))
    g.add_node(Node(id="b", label="db pool", description="db connections"))
    g.add_node(Node(id="c", label="ui button", description="ui widget"))
    r = DenseRetriever(backend="voyage", embed_fn=_fake_embed)
    scores = r.score(g, "auth login")
    assert scores["a"] == max(scores.values())
    assert scores["a"] > scores["b"]


def test_known_backends_accepted_with_injected_embedder():
    for backend in ("openai", "voyage"):
        DenseRetriever(backend=backend, embed_fn=_fake_embed)


def test_anthropic_backend_rejected():
    with pytest.raises(ValueError):
        DenseRetriever(backend="anthropic")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        DenseRetriever(backend="nope")
