"""Local static-embedding retriever with an injected fake embedder.

No network, no model download — every test plugs a toy embedder via ``embed_fn``.
"""

from __future__ import annotations

from graphex.models import KnowledgeGraph, Node
from graphex.retrieval.base import Retriever
from graphex.retrieval.local import LocalEmbeddingRetriever


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Toy 3-dim embedder: counts of 'auth', 'db', 'ui' keywords per text."""
    out = []
    for t in texts:
        low = t.lower()
        out.append([float(low.count("auth")), float(low.count("db")), float(low.count("ui"))])
    return out


def _toy_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node(Node(id="a", label="auth service", description="auth auth login"))
    g.add_node(Node(id="b", label="db pool", description="db connections"))
    g.add_node(Node(id="c", label="ui button", description="ui widget"))
    return g


def test_local_scores_match_query_axis():
    r = LocalEmbeddingRetriever(embed_fn=_fake_embed)
    scores = r.score(_toy_graph(), "auth login")
    assert scores["a"] == max(scores.values())
    assert scores["a"] > scores["b"]


def test_local_conforms_to_retriever_protocol():
    assert isinstance(LocalEmbeddingRetriever(embed_fn=_fake_embed), Retriever)


def test_local_empty_graph():
    assert LocalEmbeddingRetriever(embed_fn=_fake_embed).score(KnowledgeGraph(), "q") == {}


def test_node_embeddings_one_vector_per_node():
    r = LocalEmbeddingRetriever(embed_fn=_fake_embed)
    g = _toy_graph()
    embs = r.node_embeddings(g)
    assert set(embs) == set(g.node_ids)
    assert all(isinstance(v, list) for v in embs.values())


def test_cached_node_embeddings_match_fresh():
    r = LocalEmbeddingRetriever(embed_fn=_fake_embed)
    g = _toy_graph()
    fresh = r.score(g, "auth login")
    cached = r.score(g, "auth login", node_embeddings=r.node_embeddings(g))
    assert cached == fresh


def test_zero_vectors_do_not_crash():
    """Query/node texts with no keyword overlap embed to zero vectors -> 0.0, no error."""
    g = KnowledgeGraph()
    g.add_node(Node(id="x", label="nothing here", description="blank"))
    r = LocalEmbeddingRetriever(embed_fn=_fake_embed)
    scores = r.score(g, "also nothing")
    assert scores == {"x": 0.0}


def test_node_embeddings_empty_graph():
    assert LocalEmbeddingRetriever(embed_fn=_fake_embed).node_embeddings(KnowledgeGraph()) == {}
