"""End-to-end scoring pipeline: BM25 seeds → PPR → fuse with prior."""

from __future__ import annotations

from graphex.models import Edge, Hyperedge, KnowledgeGraph, Node
from graphex.scorer import score_nodes, score_nodes_detailed


def _auth_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node(
        Node(
            id="auth",
            label="AuthService",
            type="class",
            file_type="code",
            description="user authentication and login",
            importance=9,
            is_god=True,
            community=1,
        )
    )
    g.add_node(
        Node(
            id="login",
            label="login",
            type="function",
            file_type="code",
            description="validate credentials, create session",
            community=1,
        )
    )
    g.add_node(
        Node(
            id="logout",
            label="logout",
            type="function",
            file_type="code",
            description="destroy session",
            community=1,
        )
    )
    g.add_node(
        Node(
            id="db",
            label="ConnectionPool",
            type="class",
            file_type="code",
            description="postgres connection pooling",
            community=2,
        )
    )
    g.add_node(
        Node(
            id="ui",
            label="Button",
            type="component",
            file_type="code",
            description="reusable ui widget",
            community=3,
        )
    )
    g.add_edge(Edge(source="auth", target="login", relation="contains"))
    g.add_edge(Edge(source="auth", target="logout", relation="contains", confidence_score=0.9))
    g.add_edge(Edge(source="login", target="db", relation="calls", confidence_score=0.8))
    return g


def test_lexical_match_wins():
    g = _auth_graph()
    scores = score_nodes(g, "user login authentication")
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked[0] in {"auth", "login"}
    assert scores["ui"] == 0.0  # no lexical match, unconnected to seeds


def test_ppr_spreads_to_neighbors():
    g = _auth_graph()
    scores = score_nodes(g, "login")
    # db is two hops from the lexical seed; still outranks the unconnected ui node.
    assert scores["db"] > scores["ui"]


def test_empty_graph():
    assert score_nodes(KnowledgeGraph(), "anything") == {}


def test_no_lexical_match_returns_zeros():
    g = _auth_graph()
    # A query with zero lexical overlap returns all zeros — the honest "nothing
    # matched" answer for an LLM-context tool, rather than confident centrality
    # noise. The downstream min_score filter then yields an empty subgraph.
    scores = score_nodes(g, "квантовая запутанность")
    assert all(v == 0.0 for v in scores.values())


def test_detailed_components_present():
    g = _auth_graph()
    bd = score_nodes_detailed(g, "login authentication")
    top = max(bd.final, key=bd.final.get)
    assert 0.0 <= bd.bm25[top] <= 1.0
    assert 0.0 <= bd.ppr[top] <= 1.0
    assert 0.0 <= bd.prior[top] <= 1.0
    assert set(bd.final) == set(g.node_ids)


def test_hyperedge_lifts_co_members():
    base = _auth_graph()
    boosted = _auth_graph()
    boosted.add_hyperedge(
        Hyperedge(
            id="h",
            nodes=["auth", "login", "logout"],
            relation="participate_in",
            confidence_score=0.9,
        )
    )
    s_base = score_nodes(base, "login")
    s_boost = score_nodes(boosted, "login")
    # logout shares a hyperedge with the lexical seed (login); the clique edge
    # should lift it relative to the no-hyperedge baseline.
    assert s_boost["logout"] >= s_base["logout"]
