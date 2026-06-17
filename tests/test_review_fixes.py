"""Regression tests locking in the fixes from the adversarial review pass."""

from __future__ import annotations

import json

from click.testing import CliRunner

from apexgraph.budget import select_subgraph
from apexgraph.cli import cli
from apexgraph.loader import load_graph
from apexgraph.models import KnowledgeGraph, Node
from apexgraph.retrieval.fusion import importance_prior


def _graph(n: int = 6) -> KnowledgeGraph:
    g = KnowledgeGraph()
    for i in range(n):
        g.add_node(Node(id=f"n{i}", label=f"node {i}", description=f"thing {i}"))
    return g


# --- budget: best-single-item guarantee (density-greedy recall) -------------


def test_top_node_never_lost_to_density_greedy():
    g = KnowledgeGraph()
    # P is cheap but low value; Q is the single most relevant node and fits alone.
    g.add_node(Node(id="P", label="p", description="short"))
    g.add_node(Node(id="Q", label="q", description="a much longer description " * 6))
    scores = {"P": 0.6, "Q": 1.0}
    cost_q = select_subgraph(g, {"Q": 1.0}, budget=10_000)[1]["tokens_used"]
    sub, _ = select_subgraph(
        g, scores, budget=cost_q + 2, min_score=0.0, redundancy_weight=0.0, connectivity_bonus=0.0
    )
    assert "Q" in sub.node_ids


# --- loader: NaN / inf rejection --------------------------------------------


def test_loader_rejects_non_finite(tmp_path):
    data = {
        "nodes": [
            {"id": "a", "label": "a", "importance": "NaN"},
            {"id": "b", "label": "b", "importance": 3.0},
        ],
        "links": [{"source": "a", "target": "b", "weight": "inf"}],
    }
    p = tmp_path / "g.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    kg = load_graph(p)
    assert kg.digraph.nodes["a"]["importance"] == 0.0  # NaN → default
    assert kg.digraph.edges["a", "b"]["weight"] == 1.0  # inf → default
    prior = importance_prior(kg)
    assert all(v == v for v in prior.values())  # no NaN contamination


# --- fusion: god node doesn't erase ordinary importance ordering ------------


def test_god_node_preserves_importance_ordering():
    g = KnowledgeGraph()
    g.add_node(Node(id="god", is_god=True, importance=0.1))
    g.add_node(Node(id="a", importance=0.9))
    g.add_node(Node(id="b", importance=0.5))
    prior = importance_prior(g)
    assert prior["god"] == 1.0
    assert prior["a"] > prior["b"] > 0.0  # ordering survives, not wiped to ~0


# --- cache fingerprint: changes when node text changes ----------------------


def test_fingerprint_tracks_node_text():
    g1 = _graph(3)
    g2 = _graph(3)
    g2.digraph.nodes["n1"]["description"] = "completely different text now"
    assert g1.fingerprint() != g2.fingerprint()


# --- CLI guardrails ---------------------------------------------------------


def _write_example_graph(path: str = "graph.json") -> None:
    data = {
        "nodes": [
            {"id": "auth", "label": "auth", "description": "login authentication"},
            {"id": "db", "label": "db", "description": "database pool"},
        ],
        "links": [{"source": "auth", "target": "db"}],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_cli_rejects_empty_query():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_example_graph()
        res = runner.invoke(cli, ["", "--no-cache", "--no-audit"])
        assert res.exit_code != 0
        assert "empty" in res.output.lower()


def test_cli_rejects_nonpositive_budget():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_example_graph()
        res = runner.invoke(cli, ["auth", "-b", "0", "--no-cache", "--no-audit"])
        assert res.exit_code != 0


def test_cli_suggests_command_on_typo():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_example_graph()
        res = runner.invoke(cli, ["statss"])
        assert res.exit_code != 0
        assert "did you mean" in res.output.lower() and "stats" in res.output.lower()


def test_cli_serve_missing_graph_clean_error():
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli, ["serve", "-g", "nope.json"])
        assert res.exit_code != 0
        assert "Traceback" not in res.output


def test_cli_explain_does_not_corrupt_json():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_example_graph()
        res = runner.invoke(cli, ["auth", "-f", "json", "--explain", "--no-cache", "--no-audit"])
        assert res.exit_code == 0
        json.loads(res.output)  # must still be valid JSON despite --explain


def test_cli_no_match_message():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_example_graph()
        res = runner.invoke(cli, ["zzzz_nonexistent_term", "--no-cache", "--no-audit"])
        assert res.exit_code == 0
        assert "No nodes cleared" in res.output
