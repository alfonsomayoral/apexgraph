"""CLI smoke + behaviour tests via Click's CliRunner."""

from __future__ import annotations

import json

from click.testing import CliRunner

from apexgraph.cli import cli

_GRAPH = {
    "nodes": [
        {
            "id": "auth",
            "label": "AuthService",
            "type": "class",
            "file_type": "code",
            "description": "user authentication and login",
            "importance": 9,
            "community": 1,
        },
        {
            "id": "login",
            "label": "login",
            "type": "function",
            "file_type": "code",
            "description": "validate credentials and create a session",
            "community": 1,
        },
        {
            "id": "db",
            "label": "ConnectionPool",
            "type": "class",
            "file_type": "code",
            "description": "postgres connection pooling",
            "community": 2,
        },
    ],
    "links": [
        {"source": "auth", "target": "login", "relation": "contains"},
        {"source": "login", "target": "db", "relation": "calls", "confidence_score": 0.8},
    ],
}


def _write_graph(runner_path: str = "graph.json") -> None:
    with open(runner_path, "w", encoding="utf-8") as f:
        json.dump(_GRAPH, f)


def test_version():
    res = CliRunner().invoke(cli, ["--version"])
    assert res.exit_code == 0
    assert "apexgraph v" in res.output


def test_query_default_route_and_budget():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_graph()
        res = runner.invoke(cli, ["user login", "--budget", "300", "--no-cache", "--no-audit"])
        assert res.exit_code == 0, res.output
        assert "login" in res.output
        assert "Selected" in res.output


def test_query_json_format():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_graph()
        res = runner.invoke(cli, ["login", "-f", "json", "--no-cache", "--no-audit"])
        assert res.exit_code == 0
        payload = json.loads(res.output)
        assert "stats" in payload and "nodes" in payload


def test_explain_table_flag():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_graph()
        res = runner.invoke(cli, ["authentication", "--explain", "--no-cache", "--no-audit"])
        assert res.exit_code == 0
        assert "Score breakdown" in res.output
        assert "BM25" in res.output


def test_stats_command():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_graph()
        res = runner.invoke(cli, ["stats"])
        assert res.exit_code == 0
        assert "Nodes" in res.output and "3" in res.output


def test_explain_and_path_commands():
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_graph()
        res = runner.invoke(cli, ["explain", "login"])
        assert res.exit_code == 0 and "login" in res.output
        res = runner.invoke(cli, ["path", "auth", "db"])
        assert res.exit_code == 0
        assert "→" in res.output


def test_missing_graph_is_actionable():
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli, ["something"])
        assert res.exit_code != 0
        assert "apexgraph index" in res.output


def test_index_then_query(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        src_dir = tmp_path / "proj"
        src_dir.mkdir()
        (src_dir / "service.py").write_text(
            "def authenticate(user):\n    return True\n", encoding="utf-8"
        )
        res = runner.invoke(cli, ["index", str(src_dir), "-o", "g.json"])
        assert res.exit_code == 0, res.output
        assert "nodes" in res.output
        res = runner.invoke(cli, ["authenticate", "-g", "g.json", "--no-cache", "--no-audit"])
        assert res.exit_code == 0
        assert "authenticate" in res.output


def test_missing_backend_dep_is_clean_error(monkeypatch):
    # A semantic backend whose optional dependency isn't installed must produce a
    # clean, actionable message — not a raw traceback.
    import apexgraph.scorer as scorer

    def _boom(graph, query, backend):
        raise ImportError("The local backend requires: pip install 'apexgraph[local]'")

    monkeypatch.setattr(scorer, "_semantic_scores", _boom)
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_graph()
        res = runner.invoke(cli, ["login", "--backend", "local", "--no-cache", "--no-audit"])
        assert res.exit_code != 0
        assert "Traceback" not in res.output
        assert "apexgraph[local]" in res.output


def test_init_scaffolds_ignore():
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli, ["init"])
        assert res.exit_code == 0
        with open(".apexgraphignore", encoding="utf-8") as f:
            assert "apexgraph ignore" in f.read()
