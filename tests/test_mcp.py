"""Tests for the MCP stdio server: drive ``_handle`` and ``serve`` directly."""

from __future__ import annotations

import io
import json
from pathlib import Path

from apexgraph.cache import load_or_build
from apexgraph.mcp import _TOOLS, _handle, serve
from apexgraph.models import Edge, Hyperedge, KnowledgeGraph, Node


def _small_graph() -> KnowledgeGraph:
    """A tiny auth-flavoured graph with two communities, a god node, a hyperedge."""
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
            source_file="auth.py",
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
            id="db",
            label="ConnectionPool",
            type="class",
            file_type="code",
            description="postgres connection pooling",
            community=2,
        )
    )
    g.add_edge(Edge(source="auth", target="login", relation="calls"))
    g.add_edge(Edge(source="login", target="db", relation="uses"))
    g.add_hyperedge(Hyperedge(id="h1", nodes=["auth", "login", "db"], relation="flow"))
    return g


def _drive(msg: dict, graph: KnowledgeGraph) -> dict | None:
    """Pass one JSON-RPC message through ``_handle`` and return the parsed reply."""
    cache = load_or_build(graph, use_cache=False)
    out = io.StringIO()
    _handle(msg, graph, cache, out)
    text = out.getvalue()
    return json.loads(text) if text.strip() else None


def _req(method: str, msg_id: int = 1, **params) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}


def _call(tool: str, graph: KnowledgeGraph, **arguments) -> dict | None:
    return _drive(_req("tools/call", name=tool, arguments=arguments), graph)


# -- initialize / handshake --------------------------------------------------


def test_initialize_returns_server_info() -> None:
    reply = _drive(_req("initialize"), _small_graph())
    assert reply is not None
    info = reply["result"]["serverInfo"]
    assert info["name"] == "apexgraph"
    assert reply["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in reply["result"]["capabilities"]


def test_ping_returns_empty_result() -> None:
    reply = _drive(_req("ping"), _small_graph())
    assert reply["result"] == {}


# -- tools/list --------------------------------------------------------------


def test_tools_list_returns_all_four_tools() -> None:
    reply = _drive(_req("tools/list"), _small_graph())
    names = {t["name"] for t in reply["result"]["tools"]}
    assert names == {
        "apexgraph_query",
        "apexgraph_explain",
        "apexgraph_path",
        "apexgraph_stats",
    }
    assert len(_TOOLS) == 4


# -- apexgraph_query -----------------------------------------------------------


def test_query_returns_text_content() -> None:
    reply = _call("apexgraph_query", _small_graph(), query="authentication login")
    content = reply["result"]["content"]
    assert content[0]["type"] == "text"
    assert "AuthService" in content[0]["text"]


def test_query_missing_query_is_invalid_params() -> None:
    reply = _call("apexgraph_query", _small_graph())
    assert reply["error"]["code"] == -32602


def test_query_rejects_non_positive_budget() -> None:
    reply = _call("apexgraph_query", _small_graph(), query="auth", budget=0)
    assert reply["error"]["code"] == -32602


def test_query_coerces_whole_float_budget() -> None:
    reply = _call("apexgraph_query", _small_graph(), query="auth", budget=2000.0)
    assert "content" in reply["result"]


def test_query_json_format() -> None:
    reply = _call("apexgraph_query", _small_graph(), query="auth", format="json")
    payload = json.loads(reply["result"]["content"][0]["text"])
    assert "nodes" in payload and "stats" in payload


# -- apexgraph_stats -----------------------------------------------------------


def test_stats_reports_correct_counts() -> None:
    reply = _call("apexgraph_stats", _small_graph())
    text = reply["result"]["content"][0]["text"]
    assert "nodes: 3" in text
    assert "edges: 2" in text
    assert "hyperedges: 1" in text
    assert "communities: 2" in text
    assert "god nodes: 1" in text


# -- apexgraph_explain ---------------------------------------------------------


def test_explain_known_node() -> None:
    reply = _call("apexgraph_explain", _small_graph(), node="login")
    text = reply["result"]["content"][0]["text"]
    assert "login" in text
    # login has predecessor auth and successor db.
    assert "AuthService" in text
    assert "ConnectionPool" in text


def test_explain_unknown_node_errors() -> None:
    reply = _call("apexgraph_explain", _small_graph(), node="nope")
    assert reply["error"]["code"] == -32602


def test_explain_missing_node_arg_errors() -> None:
    reply = _call("apexgraph_explain", _small_graph())
    assert reply["error"]["code"] == -32602


# -- apexgraph_path ------------------------------------------------------------


def test_path_finds_directed_path() -> None:
    reply = _call("apexgraph_path", _small_graph(), source="auth", target="db")
    text = reply["result"]["content"][0]["text"]
    assert "AuthService" in text
    assert "ConnectionPool" in text
    assert "->" in text


def test_path_undirected_fallback() -> None:
    reply = _call("apexgraph_path", _small_graph(), source="db", target="auth")
    text = reply["result"]["content"][0]["text"]
    assert "undirected" in text


def test_path_unknown_node_errors() -> None:
    reply = _call("apexgraph_path", _small_graph(), source="auth", target="ghost")
    assert reply["error"]["code"] == -32602


# -- protocol errors ---------------------------------------------------------


def test_unknown_method_is_method_not_found() -> None:
    reply = _drive(_req("does/not/exist"), _small_graph())
    assert reply["error"]["code"] == -32601


def test_unknown_tool_errors() -> None:
    reply = _call("apexgraph_nope", _small_graph())
    assert reply["error"]["code"] == -32602


def test_notification_produces_no_response() -> None:
    # No "id" field → a notification → no reply written.
    msg = {"jsonrpc": "2.0", "method": "ping"}
    assert _drive(msg, _small_graph()) is None


def test_invalid_request_missing_method() -> None:
    reply = _drive({"jsonrpc": "2.0", "id": 7}, _small_graph())
    assert reply["error"]["code"] == -32600


# -- serve() over StringIO with a real loaded graph --------------------------


def test_serve_loads_graph_and_round_trips(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(_small_graph().to_json(), encoding="utf-8")

    requests = "\n".join(
        json.dumps(r)
        for r in (
            _req("initialize", 1),
            _req("tools/list", 2),
            _req("tools/call", 3, name="apexgraph_stats", arguments={}),
        )
    )
    out = io.StringIO()
    serve(graph_path, inp=io.StringIO(requests), out=out)

    replies = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert len(replies) == 3
    assert replies[0]["result"]["serverInfo"]["name"] == "apexgraph"
    assert len(replies[1]["result"]["tools"]) == 4
    assert "nodes: 3" in replies[2]["result"]["content"][0]["text"]


def test_serve_parse_error_on_bad_line(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(_small_graph().to_json(), encoding="utf-8")

    out = io.StringIO()
    serve(graph_path, inp=io.StringIO("{not json}\n"), out=out)
    reply = json.loads(out.getvalue().strip())
    assert reply["error"]["code"] == -32700
