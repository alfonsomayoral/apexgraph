"""An MCP stdio server exposing Apexgraph retrieval over JSON-RPC 2.0.

This module speaks the Model Context Protocol's stdio transport: newline-delimited
JSON-RPC 2.0 messages on stdin/stdout, protocol version ``2024-11-05``. It loads a
knowledge graph *once* at startup and builds the query-independent cache (BM25
index + global PageRank) *once* — every subsequent ``tools/call`` reuses that
cache, so a query costs only a BM25 lookup plus one Personalized PageRank walk
rather than re-indexing the whole graph.

Four tools are exposed:

- ``apexgraph_query`` — score → select a token-budgeted subgraph → format it.
- ``apexgraph_explain`` — focused markdown view of one node and its neighbours.
- ``apexgraph_path`` — shortest path between two nodes, rendered with labels.
- ``apexgraph_stats`` — node/edge/hyperedge/community/god-node counts.

The transport uses only the standard library (``json`` + ``sys``); there is no
MCP SDK dependency. :func:`serve` is the entry point; :func:`_handle` is the
pure dispatcher that tests drive directly without real stdio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

import networkx as nx

from apexgraph import __version__
from apexgraph.budget import select_subgraph
from apexgraph.cache import CachedArtifacts, load_or_build
from apexgraph.formatter import format_subgraph
from apexgraph.loader import ApexgraphLoadError, load_graph
from apexgraph.models import KnowledgeGraph
from apexgraph.scorer import score_nodes

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "apexgraph"

DEFAULT_BUDGET = 4000

# JSON-RPC 2.0 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "apexgraph_query",
        "description": (
            "Retrieve the highest-relevance subgraph for a natural-language query "
            "that fits within a token budget. Returns the rendered subgraph."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text query describing what to retrieve.",
                },
                "budget": {
                    "type": "integer",
                    "description": "Maximum total tokens for the rendered subgraph.",
                    "default": DEFAULT_BUDGET,
                },
                "format": {
                    "type": "string",
                    "enum": ["markdown", "json", "yaml"],
                    "description": "Output format.",
                    "default": "markdown",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "apexgraph_explain",
        "description": (
            "Explain a single node: its label, type, description and source file, "
            "plus its immediate predecessors and successors with their relations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {
                    "type": "string",
                    "description": "The id of the node to explain.",
                },
            },
            "required": ["node"],
        },
    },
    {
        "name": "apexgraph_path",
        "description": (
            "Find the shortest path between two nodes (directed first, then "
            "undirected) and render it with node labels."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source node id."},
                "target": {"type": "string", "description": "Target node id."},
            },
            "required": ["source", "target"],
        },
    },
    {
        "name": "apexgraph_stats",
        "description": (
            "Report graph-level counts: nodes, edges, hyperedges, distinct "
            "communities and god nodes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# JSON-RPC response helpers
# ---------------------------------------------------------------------------


def _result(msg_id: Any, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC success envelope."""
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error envelope."""
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _text_result(text: str) -> dict[str, Any]:
    """Wrap text in the MCP ``tools/call`` content shape."""
    return {"content": [{"type": "text", "text": text}]}


def _write(out: TextIO, payload: dict[str, Any]) -> None:
    """Write one JSON-RPC message as a newline-delimited line and flush."""
    out.write(json.dumps(payload))
    out.write("\n")
    out.flush()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


class _ToolError(Exception):
    """A user-facing tool failure carrying a JSON-RPC error code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _coerce_budget(value: Any) -> int:
    """Coerce a budget argument to a positive ``int``.

    Accepts ints and whole-number floats (e.g. ``4000.0``); rejects fractional
    floats, non-numerics, and non-positive values with :class:`_ToolError`.
    """
    if isinstance(value, bool):
        raise _ToolError(INVALID_PARAMS, "budget must be a positive integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise _ToolError(INVALID_PARAMS, "budget must be a whole number")
        value = int(value)
    if not isinstance(value, int):
        raise _ToolError(INVALID_PARAMS, "budget must be a positive integer")
    if value <= 0:
        raise _ToolError(INVALID_PARAMS, "budget must be positive")
    return value


def _tool_query(args: dict[str, Any], graph: KnowledgeGraph, cache: CachedArtifacts) -> str:
    """Run the score → select → format pipeline, reusing the prebuilt cache."""
    query = args.get("query")
    if not isinstance(query, str) or not query:
        raise _ToolError(INVALID_PARAMS, "missing required string argument 'query'")

    budget = _coerce_budget(args.get("budget", DEFAULT_BUDGET))

    fmt = args.get("format", "markdown")
    if fmt not in ("markdown", "json", "yaml"):
        raise _ToolError(INVALID_PARAMS, f"unknown format {fmt!r}; expected markdown/json/yaml")

    scores = score_nodes(graph, query, cache=cache)
    sub, stats = select_subgraph(graph, scores, budget, token_costs=cache.token_costs)
    return format_subgraph(sub, stats, format=fmt, scores=scores, query=query)


def _label_of(graph: KnowledgeGraph, node_id: str) -> str:
    """Return a node's display label, falling back to its id."""
    return str(graph.node(node_id).get("label", node_id) or node_id)


def _tool_explain(args: dict[str, Any], graph: KnowledgeGraph) -> str:
    """Render a focused markdown explanation of one node and its neighbours."""
    node_id = args.get("node")
    if not isinstance(node_id, str) or not node_id:
        raise _ToolError(INVALID_PARAMS, "missing required string argument 'node'")
    if node_id not in graph.digraph:
        raise _ToolError(INVALID_PARAMS, f"node {node_id!r} not found in graph")

    attrs = graph.node(node_id)
    label = attrs.get("label", node_id)
    node_type = attrs.get("type", "")
    lines: list[str] = [f"# {label} ({node_type})" if node_type else f"# {label}", ""]
    lines.append(f"- id: {node_id}")
    community = graph.community_of(node_id)
    if community is not None:
        lines.append(f"- community: {community}")
    if node_id in graph.god_nodes:
        lines.append("- god node: yes")
    description = attrs.get("description", "")
    if description:
        lines.append("")
        lines.append(str(description))
    source_file = attrs.get("source_file")
    if source_file:
        lines.append("")
        lines.append(f"-> File: {source_file}")

    lines.append("")
    lines.append("## Predecessors")
    preds = sorted(graph.digraph.predecessors(node_id))
    if preds:
        for pred in preds:
            relation = graph.digraph.edges[pred, node_id].get("relation", "")
            arrow = f" -[{relation}]->" if relation else " ->"
            lines.append(f"- {_label_of(graph, pred)} ({pred}){arrow} {label}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Successors")
    succs = sorted(graph.digraph.successors(node_id))
    if succs:
        for succ in succs:
            relation = graph.digraph.edges[node_id, succ].get("relation", "")
            arrow = f" -[{relation}]->" if relation else " ->"
            lines.append(f"- {label}{arrow} {_label_of(graph, succ)} ({succ})")
    else:
        lines.append("- (none)")

    return "\n".join(lines) + "\n"


def _tool_path(args: dict[str, Any], graph: KnowledgeGraph) -> str:
    """Render the shortest path between two nodes, trying undirected as a fallback."""
    source = args.get("source")
    target = args.get("target")
    if not isinstance(source, str) or not source:
        raise _ToolError(INVALID_PARAMS, "missing required string argument 'source'")
    if not isinstance(target, str) or not target:
        raise _ToolError(INVALID_PARAMS, "missing required string argument 'target'")
    if source not in graph.digraph:
        raise _ToolError(INVALID_PARAMS, f"node {source!r} not found in graph")
    if target not in graph.digraph:
        raise _ToolError(INVALID_PARAMS, f"node {target!r} not found in graph")

    digraph = graph.digraph
    try:
        path = nx.shortest_path(digraph, source, target)
        directed = True
    except nx.NetworkXNoPath:
        path = None
        directed = True
    if path is None:
        try:
            path = nx.shortest_path(digraph.to_undirected(as_view=True), source, target)
            directed = False
        except nx.NetworkXNoPath:
            return f"No path between {source!r} and {target!r}.\n"

    rendered = " -> ".join(f"{_label_of(graph, nid)} ({nid})" for nid in path)
    kind = "directed" if directed else "undirected"
    hops = len(path) - 1
    return f"Shortest {kind} path ({hops} hop(s)):\n\n{rendered}\n"


def _tool_stats(graph: KnowledgeGraph) -> str:
    """Report node/edge/hyperedge/community/god-node counts as markdown."""
    communities = len({c for c in graph.communities.values() if c is not None})
    lines = [
        "# Graph stats",
        "",
        f"- nodes: {graph.digraph.number_of_nodes()}",
        f"- edges: {graph.digraph.number_of_edges()}",
        f"- hyperedges: {len(graph.hyperedges)}",
        f"- communities: {communities}",
        f"- god nodes: {len(graph.god_nodes)}",
    ]
    return "\n".join(lines) + "\n"


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    graph: KnowledgeGraph,
    cache: CachedArtifacts,
) -> str:
    """Route a ``tools/call`` to the right implementation, returning text."""
    if name == "apexgraph_query":
        return _tool_query(args, graph, cache)
    if name == "apexgraph_explain":
        return _tool_explain(args, graph)
    if name == "apexgraph_path":
        return _tool_path(args, graph)
    if name == "apexgraph_stats":
        return _tool_stats(graph)
    raise _ToolError(INVALID_PARAMS, f"unknown tool {name!r}")


# ---------------------------------------------------------------------------
# JSON-RPC method handlers
# ---------------------------------------------------------------------------


def _handle_initialize(msg_id: Any) -> dict[str, Any]:
    return _result(
        msg_id,
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
        },
    )


def _handle_tools_call(
    msg_id: Any,
    params: dict[str, Any],
    graph: KnowledgeGraph,
    cache: CachedArtifacts,
) -> dict[str, Any]:
    name = params.get("name")
    if not isinstance(name, str) or not name:
        return _error(msg_id, INVALID_PARAMS, "tools/call requires a string 'name'")
    args = params.get("arguments") or {}
    if not isinstance(args, dict):
        return _error(msg_id, INVALID_PARAMS, "'arguments' must be an object")
    try:
        text = _dispatch_tool(name, args, graph, cache)
    except _ToolError as exc:
        return _error(msg_id, exc.code, exc.message)
    except Exception as exc:  # noqa: BLE001 — surface engine failures as JSON-RPC errors
        return _error(msg_id, INTERNAL_ERROR, f"tool {name!r} failed: {exc}")
    return _result(msg_id, _text_result(text))


def _handle(
    msg: dict[str, Any], graph: KnowledgeGraph, cache: CachedArtifacts, out: TextIO
) -> None:
    """Dispatch one parsed JSON-RPC message, writing any response to ``out``.

    Notifications (messages without an ``id``) are processed for their method but
    never produce a response, per JSON-RPC 2.0. All other messages get exactly one
    response written to ``out``.
    """
    msg_id = msg.get("id")
    is_notification = "id" not in msg

    if msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        if not is_notification:
            _write(out, _error(msg_id, INVALID_REQUEST, "invalid JSON-RPC 2.0 request"))
        return

    method = msg["method"]
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        if not is_notification:
            _write(out, _error(msg_id, INVALID_PARAMS, "'params' must be an object"))
        return

    # Notifications: act where meaningful, but never respond.
    if is_notification:
        return

    if method == "initialize":
        _write(out, _handle_initialize(msg_id))
    elif method == "ping":
        _write(out, _result(msg_id, {}))
    elif method == "tools/list":
        _write(out, _result(msg_id, {"tools": _TOOLS}))
    elif method == "tools/call":
        _write(out, _handle_tools_call(msg_id, params, graph, cache))
    else:
        _write(out, _error(msg_id, METHOD_NOT_FOUND, f"unknown method {method!r}"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(graph_path: Path, inp: TextIO | None = None, out: TextIO | None = None) -> None:
    """Run the stdio MCP server: load the graph + cache once, then loop.

    Args:
        graph_path: Path to the knowledge-graph file (``.json``/``.graphml``/``.csv``).
        inp: Input stream of newline-delimited JSON-RPC messages (defaults to stdin).
        out: Output stream for responses (defaults to stdout).

    The graph is loaded and the query-independent cache (BM25 index + global
    PageRank) built exactly once; every request reuses them. Malformed input lines
    yield a JSON-RPC parse error; blank lines are skipped.
    """
    inp = inp if inp is not None else sys.stdin
    out = out if out is not None else sys.stdout

    graph = load_graph(Path(graph_path))
    # Build the cache once, rooted at the graph's directory, and reuse it across
    # every tools/call so each query is just a BM25 lookup + one PPR walk.
    cache = load_or_build(graph, base_dir=Path(graph_path).resolve().parent)

    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _write(out, _error(None, PARSE_ERROR, "parse error: invalid JSON"))
            continue
        if not isinstance(msg, dict):
            _write(out, _error(None, INVALID_REQUEST, "invalid JSON-RPC 2.0 request"))
            continue
        _handle(msg, graph, cache, out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m apexgraph.mcp <graph-path>``."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        sys.stderr.write("usage: python -m apexgraph.mcp <graph-path>\n")
        return 2
    try:
        serve(Path(args[0]))
    except ApexgraphLoadError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
