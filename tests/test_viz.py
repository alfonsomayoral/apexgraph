"""Tests for :mod:`apexgraph.viz`."""

from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path

from apexgraph.models import Edge, KnowledgeGraph, Node
from apexgraph.viz import _VIS_NETWORK_SRI, build_html, write_html


def test_sri_constant_is_well_formed() -> None:
    # A valid SHA-384 SRI is "sha384-" + base64 of a 48-byte digest. This catches
    # a truncated/garbled hash (which would silently break the CDN script) offline,
    # without a network round-trip in CI.
    assert _VIS_NETWORK_SRI.startswith("sha384-")
    digest = base64.b64decode(_VIS_NETWORK_SRI.removeprefix("sha384-"), validate=True)
    assert len(digest) == 48


def test_sri_constant_decodes() -> None:
    try:
        base64.b64decode(_VIS_NETWORK_SRI.removeprefix("sha384-"), validate=True)
    except binascii.Error:  # pragma: no cover
        raise AssertionError("SRI hash is not valid base64") from None


def _build_graph() -> KnowledgeGraph:
    """A tiny graph with two communities, a god node, and one edge."""
    kg = KnowledgeGraph()
    kg.add_node(
        Node(
            id="a",
            label="recalcPlayerStats",
            type="function",
            description="Recomputes a player's running stats.",
            importance=0.9,
            community=0,
            is_god=True,
        )
    )
    kg.add_node(
        Node(
            id="b",
            label="Player",
            type="class",
            description="A player entity.",
            importance=0.5,
            community=1,
        )
    )
    kg.add_edge(Edge(source="a", target="b", relation="reads"))
    return kg


def _stats() -> dict[str, object]:
    return {
        "nodes_selected": 2,
        "nodes_total": 5,
        "tokens_used": 120,
        "tokens_budget": 500,
        "coverage_pct": 24,
    }


def test_build_html_basic_structure() -> None:
    out = build_html(_build_graph(), _stats(), query="how are stats computed")
    assert "<html" in out
    # CDN script is pinned to an immutable version and integrity-checked (SRI).
    assert "https://unpkg.com/vis-network@10.1.0/standalone/umd/vis-network.min.js" in out
    assert 'integrity="sha384-' in out
    assert 'crossorigin="anonymous"' in out
    assert "how are stats computed" in out
    assert "recalcPlayerStats" in out
    # Stats banner subtitle is present.
    assert "Selected 2/5 nodes · 120/500 tokens (24%)" in out


def test_build_html_embeds_node_json() -> None:
    out = build_html(_build_graph(), _stats(), query="q")
    # The embedded data is parseable JSON: extract the nodes DataSet argument.
    marker = "new vis.DataSet("
    start = out.index(marker) + len(marker)
    end = out.index(");", start)
    nodes = json.loads(out[start:end])
    ids = {n["id"] for n in nodes}
    assert ids == {"a", "b"}
    god = next(n for n in nodes if n["id"] == "a")
    assert god["shape"] == "star"  # god node gets a distinct shape


def test_scores_drive_node_size() -> None:
    scores = {"a": 1.0, "b": 0.0}
    out = build_html(_build_graph(), _stats(), scores=scores, query="q")
    marker = "new vis.DataSet("
    start = out.index(marker) + len(marker)
    end = out.index(");", start)
    nodes = json.loads(out[start:end])
    by_id = {n["id"]: n for n in nodes}
    # Higher score -> larger value.
    assert by_id["a"]["value"] > by_id["b"]["value"]


def test_special_characters_are_safely_encoded() -> None:
    kg = KnowledgeGraph()
    kg.add_node(
        Node(
            id="x",
            label="</script><img src=x>",
            type="function",
            description='He said "hi" & <b>bye</b>',
        )
    )
    out = build_html(kg, _stats(), query='inject "</script>" & stuff')

    # A raw closing script tag must never appear inside the embedded data: it
    # would terminate the inline <script> early. json.dumps + the "</" escape
    # guarantee the sequence is broken up.
    body = out.split("new vis.DataSet(", 1)[1]
    assert "</script><img" not in body

    # The escaped label round-trips through JSON intact.
    marker = "new vis.DataSet("
    start = out.index(marker) + len(marker)
    end = out.index(");", start)
    nodes = json.loads(out[start:end])
    assert nodes[0]["label"] == "</script><img src=x>"
    assert nodes[0]["title"].endswith('He said "hi" & <b>bye</b>')

    # The query in the banner is HTML-escaped (no raw quote/ampersand/script).
    assert "&amp;" in out
    assert "&quot;" in out or "&#34;" in out
    assert "<script>inject" not in out


def test_empty_graph_is_valid_and_mentions_no_nodes() -> None:
    out = build_html(KnowledgeGraph(), _stats(), query="anything")
    assert "<html" in out
    assert "</html>" in out
    assert "No nodes selected" in out


def test_write_html_creates_utf8_file(tmp_path: Path) -> None:
    target = tmp_path / "graph.html"
    result = write_html(_build_graph(), _stats(), query="café ☕", path=target)
    assert result == target
    assert result.exists()
    text = result.read_text(encoding="utf-8")
    assert "café ☕" in text
    assert "<html" in text


def test_write_html_default_tempfile() -> None:
    result = write_html(_build_graph(), _stats(), query="q")
    try:
        assert result.exists()
        assert result.suffix == ".html"
        assert "<html" in result.read_text(encoding="utf-8")
    finally:
        result.unlink(missing_ok=True)
