"""`.graphexignore` matching and node exclusion."""

from __future__ import annotations

from pathlib import Path

from graphex.ignore import GraphexIgnore, apply_ignore, load_ignore
from graphex.models import Edge, KnowledgeGraph, Node


def _graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node(Node(id="src/main.py", label="main", source_file="src/main.py"))
    g.add_node(Node(id="node:test", label="t", source_file="tests/test_x.py"))
    g.add_node(Node(id="src/util.py", label="util", source_file="src/util.py"))
    g.add_edge(Edge(source="src/main.py", target="src/util.py"))
    g.add_edge(Edge(source="src/main.py", target="node:test"))
    return g


def test_match_by_node_id():
    ignore = GraphexIgnore.from_lines(["node:test"])
    assert ignore.should_ignore({"source_file": "tests/test_x.py"}, "node:test")
    assert not ignore.should_ignore({"source_file": "src/main.py"}, "src/main.py")


def test_match_by_source_file():
    ignore = GraphexIgnore.from_lines(["tests/"])
    # id does not match the pattern, but source_file does.
    assert ignore.should_ignore({"source_file": "tests/test_x.py"}, "node:test")


def test_glob_pattern_matches_id_and_source_file():
    ignore = GraphexIgnore.from_lines(["*.py"])
    assert ignore.should_ignore({"source_file": "src/main.py"}, "src/main.py")
    # No source_file at all but id matches the glob.
    assert ignore.should_ignore({}, "src/util.py")


def test_comments_and_blanks_ignored():
    ignore = GraphexIgnore.from_lines(["# a comment", "", "   ", "src/util.py"])
    assert ignore.should_ignore({"source_file": "src/util.py"}, "src/util.py")
    # The comment text itself is not treated as a pattern.
    assert not ignore.should_ignore({}, "# a comment")


def test_should_ignore_with_plain_graphify_node_dict():
    # Mimic a graphify node dict (plain dict with id/source_file keys).
    node_dict = {"id": "src/main.py", "source_file": "src/main.py", "type": "code"}
    ignore = GraphexIgnore.from_lines(["src/main.py"])
    assert ignore.should_ignore(node_dict, node_dict["id"])


def test_no_source_file_key_does_not_raise():
    ignore = GraphexIgnore.from_lines(["*.py"])
    # Missing source_file key entirely — must not raise, falls back to id.
    assert ignore.should_ignore({"label": "x"}, "a.py")
    assert not ignore.should_ignore({"label": "x"}, "concept")


def test_load_ignore_missing_returns_none(tmp_path: Path):
    assert load_ignore(tmp_path / "nope.graphexignore") is None


def test_load_ignore_parses_file(tmp_path: Path):
    p = tmp_path / ".graphexignore"
    p.write_text("# vendored code\nsrc/util.py\n\ntests/\n", encoding="utf-8")
    ignore = load_ignore(p)
    assert ignore is not None
    assert ignore.should_ignore({"source_file": "src/util.py"}, "src/util.py")
    assert ignore.should_ignore({"source_file": "tests/test_x.py"}, "node:test")


def test_apply_ignore_none_is_noop():
    g = _graph()
    out = apply_ignore(g, None)
    assert out is g
    assert len(out) == 3


def test_apply_ignore_drops_matching_keeps_rest_and_edges():
    g = _graph()
    ignore = GraphexIgnore.from_lines(["tests/", "src/util.py"])
    out = apply_ignore(g, ignore)
    # node:test (tests/ source_file) and src/util.py dropped; main survives.
    assert set(out.node_ids) == {"src/main.py"}
    assert len(out) == 1
    # Edges to dropped nodes are gone; the induced subgraph has none left.
    assert out.digraph.number_of_edges() == 0


def test_apply_ignore_preserves_induced_edges():
    g = _graph()
    # Only drop node:test; the main -> util edge must survive.
    ignore = GraphexIgnore.from_lines(["node:test"])
    out = apply_ignore(g, ignore)
    assert set(out.node_ids) == {"src/main.py", "src/util.py"}
    assert out.digraph.has_edge("src/main.py", "src/util.py")
    assert not out.digraph.has_edge("src/main.py", "node:test")
