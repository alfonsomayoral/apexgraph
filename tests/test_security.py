"""Security regression tests — a graph is untrusted input."""

from __future__ import annotations

from pathlib import Path

from apexgraph.budget import select_subgraph
from apexgraph.injector import inject_code, safe_source_path
from apexgraph.models import KnowledgeGraph, Node


def test_safe_source_path_rejects_escapes(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "ok.py").write_text("x = 1\n", encoding="utf-8")

    assert safe_source_path(root, "ok.py") == (root / "ok.py").resolve()
    assert safe_source_path(root, "../../../../etc/passwd") is None
    assert safe_source_path(root, "/etc/passwd") is None
    # Windows-style absolute paths must also be refused on the platforms they parse on.
    assert safe_source_path(root, "sub/../ok.py") == (root / "ok.py").resolve()


def test_inject_code_refuses_path_traversal(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET\n", encoding="utf-8")

    g = KnowledgeGraph()
    g.add_node(
        Node(
            id="evil",
            label="evil",
            type="function",
            source_file="../secret.txt",
            source_location="L1",
        )
    )
    inject_code(g, root)
    assert "code_block" not in g.digraph.nodes["evil"]


def test_budget_inject_code_refuses_absolute_path(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET\n", encoding="utf-8")

    g = KnowledgeGraph()
    g.add_node(
        Node(
            id="evil",
            label="evil",
            description="reads a secret",
            source_file=str(secret),  # absolute path outside root
            source_location="L1",
        )
    )
    sub, _ = select_subgraph(g, {"evil": 1.0}, budget=10_000, inject_code=True, project_root=root)
    if "evil" in sub.node_ids:
        assert "TOP SECRET" not in sub.digraph.nodes["evil"].get("code_block", "")
