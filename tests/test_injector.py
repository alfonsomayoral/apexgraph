"""Tests for :mod:`apexgraph.injector`."""

from __future__ import annotations

from pathlib import Path

from apexgraph.injector import extract_code_block, inject_code
from apexgraph.models import KnowledgeGraph, Node

_PY_SOURCE = """\
import os


def alpha(x):
    y = x + 1
    return y


def beta():
    return 0
"""

_TS_SOURCE = """\
import { thing } from "lib";

function greet(name: string): string {
  const msg = `hi ${name}`;
  return msg;
}

const after = 1;
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_extract_python_indentation_block(tmp_path: Path) -> None:
    py = _write(tmp_path, "mod.py", _PY_SOURCE)
    # ``def alpha`` is on line 4 (1-based).
    block = extract_code_block(py, "L4")
    assert block == "def alpha(x):\n    y = x + 1\n    return y"
    # Must stop before ``def beta`` and not bleed into it.
    assert "beta" not in block


def test_extract_python_accepts_int_and_bare_string(tmp_path: Path) -> None:
    py = _write(tmp_path, "mod.py", _PY_SOURCE)
    assert extract_code_block(py, 4) == extract_code_block(py, "4")
    assert extract_code_block(py, 4) == extract_code_block(py, "L4")


def test_extract_braces_block(tmp_path: Path) -> None:
    ts = _write(tmp_path, "mod.ts", _TS_SOURCE)
    # ``function greet`` is on line 3 (1-based).
    block = extract_code_block(ts, "L3")
    assert block.startswith("function greet(name: string): string {")
    assert block.rstrip().endswith("}")
    assert "return msg;" in block
    # Stops at the closing brace, excluding the trailing ``const after``.
    assert "after" not in block


def test_extract_max_lines_cap(tmp_path: Path) -> None:
    py = _write(tmp_path, "mod.py", _PY_SOURCE)
    block = extract_code_block(py, "L4", max_lines=2)
    assert block.count("\n") == 1  # exactly 2 lines


def test_extract_returns_none_on_bad_input(tmp_path: Path) -> None:
    py = _write(tmp_path, "mod.py", _PY_SOURCE)
    assert extract_code_block(tmp_path / "missing.py", "L1") is None
    assert extract_code_block(py, None) is None
    assert extract_code_block(py, "not-a-line") is None
    assert extract_code_block(py, "L9999") is None
    assert extract_code_block(py, 0) is None


def _graph_with_nodes(n: int) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for i in range(n):
        kg.add_node(
            Node(
                id=f"n{i}",
                source_file="mod.py",
                source_location="L4",
            )
        )
    return kg


def test_inject_code_sets_code_block(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", _PY_SOURCE)
    kg = KnowledgeGraph()
    kg.add_node(Node(id="a", source_file="mod.py", source_location="L4"))
    kg.add_node(Node(id="b", source_file="mod.py", source_location="L9"))
    # Node with no location should be left alone.
    kg.add_node(Node(id="c", source_file="mod.py"))

    returned = inject_code(kg, project_root=tmp_path)
    assert returned is kg  # mutates in place, returns same object
    assert kg.node("a")["code_block"].startswith("def alpha")
    assert kg.node("b")["code_block"].startswith("def beta")
    assert "code_block" not in kg.node("c")


def test_inject_code_supports_file_path_attr(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", _PY_SOURCE)
    kg = KnowledgeGraph()
    node = Node(id="a", source_location="L4")
    node.extra["file_path"] = "mod.py"
    kg.add_node(node)
    inject_code(kg, project_root=tmp_path)
    assert kg.node("a")["code_block"].startswith("def alpha")


def test_inject_code_noop_above_max_nodes(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", _PY_SOURCE)
    kg = _graph_with_nodes(5)
    inject_code(kg, project_root=tmp_path, max_nodes=3)
    assert all("code_block" not in kg.node(nid) for nid in kg.node_ids)
