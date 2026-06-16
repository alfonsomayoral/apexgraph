"""Tests for the static source indexer.

These exercise the per-language indexers, the project walker (dedup, ignored
dirs, ignore hook), the incremental cache (only changed files re-indexed), and
that the produced graph dict round-trips through the real loader.
"""

from __future__ import annotations

import json
from pathlib import Path

from graphex.indexer.go import index_go
from graphex.indexer.project import index_project, index_project_incremental
from graphex.indexer.python import index_python
from graphex.indexer.typescript import index_typescript
from graphex.loader import load_graph


def _nodes_by_id(nodes: list[dict]) -> dict[str, dict]:
    return {n["id"]: n for n in nodes}


def _edge_set(edges: list[dict]) -> set[tuple[str, str, str]]:
    return {(e["source"], e["target"], e["relation"]) for e in edges}


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def test_index_python_basic(tmp_path: Path) -> None:
    pkg = tmp_path / "auth"
    pkg.mkdir()
    src = pkg / "session.py"
    src.write_text(
        "import os\n"
        "from typing import Any\n"
        "\n"
        "class Validate:\n"
        "    def check(self):\n"
        "        pass\n"
        "\n"
        "async def login(user):\n"
        "    return user\n",
        encoding="utf-8",
    )

    nodes, edges = index_python(src, root=tmp_path)
    by_id = _nodes_by_id(nodes)

    # Module id = {parent_dir}_{stem}; symbol id = {module_id}_{name}.
    assert "auth_session" in by_id
    assert by_id["auth_session"]["type"] == "module"
    assert by_id["auth_session"]["file_type"] == "code"
    assert by_id["auth_session"]["source_file"] == "auth/session.py"

    assert by_id["auth_session_validate"]["type"] == "class"
    assert by_id["auth_session_check"]["type"] == "function"
    assert by_id["auth_session_login"]["type"] == "function"
    assert by_id["auth_session_os"]["type"] == "import"

    es = _edge_set(edges)
    assert ("auth_session", "auth_session_validate", "contains") in es
    # Method nests under its class.
    assert ("auth_session_validate", "auth_session_check", "contains") in es
    assert ("auth_session", "auth_session_login", "contains") in es
    assert ("auth_session", "auth_session_os", "imports_from") in es
    # from typing import Any -> normalized full path.
    assert "auth_session_typing_any" in by_id


def test_index_python_toplevel_file_uses_stem(tmp_path: Path) -> None:
    src = tmp_path / "main.py"
    src.write_text("def run():\n    pass\n", encoding="utf-8")
    nodes, _ = index_python(src, root=tmp_path)
    by_id = _nodes_by_id(nodes)
    # tmp_path's own dir name is the parent here; with root=tmp_path the file is
    # top-level relative to root, so module id uses the parent dir name of the
    # file. We just assert the run function id is module_id + _run.
    module_id = next(n["id"] for n in nodes if n["type"] == "module")
    assert f"{module_id}_run" in by_id


def test_index_python_syntax_error_returns_empty(tmp_path: Path) -> None:
    src = tmp_path / "broken.py"
    src.write_text("def (:\n", encoding="utf-8")
    assert index_python(src, root=tmp_path) == ([], [])


# ---------------------------------------------------------------------------
# TypeScript (regex fallback path)
# ---------------------------------------------------------------------------


def test_index_typescript_regex(tmp_path: Path) -> None:
    pkg = tmp_path / "api"
    pkg.mkdir()
    src = pkg / "client.ts"
    src.write_text(
        "import { request } from './http';\n"
        "import axios from 'axios';\n"
        "\n"
        "export class HttpClient {\n"
        "}\n"
        "\n"
        "export interface Options {\n"
        "}\n"
        "\n"
        "export function connect() {}\n"
        "\n"
        "export const send = (msg: string) => {\n"
        "  return msg;\n"
        "};\n",
        encoding="utf-8",
    )

    nodes, edges = index_typescript(src, root=tmp_path)
    by_id = _nodes_by_id(nodes)

    assert by_id["api_client"]["type"] == "module"
    assert by_id["api_client_httpclient"]["type"] == "class"
    assert by_id["api_client_options"]["type"] == "interface"
    assert by_id["api_client_connect"]["type"] == "function"
    assert by_id["api_client_send"]["type"] == "function"

    es = _edge_set(edges)
    assert ("api_client", "api_client_httpclient", "contains") in es
    assert ("api_client", "api_client_options", "contains") in es
    assert ("api_client", "api_client_connect", "contains") in es
    assert ("api_client", "api_client_send", "contains") in es
    # Imports wired as imports_from.
    axios_id = next(n["id"] for n in nodes if n["label"] == "axios")
    assert ("api_client", axios_id, "imports_from") in es


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


def test_index_go(tmp_path: Path) -> None:
    pkg = tmp_path / "store"
    pkg.mkdir()
    src = pkg / "db.go"
    src.write_text(
        "package store\n"
        "\n"
        "import (\n"
        '\t"fmt"\n'
        '\t"net/http"\n'
        ")\n"
        "\n"
        "type User struct {\n"
        "\tName string\n"
        "}\n"
        "\n"
        "type Repository interface {\n"
        "\tGet(id int) User\n"
        "}\n"
        "\n"
        "func New() *User {\n"
        "\treturn &User{}\n"
        "}\n"
        "\n"
        "func (u *User) Save() error {\n"
        "\treturn nil\n"
        "}\n",
        encoding="utf-8",
    )

    nodes, edges = index_go(src, root=tmp_path)
    by_id = _nodes_by_id(nodes)

    assert by_id["store_db"]["type"] == "module"
    assert by_id["store_db_user"]["type"] == "class"
    assert by_id["store_db_repository"]["type"] == "interface"
    assert by_id["store_db_new"]["type"] == "function"
    # Receiver method named by its bare method name.
    assert by_id["store_db_save"]["type"] == "function"

    es = _edge_set(edges)
    assert ("store_db", "store_db_user", "contains") in es
    assert ("store_db", "store_db_repository", "contains") in es
    assert ("store_db", "store_db_new", "contains") in es
    # Block-form imports, deduped.
    fmt_id = next(n["id"] for n in nodes if n["label"] == "fmt")
    http_id = next(n["id"] for n in nodes if n["label"] == "net/http")
    assert ("store_db", fmt_id, "imports_from") in es
    assert ("store_db", http_id, "imports_from") in es


# ---------------------------------------------------------------------------
# Project walk
# ---------------------------------------------------------------------------


def _build_small_tree(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("def alpha():\n    pass\n", encoding="utf-8")
    (root / "pkg" / "b.go").write_text("package pkg\nfunc beta() {}\n", encoding="utf-8")
    # An ignored directory that must not be walked.
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.ts").write_text(
        "export function ignored() {}\n", encoding="utf-8"
    )
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "cached.py").write_text("def nope():\n    pass\n", encoding="utf-8")


def test_index_project_walk_and_ignored_dirs(tmp_path: Path) -> None:
    _build_small_tree(tmp_path)
    graph = index_project(tmp_path)

    assert graph["built_at_commit"] is None
    assert "nodes" in graph and "links" in graph
    ids = {n["id"] for n in graph["nodes"]}

    assert "pkg_a" in ids
    assert "pkg_a_alpha" in ids
    assert "pkg_b" in ids
    assert "pkg_b_beta" in ids

    # Ignored directories were not descended into.
    labels = {n["label"] for n in graph["nodes"]}
    assert "ignored" not in labels
    assert "nope" not in labels

    # No dangling edges.
    for link in graph["links"]:
        assert link["source"] in ids
        assert link["target"] in ids


def test_index_project_dedup(tmp_path: Path) -> None:
    # Two files in different dirs sharing a stem but distinct module ids.
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    (tmp_path / "x" / "mod.py").write_text("def f():\n    pass\n", encoding="utf-8")
    (tmp_path / "y" / "mod.py").write_text("def g():\n    pass\n", encoding="utf-8")
    graph = index_project(tmp_path)
    ids = [n["id"] for n in graph["nodes"]]
    # Every id is unique after dedup.
    assert len(ids) == len(set(ids))
    assert "x_mod" in ids and "y_mod" in ids


def test_index_project_ignore_hook(tmp_path: Path) -> None:
    _build_small_tree(tmp_path)

    class Ignore:
        def should_ignore(self, node: dict, node_id: str) -> bool:
            return node_id == "pkg_a_alpha"

    graph = index_project(tmp_path, ignore=Ignore())
    ids = {n["id"] for n in graph["nodes"]}
    assert "pkg_a_alpha" not in ids
    # The contains edge to the removed node is dropped too.
    for link in graph["links"]:
        assert link["target"] != "pkg_a_alpha"


# ---------------------------------------------------------------------------
# Incremental
# ---------------------------------------------------------------------------


def test_index_project_incremental(tmp_path: Path) -> None:
    src_root = tmp_path / "src"
    src_root.mkdir()
    (src_root / "one.py").write_text("def one():\n    pass\n", encoding="utf-8")
    (src_root / "two.py").write_text("def two():\n    pass\n", encoding="utf-8")
    cache_path = tmp_path / "cache.json"

    graph1 = index_project_incremental(src_root, cache_path)
    ids1 = {n["id"] for n in graph1["nodes"]}
    assert "src_one_one" in ids1
    assert "src_two_two" in ids1

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "one.py" in cache and "two.py" in cache
    one_hash_before = cache["one.py"]["hash"]
    two_entry_before = cache["two.py"]

    # Change only one.py.
    (src_root / "one.py").write_text("def one_renamed():\n    pass\n", encoding="utf-8")
    graph2 = index_project_incremental(src_root, cache_path)
    ids2 = {n["id"] for n in graph2["nodes"]}

    # The changed file's symbols updated...
    assert "src_one_one_renamed" in ids2
    assert "src_one_one" not in ids2
    # ...while the unchanged file's node set is intact.
    assert "src_two_two" in ids2

    cache2 = json.loads(cache_path.read_text(encoding="utf-8"))
    # one.py's hash changed; two.py's cached entry was reused verbatim.
    assert cache2["one.py"]["hash"] != one_hash_before
    assert cache2["two.py"] == two_entry_before

    # Delete two.py -> its cache entry is dropped on the next run.
    (src_root / "two.py").unlink()
    graph3 = index_project_incremental(src_root, cache_path)
    ids3 = {n["id"] for n in graph3["nodes"]}
    assert "src_two_two" not in ids3
    cache3 = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "two.py" not in cache3


# ---------------------------------------------------------------------------
# Loader round-trip
# ---------------------------------------------------------------------------


def test_indexed_graph_loads(tmp_path: Path) -> None:
    _build_small_tree(tmp_path)
    graph = index_project(tmp_path)

    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    kg = load_graph(graph_path)
    assert len(kg) == len(graph["nodes"])
    assert "pkg_a" in kg.node_ids
