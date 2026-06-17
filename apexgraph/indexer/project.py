"""Project-level orchestration: walk a tree and assemble one graphify graph.

Dispatches each source file to the right per-language indexer, then stitches the
results into a single graphify-compatible dict (``nodes`` + ``links``):

- nodes are deduplicated by id (first writer wins),
- edges whose endpoints aren't both real nodes are dropped,
- duplicate edges are collapsed.

:func:`index_project` does a full walk; :func:`index_project_incremental` keeps a
per-file content-hash sidecar so a re-run only re-indexes the files that changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from apexgraph.indexer.go import index_go
from apexgraph.indexer.python import index_python
from apexgraph.indexer.typescript import index_typescript

# Extension → indexer. The callable signature is
# ``(path: Path, root: Path | None, strict_ids: bool) -> tuple[list[dict], list[dict]]``.
_INDEXED_EXTENSIONS: dict[str, Callable[..., tuple[list[dict], list[dict]]]] = {
    ".py": index_python,
    ".ts": index_typescript,
    ".tsx": index_typescript,
    ".js": index_typescript,
    ".jsx": index_typescript,
    ".go": index_go,
}

# Directory names never descended into during a walk.
_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".apexgraph",
    }
)


def _is_ignored_dir(name: str) -> bool:
    """True if a directory ``name`` should be skipped (incl. ``*.egg-info``)."""
    return name in _IGNORED_DIRS or name.endswith(".egg-info")


def _iter_source_files(root: Path) -> list[Path]:
    """Yield indexable files under ``root``, skipping ignored dirs and symlinks.

    Symlinks (to files or directories) are not followed: this both avoids cycles
    and prevents escaping ``root`` via a link pointing elsewhere.
    """
    results: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_symlink():
                # Never traverse or index through a symlink — it may point
                # outside ``root``.
                continue
            if entry.is_dir():
                if _is_ignored_dir(entry.name):
                    continue
                walk(entry)
            elif entry.is_file() and entry.suffix.lower() in _INDEXED_EXTENSIONS:
                results.append(entry)

    walk(root)
    return results


def _assemble(
    raw_nodes: list[dict],
    raw_edges: list[dict],
    ignore: Any | None,
) -> dict:
    """Dedup nodes/edges, drop dangling edges, apply ``ignore``; return the graph.

    Args:
        raw_nodes: all node dicts from every file (may contain duplicates).
        raw_edges: all edge dicts from every file (may contain duplicates).
        ignore: optional duck-typed object exposing
            ``should_ignore(node, node_id) -> bool``; matching nodes (and any
            edge touching them) are dropped.
    """
    nodes_by_id: dict[str, dict] = {}
    for node in raw_nodes:
        node_id = node["id"]
        if node_id in nodes_by_id:
            # First writer wins, but a clash between two *different* source files
            # means the id scheme collapsed two distinct symbols — warn instead of
            # silently dropping one (its source coordinates would be wrong).
            kept = nodes_by_id[node_id].get("source_file")
            dropped = node.get("source_file")
            if kept and dropped and kept != dropped:
                print(
                    f"warning: node id {node_id!r} collides across "
                    f"{kept!r} and {dropped!r}; keeping the first.",
                    file=sys.stderr,
                )
            continue
        if ignore is not None and ignore.should_ignore(node, node_id):
            continue
        nodes_by_id[node_id] = node

    known = set(nodes_by_id)
    seen_edges: set[tuple[str, str, str]] = set()
    links: list[dict] = []
    for edge in raw_edges:
        source = edge["source"]
        target = edge["target"]
        if source not in known or target not in known:
            continue
        key = (source, target, edge.get("relation", ""))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        links.append(edge)

    return {
        "nodes": list(nodes_by_id.values()),
        "links": links,
        "built_at_commit": None,
    }


def index_project(root: Path, ignore: Any | None = None, strict_ids: bool = False) -> dict:
    """Statically index an entire source tree into a graphify-compatible dict.

    Walks ``root`` (skipping the standard ignore set and symlinks), runs the
    matching indexer per file, then deduplicates and prunes via
    :func:`_assemble`.

    Args:
        root: tree to index.
        ignore: optional object with ``should_ignore(node, node_id) -> bool`` to
            filter nodes out of the result.
        strict_ids: when true, indexers emit collision-free ids (full-path module
            ids and scope-qualified symbol ids); the default is the unchanged
            graphify-compatible scheme.

    Returns:
        ``{"nodes": [...], "links": [...], "built_at_commit": None}``.
    """
    root = Path(root)
    raw_nodes: list[dict] = []
    raw_edges: list[dict] = []
    for path in _iter_source_files(root):
        indexer = _INDEXED_EXTENSIONS[path.suffix.lower()]
        nodes, edges = indexer(path, root, strict_ids)
        raw_nodes.extend(nodes)
        raw_edges.extend(edges)
    return _assemble(raw_nodes, raw_edges, ignore)


# ---------------------------------------------------------------------------
# Incremental indexing
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str | None:
    """Return the sha256 hex digest of ``path``'s bytes, or ``None`` if unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, dict]:
    """Load the sidecar cache (``source_file -> {hash, nodes, edges}``).

    A missing or corrupt cache is treated as empty so a stale sidecar never
    blocks an index.
    """
    try:
        text = cache_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def index_project_incremental(
    root: Path, cache_path: Path, ignore: Any | None = None, strict_ids: bool = False
) -> dict:
    """Incrementally index ``root``, reusing cached results for unchanged files.

    A sidecar JSON at ``cache_path`` maps ``source_file -> {hash, nodes, edges}``.
    On each run a file is re-indexed only when its sha256 content hash differs
    from the cached one (or it is new); unchanged files reuse their cached
    nodes/edges, and entries for deleted files are dropped. The sidecar is
    rewritten with the current state. The returned dict has the same shape as
    :func:`index_project`.

    Args:
        root: tree to index.
        cache_path: location of the sidecar JSON (created if absent).
        ignore: optional ``should_ignore(node, node_id)`` filter, applied to the
            assembled result (cached nodes are filtered too).
        strict_ids: forwarded to the per-file indexers for newly indexed files
            (see :func:`index_project`). Keep a cache per id-scheme: mixing strict
            and default runs against one sidecar would reuse mismatched ids.
    """
    root = Path(root)
    cache_path = Path(cache_path)
    old_cache = _load_cache(cache_path)
    new_cache: dict[str, dict] = {}

    raw_nodes: list[dict] = []
    raw_edges: list[dict] = []

    for path in _iter_source_files(root):
        source_file = path.relative_to(root).as_posix() if _under(path, root) else path.as_posix()
        file_hash = _hash_file(path)
        cached = old_cache.get(source_file)

        if (
            cached is not None
            and file_hash is not None
            and cached.get("hash") == file_hash
            and "nodes" in cached
            and "edges" in cached
        ):
            nodes = cached["nodes"]
            edges = cached["edges"]
        else:
            indexer = _INDEXED_EXTENSIONS[path.suffix.lower()]
            nodes, edges = indexer(path, root, strict_ids)

        new_cache[source_file] = {"hash": file_hash, "nodes": nodes, "edges": edges}
        raw_nodes.extend(nodes)
        raw_edges.extend(edges)

    # Entries for deleted files are simply absent from ``new_cache``.
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(new_cache), encoding="utf-8")
    except OSError:
        pass

    return _assemble(raw_nodes, raw_edges, ignore)


def _under(path: Path, root: Path) -> bool:
    """True if ``path`` is located within ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
