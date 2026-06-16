"""Static Go indexer (regex-based).

Recovers, line by line:

- the ``package`` declaration → a **module** node for the file,
- ``func`` declarations → **function** nodes (receiver methods are recognized
  and named by their bare method name),
- ``type X struct`` → **class** nodes,
- ``type X interface`` → **interface** nodes,
- imports in both the single ``import "path"`` and the grouped
  ``import ( ... )`` block form, deduplicated, each wired with an
  ``imports_from`` edge.

A module node is always emitted first. IDs and node/edge shapes follow the
shared graphify convention from :mod:`graphex.indexer.python`.
"""

from __future__ import annotations

import re
from pathlib import Path

from graphex.indexer.python import (
    make_edge,
    make_node,
    module_id_for,
    relative_source,
    symbol_id_for,
)

_RE_PACKAGE = re.compile(r"^\s*package\s+([A-Za-z_]\w*)")
# func Name(...)  /  func (r Recv) Name(...)
_RE_FUNC = re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*\(")
_RE_TYPE_STRUCT = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+struct\b")
_RE_TYPE_INTERFACE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+interface\b")
_RE_IMPORT_SINGLE = re.compile(r"""^\s*import\s+(?:[A-Za-z_.]\w*\s+)?['"]([^'"]+)['"]""")
_RE_IMPORT_BLOCK_OPEN = re.compile(r"^\s*import\s*\(")
_RE_IMPORT_BLOCK_LINE = re.compile(r"""^\s*(?:[A-Za-z_.]\w*\s+)?['"]([^'"]+)['"]""")


def index_go(path: Path, root: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Statically index a single Go file into ``(nodes, edges)``.

    Emits a module node for the package, then function / struct / interface
    nodes (each with a ``contains`` edge from the module) and import nodes (each
    with an ``imports_from`` edge). Returns ``([], [])`` if the file is
    unreadable.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return [], []

    source_file = relative_source(path, root)
    module_id = module_id_for(path)

    nodes: list[dict] = [
        make_node(
            module_id,
            path.name,
            "module",
            f"Go module {source_file}",
            source_file,
            1,
        )
    ]
    edges: list[dict] = []
    seen_symbols: set[str] = set()
    seen_imports: set[str] = set()

    def emit_symbol(name: str, node_type: str, lineno: int) -> None:
        symbol_id = symbol_id_for(module_id, name)
        if symbol_id in seen_symbols:
            return
        seen_symbols.add(symbol_id)
        nodes.append(
            make_node(
                symbol_id,
                name,
                node_type,
                f"{node_type.capitalize()} {name}",
                source_file,
                lineno,
            )
        )
        edges.append(make_edge(module_id, symbol_id, "contains"))

    def emit_import(name: str, lineno: int) -> None:
        import_id = symbol_id_for(module_id, name)
        if import_id in seen_imports:
            return
        seen_imports.add(import_id)
        nodes.append(
            make_node(
                import_id,
                name,
                "import",
                f"Import of {name}",
                source_file,
                lineno,
            )
        )
        edges.append(make_edge(module_id, import_id, "imports_from"))

    in_import_block = False
    for index, line in enumerate(source.splitlines(), start=1):
        if in_import_block:
            if ")" in line:
                in_import_block = False
            match = _RE_IMPORT_BLOCK_LINE.match(line)
            if match:
                emit_import(match.group(1), index)
            continue

        if _RE_IMPORT_BLOCK_OPEN.match(line):
            in_import_block = True
            continue

        match = _RE_PACKAGE.match(line)
        if match:
            continue  # package name already captured by the module node

        match = _RE_TYPE_STRUCT.match(line)
        if match:
            emit_symbol(match.group(1), "class", index)
            continue
        match = _RE_TYPE_INTERFACE.match(line)
        if match:
            emit_symbol(match.group(1), "interface", index)
            continue
        match = _RE_FUNC.match(line)
        if match:
            emit_symbol(match.group(1), "function", index)
            continue
        match = _RE_IMPORT_SINGLE.match(line)
        if match:
            emit_import(match.group(1), index)

    return nodes, edges
