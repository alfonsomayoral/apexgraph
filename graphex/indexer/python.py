"""Static Python indexer built on the stdlib :mod:`ast` module.

Walks a single ``.py`` file into a graphify-compatible ``(nodes, edges)`` pair:

- one **module** node for the file,
- a **class** node per ``class`` definition,
- a **function** node per ``def`` / ``async def`` (methods nest under their class),
- an **import** node per imported name, with an ``imports_from`` edge.

Lexical scoping is preserved with ``contains`` edges from each enclosing scope
(module or class) to the symbols defined directly inside it. Node IDs follow the
graphify convention shared across all indexers (see :mod:`graphex.indexer.python`'s
``_make_ids`` helpers): a module id is ``{parent_dir}_{stem}`` and a symbol id is
``{module_id}_{symbol_name}``, both normalized to ``[a-z0-9_]``.
"""

from __future__ import annotations

import ast
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared ID / path helpers (the graphify-compatible naming convention)
# ---------------------------------------------------------------------------


def normalize_id(name: str) -> str:
    """Lower-case ``name`` and replace every non ``[a-z0-9_]`` char with ``_``.

    This is the single normalization rule every indexer applies to both module
    and symbol ids so statically indexed graphs line up with graphify graphs.
    """
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())


def module_id_for(path: Path) -> str:
    """Compute the module/file node id for ``path``.

    The id is ``{immediate_parent_dir}_{filename_stem}`` (only ONE level of
    parent dir; a top-level file uses just its stem), normalized to ``[a-z0-9_]``.
    """
    stem = path.stem
    parent = path.parent.name
    if parent:
        return normalize_id(f"{parent}_{stem}")
    return normalize_id(stem)


def symbol_id_for(module_id: str, symbol_name: str) -> str:
    """Compute a symbol node id as ``{module_id}_{symbol_name}`` (normalized)."""
    return normalize_id(f"{module_id}_{symbol_name}")


def relative_source(path: Path, root: Path | None) -> str:
    """Return the ``source_file`` string: path relative to ``root`` when possible."""
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def make_node(
    node_id: str,
    label: str,
    node_type: str,
    description: str,
    source_file: str,
    lineno: int,
) -> dict:
    """Build a graphify-format node dict (always ``file_type`` ``"code"``)."""
    return {
        "id": node_id,
        "label": label,
        "type": node_type,
        "description": description,
        "source_file": source_file,
        "source_location": f"L{lineno}",
        "file_type": "code",
    }


def make_edge(source: str, target: str, relation: str) -> dict:
    """Build a graphify-format edge dict."""
    return {"source": source, "target": target, "relation": relation}


# ---------------------------------------------------------------------------
# Python indexer
# ---------------------------------------------------------------------------


def index_python(path: Path, root: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Statically index a single Python file into ``(nodes, edges)``.

    Emits a module node, then walks classes, functions and imports, wiring
    ``contains`` edges from each enclosing scope to its members and
    ``imports_from`` edges from the module to each imported name.

    On a syntax error (unparseable source) or an OS error (unreadable file)
    this returns ``([], [])`` rather than raising, so a single bad file never
    aborts a whole-project index.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return [], []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []

    source_file = relative_source(path, root)
    module_id = module_id_for(path)

    nodes: list[dict] = [
        make_node(
            module_id,
            path.name,
            "module",
            f"Python module {source_file}",
            source_file,
            1,
        )
    ]
    edges: list[dict] = []
    seen_imports: set[str] = set()

    def emit_import(name: str, lineno: int) -> None:
        """Emit a (deduplicated) import node + ``imports_from`` edge."""
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

    def walk(scope_node: ast.AST, scope_id: str) -> None:
        """Recurse the direct children of ``scope_node`` (a module or class body)."""
        for child in ast.iter_child_nodes(scope_node):
            if isinstance(child, ast.ClassDef):
                class_id = symbol_id_for(module_id, child.name)
                nodes.append(
                    make_node(
                        class_id,
                        child.name,
                        "class",
                        f"Class {child.name}",
                        source_file,
                        child.lineno,
                    )
                )
                edges.append(make_edge(scope_id, class_id, "contains"))
                walk(child, class_id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_id = symbol_id_for(module_id, child.name)
                nodes.append(
                    make_node(
                        func_id,
                        child.name,
                        "function",
                        f"Function {child.name}",
                        source_file,
                        child.lineno,
                    )
                )
                edges.append(make_edge(scope_id, func_id, "contains"))
                walk(child, func_id)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    emit_import(alias.asname or alias.name, child.lineno)
            elif isinstance(child, ast.ImportFrom):
                base = child.module or ""
                for alias in child.names:
                    name = alias.asname or alias.name
                    full = f"{base}.{name}" if base else name
                    emit_import(full, child.lineno)

    walk(tree, module_id)
    return nodes, edges
