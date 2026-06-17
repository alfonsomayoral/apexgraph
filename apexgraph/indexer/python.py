"""Static Python indexer built on the stdlib :mod:`ast` module.

Walks a single ``.py`` file into a graphify-compatible ``(nodes, edges)`` pair:

- one **module** node for the file,
- a **class** node per ``class`` definition,
- a **function** node per ``def`` / ``async def`` (methods nest under their class),
- an **import** node per imported name, with an ``imports_from`` edge.

Lexical scoping is preserved with ``contains`` edges from each enclosing scope
(module or class) to the symbols defined directly inside it. Node IDs follow the
graphify convention shared across all indexers (see :mod:`apexgraph.indexer.python`'s
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


def module_id_for(path: Path, root: Path | None = None, strict_ids: bool = False) -> str:
    """Compute the module/file node id for ``path``.

    In the default (graphify-compatible) scheme the id is
    ``{immediate_parent_dir}_{filename_stem}`` (only ONE level of parent dir; a
    top-level file uses just its stem), normalized to ``[a-z0-9_]``.

    In **strict mode** the id is the FULL path relative to ``root`` with every
    component joined by ``_`` (e.g. ``a/b/util.py`` -> ``a_b_util``), so two files
    sharing only an immediate-parent dir + stem no longer collide. When ``root``
    is unknown the path is used as-is. Both schemes drop the file extension.
    """
    if strict_ids:
        rel = path
        if root is not None:
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
        parts = list(rel.with_suffix("").parts)
        return normalize_id("_".join(parts))

    stem = path.stem
    parent = path.parent.name
    if parent:
        return normalize_id(f"{parent}_{stem}")
    return normalize_id(stem)


def symbol_id_for(module_id: str, symbol_name: str, scope: list[str] | None = None) -> str:
    """Compute a symbol node id, normalized to ``[a-z0-9_]``.

    The default id is ``{module_id}_{symbol_name}``. When ``scope`` is given (the
    chain of enclosing class/function names, outermost first) the id becomes
    ``{module_id}_{scope...}_{symbol_name}`` so a method ``foo`` inside ``class C``
    (scope ``["C"]``) gets a different id than a top-level ``foo``.
    """
    parts = [module_id, *(scope or []), symbol_name]
    return normalize_id("_".join(parts))


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


def index_python(
    path: Path, root: Path | None = None, strict_ids: bool = False
) -> tuple[list[dict], list[dict]]:
    """Statically index a single Python file into ``(nodes, edges)``.

    Emits a module node, then walks classes, functions and imports, wiring
    ``contains`` edges from each enclosing scope to its members and
    ``imports_from`` edges from the module to each imported name.

    When ``strict_ids`` is true the module id uses the full relative path and
    symbol ids are scope-qualified (a method ``foo`` in ``class C`` differs from a
    top-level ``foo``), so genuinely distinct symbols never share an id. The
    default scheme is graphify-compatible and unchanged.

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
    module_id = module_id_for(path, root, strict_ids)

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

    def walk(scope_node: ast.AST, scope_id: str, scope: list[str]) -> None:
        """Recurse the direct children of ``scope_node`` (a module or class body).

        ``scope`` is the chain of enclosing class/function names (outermost
        first); in strict mode it is folded into each symbol id so nested symbols
        stay distinct. In the default mode it is left empty so ids are unchanged.
        """
        for child in ast.iter_child_nodes(scope_node):
            if isinstance(child, ast.ClassDef):
                class_id = symbol_id_for(module_id, child.name, scope)
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
                walk(child, class_id, [*scope, child.name] if strict_ids else scope)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_id = symbol_id_for(module_id, child.name, scope)
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
                walk(child, func_id, [*scope, child.name] if strict_ids else scope)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    emit_import(alias.asname or alias.name, child.lineno)
            elif isinstance(child, ast.ImportFrom):
                base = child.module or ""
                for alias in child.names:
                    name = alias.asname or alias.name
                    full = f"{base}.{name}" if base else name
                    emit_import(full, child.lineno)

    walk(tree, module_id, [])
    return nodes, edges
