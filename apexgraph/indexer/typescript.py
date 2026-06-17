"""Static TypeScript / JavaScript indexer.

Two parsing strategies, in order of preference:

1. **tree-sitter** (``tree_sitter`` + ``tree_sitter_typescript``) — a real parser
   that recovers functions, classes, interfaces, methods, arrow-function consts
   and imports. The ``tsx`` grammar is used for ``.tsx`` / ``.jsx`` files.
2. **regex fallback** — used when tree-sitter (or its grammar) is not installed
   or when the parse fails for any reason. It recovers top-level functions,
   classes, interfaces, arrow-function consts and imports. This is the path the
   test-suite exercises, since tree-sitter may be absent.

Either way a **module** node is emitted first. IDs and node/edge shapes follow
the shared graphify convention from :mod:`apexgraph.indexer.python`.
"""

from __future__ import annotations

import re
from pathlib import Path

from apexgraph.indexer.python import (
    make_edge,
    make_node,
    module_id_for,
    relative_source,
    symbol_id_for,
)

# ---------------------------------------------------------------------------
# Regex patterns (the fallback parser, and what the tests rely on)
# ---------------------------------------------------------------------------

_RE_CLASS = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"
)
_RE_INTERFACE = re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)")
_RE_FUNCTION = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s+([A-Za-z_$][\w$]*)"
)
_RE_ARROW_CONST = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*"
    r"(?:async\s+)?\([^)]*\)\s*(?::[^=]+)?=>"
)
# import ... from "module";  /  import "module";
_RE_IMPORT_FROM = re.compile(r"""^\s*import\b[^'"]*from\s+['"]([^'"]+)['"]""")
_RE_IMPORT_BARE = re.compile(r"""^\s*import\s+['"]([^'"]+)['"]""")


def _index_typescript_regex(
    path: Path, source: str, source_file: str, module_id: str
) -> tuple[list[dict], list[dict]]:
    """Line-oriented regex indexer used as the tree-sitter fallback.

    The regex parser only recovers top-level declarations, so there is no
    enclosing class/function scope to qualify against; ``strict_ids`` therefore
    only affects the module id (computed by the caller), not symbol ids here.
    """
    nodes: list[dict] = [
        make_node(
            module_id,
            path.name,
            "module",
            f"TypeScript module {source_file}",
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

    for index, line in enumerate(source.splitlines(), start=1):
        match = _RE_CLASS.match(line)
        if match:
            emit_symbol(match.group(1), "class", index)
            continue
        match = _RE_INTERFACE.match(line)
        if match:
            emit_symbol(match.group(1), "interface", index)
            continue
        match = _RE_FUNCTION.match(line)
        if match:
            emit_symbol(match.group(1), "function", index)
            continue
        match = _RE_ARROW_CONST.match(line)
        if match:
            emit_symbol(match.group(1), "function", index)
            continue
        match = _RE_IMPORT_FROM.match(line) or _RE_IMPORT_BARE.match(line)
        if match:
            emit_import(match.group(1), index)

    return nodes, edges


def _index_typescript_treesitter(
    path: Path, source: str, source_file: str, module_id: str, strict_ids: bool = False
) -> tuple[list[dict], list[dict]]:
    """tree-sitter indexer. Raises ImportError when the grammar is unavailable.

    Recovers functions, classes, interfaces, methods (nested under their class),
    arrow-function consts and imports. Any structural surprise simply yields a
    smaller graph rather than failing — the caller treats exceptions as a signal
    to fall back to the regex parser.

    In ``strict_ids`` mode a method is qualified by its enclosing class name so a
    method ``foo`` in ``class C`` no longer collides with a top-level ``foo``.
    """
    import tree_sitter_typescript as tstypescript
    from tree_sitter import Language, Parser

    suffix = path.suffix.lower()
    if suffix in {".tsx", ".jsx"}:
        language = Language(tstypescript.language_tsx())
    else:
        language = Language(tstypescript.language_typescript())
    parser = Parser(language)
    tree = parser.parse(source.encode("utf-8"))

    nodes: list[dict] = [
        make_node(
            module_id,
            path.name,
            "module",
            f"TypeScript module {source_file}",
            source_file,
            1,
        )
    ]
    edges: list[dict] = []
    seen: set[str] = set()

    def text_of(node) -> str:
        return source[node.start_byte : node.end_byte]

    def name_field(node) -> str | None:
        field = node.child_by_field_name("name")
        return text_of(field) if field is not None else None

    def emit(
        name: str,
        node_type: str,
        lineno: int,
        scope_id: str,
        relation: str,
        scope: list[str] | None = None,
    ) -> str | None:
        symbol_id = symbol_id_for(module_id, name, scope if strict_ids else None)
        if symbol_id in seen:
            return symbol_id
        seen.add(symbol_id)
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
        edges.append(make_edge(scope_id, symbol_id, relation))
        return symbol_id

    def walk(node, scope_id: str) -> None:
        for child in node.children:
            lineno = child.start_point[0] + 1
            kind = child.type
            if kind in {"function_declaration", "generator_function_declaration"}:
                name = name_field(child)
                if name:
                    emit(name, "function", lineno, scope_id, "contains")
            elif kind in {"class_declaration", "abstract_class_declaration"}:
                name = name_field(child)
                if name:
                    class_id = emit(name, "class", lineno, scope_id, "contains")
                    body = child.child_by_field_name("body")
                    if body is not None and class_id is not None:
                        for member in body.children:
                            if member.type == "method_definition":
                                mname = name_field(member)
                                if mname:
                                    emit(
                                        mname,
                                        "function",
                                        member.start_point[0] + 1,
                                        class_id,
                                        "contains",
                                        scope=[name],
                                    )
                continue
            elif kind == "interface_declaration":
                name = name_field(child)
                if name:
                    emit(name, "interface", lineno, scope_id, "contains")
            elif kind == "import_statement":
                source_node = child.child_by_field_name("source")
                if source_node is not None:
                    name = text_of(source_node).strip("'\"")
                    sid = symbol_id_for(module_id, name)
                    if sid not in seen:
                        seen.add(sid)
                        nodes.append(
                            make_node(
                                sid,
                                name,
                                "import",
                                f"Import of {name}",
                                source_file,
                                lineno,
                            )
                        )
                        edges.append(make_edge(module_id, sid, "imports_from"))
            elif kind in {"lexical_declaration", "variable_declaration"}:
                for declarator in child.children:
                    if declarator.type != "variable_declarator":
                        continue
                    value = declarator.child_by_field_name("value")
                    if value is not None and value.type == "arrow_function":
                        name = name_field(declarator)
                        if name:
                            emit(name, "function", lineno, scope_id, "contains")
            else:
                walk(child, scope_id)

    walk(tree.root_node, module_id)
    return nodes, edges


def index_typescript(
    path: Path, root: Path | None = None, strict_ids: bool = False
) -> tuple[list[dict], list[dict]]:
    """Statically index a single TypeScript / JavaScript file into ``(nodes, edges)``.

    Tries the tree-sitter parser first and falls back to the regex parser on any
    ``ImportError`` (grammar not installed) or parse failure. Returns ``([], [])``
    only when the source file itself cannot be read.

    When ``strict_ids`` is true the module id uses the full relative path and
    methods are qualified by their enclosing class, so distinct symbols never
    share an id. The default scheme is graphify-compatible and unchanged.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return [], []

    source_file = relative_source(path, root)
    module_id = module_id_for(path, root, strict_ids)

    try:
        return _index_typescript_treesitter(path, source, source_file, module_id, strict_ids)
    except Exception:
        return _index_typescript_regex(path, source, source_file, module_id)
