"""Extract real source bodies from disk and attach them to graph nodes.

graphify records *where* an entity lives (``source_file`` + ``source_location``)
but not its full text. For a small, already-selected subgraph it is worth reading
the actual code back so the formatter can show it — an LLM reasoning over the
subgraph benefits far more from a function's body than from its one-line summary.

Two extraction strategies are used, picked by file extension:

- **Python** (``.py``): indentation-based. Starting from the header line, we keep
  every following line whose indentation is *strictly greater* than the header's,
  stopping at the first non-blank line indented at or below the header. This
  captures a ``def``/``class`` block without needing to parse it.
- **Brace languages** (``.ts``/``.tsx``/``.js``/``.jsx``/``.go``): brace-balanced.
  Starting from the header, we accumulate lines until the running ``{``/``}`` depth
  returns to zero *after* the first ``{`` has been seen (checked at end of line).

Both strategies cap the result at ``max_lines`` lines.
"""

from __future__ import annotations

from pathlib import Path

from apexgraph.models import KnowledgeGraph

# Extensions handled by brace-balanced extraction; everything else (notably .py)
# uses indentation-based extraction.
_BRACE_EXTS: frozenset[str] = frozenset({".ts", ".tsx", ".js", ".jsx", ".go"})


def safe_source_path(project_root: Path, source_file: str) -> Path | None:
    """Resolve ``source_file`` under ``project_root``, or ``None`` if it escapes.

    Security boundary: a graph is untrusted input. A malicious ``source_file``
    such as ``/etc/passwd`` or ``../../secret`` would otherwise let code injection
    read arbitrary files off the host (``Path`` joins let an absolute right-hand
    side win, and ``..`` walks upward). We resolve the candidate and require it to
    stay inside ``project_root``; anything else is rejected and never opened.
    """
    root = project_root.resolve()
    try:
        candidate = (root / source_file).resolve()
        candidate.relative_to(root)
    except (ValueError, OSError):
        return None
    return candidate


def _parse_line_number(source_location: str | int | None) -> int | None:
    """Parse a line spec into a 1-based line number, or ``None`` if invalid.

    Accepts an ``int`` (used directly), a bare numeric string (``"42"``), or an
    ``L``-prefixed string (``"L42"``). Anything else — including non-positive
    numbers — yields ``None``.
    """
    if source_location is None:
        return None
    if isinstance(source_location, bool):  # bool is an int subclass; reject it
        return None
    if isinstance(source_location, int):
        return source_location if source_location >= 1 else None

    text = source_location.strip()
    if not text:
        return None
    if text[0] in ("L", "l"):
        text = text[1:]
    try:
        line = int(text)
    except ValueError:
        return None
    return line if line >= 1 else None


def _indent_width(line: str) -> int:
    """Number of leading whitespace characters (tabs count as one each)."""
    return len(line) - len(line.lstrip())


def _extract_python(lines: list[str], start: int, max_lines: int) -> str:
    """Indentation-based extraction starting at 0-based index ``start``."""
    header = lines[start]
    header_indent = _indent_width(header)
    collected = [header]

    for line in lines[start + 1 :]:
        if line.strip() == "":
            collected.append(line)
            continue
        if _indent_width(line) <= header_indent:
            break
        collected.append(line)

    # Trim trailing blank lines.
    while collected and collected[-1].strip() == "":
        collected.pop()

    return "\n".join(collected[:max_lines])


def _extract_braces(lines: list[str], start: int, max_lines: int) -> str:
    """Brace-balanced extraction starting at 0-based index ``start``.

    Accumulates lines, tracking ``{``/``}`` depth, and stops at the end of the
    line where depth returns to zero after the first ``{`` is seen. If no brace is
    ever opened, only the header line is returned.
    """
    collected: list[str] = []
    depth = 0
    seen_open = False

    for line in lines[start:]:
        collected.append(line)
        for ch in line:
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}":
                depth -= 1
        if seen_open and depth <= 0:
            break

    if not seen_open:
        collected = collected[:1]

    return "\n".join(collected[:max_lines])


def extract_code_block(
    file_path: Path,
    source_location: str | int | None,
    max_lines: int = 100,
) -> str | None:
    """Extract the source body anchored at ``source_location`` in ``file_path``.

    Args:
        file_path: Path to the source file to read.
        source_location: A 1-based line spec — ``"L42"``, ``"42"`` or ``42``.
        max_lines: Hard cap on the number of lines returned.

    Returns:
        The extracted block as a string, or ``None`` if the file is missing, the
        location is unparseable, or the line is out of range.
    """
    line_number = _parse_line_number(source_location)
    if line_number is None:
        return None

    try:
        text = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    lines = text.splitlines()
    start = line_number - 1
    if start < 0 or start >= len(lines):
        return None

    if file_path.suffix.lower() in _BRACE_EXTS:
        return _extract_braces(lines, start, max_lines)
    return _extract_python(lines, start, max_lines)


def inject_code(
    graph: KnowledgeGraph,
    project_root: Path,
    max_nodes: int = 30,
) -> KnowledgeGraph:
    """Attach extracted source bodies to nodes as a ``code_block`` attribute.

    Mutates the passed graph's ``digraph`` node attributes **in place** (no copy)
    and returns the same object for convenience.

    If the graph has more than ``max_nodes`` nodes it is considered too large to
    enrich and is returned unchanged. Otherwise, for every node carrying both a
    source file (``source_file`` or ``file_path``) and a ``source_location``, the
    file is resolved relative to ``project_root``, its body is extracted via
    :func:`extract_code_block`, and a successful extraction is stored on the node
    under the ``code_block`` key.

    Args:
        graph: The (already pruned) subgraph to enrich.
        project_root: Base directory that ``source_file`` paths are relative to.
        max_nodes: Skip enrichment entirely above this node count.

    Returns:
        The same ``graph`` instance.
    """
    if len(graph) > max_nodes:
        return graph

    for node_id in graph.node_ids:
        attrs = graph.node(node_id)
        source_file = attrs.get("source_file") or attrs.get("file_path")
        source_location = attrs.get("source_location")
        if not source_file or source_location is None:
            continue

        file_path = safe_source_path(project_root, source_file)
        if file_path is None:
            continue  # path escapes project_root — refuse to read it
        block = extract_code_block(file_path, source_location)
        if block is not None:
            attrs["code_block"] = block

    return graph
