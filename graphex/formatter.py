"""Render a selected subgraph into a token-friendly textual representation.

The retrieval pipeline hands this module a :class:`~graphex.models.KnowledgeGraph`
that has already been pruned to the selected nodes, plus a ``stats`` dict
describing the budget accounting. This module turns that into one of three output
formats — ``markdown`` (human-/LLM-facing, the default), ``json`` (machine-facing,
exact), or ``yaml`` (a hand-rolled, dependency-free emit of the same structure).

Markdown is the interesting one: it opens with a Unicode box header echoing the
query and budget, lists the selected nodes (best first) with their descriptions
and — when :func:`~graphex.injector.inject_code` has run — their source code in a
language-tagged fence, then summarises the relationships between them.
"""

from __future__ import annotations

import json
from typing import Any

from graphex.models import KnowledgeGraph

# Map a source-file extension to the language tag used in a Markdown code fence.
# Anything unrecognised falls back to an empty tag (a plain ``` ``` fence).
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
}

# YAML scalars containing any of these characters (or a leading digit/space) must
# be quoted to round-trip safely; we quote via json.dumps which is valid YAML.
_YAML_SPECIAL: frozenset[str] = frozenset(":#{}[],&*!|>'\"%@`")


def _lang_for(source_file: str | None) -> str:
    """Infer a fenced-code-block language tag from a source file's extension."""
    if not source_file:
        return ""
    lower = source_file.lower()
    for ext, lang in _EXT_TO_LANG.items():
        if lower.endswith(ext):
            return lang
    return ""


def _ordered_node_ids(graph: KnowledgeGraph, scores: dict[str, float] | None) -> list[str]:
    """Return node ids ordered for display.

    With ``scores``: descending score, ties broken by id ascending. Without
    scores: id ascending. Nodes missing from ``scores`` are treated as ``0.0``.
    """
    if scores is None:
        return sorted(graph.node_ids)
    return sorted(graph.node_ids, key=lambda nid: (-scores.get(nid, 0.0), nid))


def _node_payload(
    graph: KnowledgeGraph, node_id: str, scores: dict[str, float] | None
) -> dict[str, Any]:
    """Build the structured per-node record shared by the json/yaml emitters."""
    attrs = graph.node(node_id)
    payload: dict[str, Any] = {
        "id": node_id,
        "label": attrs.get("label", node_id),
        "type": attrs.get("type", ""),
        "description": attrs.get("description", ""),
        "importance": attrs.get("importance", 0.0),
        "file_path": attrs.get("source_file"),
    }
    if scores is not None:
        payload["score"] = round(float(scores.get(node_id, 0.0)), 4)
    return payload


def _edge_payload(graph: KnowledgeGraph) -> list[dict[str, Any]]:
    """Build the structured edge records shared by the json/yaml emitters."""
    edges: list[dict[str, Any]] = []
    for source, target, data in graph.digraph.edges(data=True):
        record: dict[str, Any] = {"source": source, "target": target}
        relation = data.get("relation")
        if relation:
            record["relation"] = relation
        if "weight" in data:
            record["weight"] = data["weight"]
        edges.append(record)
    return edges


# -- markdown ----------------------------------------------------------------


def _box_header(query: str, stats: dict[str, Any]) -> str:
    """Render the Unicode box header showing the query and budget summary."""
    budget = stats.get("tokens_budget", 0)
    title = f"Graphex subgraph for: {query}" if query else "Graphex subgraph"
    subtitle = (
        f"Selected {stats.get('nodes_selected', 0)}/{stats.get('nodes_total', 0)} "
        f"nodes ({stats.get('coverage_pct', 0)}%) · "
        f"{stats.get('tokens_used', 0)}/{budget} tokens"
    )
    width = max(len(title), len(subtitle)) + 2
    top = "┌" + "─" * width + "┐"
    bottom = "└" + "─" * width + "┘"
    title_line = "│ " + title.ljust(width - 1) + "│"
    subtitle_line = "│ " + subtitle.ljust(width - 1) + "│"
    return "\n".join((top, title_line, subtitle_line, bottom))


def _format_markdown(
    graph: KnowledgeGraph,
    stats: dict[str, Any],
    scores: dict[str, float] | None,
    query: str,
    include_header: bool = True,
) -> str:
    """Render the Markdown representation (see module docstring)."""
    lines: list[str] = [_box_header(query, stats), ""] if include_header else []

    lines.append("## Relevant Nodes")
    lines.append("")

    for node_id in _ordered_node_ids(graph, scores):
        attrs = graph.node(node_id)
        label = attrs.get("label", node_id)
        node_type = attrs.get("type", "")
        heading = f"### {label} ({node_type})"
        if scores is not None:
            heading += f" · score: {scores.get(node_id, 0.0):.2f}"
        lines.append(heading)

        description = attrs.get("description", "")
        if description:
            lines.append("")
            lines.append(description)

        source_file = attrs.get("source_file")
        if source_file:
            lines.append("")
            lines.append(f"→ File: {source_file}")

        code_block = attrs.get("code_block")
        if code_block:
            lang = _lang_for(source_file)
            lines.append("")
            lines.append(f"```{lang}")
            lines.append(code_block.rstrip("\n"))
            lines.append("```")

        lines.append("")

    lines.append("## Key Relationships")
    lines.append("")
    for source, target, data in graph.digraph.edges(data=True):
        relation = data.get("relation", "")
        lines.append(f"- {source} → {relation} → {target}")
    lines.append("")

    selected = stats.get("nodes_selected", 0)
    total = stats.get("nodes_total", 0)
    if total > selected:
        excluded = total - selected
        lines.append(
            f"_Tip: {excluded} node(s) were excluded to fit the token budget; "
            f"raise the budget to see more._"
        )

    return "\n".join(lines).rstrip("\n") + "\n"


# -- json --------------------------------------------------------------------


def _format_json(
    graph: KnowledgeGraph, stats: dict[str, Any], scores: dict[str, float] | None
) -> str:
    """Render the JSON representation with ``indent=2``."""
    payload = {
        "stats": stats,
        "nodes": [_node_payload(graph, nid, scores) for nid in _ordered_node_ids(graph, scores)],
        "edges": _edge_payload(graph),
    }
    return json.dumps(payload, indent=2)


# -- yaml --------------------------------------------------------------------


def _yaml_scalar(value: Any) -> str:
    """Render a scalar as a YAML value, quoting strings that need it.

    Strings are quoted (via :func:`json.dumps`, which produces valid YAML
    double-quoted scalars) when empty, when they begin with a digit or space, or
    when they contain a YAML metacharacter or a newline. ``None`` becomes
    ``null``; bools become ``true``/``false``; numbers are emitted bare.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value)
    needs_quote = (
        text == ""
        or text[0].isdigit()
        or text[0] == " "
        or text[-1] == " "
        or "\n" in text
        or any(ch in _YAML_SPECIAL for ch in text)
    )
    return json.dumps(text) if needs_quote else text


def _yaml_block(items: list[dict[str, Any]], indent: str) -> list[str]:
    """Render a list of flat mappings as a YAML block sequence."""
    lines: list[str] = []
    for item in items:
        first = True
        for key, value in item.items():
            prefix = f"{indent}- " if first else f"{indent}  "
            lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
            first = False
        if first:  # empty mapping
            lines.append(f"{indent}- {{}}")
    return lines


def _format_yaml(
    graph: KnowledgeGraph, stats: dict[str, Any], scores: dict[str, float] | None
) -> str:
    """Render a hand-rolled YAML representation (no PyYAML dependency)."""
    nodes = [_node_payload(graph, nid, scores) for nid in _ordered_node_ids(graph, scores)]
    edges = _edge_payload(graph)

    lines: list[str] = ["stats:"]
    for key, value in stats.items():
        lines.append(f"  {key}: {_yaml_scalar(value)}")

    lines.append("nodes:")
    lines.extend(_yaml_block(nodes, indent="  "))

    lines.append("edges:")
    lines.extend(_yaml_block(edges, indent="  "))

    return "\n".join(lines) + "\n"


# -- public API --------------------------------------------------------------


def format_subgraph(
    graph: KnowledgeGraph,
    stats: dict[str, Any],
    format: str = "markdown",
    scores: dict[str, float] | None = None,
    query: str = "",
    include_header: bool = True,
) -> str:
    """Render a selected subgraph as ``markdown``, ``json``, or ``yaml``.

    Args:
        graph: The already-pruned subgraph to render.
        stats: Budget accounting with keys ``nodes_selected``, ``nodes_total``,
            ``tokens_used``, ``tokens_budget`` and ``coverage_pct``.
        format: One of ``"markdown"`` (default), ``"json"`` or ``"yaml"``.
        scores: Optional ``{node_id: score}``. When given, nodes are ordered by
            score descending (ties by id) and scores are surfaced in the output;
            otherwise nodes are ordered by id.
        query: The originating query, echoed in the Markdown header.

    Returns:
        The rendered string.

    Raises:
        ValueError: If ``format`` is not one of the supported values.
    """
    if format == "markdown":
        return _format_markdown(graph, stats, scores, query, include_header=include_header)
    if format == "json":
        return _format_json(graph, stats, scores)
    if format == "yaml":
        return _format_yaml(graph, stats, scores)
    raise ValueError(f"unknown format {format!r}; expected 'markdown', 'json' or 'yaml'")
