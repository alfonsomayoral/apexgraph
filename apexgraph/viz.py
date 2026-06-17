"""Render a selected subgraph as a self-contained interactive HTML page.

Where :mod:`~apexgraph.formatter` produces token-friendly *text* for an LLM, this
module produces a *visual* artifact for a human: a force-directed graph the user
can open straight from disk (``file://``) and explore — pan, zoom, hover for
details. It has no Python dependencies of its own; the only runtime requirement
is the `vis-network <https://visjs.github.io/vis-network/>`_ library, which the
page pulls from a CDN via a ``<script src>`` tag. The selected nodes and edges
are embedded as JSON in an inline ``<script>``.

The contract mirrors :func:`apexgraph.formatter.format_subgraph`: it takes the
already-pruned :class:`~apexgraph.models.KnowledgeGraph`, the ``stats`` budget
dict, optional per-node ``scores`` and the originating ``query``.

Safety: every piece of graph/user text reaches the page through
:func:`json.dumps` (for the embedded data) or :func:`html.escape` (for the
header markup) — raw text is never concatenated into HTML attributes or the
script body, so labels containing quotes, ``&`` or even ``"</script>"`` cannot
break out of their context.
"""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path
from typing import Any

from apexgraph.models import KnowledgeGraph

# vis-network is loaded from a pinned, integrity-checked CDN URL. The version is
# immutable on unpkg, so the SHA-384 below stays valid; regenerate it only when
# bumping the version (see the note at the <script> tag).
_VIS_NETWORK_VERSION = "10.1.0"
_VIS_NETWORK_SRI = "sha384-Kp7cMaDnHOrgpE8FT6l7tUuGIo7kBcBVcttockpXN/whrsQBcy9ZcpKmr/1a/nMo"

# A small categorical palette cycled by ``community`` index. Picked to read well
# on the dark canvas below; god nodes override the border (see _node_records).
_PALETTE: tuple[str, ...] = (
    "#4f9dff",  # blue
    "#ff6b6b",  # red
    "#3ddc97",  # green
    "#ffd166",  # amber
    "#c792ea",  # violet
    "#ff9f43",  # orange
    "#2ec4b6",  # teal
    "#f78fb3",  # pink
)

# Border colour / shape used to single out god (high-centrality) nodes.
_GOD_BORDER = "#ffffff"
_GOD_SHAPE = "star"
_DEFAULT_SHAPE = "dot"

# Node radius bounds (vis-network "value" is scaled between these in px-ish).
_MIN_SIZE = 10.0
_MAX_SIZE = 40.0


def _color_for(community: Any) -> str:
    """Pick a palette colour for a community index (``None`` -> first colour)."""
    if community is None:
        return _PALETTE[0]
    try:
        idx = int(community)
    except (TypeError, ValueError):
        return _PALETTE[0]
    return _PALETTE[idx % len(_PALETTE)]


def _size_for(
    node_id: str,
    graph: KnowledgeGraph,
    scores: dict[str, float] | None,
    score_range: tuple[float, float],
) -> float:
    """Scale a node's radius by score (if provided) else by total degree.

    Scores are min-max normalised across the subgraph; degree is normalised by
    the maximum degree. Either way the result lands in ``[_MIN_SIZE, _MAX_SIZE]``.
    """
    if scores is not None:
        lo, hi = score_range
        value = scores.get(node_id, 0.0)
        frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    else:
        degree = graph.digraph.degree(node_id)
        max_degree = max((graph.digraph.degree(n) for n in graph.node_ids), default=0)
        frac = 0.0 if max_degree <= 0 else degree / max_degree
    return _MIN_SIZE + frac * (_MAX_SIZE - _MIN_SIZE)


def _hover_title(attrs: dict[str, Any], score: float | None) -> str:
    """Plain-text hover tooltip: type, description and (optionally) score.

    Returned raw — it is embedded via :func:`json.dumps`, never interpolated
    into HTML, so no manual escaping is needed here.
    """
    parts: list[str] = []
    node_type = attrs.get("type", "")
    if node_type:
        parts.append(str(node_type))
    description = attrs.get("description", "")
    if description:
        parts.append(str(description))
    if score is not None:
        parts.append(f"score: {score:.3f}")
    return "\n".join(parts)


def _node_records(graph: KnowledgeGraph, scores: dict[str, float] | None) -> list[dict[str, Any]]:
    """Build the list of vis-network node objects for the subgraph."""
    if scores:
        values = list(scores.values())
        score_range = (min(values), max(values))
    else:
        score_range = (0.0, 0.0)

    records: list[dict[str, Any]] = []
    for node_id in graph.node_ids:
        attrs = graph.node(node_id)
        is_god = bool(attrs.get("is_god", False))
        score = scores.get(node_id, 0.0) if scores is not None else None
        record: dict[str, Any] = {
            "id": node_id,
            "label": str(attrs.get("label", node_id)),
            "title": _hover_title(attrs, score),
            "value": _size_for(node_id, graph, scores, score_range),
            "shape": _GOD_SHAPE if is_god else _DEFAULT_SHAPE,
            "color": {
                "background": _color_for(attrs.get("community")),
                "border": _GOD_BORDER if is_god else "#1b1f27",
            },
            "borderWidth": 4 if is_god else 1,
        }
        records.append(record)
    return records


def _edge_records(graph: KnowledgeGraph) -> list[dict[str, Any]]:
    """Build the list of vis-network edge objects for the subgraph."""
    records: list[dict[str, Any]] = []
    for source, target, data in graph.digraph.edges(data=True):
        relation = str(data.get("relation", ""))
        record: dict[str, Any] = {
            "from": source,
            "to": target,
            "title": relation,
            "arrows": "to",
        }
        if relation:
            record["label"] = relation
        records.append(record)
    return records


def _stats_line(stats: dict[str, Any]) -> str:
    """The ``Selected X/Y nodes · U/B tokens (C%)`` banner subtitle."""
    return (
        f"Selected {stats.get('nodes_selected', 0)}/{stats.get('nodes_total', 0)} "
        f"nodes · {stats.get('tokens_used', 0)}/{stats.get('tokens_budget', 0)} "
        f"tokens ({stats.get('coverage_pct', 0)}%)"
    )


# Shared dark stylesheet for both the populated and empty pages.
_STYLE = """
  html, body { margin: 0; padding: 0; height: 100%; background: #0d1117;
    color: #e6edf3; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
  #banner { position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    padding: 10px 16px; background: rgba(22, 27, 34, 0.92);
    border-bottom: 1px solid #30363d; box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
  #banner .query { font-size: 15px; font-weight: 600; color: #58a6ff; }
  #banner .stats { font-size: 12px; color: #8b949e; margin-top: 2px; }
  #graph { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }
  #empty { display: flex; align-items: center; justify-content: center;
    height: 100%; font-size: 18px; color: #8b949e; }
""".strip()


def _empty_html(query: str) -> str:
    """A valid standalone page used when the subgraph has no nodes."""
    title = html.escape(query) if query else "Apexgraph"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_STYLE}\n</style>\n"
        "</head>\n<body>\n"
        '<div id="empty">No nodes selected.</div>\n'
        "</body>\n</html>\n"
    )


def build_html(
    graph: KnowledgeGraph,
    stats: dict[str, Any],
    scores: dict[str, float] | None = None,
    query: str = "",
) -> str:
    """Render the subgraph as a self-contained interactive HTML document.

    The returned string is a complete page: it loads ``vis-network`` from a CDN
    and embeds the selected nodes/edges as JSON, so it can be written to disk and
    opened directly in a browser with no server.

    Args:
        graph: The already-pruned subgraph to visualise.
        stats: Budget accounting with keys ``nodes_selected``, ``nodes_total``,
            ``tokens_used``, ``tokens_budget`` and ``coverage_pct``.
        scores: Optional ``{node_id: score}``. When given, node size scales by
            (normalised) score and the score is shown in the hover tooltip;
            otherwise size scales by node degree.
        query: The originating query, echoed in the fixed banner.

    Returns:
        A complete HTML document as a string. For an empty graph this is a valid
        page stating that no nodes were selected.
    """
    if len(graph) == 0:
        return _empty_html(query)

    nodes = _node_records(graph, scores)
    edges = _edge_records(graph)

    # json.dumps handles every quoting concern. The "</" -> "<\\/" replacement
    # closes the one remaining HTML hazard: a literal "</script>" inside a string
    # value would otherwise terminate the inline <script> element early.
    nodes_json = json.dumps(nodes).replace("</", "<\\/")
    edges_json = json.dumps(edges).replace("</", "<\\/")

    banner_query = html.escape(query) if query else "Apexgraph subgraph"
    banner_stats = html.escape(_stats_line(stats))
    title = html.escape(query) if query else "Apexgraph subgraph"

    # Pinned + Subresource Integrity: the version URL is immutable and the browser
    # refuses to run the script unless its hash matches, so a compromised CDN can't
    # inject code into a Apexgraph-generated page. Regenerate the hash if bumping the
    # version: curl -sL <url> | openssl dgst -sha384 -binary | openssl base64
    cdn = f"https://unpkg.com/vis-network@{_VIS_NETWORK_VERSION}/standalone/umd/vis-network.min.js"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{cdn}" integrity="{_VIS_NETWORK_SRI}" crossorigin="anonymous"></script>
<style>
{_STYLE}
</style>
</head>
<body>
<div id="banner">
  <div class="query">{banner_query}</div>
  <div class="stats">{banner_stats}</div>
</div>
<div id="graph"></div>
<script>
  const nodes = new vis.DataSet({nodes_json});
  const edges = new vis.DataSet({edges_json});
  const container = document.getElementById("graph");
  const options = {{
    nodes: {{
      shape: "dot",
      scaling: {{ min: {int(_MIN_SIZE)}, max: {int(_MAX_SIZE)} }},
      font: {{ color: "#e6edf3", size: 14, strokeWidth: 2, strokeColor: "#0d1117" }}
    }},
    edges: {{
      color: {{ color: "#5b6470", highlight: "#58a6ff" }},
      arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
      font: {{ color: "#8b949e", size: 10, strokeWidth: 3, strokeColor: "#0d1117" }},
      smooth: {{ type: "continuous" }}
    }},
    physics: {{
      solver: "forceAtlas2Based",
      stabilization: {{ iterations: 200 }}
    }},
    interaction: {{ hover: true, tooltipDelay: 120 }}
  }};
  new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);
</script>
</body>
</html>
"""


def write_html(
    graph: KnowledgeGraph,
    stats: dict[str, Any],
    scores: dict[str, float] | None = None,
    query: str = "",
    path: Path | None = None,
) -> Path:
    """Render the subgraph and write it to a UTF-8 ``.html`` file.

    Does *not* open a browser — the CLI owns that decision.

    Args:
        graph: The subgraph to visualise.
        stats: Budget accounting (see :func:`build_html`).
        scores: Optional ``{node_id: score}`` (see :func:`build_html`).
        query: The originating query, echoed in the banner.
        path: Destination file. When ``None``, a ``NamedTemporaryFile`` with a
            ``.html`` suffix is created and its path returned.

    Returns:
        The :class:`~pathlib.Path` the HTML was written to.
    """
    document = build_html(graph, stats, scores=scores, query=query)

    if path is None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", encoding="utf-8", delete=False
        ) as handle:
            handle.write(document)
            return Path(handle.name)

    target = Path(path)
    target.write_text(document, encoding="utf-8")
    return target
