"""Multi-format loaders that materialize a :class:`KnowledgeGraph`.

Graphex reads knowledge graphs from several on-disk shapes and normalizes them
all onto the single contract in :mod:`graphex.models`:

- ``.json`` — graphify's native serialization (``nodes`` + ``links`` +
  ``hyperedges``) or a generic ``nodes`` + ``edges`` dict. This is the primary,
  richest format: it carries edge weights, confidence scores, community
  membership, importance priors, god-node flags and higher-order hyperedges.
- ``.graphml`` — GraphML via NetworkX. Attribute names are canonicalized back
  onto the graphify field names.
- ``.csv`` — Neo4j-style node/relationship CSV pairs (both the native bulk-import
  ``:ID`` / ``:START_ID`` columns and a simplified ``nodeId`` / ``startNodeId``
  flavour).

Anything that cannot be parsed into a usable graph raises
:class:`GraphexLoadError` with a message that names the offending path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import networkx as nx

from graphex.models import Edge, Hyperedge, KnowledgeGraph, Node

__all__ = [
    "GraphexLoadError",
    "load_graph",
    "load_graph_neo4j",
    "convert_graph",
    "detect_format",
]


class GraphexLoadError(Exception):
    """Raised when a graph cannot be loaded or converted.

    The message always includes the offending path so failures are actionable.
    """


# ---------------------------------------------------------------------------
# Field maps and small helpers
# ---------------------------------------------------------------------------

# Node keys that map directly onto :class:`Node` constructor arguments. Every
# other key on a node payload is preserved verbatim in ``Node.extra``.
_NODE_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "label",
        "type",
        "file_type",
        "description",
        "importance",
        "source_file",
        "source_location",
        "source_url",
        "community",
        "is_god",
    }
)

# Edge keys that map directly onto :class:`Edge`. Endpoints are handled
# separately. Everything else lands in ``Edge.extra``.
_EDGE_FIELDS: frozenset[str] = frozenset(
    {
        "source",
        "target",
        "relation",
        "weight",
        "confidence",
        "confidence_score",
    }
)

# Rich node keys whose mere presence implies the graphify producer.
_RICH_NODE_KEYS: frozenset[str] = frozenset(
    {"type", "description", "importance", "file_type", "source_file", "community"}
)


def _to_float(value: Any, default: float) -> float:
    """Best-effort float coercion that never raises."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int_or_none(value: Any) -> int | None:
    """Coerce a community marker to ``int`` (or ``None`` when absent/invalid)."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    """Coerce a truthiness marker, tolerating GraphML/CSV string booleans."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _pick(row: dict[str, Any], candidates: list[str]) -> Any:
    """Return the first non-empty value among ``candidates`` in ``row``.

    Used to bridge the native Neo4j export column names (``:ID``, ``:START_ID``,
    …) and the simplified ones (``nodeId``, ``startNodeId``, …) transparently.
    """
    for key in candidates:
        if key in row:
            value = row[key]
            if value is not None and value != "":
                return value
    return None


# ---------------------------------------------------------------------------
# Node / edge / hyperedge builders shared by every JSON-ish format
# ---------------------------------------------------------------------------


def _build_node(raw: dict[str, Any], path: Path) -> Node:
    """Construct a :class:`Node` from a raw payload, routing extras to ``extra``."""
    if "id" not in raw or raw["id"] in (None, ""):
        raise GraphexLoadError(f"{path}: node is missing required 'id' field")

    node_id = str(raw["id"])
    is_god = bool(raw.get("is_god", raw.get("god", False)))
    extra = {key: value for key, value in raw.items() if key not in _NODE_FIELDS and key != "god"}

    return Node(
        id=node_id,
        label=str(raw.get("label", "") or ""),
        type=str(raw.get("type", "") or ""),
        file_type=str(raw.get("file_type", "concept") or "concept"),
        description=str(raw.get("description", "") or ""),
        importance=_to_float(raw.get("importance"), 0.0),
        source_file=raw.get("source_file"),
        source_location=raw.get("source_location"),
        source_url=raw.get("source_url"),
        community=_to_int_or_none(raw.get("community")),
        is_god=is_god,
        extra=extra,
    )


def _build_edge(raw: dict[str, Any]) -> Edge | None:
    """Construct an :class:`Edge`, or ``None`` if endpoints are missing.

    Caller is responsible for skipping edges whose endpoints aren't real nodes.
    """
    source = raw.get("source")
    target = raw.get("target")
    if source in (None, "") or target in (None, ""):
        return None

    extra = {key: value for key, value in raw.items() if key not in _EDGE_FIELDS}
    return Edge(
        source=str(source),
        target=str(target),
        relation=str(raw.get("relation", "") or ""),
        weight=_to_float(raw.get("weight"), 1.0),
        confidence=str(raw.get("confidence", "EXTRACTED") or "EXTRACTED"),
        confidence_score=_to_float(raw.get("confidence_score"), 1.0),
        extra=extra,
    )


def _build_hyperedge(raw: dict[str, Any], path: Path) -> Hyperedge:
    """Construct a :class:`Hyperedge` from a raw payload."""
    if "id" not in raw or raw["id"] in (None, ""):
        raise GraphexLoadError(f"{path}: hyperedge is missing required 'id' field")
    nodes = raw.get("nodes") or []
    return Hyperedge(
        id=str(raw["id"]),
        label=str(raw.get("label", "") or ""),
        nodes=[str(n) for n in nodes],
        relation=str(raw.get("relation", "") or ""),
        confidence_score=_to_float(raw.get("confidence_score"), 1.0),
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def detect_format(data: dict[str, Any]) -> str:
    """Classify a parsed JSON graph as ``"graphify"`` or ``"generic"``.

    A graph is treated as graphify-flavoured when it uses NetworkX's ``links``
    key, carries ``hyperedges``, or when any node exposes a rich graphify-only
    attribute (type / description / importance / file_type / source_file /
    community). Otherwise it is a plain ``nodes`` + ``edges`` document.
    """
    if data.get("links") or data.get("hyperedges"):
        return "graphify"
    for node in data.get("nodes", []) or []:
        if isinstance(node, dict) and _RICH_NODE_KEYS.intersection(node.keys()):
            return "graphify"
    return "generic"


def _load_json(path: Path) -> KnowledgeGraph:
    """Load a graphify or generic JSON graph file into a :class:`KnowledgeGraph`."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise GraphexLoadError(f"{path}: file not found") from exc
    except OSError as exc:
        raise GraphexLoadError(f"{path}: could not read file ({exc})") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraphexLoadError(f"{path}: invalid JSON ({exc})") from exc

    if not isinstance(data, dict):
        raise GraphexLoadError(f"{path}: top-level JSON must be an object with a 'nodes' array")

    raw_nodes = data.get("nodes")
    if not raw_nodes:
        raise GraphexLoadError(f"{path}: graph has no nodes")
    if not isinstance(raw_nodes, list):
        raise GraphexLoadError(f"{path}: 'nodes' must be an array")

    kg = KnowledgeGraph()
    node_ids: set[str] = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise GraphexLoadError(f"{path}: every node must be an object")
        node = _build_node(raw, path)
        kg.add_node(node)
        node_ids.add(node.id)

    # graphify (NetworkX) uses "links"; generic documents use "edges". Accept
    # both, preferring whichever is populated.
    raw_edges = data.get("links")
    if not raw_edges:
        raw_edges = data.get("edges") or []
    for raw in raw_edges:
        if not isinstance(raw, dict):
            continue
        edge = _build_edge(raw)
        if edge is None:
            continue
        # Skip dangling edges whose endpoints aren't present as nodes.
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        kg.add_edge(edge)

    for raw in data.get("hyperedges") or []:
        if not isinstance(raw, dict):
            continue
        kg.add_hyperedge(_build_hyperedge(raw, path))

    return kg


# ---------------------------------------------------------------------------
# GraphML
# ---------------------------------------------------------------------------


def _load_graphml(path: Path) -> KnowledgeGraph:
    """Load a GraphML file, canonicalizing attribute names onto the contract."""
    if not path.exists():
        raise GraphexLoadError(f"{path}: file not found")
    try:
        graph = nx.read_graphml(path)
    except Exception as exc:  # nx raises a grab-bag of parse errors
        raise GraphexLoadError(f"{path}: could not parse GraphML ({exc})") from exc

    kg = KnowledgeGraph()
    node_ids: set[str] = set()
    for raw_id, attrs in graph.nodes(data=True):
        node_id = str(raw_id)
        label = attrs.get("label") or attrs.get("name") or node_id
        ntype = attrs.get("type") or attrs.get("kind") or ""
        description = attrs.get("description") or attrs.get("desc") or ""
        consumed = {
            "label",
            "name",
            "type",
            "kind",
            "description",
            "desc",
            "file_type",
            "importance",
            "source_file",
            "source_location",
            "source_url",
            "community",
            "is_god",
        }
        extra = {k: v for k, v in attrs.items() if k not in consumed}
        node = Node(
            id=node_id,
            label=str(label),
            type=str(ntype),
            file_type=str(attrs.get("file_type", "concept") or "concept"),
            description=str(description),
            importance=_to_float(attrs.get("importance"), 0.0),
            source_file=attrs.get("source_file"),
            source_location=attrs.get("source_location"),
            source_url=attrs.get("source_url"),
            community=_to_int_or_none(attrs.get("community")),
            is_god=_to_bool(attrs.get("is_god", False)),
            extra=extra,
        )
        kg.add_node(node)
        node_ids.add(node_id)

    for raw_u, raw_v, attrs in graph.edges(data=True):
        u, v = str(raw_u), str(raw_v)
        if u not in node_ids or v not in node_ids:
            continue
        relation = attrs.get("relation") or attrs.get("label") or ""
        consumed = {"relation", "label", "weight", "confidence", "confidence_score"}
        extra = {k: val for k, val in attrs.items() if k not in consumed}
        edge = Edge(
            source=u,
            target=v,
            relation=str(relation),
            weight=_to_float(attrs.get("weight"), 1.0),
            confidence=str(attrs.get("confidence", "EXTRACTED") or "EXTRACTED"),
            confidence_score=_to_float(attrs.get("confidence_score"), 1.0),
            extra=extra,
        )
        kg.add_edge(edge)

    return kg


# ---------------------------------------------------------------------------
# Neo4j CSV
# ---------------------------------------------------------------------------

# Column candidate lists bridging native bulk-import and simplified exports.
_NODE_ID_COLS = [":ID", "nodeId", "id", "_id"]
_NODE_LABEL_COLS = [":LABEL", "labels", "label"]
_NODE_NAME_COLS = ["name", "label"]
_REL_START_COLS = [":START_ID", "startNodeId", "start", "source"]
_REL_END_COLS = [":END_ID", "endNodeId", "end", "target"]
_REL_TYPE_COLS = [":TYPE", "type", "relation", "label"]

# Reserved relationship columns excluded from edge extras.
_REL_RESERVED = {
    ":START_ID",
    "startNodeId",
    "start",
    "source",
    ":END_ID",
    "endNodeId",
    "end",
    "target",
    ":TYPE",
    "type",
    "relation",
    "label",
    "weight",
    "confidence",
    "confidence_score",
}


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV into (fieldnames, rows). Raises on a missing file."""
    if not path.exists():
        raise GraphexLoadError(f"{path}: file not found")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [row for row in reader]
    except OSError as exc:
        raise GraphexLoadError(f"{path}: could not read CSV ({exc})") from exc
    return fieldnames, rows


def _neo4j_node_from_row(row: dict[str, str]) -> Node:
    """Build a :class:`Node` from a Neo4j node-CSV row."""
    node_id = str(_pick(row, _NODE_ID_COLS))
    label_marker = _pick(row, _NODE_NAME_COLS) or _pick(row, _NODE_LABEL_COLS)
    type_marker = _pick(row, _NODE_LABEL_COLS) or ""

    reserved = (
        set(_NODE_ID_COLS)
        | set(_NODE_LABEL_COLS)
        | set(_NODE_NAME_COLS)
        | {
            "file_type",
            "description",
            "importance",
            "source_file",
            "source_location",
            "source_url",
            "community",
            "is_god",
            "god",
        }
    )
    extra = {
        key: value for key, value in row.items() if key not in reserved and value not in (None, "")
    }

    return Node(
        id=node_id,
        label=str(label_marker or node_id),
        type=str(type_marker or ""),
        file_type=str(row.get("file_type", "concept") or "concept"),
        description=str(row.get("description", "") or ""),
        importance=_to_float(row.get("importance"), 0.0),
        source_file=row.get("source_file") or None,
        source_location=row.get("source_location") or None,
        source_url=row.get("source_url") or None,
        community=_to_int_or_none(row.get("community")),
        is_god=_to_bool(row.get("is_god") or row.get("god") or False),
        extra=extra,
    )


def load_graph_neo4j(nodes_path: Path, relationships_path: Path | None = None) -> KnowledgeGraph:
    """Load a Neo4j CSV node file plus an optional relationships file.

    Supports both the native bulk-import column names (``:ID``, ``:LABEL``,
    ``:START_ID``, ``:END_ID``, ``:TYPE``) and a simplified flavour
    (``nodeId``, ``labels``, ``name``, ``startNodeId``, ``endNodeId``, ``type``).
    Unknown columns are preserved as extras; malformed relationship rows are
    skipped silently.

    Raises:
        GraphexLoadError: if the nodes file is missing, has no recognizable ID
            column, or contains no data rows.
    """
    fieldnames, rows = _read_csv_rows(nodes_path)
    if not any(col in fieldnames for col in _NODE_ID_COLS):
        raise GraphexLoadError(
            f"{nodes_path}: no node-id column found " f"(expected one of {_NODE_ID_COLS})"
        )
    if not rows:
        raise GraphexLoadError(f"{nodes_path}: nodes CSV has no data rows")

    kg = KnowledgeGraph()
    node_ids: set[str] = set()
    for row in rows:
        if _pick(row, _NODE_ID_COLS) is None:
            continue
        node = _neo4j_node_from_row(row)
        kg.add_node(node)
        node_ids.add(node.id)

    if relationships_path is not None and relationships_path.exists():
        _, rel_rows = _read_csv_rows(relationships_path)
        for row in rel_rows:
            start = _pick(row, _REL_START_COLS)
            end = _pick(row, _REL_END_COLS)
            if start is None or end is None:
                continue
            source, target = str(start), str(end)
            if source not in node_ids or target not in node_ids:
                continue
            extra = {
                key: value
                for key, value in row.items()
                if key not in _REL_RESERVED and value not in (None, "")
            }
            edge = Edge(
                source=source,
                target=target,
                relation=str(_pick(row, _REL_TYPE_COLS) or ""),
                weight=_to_float(row.get("weight"), 1.0),
                confidence=str(row.get("confidence", "EXTRACTED") or "EXTRACTED"),
                confidence_score=_to_float(row.get("confidence_score"), 1.0),
                extra=extra,
            )
            kg.add_edge(edge)

    return kg


def _discover_relationships_csv(nodes_path: Path) -> Path | None:
    """Find a sibling relationships CSV for a Neo4j nodes CSV, if one exists."""
    directory = nodes_path.parent
    stem = nodes_path.stem
    candidates = [
        directory / "relationships.csv",
        directory / f"{stem}_relationships.csv",
    ]
    if stem.endswith("_nodes"):
        base = stem[: -len("_nodes")]
        candidates.append(directory / f"{base}_relationships.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------


def load_graph(path: Path) -> KnowledgeGraph:
    """Load a knowledge graph, dispatching on file extension.

    Supported extensions:

    - ``.json`` — graphify or generic JSON.
    - ``.graphml`` — GraphML.
    - ``.csv`` — Neo4j nodes CSV; a sibling relationships CSV is auto-discovered.

    Raises:
        GraphexLoadError: for a missing file, an unsupported extension, or any
            format-specific failure.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return _load_json(path)
    if suffix == ".graphml":
        return _load_graphml(path)
    if suffix == ".csv":
        if not path.exists():
            raise GraphexLoadError(f"{path}: file not found")
        relationships = _discover_relationships_csv(path)
        return load_graph_neo4j(path, relationships)

    raise GraphexLoadError(
        f"{path}: unsupported extension '{suffix}' " f"(expected .json, .graphml or .csv)"
    )


# ---------------------------------------------------------------------------
# Export / conversion
# ---------------------------------------------------------------------------


def _graphml_safe(value: Any) -> Any:
    """Coerce a value to a GraphML-serializable scalar.

    GraphML only supports primitive scalars, so ``None`` becomes ``""`` and
    containers are JSON-encoded.
    """
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, default=str)


def _convert_graphml(kg: KnowledgeGraph, output_path: Path) -> list[Path]:
    """Write ``kg`` to a single GraphML file."""
    graph = nx.DiGraph()
    for node_id in kg.digraph.nodes:
        attrs = {k: _graphml_safe(v) for k, v in kg.digraph.nodes[node_id].items()}
        graph.add_node(str(node_id), **attrs)
    for u, v, data in kg.digraph.edges(data=True):
        attrs = {k: _graphml_safe(val) for k, val in data.items()}
        graph.add_edge(str(u), str(v), **attrs)

    output_path = output_path.with_suffix(".graphml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        nx.write_graphml(graph, output_path)
    except Exception as exc:
        raise GraphexLoadError(f"{output_path}: could not write GraphML ({exc})") from exc
    return [output_path]


def _convert_neo4j(kg: KnowledgeGraph, output_path: Path) -> list[Path]:
    """Write ``kg`` to a Neo4j-style node CSV and relationship CSV pair."""
    base = output_path.with_suffix("")
    nodes_path = base.parent / f"{base.name}_nodes.csv"
    rels_path = base.parent / f"{base.name}_relationships.csv"
    nodes_path.parent.mkdir(parents=True, exist_ok=True)

    node_columns = [
        ":ID",
        "name",
        ":LABEL",
        "file_type",
        "description",
        "importance",
        "community",
        "is_god",
    ]
    try:
        with nodes_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(node_columns)
            for node_id in kg.digraph.nodes:
                attrs = kg.digraph.nodes[node_id]
                writer.writerow(
                    [
                        node_id,
                        attrs.get("label", node_id),
                        attrs.get("type", ""),
                        attrs.get("file_type", "concept"),
                        attrs.get("description", ""),
                        attrs.get("importance", 0.0),
                        "" if attrs.get("community") is None else attrs.get("community"),
                        attrs.get("is_god", False),
                    ]
                )

        rel_columns = [":START_ID", ":END_ID", ":TYPE", "weight", "confidence_score"]
        with rels_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(rel_columns)
            for u, v, data in kg.digraph.edges(data=True):
                writer.writerow(
                    [
                        u,
                        v,
                        data.get("relation", ""),
                        data.get("weight", 1.0),
                        data.get("confidence_score", 1.0),
                    ]
                )
    except OSError as exc:
        raise GraphexLoadError(f"{output_path}: could not write Neo4j CSVs ({exc})") from exc

    return [nodes_path, rels_path]


def convert_graph(kg: KnowledgeGraph, output_path: Path, format: str = "graphml") -> list[Path]:
    """Export ``kg`` to disk in the requested format.

    Args:
        kg: The graph to export.
        output_path: Base output path; the concrete suffix(es) are derived from
            ``format``.
        format: ``"graphml"`` (one ``.graphml`` file) or ``"neo4j"`` (a
            ``*_nodes.csv`` / ``*_relationships.csv`` pair).

    Returns:
        The list of file paths actually written.

    Raises:
        GraphexLoadError: for an unknown format or a write failure.
    """
    output_path = Path(output_path)
    fmt = format.lower()
    if fmt == "graphml":
        return _convert_graphml(kg, output_path)
    if fmt == "neo4j":
        return _convert_neo4j(kg, output_path)
    raise GraphexLoadError(
        f"{output_path}: unknown export format '{format}' " f"(expected 'graphml' or 'neo4j')"
    )
