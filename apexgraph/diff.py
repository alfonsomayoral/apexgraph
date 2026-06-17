"""Compare two versions of a knowledge graph and surface the impact of the change.

When a codebase is re-indexed, the graph shifts: nodes appear and disappear,
existing entities get re-described, and edges are rewired. This module answers
two questions:

1. *What changed?* — :func:`diff_graphs` returns a :class:`GraphDiff` of node and
   edge set differences, treating a node present in both versions as *modified*
   when any of its identifying attributes (label/description/type/source
   location) differ.
2. *What is affected?* — :func:`affected_subgraph` grows a ``hops``-neighborhood
   outward from every added or modified node, so a reviewer sees not just the
   change but the part of the graph that lives next to it.

:func:`format_diff` renders a :class:`GraphDiff` as a compact Markdown summary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from apexgraph.models import KnowledgeGraph

# Node attributes whose change marks a node as "modified". An identity-level set:
# a different label/type/description/location means the entity meaningfully moved,
# whereas a shifted importance score or community index does not.
_MODIFIED_KEYS: tuple[str, ...] = ("label", "description", "type", "source_location")


@dataclass
class GraphDiff:
    """The structural delta between an ``old`` and a ``new`` knowledge graph.

    Node lists hold ids; edge lists hold ``(source, target)`` pairs. A node is in
    exactly one of ``added`` / ``removed`` / ``modified`` (or none, if unchanged).
    """

    added_nodes: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    modified_nodes: list[str] = field(default_factory=list)
    added_edges: list[tuple[str, str]] = field(default_factory=list)
    removed_edges: list[tuple[str, str]] = field(default_factory=list)


def _node_signature(graph: KnowledgeGraph, node_id: str) -> tuple[object, ...]:
    """The identity-level attribute tuple compared to detect modification."""
    attrs = graph.node(node_id)
    return tuple(attrs.get(key) for key in _MODIFIED_KEYS)


def diff_graphs(old: KnowledgeGraph, new: KnowledgeGraph) -> GraphDiff:
    """Compute the node/edge delta from ``old`` to ``new``.

    Nodes are diffed by id set difference; nodes present in both versions are
    *modified* when their :data:`_MODIFIED_KEYS` signature differs. Edges are
    diffed by ``(source, target)`` set difference.

    Args:
        old: The previous graph version.
        new: The current graph version.

    Returns:
        A :class:`GraphDiff`. All lists are sorted for stable output.
    """
    old_nodes = set(old.node_ids)
    new_nodes = set(new.node_ids)

    added_nodes = sorted(new_nodes - old_nodes)
    removed_nodes = sorted(old_nodes - new_nodes)
    modified_nodes = sorted(
        nid
        for nid in old_nodes & new_nodes
        if _node_signature(old, nid) != _node_signature(new, nid)
    )

    old_edges = {(u, v) for u, v in old.digraph.edges()}
    new_edges = {(u, v) for u, v in new.digraph.edges()}

    added_edges = sorted(new_edges - old_edges)
    removed_edges = sorted(old_edges - new_edges)

    return GraphDiff(
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        modified_nodes=modified_nodes,
        added_edges=added_edges,
        removed_edges=removed_edges,
    )


def _label(new: KnowledgeGraph, node_id: str) -> str:
    """Best-effort display label for a node id, falling back to the id itself."""
    if node_id in new.digraph:
        return str(new.node(node_id).get("label", node_id))
    return node_id


def format_diff(diff: GraphDiff, new: KnowledgeGraph) -> str:
    """Render a :class:`GraphDiff` as a readable Markdown summary.

    Node labels are taken from ``new`` where the node still exists (removed nodes
    fall back to their id). The header carries the headline counts.

    Args:
        diff: The computed delta.
        new: The current graph, used to resolve node labels.

    Returns:
        A Markdown string.
    """
    lines: list[str] = [
        "## Graph Diff",
        "",
        (
            f"+{len(diff.added_nodes)} added / "
            f"-{len(diff.removed_nodes)} removed / "
            f"~{len(diff.modified_nodes)} modified "
            f"(nodes)"
        ),
        (f"+{len(diff.added_edges)} added / " f"-{len(diff.removed_edges)} removed " f"(edges)"),
        "",
    ]

    sections: list[tuple[str, list[str]]] = [
        ("Added nodes", [f"- {_label(new, nid)} (`{nid}`)" for nid in diff.added_nodes]),
        ("Removed nodes", [f"- {_label(new, nid)} (`{nid}`)" for nid in diff.removed_nodes]),
        ("Modified nodes", [f"- {_label(new, nid)} (`{nid}`)" for nid in diff.modified_nodes]),
        ("Added edges", [f"- {u} → {v}" for u, v in diff.added_edges]),
        ("Removed edges", [f"- {u} → {v}" for u, v in diff.removed_edges]),
    ]

    for title, items in sections:
        if not items:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(items)
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def affected_subgraph(new: KnowledgeGraph, diff: GraphDiff, hops: int = 2) -> KnowledgeGraph:
    """Return the ``hops``-neighborhood of the changed nodes, induced on ``new``.

    Seeds are the added and modified nodes (the ones that exist in ``new``). The
    search walks *undirected* adjacency in ``new.digraph`` — predecessors and
    successors alike — out to ``hops`` levels, so the result captures both what a
    changed node points at and what points at it.

    Args:
        new: The current graph.
        diff: The delta whose added/modified nodes seed the search.
        hops: Number of edges to expand outward (``0`` returns just the seeds).

    Returns:
        A :class:`KnowledgeGraph` induced on the reached node set.
    """
    digraph = new.digraph
    seeds = [nid for nid in (*diff.added_nodes, *diff.modified_nodes) if nid in digraph]

    reached: set[str] = set(seeds)
    frontier: deque[tuple[str, int]] = deque((nid, 0) for nid in seeds)

    while frontier:
        node_id, depth = frontier.popleft()
        if depth >= hops:
            continue
        neighbors = set(digraph.predecessors(node_id)) | set(digraph.successors(node_id))
        for nbr in neighbors:
            if nbr not in reached:
                reached.add(nbr)
                frontier.append((nbr, depth + 1))

    return new.induced_subgraph(reached)
