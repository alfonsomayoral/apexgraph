"""Node exclusion via a ``.apexgraphignore`` file (gitignore syntax).

Graphs produced by graphify often carry nodes you never want surfaced — vendored
code, generated files, scratch notes. Rather than hard-coding skip rules, this
module lets a project drop a ``.apexgraphignore`` file using the exact same syntax
as ``.gitignore`` (blank lines and ``#`` comments ignored, ``!`` negation, glob
wildcards, directory anchors).

Patterns are matched against *two* strings per node: its ``id`` and its
``source_file`` attribute. A match on either excludes the node. The compiled
matcher is a :class:`pathspec.PathSpec` built with the ``gitignore`` factory,
so behaviour tracks Git's own ignore semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pathspec

from apexgraph.models import KnowledgeGraph


class ApexgraphIgnore:
    """A compiled set of ignore patterns matched against nodes.

    Wraps a :class:`pathspec.PathSpec` (``gitignore``). A node is ignored when
    either its id or its ``source_file`` matches any pattern.
    """

    def __init__(self, spec: pathspec.PathSpec) -> None:
        self.spec = spec

    @classmethod
    def from_lines(cls, lines: list[str]) -> ApexgraphIgnore:
        """Compile a :class:`ApexgraphIgnore` from raw pattern lines.

        ``pathspec`` itself skips blank lines and ``#`` comments, so the caller
        may pass the file's lines verbatim.
        """
        spec = pathspec.PathSpec.from_lines("gitignore", lines)
        return cls(spec)

    def should_ignore(self, node_attrs: dict[str, Any], node_id: str) -> bool:
        """Return ``True`` if the node should be excluded.

        ``node_attrs`` is a plain dict (e.g. a graphify node dict, or the
        attribute dict stored on the NetworkX graph). The node id and the
        ``source_file`` attribute are both tested against the patterns; a match
        on either is enough.
        """
        if self.spec.match_file(node_id):
            return True
        source_file = node_attrs.get("source_file")
        return bool(source_file and self.spec.match_file(str(source_file)))


def load_ignore(path: Path) -> ApexgraphIgnore | None:
    """Load a :class:`ApexgraphIgnore` from ``path`` if it exists.

    Returns ``None`` when the file is absent — a ``None`` ignore means "ignore
    nothing". Blank lines and ``#`` comments are handled by ``pathspec``.
    """
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    return ApexgraphIgnore.from_lines(lines)


def apply_ignore(graph: KnowledgeGraph, ignore: ApexgraphIgnore | None) -> KnowledgeGraph:
    """Return a graph with ignored nodes removed.

    If ``ignore`` is ``None`` the input graph is returned unchanged. Otherwise a
    new induced subgraph is returned containing only the nodes for which
    :meth:`ApexgraphIgnore.should_ignore` is ``False`` (carrying over the induced
    edges and side metadata via :meth:`KnowledgeGraph.induced_subgraph`).
    """
    if ignore is None:
        return graph
    kept_ids = [
        node_id
        for node_id in graph.node_ids
        if not ignore.should_ignore(graph.digraph.nodes[node_id], node_id)
    ]
    return graph.induced_subgraph(kept_ids)
