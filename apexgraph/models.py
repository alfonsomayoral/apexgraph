"""Core data model — a typed superset of the graphify knowledge-graph schema.

This module is the *contract* every other module builds against. It deliberately
preserves the rich signals graphify emits that simpler tools discard:

- edge ``weight`` and ``confidence_score`` (used as transition weights for PPR),
- ``hyperedges`` (3+ node co-participation, expanded to weighted cliques),
- ``community`` membership (used to diversify selection),
- ``importance`` / god-node flags (used as a relevance prior),
- the six graphify ``file_type`` values, not just ``code``.

The graph itself is stored as a :class:`networkx.DiGraph` (so we get battle-tested
traversal/serialization for free), with side metadata for the structures NetworkX
does not model natively (hyperedges, communities, god nodes).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx

# graphify's six canonical file_type values. Anything else is coerced to "concept".
FileType = Literal["code", "document", "paper", "image", "rationale", "concept"]
_VALID_FILE_TYPES: frozenset[str] = frozenset(
    {"code", "document", "paper", "image", "rationale", "concept"}
)

# graphify edge confidence tiers.
Confidence = Literal["EXTRACTED", "INFERRED", "AMBIGUOUS"]


@dataclass(slots=True)
class Node:
    """A single entity in the knowledge graph.

    ``id`` is the only strictly required field. ``label`` defaults to ``id`` so a
    bare ``{"id": ...}`` node is always renderable.
    """

    id: str
    label: str = ""
    type: str = ""
    file_type: str = "concept"
    description: str = ""
    importance: float = 0.0
    source_file: str | None = None
    source_location: str | None = None
    source_url: str | None = None
    community: int | None = None
    is_god: bool = False
    # Any attribute graphify (or another producer) emits that we don't model
    # explicitly is preserved here so nothing is lost on round-trips.
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.id
        if self.file_type not in _VALID_FILE_TYPES:
            self.file_type = "concept"

    def text(self) -> str:
        """Concatenated searchable text for lexical/semantic scoring."""
        return " ".join(p for p in (self.label, self.type, self.description) if p)

    def attrs(self) -> dict[str, Any]:
        """Node attributes for storage on the NetworkX graph (everything but ``id``)."""
        data: dict[str, Any] = {
            "label": self.label,
            "type": self.type,
            "file_type": self.file_type,
            "description": self.description,
            "importance": self.importance,
            "source_file": self.source_file,
            "source_location": self.source_location,
            "source_url": self.source_url,
            "community": self.community,
            "is_god": self.is_god,
        }
        data.update(self.extra)
        return data


@dataclass(slots=True)
class Edge:
    """A directed relationship between two nodes.

    The *effective transition weight* used by structural scoring is
    ``weight * confidence_score`` — an INFERRED edge with confidence 0.5 carries
    half the pull of an EXTRACTED edge with the same nominal weight.
    """

    source: str
    target: str
    relation: str = ""
    weight: float = 1.0
    confidence: str = "EXTRACTED"
    confidence_score: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def transition_weight(self) -> float:
        """Weight used to build the PPR transition matrix. Always non-negative."""
        return max(0.0, self.weight) * max(0.0, self.confidence_score)

    def attrs(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "relation": self.relation,
            "weight": self.weight,
            "confidence": self.confidence,
            "confidence_score": self.confidence_score,
        }
        data.update(self.extra)
        return data


@dataclass(slots=True)
class Hyperedge:
    """A higher-order relationship binding 3+ nodes that share a concept/flow.

    Expanded into weighted clique edges for the random walk so co-participation
    propagates relevance even when no pairwise edge exists.
    """

    id: str
    label: str = ""
    nodes: list[str] = field(default_factory=list)
    relation: str = ""
    confidence_score: float = 1.0

    def clique_weight(self) -> float:
        """Per-pair weight when exploded to a clique.

        Normalised by ``1/(k-1)`` so a large hyperedge doesn't swamp the graph:
        each member distributes a unit of pull across its co-members.
        """
        k = len(self.nodes)
        if k < 2:
            return 0.0
        return self.confidence_score / (k - 1)


class KnowledgeGraph:
    """Container around a :class:`networkx.DiGraph` plus graphify side metadata.

    Attributes:
        digraph: The directed graph. Node IDs are keys; attributes follow
            :meth:`Node.attrs`. Edge attributes follow :meth:`Edge.attrs`.
        hyperedges: Higher-order relationships not expressible as pairwise edges.
        communities: ``{node_id: community_index}`` (from graphify clustering).
        god_nodes: IDs of the most central / important nodes.
    """

    def __init__(self) -> None:
        self.digraph: nx.DiGraph = nx.DiGraph()
        self.hyperedges: list[Hyperedge] = []
        self.communities: dict[str, int] = {}
        self.god_nodes: set[str] = set()

    # -- construction --------------------------------------------------------

    def add_node(self, node: Node) -> None:
        self.digraph.add_node(node.id, **node.attrs())
        if node.community is not None:
            self.communities[node.id] = node.community
        if node.is_god:
            self.god_nodes.add(node.id)

    def add_edge(self, edge: Edge) -> None:
        self.digraph.add_edge(edge.source, edge.target, **edge.attrs())

    def add_hyperedge(self, hyperedge: Hyperedge) -> None:
        self.hyperedges.append(hyperedge)

    # -- accessors -----------------------------------------------------------

    def __len__(self) -> int:
        return self.digraph.number_of_nodes()

    @property
    def node_ids(self) -> list[str]:
        return list(self.digraph.nodes)

    def node(self, node_id: str) -> dict[str, Any]:
        """Return the stored attribute dict for ``node_id``."""
        return self.digraph.nodes[node_id]

    def node_text(self, node_id: str) -> str:
        """Searchable text (label + type + description) for a stored node."""
        a = self.digraph.nodes[node_id]
        return " ".join(
            str(p)
            for p in (a.get("label", node_id), a.get("type", ""), a.get("description", ""))
            if p
        )

    def community_of(self, node_id: str) -> int | None:
        return self.communities.get(node_id)

    # -- structure -----------------------------------------------------------

    def clique_edges(self) -> list[tuple[str, str, float]]:
        """Hyperedges exploded into ``(source, target, weight)`` clique triples.

        Only emits pairs whose endpoints both exist in the graph.
        """
        out: list[tuple[str, str, float]] = []
        present = set(self.digraph.nodes)
        for he in self.hyperedges:
            members = [n for n in he.nodes if n in present]
            w = he.clique_weight()
            if w <= 0.0:
                continue
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    out.append((a, b, w))
                    out.append((b, a, w))
        return out

    def induced_subgraph(self, node_ids: set[str] | list[str]) -> KnowledgeGraph:
        """Return a new :class:`KnowledgeGraph` induced on ``node_ids``.

        Carries over the induced edges, the side metadata for the kept nodes
        (communities, god nodes), and any hyperedge all of whose members survive.
        Node/edge attribute dicts are copied so mutating the subgraph (e.g.
        injecting code blocks) never touches the source graph.
        """
        keep = set(node_ids) & set(self.digraph.nodes)
        sub = KnowledgeGraph()
        for nid in self.digraph.nodes:
            if nid in keep:
                sub.digraph.add_node(nid, **dict(self.digraph.nodes[nid]))
        for u, v, data in self.digraph.edges(data=True):
            if u in keep and v in keep:
                sub.digraph.add_edge(u, v, **dict(data))
        sub.communities = {n: c for n, c in self.communities.items() if n in keep}
        sub.god_nodes = {n for n in self.god_nodes if n in keep}
        for he in self.hyperedges:
            if he.nodes and all(n in keep for n in he.nodes):
                sub.hyperedges.append(he)
        return sub

    def fingerprint(self) -> str:
        """Stable content hash, used to invalidate the on-disk cache.

        Derived from the sorted node IDs, their searchable text, edge tuples
        (with transition weights), and hyperedge membership — i.e. everything
        that feeds the cached BM25 index and PageRank. Node text is included so
        that editing a label/description (id unchanged) correctly invalidates
        the cached lexical scores.
        """
        h = hashlib.sha256()
        for nid in sorted(self.digraph.nodes):
            h.update(nid.encode("utf-8"))
            h.update(b"\x00")
            h.update(self.node_text(nid).encode("utf-8"))
            h.update(b"\x00")
        for u, v, data in sorted(self.digraph.edges(data=True)):
            w = float(data.get("weight", 1.0)) * float(data.get("confidence_score", 1.0))
            h.update(f"{u}>{v}:{w:.6f}".encode())
            h.update(b"\x00")
        for he in sorted(self.hyperedges, key=lambda x: x.id):
            h.update(he.id.encode("utf-8"))
            h.update(",".join(sorted(he.nodes)).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a graphify-compatible dict (nodes / links / hyperedges)."""
        nodes = [{"id": nid, **self.digraph.nodes[nid]} for nid in self.digraph.nodes]
        links = [{"source": u, "target": v, **data} for u, v, data in self.digraph.edges(data=True)]
        hyperedges = [
            {
                "id": he.id,
                "label": he.label,
                "nodes": he.nodes,
                "relation": he.relation,
                "confidence_score": he.confidence_score,
            }
            for he in self.hyperedges
        ]
        return {"nodes": nodes, "links": links, "hyperedges": hyperedges}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
