"""Fuse the retrieval signals into one relevance score per node.

The default pipeline:

1. BM25 picks lexical *seeds* (which nodes the query is literally about).
2. Personalized PageRank spreads that signal across the weighted graph.
3. A light importance/god-node prior nudges genuinely central entities up.

When a dense backend is available, its ranking is folded in with Reciprocal
Rank Fusion (RRF) — rank-based, so it needs no score calibration between the
lexical and dense scales.
"""

from __future__ import annotations

from graphex.models import KnowledgeGraph
from graphex.retrieval.base import normalize


def importance_prior(graph: KnowledgeGraph) -> dict[str, float]:
    """A ``[0, 1]`` prior from graphify ``importance`` and god-node flags.

    Real importances are normalized first (so ordinary nodes keep their relative
    ordering), then god nodes are pinned to 1.0 on top. Graphs with no importance
    signal and no god nodes return all zeros (the prior then contributes nothing).
    """
    importances = {
        nid: float(graph.digraph.nodes[nid].get("importance", 0.0) or 0.0) for nid in graph.node_ids
    }
    prior = normalize(importances)
    for nid in graph.node_ids:
        if graph.digraph.nodes[nid].get("is_god"):
            prior[nid] = 1.0
    return prior


def fuse(
    ppr: dict[str, float],
    prior: dict[str, float],
    gamma: float = 0.1,
) -> dict[str, float]:
    """Combine the spread relevance with the importance prior.

    ``score(n) = normalize(ppr)(n) + gamma * prior(n)``.

    PPR is the primary signal; ``gamma`` keeps the prior a gentle nudge rather
    than a second opinion that can override a strong query match.
    """
    ppr_n = normalize(ppr)
    out: dict[str, float] = {}
    for nid in ppr_n:
        out[nid] = ppr_n[nid] + gamma * prior.get(nid, 0.0)
    return out


def reciprocal_rank_fusion(
    rankings: list[dict[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """Combine several score maps by Reciprocal Rank Fusion.

    For each ranking, a node at rank ``r`` (1-based, by descending score)
    contributes ``1 / (k + r)``. Robust to mismatched score scales because it
    only uses the order. Nodes absent from a ranking contribute nothing for it.
    """
    agg: dict[str, float] = {}
    for ranking in rankings:
        ordered = sorted(ranking.items(), key=lambda kv: (-kv[1], kv[0]))
        for rank, (nid, _score) in enumerate(ordered, start=1):
            agg[nid] = agg.get(nid, 0.0) + 1.0 / (k + rank)
    return agg
