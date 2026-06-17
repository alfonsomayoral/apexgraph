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

from apexgraph.models import KnowledgeGraph
from apexgraph.retrieval.base import normalize


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
    global_pr: dict[str, float] | None = None,
    delta: float = 0.05,
) -> dict[str, float]:
    """Combine the spread relevance with the importance and structural priors.

    ``score(n) = normalize(ppr)(n) + gamma * prior(n) + delta * global_pr(n)``.

    PPR is the primary signal. ``gamma`` keeps the importance/god-node prior a
    gentle nudge; ``delta`` adds a small query-independent centrality tiebreak
    from the (cached) global PageRank. ``global_pr`` is normalized to ``[0, 1]``
    here; when omitted, the structural term contributes nothing.

    Priors apply only to nodes the walk actually reached (``ppr > 0``): they
    refine the ranking of relevant nodes without resurrecting irrelevant ones,
    so a node unreachable from the query seeds stays at exactly ``0`` (preserving
    the honest "nothing matched" behaviour downstream).
    """
    ppr_n = normalize(ppr)
    gpr_n = normalize(global_pr) if global_pr else {}
    out: dict[str, float] = {}
    for nid, base in ppr_n.items():
        if base > 0.0:
            base += gamma * prior.get(nid, 0.0) + delta * gpr_n.get(nid, 0.0)
        out[nid] = base
    return out


def seeds_from_scores(scores: dict[str, float], k: int = 10) -> dict[str, float]:
    """Top-``k`` nodes by score as a restart distribution (summing to 1).

    Used to turn a fused ranking (e.g. RRF of BM25 + semantic) into Personalized
    PageRank seeds. Keeps only positive scores; returns ``{}`` if none.
    """
    positive = [(nid, s) for nid, s in scores.items() if s > 0.0]
    if not positive:
        return {}
    positive.sort(key=lambda kv: (-kv[1], kv[0]))
    top = positive[:k]
    total = sum(s for _, s in top)
    if total <= 0.0:
        return {}
    return {nid: s / total for nid, s in top}


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
