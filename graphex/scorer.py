"""Orchestrates retrieval into one relevance score per node.

Pipeline (default ``bm25`` backend):

    BM25 seeds ──▶ Personalized PageRank ──▶ fuse with importance prior

The lexical layer decides *what the query is about*; PageRank decides *what's
near and structurally relevant to that*; the prior gives genuinely central
entities a small edge. One principled number per node — not a hand-tuned mix of
independent axes.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphex.cache import CachedArtifacts, load_or_build
from graphex.models import KnowledgeGraph
from graphex.retrieval import fusion
from graphex.retrieval.bm25 import BM25Index
from graphex.retrieval.ppr import normalize_max, personalized_pagerank

DEFAULT_SEEDS = 10
DEFAULT_GAMMA = 0.1


@dataclass(slots=True)
class ScoreBreakdown:
    """Per-node component scores, surfaced by ``graphex query --explain``."""

    final: dict[str, float]
    bm25: dict[str, float]
    ppr: dict[str, float]
    prior: dict[str, float]


def _artifacts(
    graph: KnowledgeGraph,
    cache: CachedArtifacts | None,
) -> CachedArtifacts:
    """Reuse caller-provided artifacts, else build them in-memory (no disk)."""
    if cache is not None:
        return cache
    return load_or_build(graph, use_cache=False)


def _compute(
    graph: KnowledgeGraph,
    query: str,
    *,
    cache: CachedArtifacts | None,
    k_seeds: int,
    gamma: float,
) -> ScoreBreakdown:
    if len(graph) == 0:
        return ScoreBreakdown({}, {}, {}, {})

    artifacts = _artifacts(graph, cache)
    bm25: BM25Index = artifacts.bm25

    bm25_norm = bm25.normalized_scores(query)
    seeds = bm25.seeds(query, k=k_seeds)
    prior = fusion.importance_prior(graph)

    # No lexical hit at all → return zeros rather than confidently surfacing
    # centrality noise. For an LLM-context tool, "nothing matched" is the honest
    # answer; the downstream min_score filter then yields an empty subgraph.
    if not seeds:
        zeros = dict.fromkeys(graph.node_ids, 0.0)
        return ScoreBreakdown(final=zeros, bm25=bm25_norm, ppr=dict(zeros), prior=prior)

    ppr = personalized_pagerank(graph, seeds)
    final = fusion.fuse(ppr, prior, gamma=gamma)

    return ScoreBreakdown(
        final=final,
        bm25=bm25_norm,
        ppr=normalize_max(ppr),
        prior=prior,
    )


def score_nodes(
    graph: KnowledgeGraph,
    query: str,
    *,
    cache: CachedArtifacts | None = None,
    k_seeds: int = DEFAULT_SEEDS,
    gamma: float = DEFAULT_GAMMA,
) -> dict[str, float]:
    """Return ``{node_id: relevance}`` for every node, blending the signals.

    Args:
        graph: The knowledge graph to score.
        query: Free-text query.
        cache: Precomputed :class:`CachedArtifacts` (BM25 index + global PR). When
            omitted, they are built in-memory for this call.
        k_seeds: How many BM25 hits seed the random walk.
        gamma: Weight of the importance prior relative to the spread relevance.
    """
    return _compute(graph, query, cache=cache, k_seeds=k_seeds, gamma=gamma).final


def score_nodes_detailed(
    graph: KnowledgeGraph,
    query: str,
    *,
    cache: CachedArtifacts | None = None,
    k_seeds: int = DEFAULT_SEEDS,
    gamma: float = DEFAULT_GAMMA,
) -> ScoreBreakdown:
    """Like :func:`score_nodes` but also returns the BM25/PPR/prior components."""
    return _compute(graph, query, cache=cache, k_seeds=k_seeds, gamma=gamma)
