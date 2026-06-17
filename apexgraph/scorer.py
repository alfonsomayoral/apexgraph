"""Orchestrates retrieval into one relevance score per node.

Pipeline (default ``bm25`` backend):

    BM25 seeds ──▶ Personalized PageRank ──▶ fuse with importance + structural priors

The lexical layer decides *what the query is about*; PageRank decides *what's
near and structurally relevant to that*; the priors give genuinely central
entities a small edge. One principled number per node — not a hand-tuned mix of
independent axes.

With a semantic backend (``local`` model2vec embeddings, or the cloud ``openai``
/ ``voyage`` backends), the seeds come from a Reciprocal Rank Fusion of the BM25
and embedding rankings — so a node the query is *about* but shares no tokens with
(e.g. "login" vs "sign in") can still seed the walk. RRF is rank-based, so the
two scales need no calibration.

Note: semantic backends behave like nearest-neighbour vector search — they always
surface the closest nodes rather than returning empty for an unrelated query. The
honest "nothing matched → empty result" guarantee applies to the default ``bm25``
backend; pick it when a confident "no relevant nodes" answer matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from apexgraph.cache import CachedArtifacts, load_or_build
from apexgraph.models import KnowledgeGraph
from apexgraph.retrieval import fusion
from apexgraph.retrieval.bm25 import BM25Index
from apexgraph.retrieval.ppr import normalize_max, personalized_pagerank

DEFAULT_SEEDS = 10
DEFAULT_GAMMA = 0.1
# Weight of the query-independent global-PageRank tiebreak in the final score.
DEFAULT_DELTA = 0.05
# Selectable scoring backends. "bm25" is fully local + dependency-light; "local"
# adds offline semantic recall (model2vec); "openai"/"voyage" are cloud embeddings.
BACKENDS = ("bm25", "local", "openai", "voyage")


@dataclass(slots=True)
class ScoreBreakdown:
    """Per-node component scores, surfaced by ``apexgraph query --explain``."""

    final: dict[str, float]
    bm25: dict[str, float]
    ppr: dict[str, float]
    prior: dict[str, float]
    # Semantic similarity per node (empty for the bm25 backend).
    semantic: dict[str, float] = field(default_factory=dict)


def _artifacts(graph: KnowledgeGraph, cache: CachedArtifacts | None) -> CachedArtifacts:
    """Reuse caller-provided artifacts, else build them in-memory (no disk)."""
    if cache is not None:
        return cache
    return load_or_build(graph, use_cache=False)


def _semantic_scores(graph: KnowledgeGraph, query: str, backend: str) -> dict[str, float]:
    """Normalized semantic similarity per node from the chosen embedding backend.

    Retrievers are imported lazily so the default ``bm25`` path never touches the
    optional embedding dependencies.
    """
    if backend == "local":
        from apexgraph.retrieval.local import LocalEmbeddingRetriever

        return LocalEmbeddingRetriever().score(graph, query)
    from apexgraph.retrieval.dense import DenseRetriever

    return DenseRetriever(backend=backend).score(graph, query)


def _compute(
    graph: KnowledgeGraph,
    query: str,
    *,
    cache: CachedArtifacts | None,
    k_seeds: int,
    gamma: float,
    backend: str,
) -> ScoreBreakdown:
    if len(graph) == 0:
        return ScoreBreakdown({}, {}, {}, {})

    artifacts = _artifacts(graph, cache)
    bm25: BM25Index = artifacts.bm25
    bm25_norm = bm25.normalized_scores(query)
    prior = fusion.importance_prior(graph)
    semantic: dict[str, float] = {}

    if backend == "bm25":
        seeds = bm25.seeds(query, k=k_seeds)
    else:
        # Fuse the lexical and semantic rankings, then seed from the top of the
        # combination — a purely semantic hit (no shared tokens) can still seed.
        # Only BM25's actual matches enter the fusion: a no-match query then seeds
        # purely from semantics rather than from a sea of tied zero-score nodes.
        semantic = _semantic_scores(graph, query, backend)
        bm25_hits = {nid: s for nid, s in bm25_norm.items() if s > 0.0}
        combined = fusion.reciprocal_rank_fusion([bm25_hits, semantic])
        seeds = fusion.seeds_from_scores(combined, k=k_seeds)

    # Nothing matched at all → return zeros rather than confidently surfacing
    # centrality noise. The downstream min_score filter then yields an empty
    # subgraph — an honest "nothing relevant" for an LLM-context tool.
    if not seeds:
        zeros = dict.fromkeys(graph.node_ids, 0.0)
        return ScoreBreakdown(
            final=zeros, bm25=bm25_norm, ppr=dict(zeros), prior=prior, semantic=semantic
        )

    ppr = personalized_pagerank(graph, seeds)
    final = fusion.fuse(
        ppr, prior, gamma=gamma, global_pr=artifacts.global_pagerank, delta=DEFAULT_DELTA
    )

    return ScoreBreakdown(
        final=final, bm25=bm25_norm, ppr=normalize_max(ppr), prior=prior, semantic=semantic
    )


def score_nodes(
    graph: KnowledgeGraph,
    query: str,
    *,
    cache: CachedArtifacts | None = None,
    k_seeds: int = DEFAULT_SEEDS,
    gamma: float = DEFAULT_GAMMA,
    backend: str = "bm25",
) -> dict[str, float]:
    """Return ``{node_id: relevance}`` for every node, blending the signals.

    Args:
        graph: The knowledge graph to score.
        query: Free-text query.
        cache: Precomputed :class:`CachedArtifacts` (BM25 index + global PR). When
            omitted, they are built in-memory for this call.
        k_seeds: How many top hits seed the random walk.
        gamma: Weight of the importance prior relative to the spread relevance.
        backend: ``"bm25"`` (default, local lexical), ``"local"`` (offline
            embeddings), or ``"openai"``/``"voyage"`` (cloud embeddings).
    """
    return _compute(graph, query, cache=cache, k_seeds=k_seeds, gamma=gamma, backend=backend).final


def score_nodes_detailed(
    graph: KnowledgeGraph,
    query: str,
    *,
    cache: CachedArtifacts | None = None,
    k_seeds: int = DEFAULT_SEEDS,
    gamma: float = DEFAULT_GAMMA,
    backend: str = "bm25",
) -> ScoreBreakdown:
    """Like :func:`score_nodes` but also returns the BM25/PPR/prior components."""
    return _compute(graph, query, cache=cache, k_seeds=k_seeds, gamma=gamma, backend=backend)
