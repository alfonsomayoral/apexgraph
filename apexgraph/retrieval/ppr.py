"""Personalized PageRank / Random Walk with Restart over the weighted graph.

This module turns a seed distribution (e.g. the BM25 top-k) into a structural
relevance score for *every* node, by simulating a random surfer that — at each
step — either follows a weighted out-edge with probability ``alpha`` or teleports
back to the seed distribution with probability ``1 - alpha``. The query-dependent
variant (:func:`personalized_pagerank`) restarts on the caller's seeds; the
query-independent variant (:func:`global_pagerank`) restarts uniformly and serves
as a mild centrality prior elsewhere.

Transition structure
--------------------
The walk runs over a *combined* weighted adjacency built from two sources:

- the directed edges of ``kg.digraph``, each contributing an effective weight of
  ``max(0, weight) * max(0, confidence_score)`` (see :attr:`apexgraph.models.Edge`),
- the exploded clique pairs returned by :meth:`KnowledgeGraph.clique_edges`, so
  hyperedge co-participation propagates relevance even without a pairwise edge.

When a pair ``(u, v)`` appears in both sources, the weights are *summed*.

The implementation is pure Python over NetworkX structures (no numpy): the
predecessor lists and out-weights are precomputed once, then reused across every
power-iteration step.
"""

from __future__ import annotations

from apexgraph.models import KnowledgeGraph

__all__ = [
    "personalized_pagerank",
    "global_pagerank",
    "normalize_max",
]


def _build_adjacency(
    graph: KnowledgeGraph,
) -> tuple[
    list[str],
    dict[str, list[tuple[str, float]]],
    dict[str, float],
]:
    """Build the combined weighted transition structure for the walk.

    Combines the directed edges of ``graph.digraph`` (weighted by
    ``max(0, weight) * max(0, confidence_score)``) with the exploded clique pairs
    from :meth:`KnowledgeGraph.clique_edges`; weights are summed when a pair
    appears in both. Zero/negative-weight pairs are dropped.

    Returns:
        A ``(nodes, predecessors, out_weight)`` triple where:

        - ``nodes`` is the list of node ids (the iteration order),
        - ``predecessors`` maps each node ``n`` to a list of ``(p, w_pn)`` pairs,
          i.e. the incoming edges used to pull rank *into* ``n``,
        - ``out_weight`` maps each node ``p`` to its total outgoing weight
          ``Σ_n w_pn`` (used to normalise the pull and to detect dangling nodes).
    """
    nodes: list[str] = list(graph.digraph.nodes)

    # Accumulate combined edge weights into a {source: {target: weight}} map so
    # duplicate pairs (digraph + clique) are summed exactly once.
    combined: dict[str, dict[str, float]] = {n: {} for n in nodes}

    for u, v, data in graph.digraph.edges(data=True):
        w = max(0.0, float(data.get("weight", 1.0))) * max(
            0.0, float(data.get("confidence_score", 1.0))
        )
        if w > 0.0:
            combined[u][v] = combined[u].get(v, 0.0) + w

    for u, v, w in graph.clique_edges():
        w = max(0.0, float(w))
        if w > 0.0:
            combined[u][v] = combined[u].get(v, 0.0) + w

    predecessors: dict[str, list[tuple[str, float]]] = {n: [] for n in nodes}
    out_weight: dict[str, float] = {n: 0.0 for n in nodes}

    for u, targets in combined.items():
        for v, w in targets.items():
            out_weight[u] += w
            predecessors[v].append((u, w))

    return nodes, predecessors, out_weight


def _restart_distribution(nodes: list[str], seeds: dict[str, float]) -> dict[str, float]:
    """Normalise ``seeds`` into a restart distribution over ``nodes``.

    Seed ids absent from the graph are ignored. If the surviving seeds are empty
    or sum to a non-positive total, falls back to the uniform distribution.
    """
    present = set(nodes)
    kept = {n: float(w) for n, w in seeds.items() if n in present and float(w) > 0.0}
    total = sum(kept.values())
    if total <= 0.0:
        u = 1.0 / len(nodes)
        return {n: u for n in nodes}
    return {n: kept.get(n, 0.0) / total for n in nodes}


def _power_iteration(
    nodes: list[str],
    predecessors: dict[str, list[tuple[str, float]]],
    out_weight: dict[str, float],
    restart: dict[str, float],
    alpha: float,
    max_iter: int,
    tol: float,
) -> dict[str, float]:
    """Run restarted power iteration to convergence.

    Implements

        r_n^{t+1} = alpha * Σ_{p→n} r_p^t * w_pn / outw_p
                  + alpha * (Σ_{d dangling} r_d^t) * restart_n
                  + (1 - alpha) * restart_n

    where dangling nodes (zero total out-weight) redistribute their mass along
    ``restart`` — the personalized variant teleports dangling mass to the seed
    distribution, not uniformly. Stops when ``Σ_n |r^{t+1} - r^t| < N * tol`` or
    after ``max_iter`` iterations.
    """
    n_nodes = len(nodes)
    dangling = [n for n in nodes if out_weight[n] <= 0.0]

    # Start from the restart distribution (a sensible, already-normalised prior).
    rank: dict[str, float] = dict(restart)

    for _ in range(max_iter):
        dangling_mass = sum(rank[n] for n in dangling)
        base = alpha * dangling_mass

        new_rank: dict[str, float] = {}
        for n in nodes:
            pull = 0.0
            for p, w in predecessors[n]:
                pull += rank[p] * w / out_weight[p]
            new_rank[n] = alpha * pull + (base + (1.0 - alpha)) * restart[n]

        delta = sum(abs(new_rank[n] - rank[n]) for n in nodes)
        rank = new_rank
        if delta < n_nodes * tol:
            break

    # Guard against tiny floating-point drift so the result sums to ~1.
    total = sum(rank.values())
    if total > 0.0:
        rank = {n: v / total for n, v in rank.items()}
    return rank


def personalized_pagerank(
    graph: KnowledgeGraph,
    seeds: dict[str, float],
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Personalized PageRank (Random Walk with Restart) over the weighted graph.

    Args:
        graph: The knowledge graph to walk.
        seeds: A ``{node_id: weight}`` restart distribution (e.g. BM25 top-k).
            Normalised to sum 1 over node ids present in the graph; ids absent
            from the graph are ignored. If the seeds are empty or sum to 0, the
            uniform distribution is used instead.
        alpha: Damping / continuation probability (``1 - alpha`` is the restart
            probability). Defaults to ``0.85``.
        max_iter: Maximum power-iteration steps.
        tol: Per-node convergence tolerance; iteration stops once the total L1
            change drops below ``len(graph) * tol``.

    Returns:
        A ``{node_id: rank}`` mapping summing to ~1.0. Empty graph yields ``{}``;
        a single-node graph yields ``{that_node: 1.0}``.
    """
    nodes, predecessors, out_weight = _build_adjacency(graph)
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: 1.0}

    restart = _restart_distribution(nodes, seeds)
    return _power_iteration(nodes, predecessors, out_weight, restart, alpha, max_iter, tol)


def global_pagerank(
    graph: KnowledgeGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Query-independent global PageRank with a uniform restart distribution.

    Identical machinery to :func:`personalized_pagerank` but teleporting to the
    uniform distribution, so the result is a mild centrality prior independent of
    any query. Empty graph yields ``{}``; a single node yields ``{that: 1.0}``.
    """
    nodes, predecessors, out_weight = _build_adjacency(graph)
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: 1.0}

    u = 1.0 / len(nodes)
    restart = {n: u for n in nodes}
    return _power_iteration(nodes, predecessors, out_weight, restart, alpha, max_iter, tol)


def normalize_max(scores: dict[str, float]) -> dict[str, float]:
    """Scale scores to ``[0, 1]`` by dividing by the max.

    Lets PPR ranks be blended predictably with other ``[0, 1]`` signals. All-zero
    (or empty) input is returned unchanged.
    """
    if not scores:
        return {}
    hi = max(scores.values())
    if hi <= 0.0:
        return {k: 0.0 for k in scores}
    return {k: v / hi for k, v in scores.items()}
