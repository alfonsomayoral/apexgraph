"""Select the best subgraph that fits a token budget.

Picking the highest-value set of nodes under a token ceiling is a 0/1 knapsack.
Greedy-by-score (what simpler tools do) ignores cost — a 0.9 node worth 400
tokens is a worse buy than two 0.6 nodes worth 80 each. Graphex selects by
*marginal value per token* and shapes the result with two extra terms:

    gain(n | S) = relevance(n)
                − redundancy_weight · sim(n, S)        # MMR: avoid saying it twice
                + connectivity_bonus · adjacent(n, S)  # keep the subgraph coherent

so the output is diverse and connected, not a bag of redundant islands.

Token accounting is *honest*: a node's cost is the size of its final rendered
form, including any injected source code — so ``tokens_used`` never lies and the
result never overflows the budget you asked for.
"""

from __future__ import annotations

import functools
from pathlib import Path

import networkx as nx
import tiktoken

from graphex.injector import extract_code_block, safe_source_path
from graphex.models import KnowledgeGraph
from graphex.retrieval.base import normalize

# Flat per-node allowance for the markdown heading, file line, score, and a
# relationship line — keeps cost a slight over-estimate so output never overflows.
_NODE_OVERHEAD_TOKENS = 12

# Default relevance floor. Positive so that queries with no real match select
# nothing (an honest "no relevant nodes") instead of padding the budget with
# centrality noise. Shared by the CLI, MCP server, and benchmark.
DEFAULT_MIN_SCORE = 0.05


@functools.lru_cache(maxsize=8)
def _encoding(model: str) -> tiktoken.Encoding:
    """Resolve a tiktoken encoding from either an encoding name or a model id."""
    try:
        return tiktoken.get_encoding(model)
    except (ValueError, KeyError):
        pass
    try:
        return tiktoken.encoding_for_model(model)
    except (ValueError, KeyError):
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Token count for ``text`` under a tiktoken encoding (encoding is cached)."""
    if not text:
        return 0
    return len(_encoding(model).encode(text))


def _node_body(graph: KnowledgeGraph, node_id: str, code_block: str | None) -> str:
    """The text we will actually render for a node — what its cost must reflect."""
    a = graph.digraph.nodes[node_id]
    parts: list[str] = [str(node_id)]
    if label := a.get("label"):
        parts.append(str(label))
    if ntype := a.get("type"):
        parts.append(f"({ntype})")
    if desc := a.get("description"):
        parts.append(str(desc))
    if fpath := (a.get("source_file") or a.get("file_path")):
        parts.append(f"-> {fpath}")
    text = " ".join(parts)
    if code_block:
        text = f"{text}\n{code_block}"
    return text


def _node_cost(
    graph: KnowledgeGraph,
    node_id: str,
    code_block: str | None,
    model: str,
    token_costs: dict[str, int] | None,
) -> int:
    """Token cost for a candidate: base body cost plus any injected code.

    The *base* cost is the node body WITHOUT code (``_node_body(graph, nid, None)``)
    plus :data:`_NODE_OVERHEAD_TOKENS`. It is taken from ``token_costs`` when the
    caller precomputed it, else recomputed inline — the two paths agree by
    construction. Injected code is always counted separately and added on top, so
    the total is identical whether or not the base was precomputed.
    """
    if token_costs is not None and node_id in token_costs:
        base = token_costs[node_id]
    else:
        base = count_tokens(_node_body(graph, node_id, None), model) + _NODE_OVERHEAD_TOKENS
    if code_block:
        base += count_tokens(code_block, model)
    return base


def _extract_codes(
    graph: KnowledgeGraph,
    candidates: list[str],
    project_root: Path,
) -> dict[str, str]:
    """Pull source bodies for candidates that carry source coordinates."""
    codes: dict[str, str] = {}
    for nid in candidates:
        a = graph.digraph.nodes[nid]
        src = a.get("source_file") or a.get("file_path")
        loc = a.get("source_location")
        if not src or loc is None:
            continue
        # Contain reads to project_root: an untrusted graph must not be able to
        # exfiltrate arbitrary host files via a crafted source_file path.
        path = safe_source_path(project_root, src)
        if path is None:
            continue
        block = extract_code_block(path, loc)
        if block:
            codes[nid] = block
    return codes


def _token_set(graph: KnowledgeGraph, node_id: str) -> frozenset[str]:
    """Lowercased word set of a node's text, for redundancy (MMR) similarity."""
    text = graph.node_text(node_id).lower()
    return frozenset(w for w in text.replace("_", " ").split() if w)


def _similarity(
    graph: KnowledgeGraph,
    a: str,
    b: str,
    tokens: dict[str, frozenset[str]],
) -> float:
    """Redundancy similarity in ``[0, 1]``: half community match, half token Jaccard."""
    comm_a, comm_b = graph.community_of(a), graph.community_of(b)
    comm = 1.0 if comm_a is not None and comm_a == comm_b else 0.0
    ta, tb = tokens[a], tokens[b]
    jac = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return 0.5 * comm + 0.5 * jac


def _empty_stats(nodes_total: int, budget: int) -> dict:
    return {
        "nodes_selected": 0,
        "nodes_total": nodes_total,
        "tokens_used": 0,
        "tokens_budget": budget,
        "coverage_pct": 0.0,
    }


def select_subgraph(
    graph: KnowledgeGraph,
    scores: dict[str, float],
    budget: int,
    *,
    model: str = "cl100k_base",
    min_score: float = DEFAULT_MIN_SCORE,
    redundancy_weight: float = 0.3,
    connectivity_bonus: float = 0.2,
    inject_code: bool = False,
    project_root: Path | None = None,
    strategy: str = "greedy",
    token_costs: dict[str, int] | None = None,
    connected: bool = False,
) -> tuple[KnowledgeGraph, dict]:
    """Choose a subgraph maximising relevant, diverse, connected coverage under ``budget``.

    Args:
        graph: Source knowledge graph.
        scores: ``{node_id: relevance}`` from :func:`graphex.scorer.score_nodes`.
        budget: Maximum total tokens for the rendered subgraph.
        model: tiktoken encoding for counting.
        min_score: Drop candidates scoring below this before selecting.
        redundancy_weight: MMR penalty (λ) for similarity to already-picked nodes.
        connectivity_bonus: Reward (μ) for being adjacent to the selected set.
        inject_code: Extract and include source bodies (counted in the budget).
        project_root: Root for resolving ``source_file`` when ``inject_code``.
        strategy: ``"greedy"`` (cost-aware MMR, default) or ``"exact"`` (DP knapsack
            on relevance only — no diversity, for benchmarking the value ceiling).
        token_costs: Optional ``{node_id: base_cost}`` precomputed by a caching layer.
            Each value is the *base* cost — ``_node_body`` WITHOUT code plus
            ``_NODE_OVERHEAD_TOKENS`` — used in place of recomputing it. When
            ``inject_code`` is on, the injected code's own token cost is still added
            on top of the supplied base. The result is identical to computing inline.
        connected: When ``True``, repair the greedy result into a single connected
            subgraph by adding bridge nodes (approximate Steiner) — never exceeding
            ``budget`` (best-effort: adds only bridges that fit).

    Returns:
        ``(subgraph, stats)``. ``stats`` has ``nodes_selected, nodes_total,
        tokens_used, tokens_budget, coverage_pct``. ``tokens_used <= budget`` always.
    """
    nodes_total = graph.digraph.number_of_nodes()
    if nodes_total == 0 or budget <= 0:
        return KnowledgeGraph(), _empty_stats(nodes_total, budget)

    candidates = [nid for nid in graph.node_ids if scores.get(nid, 0.0) >= min_score]
    if not candidates:
        return KnowledgeGraph(), _empty_stats(nodes_total, budget)

    codes: dict[str, str] = {}
    if inject_code:
        root = project_root or Path.cwd()
        codes = _extract_codes(graph, candidates, root)

    cost: dict[str, int] = {
        nid: _node_cost(graph, nid, codes.get(nid), model, token_costs) for nid in candidates
    }
    # A node that cannot fit even alone is unselectable; drop it up front.
    candidates = [nid for nid in candidates if cost[nid] <= budget]
    if not candidates:
        return KnowledgeGraph(), _empty_stats(nodes_total, budget)

    if strategy == "exact":
        selected, tokens_used = _knapsack_exact(candidates, scores, cost, budget)
    else:
        selected, tokens_used = _greedy_mmr(
            graph,
            candidates,
            scores,
            cost,
            budget,
            redundancy_weight=redundancy_weight,
            connectivity_bonus=connectivity_bonus,
        )

    if connected and selected:
        bridges, tokens_used = _connect_components(
            graph, selected, tokens_used, budget, cost, model, token_costs
        )
        if bridges:
            selected = selected + bridges

    sub = graph.induced_subgraph(selected)
    if inject_code:
        for nid in selected:
            if nid in codes:
                sub.digraph.nodes[nid]["code_block"] = codes[nid]

    stats = {
        "nodes_selected": len(selected),
        "nodes_total": nodes_total,
        "tokens_used": tokens_used,
        "tokens_budget": budget,
        "coverage_pct": round(len(selected) / nodes_total * 100, 1),
    }
    return sub, stats


def _greedy_mmr(
    graph: KnowledgeGraph,
    candidates: list[str],
    scores: dict[str, float],
    cost: dict[str, int],
    budget: int,
    *,
    redundancy_weight: float,
    connectivity_bonus: float,
) -> tuple[list[str], int]:
    """Cost-aware greedy: repeatedly take the best gain-per-token that still fits.

    Redundancy (max similarity to the selected set) and connectivity (adjacency
    to the selected set) are tracked incrementally, so each round is O(candidates).
    """
    rel = normalize(scores)
    tokens = {nid: _token_set(graph, nid) for nid in candidates}

    # Adjacency among candidates (directed edges either way, plus hyperedge cliques).
    cand_set = set(candidates)
    neighbors: dict[str, set[str]] = {nid: set() for nid in candidates}
    for nid in candidates:
        for nbr in set(graph.digraph.predecessors(nid)) | set(graph.digraph.successors(nid)):
            if nbr in cand_set:
                neighbors[nid].add(nbr)
    for u, v, _w in graph.clique_edges():
        if u in cand_set and v in cand_set:
            neighbors[u].add(v)

    remaining = set(candidates)
    selected: list[str] = []
    tokens_used = 0
    max_redundancy: dict[str, float] = {nid: 0.0 for nid in candidates}
    connected: set[str] = set()

    while remaining:
        best_nid: str | None = None
        best_priority = float("-inf")
        budget_left = budget - tokens_used

        for nid in remaining:
            if cost[nid] > budget_left:
                continue
            gain = rel.get(nid, 0.0)
            gain -= redundancy_weight * max_redundancy[nid]
            if nid in connected:
                gain += connectivity_bonus
            priority = gain / cost[nid]
            if priority > best_priority:
                best_priority = priority
                best_nid = nid

        if best_nid is None:
            break  # nothing left fits the remaining budget

        selected.append(best_nid)
        tokens_used += cost[best_nid]
        remaining.discard(best_nid)

        # Incremental updates for the newly selected node.
        for nid in remaining:
            sim = _similarity(graph, nid, best_nid, tokens)
            if sim > max_redundancy[nid]:
                max_redundancy[nid] = sim
        connected.update(neighbors[best_nid] & remaining)

    # Density-greedy can leave the single most relevant node out when a cheap,
    # low-value node was taken first (classic knapsack failure). Guarantee the
    # standard `max(greedy, best single item)` bound so the top hit is never lost.
    best_single = max(candidates, key=lambda n: rel.get(n, 0.0))
    if rel.get(best_single, 0.0) > sum(rel.get(n, 0.0) for n in selected):
        return [best_single], cost[best_single]

    return selected, tokens_used


def _connect_components(
    graph: KnowledgeGraph,
    selected: list[str],
    tokens_used: int,
    budget: int,
    cost: dict[str, int],
    model: str,
    token_costs: dict[str, int] | None,
) -> tuple[list[str], int]:
    """Approximate-Steiner repair: stitch the selected set into one component.

    Treats ``graph.digraph`` as undirected. While the selected nodes span more than
    one connected component (in the full-graph undirected view), it finds the two
    nearest components and the shortest bridging path between them, then adds the
    intermediate "bridge" nodes — but only if their combined cost fits the budget
    that is still free. Bridge nodes were not candidates, so their cost is computed
    here (base from ``token_costs``/recompute; no code injected for bridges).

    Best-effort and budget-safe: it never adds a bridge that would push
    ``tokens_used`` past ``budget`` and stops when nothing more fits, leaving the
    result possibly still disconnected rather than overflowing.

    Returns ``(bridge_nodes_added, new_tokens_used)``.
    """
    undirected = graph.digraph.to_undirected(as_view=True)
    in_subgraph = set(selected)
    bridge_cost: dict[str, int] = {}

    def node_cost(nid: str) -> int:
        if nid not in bridge_cost:
            bridge_cost[nid] = _node_cost(graph, nid, None, model, token_costs)
        return bridge_cost[nid]

    added: list[str] = []
    while True:
        # Components of the current selection, viewed in the full undirected graph.
        comps = list(nx.connected_components(undirected.subgraph(in_subgraph)))
        if len(comps) <= 1:
            break

        # Find the cheapest (fewest-bridge) path joining any two components, using a
        # multi-source BFS from one component out to the nearest other component.
        best_bridges: list[str] | None = None
        comp_id = {nid: i for i, comp in enumerate(comps) for nid in comp}
        for i, comp in enumerate(comps):
            try:
                # Shortest hop path from any node in comp to any node in another comp.
                bridges = _nearest_bridge(undirected, comp, comp_id, i)
            except nx.NetworkXNoPath:
                bridges = None
            if bridges is not None and (best_bridges is None or len(bridges) < len(best_bridges)):
                best_bridges = bridges

        if best_bridges is None:
            break  # components are unreachable from each other; cannot connect

        # Only the truly new bridge nodes cost anything.
        new_bridges = [b for b in best_bridges if b not in in_subgraph]
        extra = sum(node_cost(b) for b in new_bridges)
        if tokens_used + extra > budget:
            break  # this bridge doesn't fit; best-effort stops here

        tokens_used += extra
        for b in new_bridges:
            in_subgraph.add(b)
            added.append(b)

    return added, tokens_used


def _nearest_bridge(
    undirected,
    source_comp: set[str],
    comp_id: dict[str, int],
    source_id: int,
) -> list[str] | None:
    """Shortest path (incl. endpoints) from ``source_comp`` to the nearest other
    selected component, as a node list — or ``None`` if no path reaches one.

    Multi-source BFS over the full undirected graph: it expands outward from the
    whole source component and stops at the first node belonging to a different
    selected component, reconstructing the bridging path back to the source.
    """
    visited: dict[str, str | None] = {n: None for n in source_comp}
    frontier: list[str] = list(source_comp)
    target: str | None = None
    while frontier and target is None:
        nxt: list[str] = []
        for u in frontier:
            for v in undirected.neighbors(u):
                if v in visited:
                    continue
                visited[v] = u
                if comp_id.get(v, source_id) != source_id:
                    target = v
                    break
                nxt.append(v)
            if target is not None:
                break
        frontier = nxt
    if target is None:
        return None
    # Reconstruct path target -> ... -> source root.
    path: list[str] = []
    cur: str | None = target
    while cur is not None:
        path.append(cur)
        cur = visited[cur]
    path.reverse()
    return path


def _knapsack_exact(
    candidates: list[str],
    scores: dict[str, float],
    cost: dict[str, int],
    budget: int,
) -> tuple[list[str], int]:
    """Exact 0/1 knapsack maximising total relevance (no diversity terms).

    Falls back to nothing fancy — used for benchmarking the achievable value
    ceiling on small candidate sets. Guarded against pathological sizes.
    """
    if budget * len(candidates) > 5_000_000:
        # Too large for the DP table; defer to the caller's greedy path instead.
        return _greedy_mmr_relevance_only(candidates, scores, cost, budget)

    # dp[c] = best (value, frozenset of chosen) achievable with cost exactly ≤ c.
    dp_val = [0.0] * (budget + 1)
    dp_set: list[tuple[str, ...]] = [()] * (budget + 1)
    for nid in candidates:
        w = cost[nid]
        v = scores.get(nid, 0.0)
        for c in range(budget, w - 1, -1):
            cand = dp_val[c - w] + v
            if cand > dp_val[c]:
                dp_val[c] = cand
                dp_set[c] = dp_set[c - w] + (nid,)
    chosen = list(dp_set[budget])
    tokens_used = sum(cost[n] for n in chosen)
    return chosen, tokens_used


def _greedy_mmr_relevance_only(
    candidates: list[str],
    scores: dict[str, float],
    cost: dict[str, int],
    budget: int,
) -> tuple[list[str], int]:
    """Density-greedy by relevance/cost — the exact-path fallback for huge inputs."""
    ordered = sorted(candidates, key=lambda n: scores.get(n, 0.0) / cost[n], reverse=True)
    selected: list[str] = []
    tokens_used = 0
    for nid in ordered:
        if tokens_used + cost[nid] <= budget:
            selected.append(nid)
            tokens_used += cost[nid]
    return selected, tokens_used
