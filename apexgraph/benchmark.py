"""Measure the *honest* value of token-budgeted graph retrieval.

The temptation with a tool like Apexgraph is to report a single flattering number:
"we saved 92% of your tokens!" That number is meaningless on its own — a tool
that returns nothing saves 100% of the tokens and answers 0% of the questions.
The only number that matters is the trade-off: what fraction of the *relevant*
content did the budgeted subgraph actually keep, for the tokens it spent?

So every ``(query, budget)`` pair is scored on two axes:

1. ``token_savings`` — ``1 − tokens_used / full_graph_tokens``, where
   ``full_graph_tokens`` is the cost of injecting the whole graph (the naive
   baseline). Higher is cheaper. Easy to game by retrieving less.
2. ``recall_at_budget`` — of the top-``k`` nodes by full-graph relevance (the
   retrieval target), what fraction did the budgeted subgraph capture? Higher is
   better. *This* is the metric that exposes under-retrieval: a tiny budget posts
   a gorgeous ``token_savings`` and a damning ``recall_at_budget``.

Read them together. Savings without recall is just throwing the answer away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apexgraph.budget import (
    _NODE_OVERHEAD_TOKENS,
    _node_body,
    count_tokens,
    select_subgraph,
)
from apexgraph.cache import CachedArtifacts, load_or_build
from apexgraph.models import KnowledgeGraph
from apexgraph.scorer import score_nodes

DEFAULT_BUDGETS: tuple[int, ...] = (2000, 4000, 8000)
DEFAULT_K_RELEVANT = 10


@dataclass(slots=True)
class BenchmarkRow:
    """One ``(query, budget)`` measurement.

    Attributes:
        query: The query that was scored.
        budget: The token ceiling the subgraph was selected under.
        nodes_selected: How many nodes the budgeted subgraph kept.
        tokens_used: Rendered token cost of the budgeted subgraph.
        full_graph_tokens: Token cost of injecting every node (the baseline).
        token_savings: ``1 − tokens_used / full_graph_tokens`` in ``[0, 1]``.
        recall_at_budget: Fraction of the relevant top-k set captured, in ``[0, 1]``.
    """

    query: str
    budget: int
    nodes_selected: int
    tokens_used: int
    full_graph_tokens: int
    token_savings: float
    recall_at_budget: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-friendly mapping."""
        return {
            "query": self.query,
            "budget": self.budget,
            "nodes_selected": self.nodes_selected,
            "tokens_used": self.tokens_used,
            "full_graph_tokens": self.full_graph_tokens,
            "token_savings": self.token_savings,
            "recall_at_budget": self.recall_at_budget,
        }


@dataclass(slots=True)
class BenchmarkResult:
    """The full benchmark: one :class:`BenchmarkRow` per ``(query, budget)``."""

    rows: list[BenchmarkRow] = field(default_factory=list)

    @property
    def mean_token_savings(self) -> float:
        """Average ``token_savings`` across all rows (0.0 if empty)."""
        if not self.rows:
            return 0.0
        return sum(r.token_savings for r in self.rows) / len(self.rows)

    @property
    def mean_recall(self) -> float:
        """Average ``recall_at_budget`` across all rows (0.0 if empty)."""
        if not self.rows:
            return 0.0
        return sum(r.recall_at_budget for r in self.rows) / len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict (rows plus the aggregate means)."""
        return {
            "rows": [r.to_dict() for r in self.rows],
            "mean_token_savings": self.mean_token_savings,
            "mean_recall": self.mean_recall,
        }


def _full_graph_tokens(graph: KnowledgeGraph, model: str) -> int:
    """Token cost of injecting the whole graph — the naive baseline to beat.

    Computed with the *same* per-node accounting as the budgeted selection
    (:func:`apexgraph.budget._node_body` + per-node overhead), so ``token_savings``
    is a fair like-for-like comparison rather than a pessimistic one.
    """
    return sum(
        count_tokens(_node_body(graph, nid, None), model) + _NODE_OVERHEAD_TOKENS
        for nid in graph.node_ids
    )


def _relevant_set(scores: dict[str, float], k: int) -> set[str]:
    """The retrieval target: the top-``k`` nodes by full-graph relevance.

    Ties are broken by node id (ascending) so the set is deterministic. Nodes
    with a non-positive score are never relevant — an empty graph (or one where
    nothing matched) yields an empty target rather than padding with noise.
    """
    ranked = sorted(
        (nid for nid, s in scores.items() if s > 0.0),
        key=lambda nid: (-scores[nid], nid),
    )
    return set(ranked[:k])


def _measure(
    graph: KnowledgeGraph,
    query: str,
    budget: int,
    scores: dict[str, float],
    relevant: set[str],
    full_tokens: int,
    *,
    model: str,
    min_score: float,
) -> BenchmarkRow:
    """Score a single ``(query, budget)`` pair into a :class:`BenchmarkRow`."""
    sub, stats = select_subgraph(graph, scores, budget, model=model, min_score=min_score)
    tokens_used = int(stats["tokens_used"])

    token_savings = 1.0 - tokens_used / full_tokens if full_tokens > 0 else 0.0
    token_savings = max(0.0, min(1.0, token_savings))

    if relevant:
        captured = len(set(sub.node_ids) & relevant)
        recall = captured / len(relevant)
    else:
        recall = 0.0

    return BenchmarkRow(
        query=query,
        budget=budget,
        nodes_selected=int(stats["nodes_selected"]),
        tokens_used=tokens_used,
        full_graph_tokens=full_tokens,
        token_savings=round(token_savings, 4),
        recall_at_budget=round(recall, 4),
    )


def run_benchmark(
    graph: KnowledgeGraph,
    queries: list[str],
    budgets: list[int] | tuple[int, ...] = DEFAULT_BUDGETS,
    *,
    model: str = "cl100k_base",
    k_relevant: int = DEFAULT_K_RELEVANT,
    min_score: float = 0.0,
    cache: CachedArtifacts | None = None,
) -> BenchmarkResult:
    """Benchmark budgeted retrieval over a grid of ``queries × budgets``.

    For each query the full-graph relevance is scored once; the top-``k_relevant``
    nodes define the retrieval target. Each budget then selects a subgraph and is
    scored on ``token_savings`` and ``recall_at_budget`` against that target.

    The query-independent BM25/PageRank artifacts are computed once and reused
    across every query, so the grid costs one walk per query, not per cell.

    Args:
        graph: The knowledge graph to benchmark against.
        queries: Free-text queries to evaluate.
        budgets: Token ceilings to sweep (default ``(2000, 4000, 8000)``).
        model: tiktoken encoding for token counting.
        k_relevant: Size of the relevant top-k target set per query.
        min_score: Drop candidates below this score before selection.
        cache: Precomputed artifacts to reuse. Built in-memory when omitted.

    Returns:
        A :class:`BenchmarkResult` with one row per ``(query, budget)`` pair.
    """
    artifacts = cache if cache is not None else load_or_build(graph, use_cache=False)
    full_tokens = _full_graph_tokens(graph, model)

    rows: list[BenchmarkRow] = []
    for query in queries:
        scores = score_nodes(graph, query, cache=artifacts)
        relevant = _relevant_set(scores, k_relevant)
        for budget in budgets:
            rows.append(
                _measure(
                    graph,
                    query,
                    budget,
                    scores,
                    relevant,
                    full_tokens,
                    model=model,
                    min_score=min_score,
                )
            )
    return BenchmarkResult(rows=rows)


# -- formatting --------------------------------------------------------------

_HEADERS: tuple[str, ...] = (
    "query",
    "budget",
    "nodes",
    "tokens_used",
    "full_tokens",
    "token_savings",
    "recall@budget",
)


def _truncate(text: str, width: int) -> str:
    """Clip ``text`` to ``width`` columns, marking elision with an ellipsis."""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def format_benchmark(result: BenchmarkResult, *, max_query_width: int = 28) -> str:
    """Render a benchmark as a plain, aligned text table.

    Shows every ``(query, budget)`` row plus the aggregate means, and closes with
    a one-line reminder that high savings with low recall means under-retrieval.

    Args:
        result: The benchmark to render.
        max_query_width: Column width cap for the (possibly long) query text.

    Returns:
        A multi-line string ending in a newline.
    """
    rows = result.rows
    cells: list[tuple[str, ...]] = []
    for r in rows:
        cells.append(
            (
                _truncate(r.query, max_query_width),
                str(r.budget),
                str(r.nodes_selected),
                str(r.tokens_used),
                str(r.full_graph_tokens),
                f"{r.token_savings:.1%}",
                f"{r.recall_at_budget:.1%}",
            )
        )

    widths = [len(h) for h in _HEADERS]
    for row in cells:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def _fmt(values: tuple[str, ...] | list[str]) -> str:
        # First column left-aligned (text); the rest right-aligned (numbers).
        parts = [values[0].ljust(widths[0])]
        parts.extend(values[i].rjust(widths[i]) for i in range(1, len(values)))
        return "  ".join(parts)

    sep = "  ".join("-" * w for w in widths)
    lines: list[str] = ["Apexgraph retrieval benchmark", "", _fmt(_HEADERS), sep]
    lines.extend(_fmt(row) for row in cells)
    if not cells:
        lines.append("(no rows)")
    lines.append(sep)
    lines.append(
        f"mean token_savings: {result.mean_token_savings:.1%}   "
        f"mean recall@budget: {result.mean_recall:.1%}"
    )
    lines.append("")
    lines.append(
        "Note: high token-savings with low recall@budget means under-retrieval — "
        "the budget is too tight to keep the relevant nodes."
    )
    return "\n".join(lines) + "\n"
