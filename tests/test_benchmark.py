"""Honest benchmark metrics: token_savings vs recall@budget, swept over budgets.

The point of these tests is the *trade-off*, not either number alone: as the
budget grows, recall must not fall and savings must not rise — and a starvation
budget must recall strictly less than a generous one.
"""

from __future__ import annotations

from apexgraph.benchmark import (
    BenchmarkResult,
    BenchmarkRow,
    format_benchmark,
    run_benchmark,
)
from apexgraph.models import Edge, KnowledgeGraph, Node

# A token budget so tight only one or two small nodes fit.
TINY_BUDGET = 60
# A budget generous enough to hold the whole graph comfortably.
HUGE_BUDGET = 100_000


def _topic_graph(n: int = 12) -> KnowledgeGraph:
    """A chain of nodes, half clearly about 'authentication', half about 'billing'.

    Descriptions are verbose so each node has a non-trivial, comparable token
    cost — that's what makes a budget sweep meaningful.
    """
    g = KnowledgeGraph()
    for i in range(n):
        if i % 2 == 0:
            topic = (
                "authentication login session token user credential password "
                "oauth verify identity access control sign in flow"
            )
        else:
            topic = (
                "billing invoice payment charge subscription receipt refund "
                "credit card transaction checkout price plan accounting"
            )
        g.add_node(
            Node(
                id=f"n{i}",
                label=f"node{i}",
                type="function",
                file_type="code",
                description=f"handles {topic} for component number {i} in the system",
                community=i % 3,
            )
        )
    for i in range(n - 1):
        g.add_edge(Edge(source=f"n{i}", target=f"n{i+1}", relation="calls"))
    return g


def test_metrics_in_unit_range():
    g = _topic_graph()
    result = run_benchmark(g, ["authentication login"], budgets=[TINY_BUDGET, 1000, HUGE_BUDGET])
    for row in result.rows:
        assert 0.0 <= row.token_savings <= 1.0
        assert 0.0 <= row.recall_at_budget <= 1.0


def test_recall_non_decreasing_in_budget():
    g = _topic_graph()
    budgets = [TINY_BUDGET, 200, 600, 1500, HUGE_BUDGET]
    result = run_benchmark(g, ["authentication login session"], budgets=budgets)
    recalls = [r.recall_at_budget for r in result.rows]
    for lo, hi in zip(recalls, recalls[1:], strict=False):
        assert hi >= lo - 1e-9, f"recall dropped as budget grew: {recalls}"


def test_savings_non_increasing_in_budget():
    g = _topic_graph()
    budgets = [TINY_BUDGET, 200, 600, 1500, HUGE_BUDGET]
    result = run_benchmark(g, ["billing payment invoice"], budgets=budgets)
    savings = [r.token_savings for r in result.rows]
    for lo, hi in zip(savings, savings[1:], strict=False):
        assert hi <= lo + 1e-9, f"savings rose as budget grew: {savings}"


def test_tiny_budget_recalls_less_than_huge():
    g = _topic_graph()
    result = run_benchmark(g, ["authentication login token"], budgets=[TINY_BUDGET, HUGE_BUDGET])
    tiny, huge = result.rows[0], result.rows[1]
    assert tiny.recall_at_budget < huge.recall_at_budget
    # The huge budget fits everything, so it captures the entire relevant set.
    assert huge.recall_at_budget == 1.0
    # ...and the starvation budget saves more tokens than the generous one.
    assert tiny.token_savings > huge.token_savings


def test_huge_budget_saves_little():
    g = _topic_graph()
    result = run_benchmark(g, ["authentication"], budgets=[HUGE_BUDGET])
    # Injecting essentially the whole graph means little is saved vs the baseline.
    assert result.rows[0].token_savings < 0.5


def test_row_count_matches_grid():
    g = _topic_graph()
    queries = ["authentication", "billing", "session token"]
    budgets = [500, 1000, 4000]
    result = run_benchmark(g, queries, budgets=budgets)
    assert len(result.rows) == len(queries) * len(budgets)


def test_aggregate_means():
    g = _topic_graph()
    result = run_benchmark(g, ["authentication", "billing"], budgets=[500, HUGE_BUDGET])
    expected_savings = sum(r.token_savings for r in result.rows) / len(result.rows)
    expected_recall = sum(r.recall_at_budget for r in result.rows) / len(result.rows)
    assert abs(result.mean_token_savings - expected_savings) < 1e-9
    assert abs(result.mean_recall - expected_recall) < 1e-9


def test_empty_result_means_are_zero():
    empty = BenchmarkResult(rows=[])
    assert empty.mean_token_savings == 0.0
    assert empty.mean_recall == 0.0


def test_to_dict_round_trips():
    g = _topic_graph()
    result = run_benchmark(g, ["authentication", "billing"], budgets=[500, 2000])
    d = result.to_dict()

    assert set(d) == {"rows", "mean_token_savings", "mean_recall"}
    assert len(d["rows"]) == len(result.rows)
    assert abs(d["mean_token_savings"] - result.mean_token_savings) < 1e-9
    assert abs(d["mean_recall"] - result.mean_recall) < 1e-9

    # Each row dict carries every field and reconstructs an equal BenchmarkRow.
    for row_dict, row in zip(d["rows"], result.rows, strict=False):
        assert BenchmarkRow(**row_dict) == row


def test_format_benchmark_non_empty_and_mentions_queries():
    g = _topic_graph()
    queries = ["authentication", "billing"]
    result = run_benchmark(g, queries, budgets=[500, 2000])
    text = format_benchmark(result)

    assert isinstance(text, str) and text.strip()
    for query in queries:
        assert query in text
    # The honest-trade-off note and the aggregate line are both present.
    assert "recall" in text.lower()
    assert "under-retrieval" in text.lower()
    assert "mean token_savings" in text


def test_format_benchmark_handles_empty():
    text = format_benchmark(BenchmarkResult(rows=[]))
    assert isinstance(text, str) and text.strip()


def test_k_relevant_caps_target_set():
    g = _topic_graph(12)
    # With k=2 the relevant set has 2 nodes; a huge budget captures both → recall 1.0.
    result = run_benchmark(g, ["authentication login"], budgets=[HUGE_BUDGET], k_relevant=2)
    assert result.rows[0].recall_at_budget == 1.0
