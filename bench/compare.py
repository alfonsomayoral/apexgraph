"""Reproducible benchmark: Graphex (bm25 / local) vs slurp on recall@budget.

For each labeled query (see ``bench/queries.json``) and each token budget, we run
three retrievers over the same graph and measure how much of the human-labeled
*relevant* set each one recovers within the budget:

  - Graphex bm25  : in-process lexical retrieval (the tool's default backend).
  - Graphex local : in-process bm25 + offline semantic embeddings (model2vec),
                    fused by reciprocal rank fusion.
  - slurp         : the prior-art tool, run as an opaque black box from PyPI via
                    ``uvx --from slurp-graph slurp ...``. TF-IDF retrieval.

Primary metric is recall@budget = |selected ∩ relevant| / |relevant|. We also
report precision = |selected ∩ relevant| / |selected| and the token cost, split
by query type (lexical vs semantic) because the semantic queries — which share NO
tokens with their relevant nodes — are the discriminator between lexical and
semantic retrieval.

Run:  uv run python bench/compare.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Resolve paths relative to the repo root (parent of bench/), so the harness runs
# from any working directory.
BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
GRAPH_PATH = REPO_ROOT / "examples" / "sample_graph.json"
QUERIES_PATH = BENCH_DIR / "queries.json"
RESULTS_PATH = BENCH_DIR / "results.json"

BUDGETS: tuple[int, ...] = (500, 1500, 4000)

# Graphex is imported from the repo it lives in; make sure it's importable when
# the script is invoked as a file.
sys.path.insert(0, str(REPO_ROOT))

from graphex.budget import select_subgraph  # noqa: E402
from graphex.cache import load_or_build  # noqa: E402
from graphex.loader import load_graph  # noqa: E402
from graphex.scorer import score_nodes  # noqa: E402

SLURP_TIMEOUT_S = 300  # first call installs slurp; generous to be safe.


def load_queries() -> list[dict]:
    """Load the labeled query set, dropping the leading ``_comment`` metadata."""
    data = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return data["queries"]


def graphex_select(graph, cache, query: str, budget: int, backend: str) -> tuple[set[str], int, int]:
    """Run one Graphex backend for (query, budget). Returns (ids, n_selected, tokens)."""
    scores = score_nodes(graph, query, cache=cache, backend=backend)
    sub, stats = select_subgraph(
        graph, scores, budget, token_costs=cache.token_costs
    )
    return set(sub.node_ids), int(stats["nodes_selected"]), int(stats["tokens_used"])


def slurp_select(query: str, budget: int) -> tuple[set[str], int, int] | None:
    """Run slurp as a subprocess. Returns (ids, n_selected, tokens) or None if unavailable."""
    cmd = [
        "uvx", "--from", "slurp-graph", "slurp", query,
        "-g", str(GRAPH_PATH), "-b", str(budget), "-f", "json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SLURP_TIMEOUT_S,
            cwd=str(REPO_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  [slurp unavailable: {exc}]", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"  [slurp failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}]", file=sys.stderr)
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"  [slurp output not JSON: {exc}]", file=sys.stderr)
        return None
    ids = {n["id"] for n in data.get("nodes", []) if isinstance(n, dict) and "id" in n}
    stats = data.get("stats", {}) or {}
    n_selected = int(stats.get("nodes_selected", len(ids)))
    tokens = int(stats.get("tokens_used", 0))
    return ids, n_selected, tokens


def metrics(selected: set[str], relevant: set[str]) -> tuple[float, float]:
    """recall = hits/|relevant|, precision = hits/|selected|."""
    if not relevant:
        return 0.0, 0.0
    hits = len(selected & relevant)
    recall = hits / len(relevant)
    precision = hits / max(1, len(selected))
    return recall, precision


def main() -> int:
    graph = load_graph(GRAPH_PATH)
    cache = load_or_build(graph, use_cache=False)
    queries = load_queries()

    rows: list[dict] = []
    # slurp results are independent of budget order; cache per (query, budget) so a
    # rerun within one invocation doesn't re-shell out.
    slurp_cache: dict[tuple[str, int], tuple[set[str], int, int] | None] = {}

    for q in queries:
        query = q["query"]
        qtype = q["type"]
        relevant = set(q["relevant"])
        for budget in BUDGETS:
            # Graphex bm25
            gx_ids, gx_n, gx_tok = graphex_select(graph, cache, query, budget, "bm25")
            r, p = metrics(gx_ids, relevant)
            rows.append(_row("graphex-bm25", query, qtype, budget, gx_n, gx_tok, r, p, len(relevant)))

            # Graphex local (semantic)
            lx_ids, lx_n, lx_tok = graphex_select(graph, cache, query, budget, "local")
            r, p = metrics(lx_ids, relevant)
            rows.append(_row("graphex-local", query, qtype, budget, lx_n, lx_tok, r, p, len(relevant)))

            # slurp (black box)
            key = (query, budget)
            if key not in slurp_cache:
                slurp_cache[key] = slurp_select(query, budget)
            sl = slurp_cache[key]
            if sl is None:
                rows.append(_row("slurp", query, qtype, budget, None, None, None, None, len(relevant),
                                 available=False))
            else:
                sl_ids, sl_n, sl_tok = sl
                r, p = metrics(sl_ids, relevant)
                rows.append(_row("slurp", query, qtype, budget, sl_n, sl_tok, r, p, len(relevant)))

    agg = aggregate(rows)
    out = {"budgets": list(BUDGETS), "rows": rows, "aggregate": agg}
    RESULTS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(render_per_query(rows))
    print()
    print(render_aggregate(agg))
    print(f"\nWrote {RESULTS_PATH}")
    return 0


def _row(tool, query, qtype, budget, n_selected, tokens, recall, precision, n_relevant,
         available=True) -> dict:
    return {
        "tool": tool,
        "query": query,
        "type": qtype,
        "budget": budget,
        "nodes_selected": n_selected,
        "tokens_used": tokens,
        "n_relevant": n_relevant,
        "recall": None if recall is None else round(recall, 4),
        "precision": None if precision is None else round(precision, 4),
        "available": available,
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean recall/precision per tool, split by query type and overall."""
    # buckets[(tool, scope)] = list of (recall, precision)
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if not r["available"] or r["recall"] is None:
            continue
        tool = r["tool"]
        for scope in (r["type"], "overall"):
            buckets[(tool, scope)].append((r["recall"], r["precision"]))

    tools = sorted({r["tool"] for r in rows})
    result: dict[str, dict] = {}
    for tool in tools:
        result[tool] = {}
        for scope in ("lexical", "semantic", "overall"):
            vals = buckets.get((tool, scope), [])
            if vals:
                mr = sum(v[0] for v in vals) / len(vals)
                mp = sum(v[1] for v in vals) / len(vals)
                result[tool][scope] = {
                    "mean_recall": round(mr, 4),
                    "mean_precision": round(mp, 4),
                    "n": len(vals),
                }
            else:
                result[tool][scope] = {"mean_recall": None, "mean_precision": None, "n": 0}
    return result


# -- rendering ---------------------------------------------------------------

def _cell(v, pct=False):
    if v is None:
        return "n/a"
    return f"{v:.0%}" if pct else str(v)


def render_per_query(rows: list[dict]) -> str:
    headers = ("query", "type", "budget", "tool", "sel", "tokens", "recall", "prec")
    table: list[tuple[str, ...]] = []
    for r in rows:
        table.append((
            r["query"][:24],
            r["type"],
            str(r["budget"]),
            r["tool"],
            _cell(r["nodes_selected"]),
            _cell(r["tokens_used"]),
            _cell(r["recall"], pct=True),
            _cell(r["precision"], pct=True),
        ))
    widths = [len(h) for h in headers]
    for row in table:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt(vals):
        parts = [str(vals[0]).ljust(widths[0])]
        parts += [str(vals[i]).rjust(widths[i]) for i in range(1, len(vals))]
        return "  ".join(parts)

    lines = ["Per-query results (recall@budget primary metric)", "", fmt(headers),
             "  ".join("-" * w for w in widths)]
    lines += [fmt(r) for r in table]
    return "\n".join(lines)


def render_aggregate(agg: dict) -> str:
    headers = ("tool", "scope", "mean_recall", "mean_precision", "n")
    table: list[tuple[str, ...]] = []
    for tool in sorted(agg):
        for scope in ("lexical", "semantic", "overall"):
            m = agg[tool][scope]
            table.append((
                tool, scope,
                _cell(m["mean_recall"], pct=True),
                _cell(m["mean_precision"], pct=True),
                str(m["n"]),
            ))
    widths = [len(h) for h in headers]
    for row in table:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt(vals):
        parts = [str(vals[0]).ljust(widths[0]), str(vals[1]).ljust(widths[1])]
        parts += [str(vals[i]).rjust(widths[i]) for i in range(2, len(vals))]
        return "  ".join(parts)

    lines = ["Aggregate (mean over queries x budgets)", "", fmt(headers),
             "  ".join("-" * w for w in widths)]
    lines += [fmt(r) for r in table]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
