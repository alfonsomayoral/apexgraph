"""Command-line interface for Apexgraph.

``apexgraph QUERY`` is the default: any unrecognised first argument is routed to
the hidden ``query`` command, so ``apexgraph "how does auth work"`` just works.
"""

from __future__ import annotations

import contextlib
import difflib
import sys
import tempfile
import webbrowser
from io import StringIO
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from apexgraph import __version__

_GRAPH_SEARCH_PATHS = [
    Path("graph.json"),
    Path("graphify-out/graph.json"),
    Path(".graphify/graph.json"),
    Path(".apexgraph/graph.json"),
]

_BANNER = r"""
                                       _
 __ _ _ __  _____ ____ _ _ _ __ _ _ __| |_
/ _` | '_ \/ -_) \ / _` | '_/ _` | '_ \ ' \
\__,_| .__/\___/_\_\__, |_| \__,_| .__/_||_|
     |_|           |___/         |_|  apex-relevance subgraph retrieval
"""


def _force_utf8() -> None:
    """Make stdout/stderr UTF-8 so the box-drawing output renders on Windows too."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")


def _find_graph() -> Path | None:
    for candidate in _GRAPH_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    return None


def _resolve_graph(graph: str | None) -> Path:
    path = Path(graph) if graph else _find_graph()
    if path is None:
        raise click.ClickException(
            "No graph found. Run `apexgraph index .` to build one, or pass --graph PATH.\n"
            "Searched: " + ", ".join(str(p) for p in _GRAPH_SEARCH_PATHS)
        )
    return path


def _load(graph_path: Path):
    from apexgraph.loader import ApexgraphLoadError, load_graph

    try:
        return load_graph(graph_path)
    except ApexgraphLoadError as exc:
        raise click.ClickException(str(exc)) from exc


def _echo_rich(renderable) -> None:
    """Render a rich object to a string and echo it (safe under CliRunner)."""
    buf = StringIO()
    Console(file=buf, width=100, highlight=False).print(renderable)
    click.echo(buf.getvalue(), nl=False)


def _version_callback(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(_BANNER)
    click.echo(f"apexgraph v{__version__}\n")
    ctx.exit()


class _ApexgraphGroup(click.Group):
    """Route an unrecognised first argument to the hidden ``query`` command.

    A single-token first argument that closely resembles a real command (e.g.
    ``statss``) is treated as a typo and gets a "did you mean" hint, so a
    mistyped subcommand isn't silently run as a search query.
    """

    def resolve_command(self, ctx: click.Context, args: list):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if not args:
                raise
            first = args[0]
            visible = [c for c in self.list_commands(ctx) if c != "query"]
            if first and " " not in first and not first.startswith("-"):
                # High cutoff so real one-word queries (e.g. "auth") still search;
                # only near-identical typos (e.g. "statss") are flagged.
                close = difflib.get_close_matches(first, visible, n=1, cutoff=0.8)
                if close:
                    raise click.UsageError(
                        f"No such command {first!r}. Did you mean {close[0]!r}? "
                        f'(Use quotes to search, e.g. apexgraph "{first}".)'
                    ) from None
            query_cmd = self.get_command(ctx, "query")
            if query_cmd is not None:
                return "query", query_cmd, list(args)
            raise


@click.group(cls=_ApexgraphGroup, invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_version_callback,
    help="Show version and exit.",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Apexgraph - apex-relevance subgraph retrieval for AI agents.

    Select the most relevant subgraph for your query within a token budget.

        apexgraph "how does auth work" --budget 4000
    """
    _force_utf8()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# query (default)
# ---------------------------------------------------------------------------


@cli.command("query", hidden=True)
@click.argument("query")
@click.option(
    "--graph", "-g", default=None, help="Path to graph.json (auto-discovered if omitted)."
)
@click.option(
    "--budget",
    "-b",
    type=click.IntRange(min=1),
    default=4000,
    show_default=True,
    help="Token budget.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["markdown", "json", "yaml"]),
    default="markdown",
    show_default=True,
)
@click.option("--model", "-m", default="cl100k_base", show_default=True, help="tiktoken encoding.")
@click.option("--explain", is_flag=True, help="Show a per-node score breakdown table.")
@click.option(
    "--min-score", default=0.05, show_default=True, help="Drop candidates below this score."
)
@click.option(
    "--redundancy-weight", default=0.3, show_default=True, help="MMR diversity penalty (λ)."
)
@click.option(
    "--connectivity-bonus", default=0.2, show_default=True, help="Reward adjacency to the set (μ)."
)
@click.option(
    "--inject-code", is_flag=True, help="Include source code bodies (counted in the budget)."
)
@click.option(
    "--project-root", default=None, help="Root for resolving source_file (default: graph dir)."
)
@click.option(
    "--strategy", type=click.Choice(["greedy", "exact"]), default="greedy", show_default=True
)
@click.option(
    "--backend",
    type=click.Choice(["bm25", "local", "openai", "voyage"]),
    default="bm25",
    show_default=True,
    help="Scoring backend. 'local' adds offline semantic recall (needs [local]); "
    "'openai'/'voyage' use cloud embeddings (need [dense] + an API key).",
)
@click.option(
    "--connected",
    is_flag=True,
    help="Stitch the result toward a connected subgraph by adding minimal bridge "
    "nodes within budget (best-effort; can't bridge already-disconnected components).",
)
@click.option("--ignore-file", default=".apexgraphignore", show_default=True)
@click.option("--no-cache", is_flag=True, help="Skip the on-disk cache.")
@click.option("--no-audit", is_flag=True, help="Skip writing to the audit log.")
@click.option("--viz", is_flag=True, help="Open an interactive visualisation in the browser.")
def query_cmd(
    query: str,
    graph: str | None,
    budget: int,
    fmt: str,
    model: str,
    explain: bool,
    min_score: float,
    redundancy_weight: float,
    connectivity_bonus: float,
    inject_code: bool,
    project_root: str | None,
    strategy: str,
    backend: str,
    connected: bool,
    ignore_file: str,
    no_cache: bool,
    no_audit: bool,
    viz: bool,
) -> None:
    """Retrieve the apex subgraph for QUERY within a token budget."""
    from apexgraph.budget import select_subgraph
    from apexgraph.cache import load_or_build
    from apexgraph.ignore import apply_ignore, load_ignore
    from apexgraph.scorer import score_nodes, score_nodes_detailed

    if not query.strip():
        raise click.ClickException("Query must not be empty.")

    graph_path = _resolve_graph(graph)
    kg = _load(graph_path)
    kg = apply_ignore(kg, load_ignore(Path(ignore_file)))

    cache = load_or_build(kg, base_dir=graph_path.parent, use_cache=not no_cache)

    breakdown = None
    try:
        if explain:
            breakdown = score_nodes_detailed(kg, query, cache=cache, backend=backend)
            scores = breakdown.final
        else:
            scores = score_nodes(kg, query, cache=cache, backend=backend)
    except ImportError as exc:
        # A semantic backend whose optional dependency isn't installed — surface
        # the retriever's actionable message cleanly instead of a traceback.
        raise click.ClickException(str(exc)) from exc

    # Reuse the precomputed base token costs when the encoding matches the cached one.
    token_costs = cache.token_costs if model == cache.token_model else None

    root = Path(project_root) if project_root else graph_path.parent
    sub, stats = select_subgraph(
        kg,
        scores,
        budget=budget,
        model=model,
        min_score=min_score,
        redundancy_weight=redundancy_weight,
        connectivity_bonus=connectivity_bonus,
        inject_code=inject_code,
        project_root=root,
        strategy=strategy,
        token_costs=token_costs,
        connected=connected,
    )

    from apexgraph.formatter import format_subgraph

    click.echo(format_subgraph(sub, stats, format=fmt, scores=scores, query=query))

    if stats["nodes_selected"] == 0 and fmt == "markdown":
        click.echo(
            f"\nNo nodes cleared the relevance threshold (--min-score {min_score}) "
            "for this query — try different terms or lower --min-score.",
            err=True,
        )

    if not no_audit:
        from apexgraph.audit import log_query

        top = sorted(sub.node_ids, key=lambda n: scores.get(n, 0.0), reverse=True)
        log_query(query, graph_path, stats, top, audit_dir=graph_path.parent / ".apexgraph")

    # The breakdown table is markdown-only — appending it to json/yaml would
    # corrupt machine-readable output.
    if explain and breakdown is not None and fmt == "markdown":
        _echo_rich(_explain_table(sub, breakdown))

    if viz:
        _open_viz(sub, stats, scores, query)


def _explain_table(sub, breakdown) -> Table:
    # Only show the Semantic column when a semantic backend actually contributed.
    show_semantic = any(v > 0.0 for v in breakdown.semantic.values())
    table = Table(title="Score breakdown", show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Label")
    table.add_column("Final", justify="right")
    table.add_column("BM25", justify="right")
    if show_semantic:
        table.add_column("Semantic", justify="right")
    table.add_column("PPR", justify="right")
    table.add_column("Prior", justify="right")
    table.add_column("Type", style="dim")
    ranked = sorted(sub.node_ids, key=lambda n: breakdown.final.get(n, 0.0), reverse=True)
    for i, nid in enumerate(ranked, 1):
        a = sub.digraph.nodes[nid]
        row = [
            str(i),
            a.get("label", nid),
            f"{breakdown.final.get(nid, 0.0):.3f}",
            f"{breakdown.bm25.get(nid, 0.0):.3f}",
        ]
        if show_semantic:
            row.append(f"{breakdown.semantic.get(nid, 0.0):.3f}")
        row += [
            f"{breakdown.ppr.get(nid, 0.0):.3f}",
            f"{breakdown.prior.get(nid, 0.0):.3f}",
            a.get("type") or a.get("file_type") or "",
        ]
        table.add_row(*row)
    return table


def _open_viz(sub, stats, scores, query: str) -> None:
    from apexgraph.viz import build_html

    html = build_html(sub, stats, scores, query)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        path = Path(f.name)
    webbrowser.open(path.as_uri())


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--graph", "-g", default=None, help="Path to graph.json (auto-discovered if omitted)."
)
def stats(graph: str | None) -> None:
    """Show summary statistics for a graph."""
    graph_path = _resolve_graph(graph)
    kg = _load(graph_path)
    communities = len(set(kg.communities.values())) if kg.communities else 0
    click.echo(f"Graph: {graph_path}")
    table = Table(show_header=False)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("Nodes", str(kg.digraph.number_of_nodes()))
    table.add_row("Edges", str(kg.digraph.number_of_edges()))
    table.add_row("Hyperedges", str(len(kg.hyperedges)))
    table.add_row("Communities", str(communities))
    table.add_row("God nodes", str(len(kg.god_nodes)))
    _echo_rich(table)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@cli.command("index")
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output graph.json (default: <path>/graphify-out/graph.json).",
)
@click.option("--ignore-file", default=".apexgraphignore", show_default=True)
@click.option("--incremental", is_flag=True, help="Only re-index files whose content changed.")
@click.option(
    "--strict-ids",
    is_flag=True,
    help="Collision-free node ids (full-path module ids, scope-qualified symbols).",
)
def index_cmd(
    path: str, output: str | None, ignore_file: str, incremental: bool, strict_ids: bool
) -> None:
    """Build a graph.json by statically indexing a source tree (no LLM)."""
    import json

    from apexgraph.ignore import load_ignore
    from apexgraph.indexer.project import index_project, index_project_incremental

    root = Path(path).resolve()
    out = Path(output) if output else root / "graphify-out" / "graph.json"
    ignore = load_ignore(Path(ignore_file))

    click.echo(f"Indexing {root} ...")
    if incremental:
        cache_path = root / ".apexgraph" / "index_cache.json"
        graph = index_project_incremental(root, cache_path, ignore, strict_ids=strict_ids)
    else:
        graph = index_project(root, ignore, strict_ids=strict_ids)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    files = len({n.get("source_file") for n in graph["nodes"] if n.get("source_file")})
    click.echo(f"  {len(graph['nodes'])} nodes · {len(graph['links'])} edges · {files} files")
    click.echo(f"  Saved: {out}")
    click.echo(f'\nNext: apexgraph "your query" --graph {out}')


# ---------------------------------------------------------------------------
# serve (MCP)
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--graph", "-g", default=None, help="Path to graph.json (auto-discovered if omitted)."
)
def serve(graph: str | None) -> None:
    """Start an MCP stdio server exposing the apexgraph_* tools."""
    from apexgraph import mcp

    graph_path = _resolve_graph(graph)
    _load(graph_path)  # fail fast with a clean error before entering the stdio loop
    mcp.serve(graph_path)


# ---------------------------------------------------------------------------
# explain / path
# ---------------------------------------------------------------------------


@cli.command("explain")
@click.argument("node")
@click.option("--graph", "-g", default=None)
def explain_cmd(node: str, graph: str | None) -> None:
    """Explain a single NODE and its immediate neighbourhood."""
    graph_path = _resolve_graph(graph)
    kg = _load(graph_path)
    if node not in kg.digraph:
        raise click.ClickException(f"Node not found: {node!r}")
    a = kg.digraph.nodes[node]
    lines = [f"# {a.get('label', node)}  ({a.get('type') or a.get('file_type') or 'node'})"]
    if a.get("description"):
        lines.append(a["description"])
    if a.get("source_file"):
        loc = a.get("source_location") or ""
        lines.append(f"→ {a['source_file']} {loc}".rstrip())
    preds = list(kg.digraph.predecessors(node))
    succs = list(kg.digraph.successors(node))
    if preds:
        lines.append("\n## Used by")
        lines += [
            f"- {kg.digraph.nodes[p].get('label', p)} → {kg.digraph.edges[p, node].get('relation', '')}".rstrip()
            for p in preds
        ]
    if succs:
        lines.append("\n## Depends on")
        lines += [
            f"- {kg.digraph.edges[node, s].get('relation', '')} → {kg.digraph.nodes[s].get('label', s)}".strip()
            for s in succs
        ]
    click.echo("\n".join(lines))


@cli.command("path")
@click.argument("source")
@click.argument("target")
@click.option("--graph", "-g", default=None)
def path_cmd(source: str, target: str, graph: str | None) -> None:
    """Show the shortest path between two nodes."""
    import networkx as nx

    graph_path = _resolve_graph(graph)
    kg = _load(graph_path)
    for nid in (source, target):
        if nid not in kg.digraph:
            raise click.ClickException(f"Node not found: {nid!r}")
    try:
        nodes = nx.shortest_path(kg.digraph, source, target)
    except nx.NetworkXNoPath:
        try:
            nodes = nx.shortest_path(kg.digraph.to_undirected(as_view=True), source, target)
        except nx.NetworkXNoPath:
            raise click.ClickException(f"No path between {source!r} and {target!r}.") from None
    labels = [kg.digraph.nodes[n].get("label", n) for n in nodes]
    click.echo(" → ".join(labels))


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@cli.command("diff")
@click.argument("old_graph", type=click.Path(exists=True))
@click.argument("new_graph", type=click.Path(exists=True))
@click.option("--hops", default=2, show_default=True, help="Impact-neighbourhood depth.")
@click.option(
    "--budget",
    "-b",
    default=None,
    type=click.IntRange(min=1),
    help="Token-budget the affected area.",
)
@click.option("--viz", is_flag=True, help="Open the affected area in the browser.")
def diff_cmd(old_graph: str, new_graph: str, hops: int, budget: int | None, viz: bool) -> None:
    """Compare two graph versions and show the impact of the changes."""
    from apexgraph.diff import affected_subgraph, diff_graphs, format_diff

    old = _load(Path(old_graph))
    new = _load(Path(new_graph))
    diff = diff_graphs(old, new)
    click.echo(format_diff(diff, new))

    if budget is not None:
        from apexgraph.budget import select_subgraph
        from apexgraph.formatter import format_subgraph
        from apexgraph.scorer import score_nodes

        affected = affected_subgraph(new, diff, hops)
        if len(affected) > 0:
            seeds = (diff.added_nodes + diff.modified_nodes)[:5]
            query = " ".join(
                new.digraph.nodes[n].get("label", n) for n in seeds if n in new.digraph
            )
            scores = score_nodes(affected, query or "diff")
            sub, stats = select_subgraph(affected, scores, budget=budget)
            click.echo("\n## Token-budgeted affected subgraph\n")
            click.echo(format_subgraph(sub, stats, query=query))

    if viz:
        affected = affected_subgraph(new, diff, hops)
        scores = {n: 1.0 for n in affected.node_ids}
        stats = {
            "nodes_selected": len(affected),
            "nodes_total": new.digraph.number_of_nodes(),
            "tokens_used": 0,
            "tokens_budget": 0,
            "coverage_pct": 0.0,
        }
        _open_viz(affected, stats, scores, "diff")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@cli.command("export")
@click.argument("query")
@click.option("--graph", "-g", default=None)
@click.option("--budget", "-b", type=click.IntRange(min=1), default=4000, show_default=True)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["claude", "chatgpt", "claudemd"]),
    default="claude",
    show_default=True,
)
@click.option("--output", "-o", default=None, help="Write to a file instead of stdout.")
@click.option("--min-score", default=0.05, show_default=True)
def export_cmd(
    query: str, graph: str | None, budget: int, fmt: str, output: str | None, min_score: float
) -> None:
    """Export a context block ready to paste into a system prompt or CLAUDE.md."""
    from apexgraph.budget import select_subgraph
    from apexgraph.cache import load_or_build
    from apexgraph.exporter import export_context
    from apexgraph.scorer import score_nodes

    if not query.strip():
        raise click.ClickException("Query must not be empty.")

    graph_path = _resolve_graph(graph)
    kg = _load(graph_path)
    cache = load_or_build(kg, base_dir=graph_path.parent)
    scores = score_nodes(kg, query, cache=cache)
    sub, stats = select_subgraph(kg, scores, budget=budget, min_score=min_score)
    result = export_context(sub, stats, query, format=fmt, scores=scores)

    if output:
        Path(output).write_text(result, encoding="utf-8")
        click.echo(f"Exported to {output} ({len(result):,} chars)")
    else:
        click.echo(result)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--top-nodes", "top_n", default=10, show_default=True)
@click.option(
    "--audit-dir", default=None, help="Audit directory (default: the graph's .apexgraph sidecar)."
)
def audit(top_n: int, audit_dir: str | None) -> None:
    """Show query history and the most frequently selected nodes."""
    from apexgraph.audit import read_audit, top_nodes_from_audit

    # Queries log to the discovered graph's sibling .apexgraph; default there so
    # `apexgraph audit` reads the same place `apexgraph query` wrote to.
    if audit_dir is None:
        found = _find_graph()
        audit_dir = str((found.parent / ".apexgraph") if found else Path(".apexgraph"))

    entries = read_audit(audit_dir)
    if not entries:
        click.echo("No audit entries found.")
        return

    recent = entries[-20:]
    qt = Table(title=f"Query history (last {len(recent)} of {len(entries)})", show_header=True)
    qt.add_column("#", style="dim", width=4)
    qt.add_column("Timestamp")
    qt.add_column("Query")
    qt.add_column("Nodes", justify="right")
    qt.add_column("Tokens", justify="right")
    for i, e in enumerate(recent, 1):
        qt.add_row(
            str(i),
            str(e.get("timestamp", ""))[:19],
            str(e.get("query", "")),
            str(e.get("nodes_selected", "")),
            str(e.get("tokens_used", "")),
        )
    _echo_rich(qt)

    top = top_nodes_from_audit(audit_dir, n=top_n)
    if top:
        nt = Table(title=f"Top {top_n} most selected nodes", show_header=True)
        nt.add_column("Rank", style="dim", width=5)
        nt.add_column("Node")
        nt.add_column("Times", justify="right")
        for rank, (node, count) in enumerate(top, 1):
            nt.add_row(str(rank), node, str(count))
        _echo_rich(nt)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@cli.command("benchmark")
@click.option("--graph", "-g", default=None)
@click.option(
    "--query", "-q", "queries", multiple=True, required=True, help="Query to test (repeatable)."
)
@click.option(
    "--budget",
    "-b",
    "budgets",
    multiple=True,
    type=click.IntRange(min=1),
    default=(2000, 4000, 8000),
    show_default=True,
)
@click.option(
    "--k-relevant", default=10, show_default=True, help="Size of the relevant set for recall."
)
@click.option("--output", "-o", default=None, help="Write raw results to a JSON file.")
def benchmark_cmd(
    graph: str | None,
    queries: tuple[str, ...],
    budgets: tuple[int, ...],
    k_relevant: int,
    output: str | None,
) -> None:
    """Measure recall@budget and token savings across queries and budgets."""
    import json

    from apexgraph.benchmark import format_benchmark, run_benchmark
    from apexgraph.cache import load_or_build

    graph_path = _resolve_graph(graph)
    kg = _load(graph_path)
    cache = load_or_build(kg, base_dir=graph_path.parent)
    result = run_benchmark(kg, list(queries), list(budgets), k_relevant=k_relevant, cache=cache)
    click.echo(format_benchmark(result))

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        click.echo(f"\nResults saved to {output}")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

_DEFAULT_IGNORE = """\
# apexgraph ignore — gitignore syntax, matched against node id and source_file
*/tests/*
*/__pycache__/*
*.test.ts
*.spec.ts
"""


@cli.command()
def init() -> None:
    """Scaffold a .apexgraphignore in the current directory."""
    path = Path(".apexgraphignore")
    if path.exists():
        click.echo(".apexgraphignore already exists — leaving it untouched.")
        return
    path.write_text(_DEFAULT_IGNORE, encoding="utf-8")
    click.echo(f"Created {path}")
