<div align="center">

# Graphex

**Apex-relevance subgraph retrieval for AI agents.**

Feed your LLM the *peak* of your knowledge graph — sized to a token budget.

[![CI](https://github.com/alfonsomayoral/graphex/actions/workflows/ci.yml/badge.svg)](https://github.com/alfonsomayoral/graphex/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

Knowledge graphs grow large. When an agent needs context about one corner of a
codebase, dumping the whole graph into the prompt wastes tokens and money — and
buries the relevant nodes in noise. **Graphex scores every node against your
query and returns the most relevant, connected subgraph that fits within a token
budget**, ready to paste into a prompt or serve over MCP.

```bash
graphex index .                            # build a graph from your code (no LLM)
graphex "how does auth work" --budget 4000 # retrieve the apex subgraph
graphex serve                              # expose it to agents over MCP
```

Graphex reads the graphs produced by **graphify** and uses the rich signals
graphify emits — edge weights, confidence, hyperedges, communities, and god
nodes — that simpler tools throw away.

## Install

```bash
uv tool install graphex            # or: pipx install graphex
# optional extras:
uv tool install "graphex[ts]"      # better TypeScript indexing (tree-sitter)
uv tool install "graphex[dense]"   # OpenAI/Anthropic embedding backend
```

Requires Python 3.12+.

## How it works

A five-stage pipeline, each stage a single-responsibility module:

```
  load ─▶ score ─▶ select ─▶ inject ─▶ render
   │        │         │         │         │
 multi-   BM25 →    cost-aware  source-   markdown /
 format   PPR +     MMR under   code      json / yaml
 loader   prior     budget      bodies
                        ▲
   index ───────────────┘   build a graph straight from code (no graphify)
```

**Relevance is one principled number, not a hand-tuned mix.** BM25 finds the
nodes the query is literally about; those seed a **Personalized PageRank** walk
that spreads relevance across the weighted graph (edge `weight × confidence`,
plus hyperedge cliques); a light importance/god-node prior nudges genuinely
central entities up. The query-independent half — global PageRank, the BM25
inverted index — is precomputed once and cached, invalidated by content hash, so
a query is just a lookup plus one walk.

**Selection is a budgeted knapsack, solved as one.** Picking the highest-value
set of nodes under a token ceiling is the 0/1 knapsack problem. Graphex selects
by *marginal value per token* and shapes the result with two terms — an MMR
penalty so it doesn't say the same thing twice, and a connectivity bonus so the
result is a coherent connected subgraph, not a bag of redundant islands. An exact
DP-knapsack mode is available for benchmarking the value ceiling.

**Token accounting is honest.** A node's cost is the size of its *final rendered
form*, including any injected source code — so `tokens_used` never lies and the
output never overflows the budget you asked for.

## Usage

```bash
# Index a project into a graphify-compatible graph.json (Python / TS / Go)
graphex index ./src -o graph.json
graphex index ./src --incremental          # re-index only changed files

# Query (any unrecognised first arg routes here)
graphex "session token validation" -b 2000
graphex "auth flow" --explain               # per-node BM25 / PPR / prior breakdown
graphex "auth flow" --inject-code           # include real function bodies, still in budget
graphex "auth flow" --viz                   # interactive force-directed HTML

# Inspect (node ids come from your indexed graph; these match examples/)
graphex stats -g examples/sample_graph.json
graphex explain auth_service_login -g examples/sample_graph.json
graphex path auth_service auth_service_login -g examples/sample_graph.json

# Export a context block to paste into a system prompt / CLAUDE.md
graphex export "auth flow" -f claudemd -o CONTEXT.md

# Measure quality honestly (recall@budget, not just tokens saved)
graphex benchmark -q "auth flow" -q "db pooling" -b 1000 -b 4000

# Compare two graph versions and see the change impact
graphex diff old.json new.json --budget 2000
```

See [`examples/`](examples/) for a full walkthrough on a sample project.

## MCP server

Graphex speaks the Model Context Protocol over stdio (stdlib only, no SDK):

```bash
graphex serve --graph graph.json
```

It exposes four tools: `graphex_query`, `graphex_explain`, `graphex_path`,
`graphex_stats`. Register it with Claude Code:

```bash
claude mcp add graphex -- graphex serve --graph /abs/path/to/graph.json
```

## Honest benchmarking

"Tokens saved" is a vanity metric — a tool that returns nothing saves 100%.
Graphex reports **recall@budget** alongside it: how much of the relevant set the
budgeted subgraph actually captures. High savings with low recall means
under-retrieval, and the benchmark makes that trade-off visible.

## Development

```bash
uv sync
uv run pytest          # test suite
uv run ruff check .    # lint
uv run black .         # format
```

## License

MIT © Alfonso Mayoral
