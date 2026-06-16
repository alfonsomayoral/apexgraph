<div align="center">

# ◢◤ Graphex

### Apex-relevance subgraph retrieval for AI agents

**Stop dumping your whole knowledge graph into the prompt.**
Graphex hands your LLM the *peak* of the graph — the smallest, most relevant
subgraph that answers the query — sized to an exact token budget.

[![PyPI](https://img.shields.io/pypi/v/apexgraph?color=2b8a3e&label=apexgraph)](https://pypi.org/project/apexgraph/)
[![CI](https://github.com/alfonsomayoral/graphex/actions/workflows/ci.yml/badge.svg)](https://github.com/alfonsomayoral/graphex/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Tests](https://img.shields.io/badge/tests-229%20passing-2b8a3e)
![License](https://img.shields.io/badge/license-MIT-blue)

```bash
uv tool install "apexgraph[local]"
graphex index .                       # build a code graph (no LLM)
graphex "how does auth work" -b 4000  # retrieve the apex subgraph
```

</div>

---

## 🎯 The problem

Knowledge graphs — the kind [`graphify`](https://github.com/) builds from a
codebase — get **big**. A real app app indexes to ~9,000 nodes. When an
agent needs context about *one* corner of it, the usual options are both bad:

- **Dump the whole graph** → tens of thousands of tokens, most of them irrelevant,
  and the few nodes that matter are buried in noise.
- **Naive keyword match + BFS** → walks into the wrong neighbourhood and returns a
  pile of off-topic nodes (translation strings, unrelated helpers).

You want the *opposite*: a tight, on-topic, **connected** slice of the graph that
fits the budget you have — and you want it in milliseconds, offline, every query.

## ✨ What Graphex does

Graphex scores every node against your query, then selects the highest-value
subgraph that fits a token ceiling. One command, one principled relevance number
per node, a budget that's never exceeded.

```
   graph.json ──load─▶ ┌───────────────────────────────────────────────┐
  (graphify, or        │  SCORE   BM25 + stemming ──seed──▶ Personalized │
   `graphex index`)    │          PageRank  (+ optional local / cloud    │
   query ────────────▶ │          embeddings, fused by RRF) + priors     │
                       └─────────────────────┬─────────────────────────┘
                                             ▼
                       ┌───────────────────────────────────────────────┐
                       │  SELECT  cost-aware MMR knapsack under a token  │
                       │          budget  (+ connectivity, honest cost)  │
                       └─────────────────────┬─────────────────────────┘
                                             ▼
                 RENDER  markdown · json · yaml   │   MCP server   │   HTML viz
```

It reads graphify's graphs **natively** and uses the rich signals graphify emits
— edge weights, confidence, hyperedges, communities, god nodes — that simpler
tools throw away. Or skip graphify entirely: `graphex index` builds a clean,
code-only graph from your source in ~1.5s.

> **One name note:** the PyPI package is `apexgraph`; the command and import are
> both `graphex`.

## 🧩 Capabilities

| | |
|---|---|
| 🎯 **Principled relevance** | BM25 (with stemming) seeds a Personalized PageRank walk over the weighted graph. One unified score — not a hand-tuned mix of independent axes. |
| 🧠 **Semantic recall, offline** | `--backend local` (model2vec) finds what the query is *about* even with **zero shared tokens** — "authorization gate" surfaces the auth code. No API key, no network. Cloud `openai` / `voyage` also available. |
| 📐 **Budget solved as a knapsack** | Selection maximises value *per token* with an MMR diversity penalty and a connectivity bonus — a tight, non-redundant slice, not a bag of islands. Exact DP mode for the value ceiling. |
| 💯 **Honest token accounting** | A node's cost is its *final rendered form*, including injected source code — so `tokens_used` never lies and the output never overflows the budget. |
| ⚡ **Fast & cached** | Query-independent work (global PageRank, the BM25 index, token costs) is precomputed once and cached, invalidated by content hash. A query is a lookup plus one walk — ~0.1s on a 9k-node graph. |
| 🔌 **MCP server** | Stdlib JSON-RPC over stdio (no SDK). Exposes `graphex_query`, `graphex_explain`, `graphex_path`, `graphex_stats` to Claude Code and any MCP agent. |
| 🏗️ **Built-in indexer** | Python (`ast`), TypeScript/JS (tree-sitter → regex), Go (regex). `--strict-ids` for collision-free ids; incremental re-index by file hash. |
| 🧷 **Connected output** | `--connected` stitches the result toward a single connected subgraph (approximate Steiner) within budget. |
| 🔒 **Safe by default** | Code injection is contained to the project root (no path-traversal exfiltration); the HTML viz pins its CDN script with Subresource Integrity. |
| 📤 **Drops in anywhere** | Render to markdown / json / yaml, or `export` a context block ready to paste into a Claude / ChatGPT system prompt or a `CLAUDE.md`. |

## ⚙️ How it works

**Relevance is one number, computed properly.** BM25 finds the nodes the query is
literally about; those seed a **Personalized PageRank** random walk that spreads
relevance across the weighted graph — edge `weight × confidence`, plus hyperedges
exploded into weighted cliques. A light importance / god-node prior and a global
PageRank tiebreak refine the ranking *only among nodes the walk reached*, so a
node unrelated to the query stays at exactly zero (an honest "nothing matched").

**Semantic recall, when you want it.** Add `--backend local` and BM25's ranking is
fused with offline embedding similarity via Reciprocal Rank Fusion (rank-based, so
the two scales need no calibration). A query like *"sign in flow"* then seeds the
walk from the login code even though they share no tokens.

**Selection is a budgeted 0/1 knapsack, solved as one.** Picking the best set of
nodes under a token ceiling is exactly the knapsack problem. Graphex selects by
*marginal value per token* and shapes the result with two terms — an MMR penalty
so it doesn't say the same thing twice, and a connectivity bonus so the subgraph
holds together. The single most relevant node is guaranteed to survive.

## 📊 Results

### vs `slurp` (the prior-art tool Graphex improves on)

A reproducible head-to-head lives in [`bench/`](bench/). On a labeled query set,
measuring **precision and recall together** (because recall alone is gameable):

| tool | precision | what it does |
|------|-----------|--------------|
| **graphex** | **32–38%** | returns a tight, on-topic subgraph |
| slurp | ~8% | pads the budget with low-relevance nodes to inflate raw recall |

> slurp posts higher *raw* recall — by returning almost the whole graph. ~90% of
> what it hands the model is off-topic. Graphex is **4–10× more precise** under
> budget, and its `local` backend recovers relevant nodes on semantic queries
> where lexical retrieval (slurp's TF-IDF *and* Graphex's own BM25) scores zero.

### On a real codebase (a ~9k-node graph of a app app)

| | graphify query | graphex |
|---|---|---|
| nodes returned for an "auth" query | 89 (mostly i18n translation strings) | **12 — all actual auth code** |
| latency | ~0.9s | **~0.1s** (≈10× faster) |
| build your own graph | — | **5,178 code nodes from 391 files in ~1.5s** |

Querying Graphex's own clean code index returns exactly the feature code, with
**zero translation-string noise**:

```text
"AI coaching"        → Component, Component, Component, store
"streak tracking"    → Component, Component, Component, Store
"workout tracking"   → Component, Component, Component
```

> Honest caveat: a single "accuracy %" on a real repo is unreliable to measure
> automatically — features sprawl across `app/`, `components/`, `store/` and
> backend functions, so any one-directory ground truth undercounts correct hits.
> The objective wins (focus, speed, clean graph, on-topic ranking) are what hold.

---

## 🚀 Usage guide

### Install

```bash
uv tool install apexgraph              # core (fully local, lexical)
uv tool install "apexgraph[local]"     # + offline semantic recall (model2vec)
uv tool install "apexgraph[ts]"        # + precise TypeScript indexing (tree-sitter)
uv tool install "apexgraph[dense]"     # + cloud embeddings (OpenAI / Voyage AI)
# or: pipx install apexgraph
```

Requires Python 3.12+. The command is `graphex`.

### 1 · Get a graph

Either point Graphex at a graph `graphify` already built, **or** build one from
source with no LLM:

```bash
graphex index ./src                    # → ./src/graphify-out/graph.json
graphex index ./src --strict-ids       # collision-free node ids
graphex index ./src --incremental      # re-index only changed files
graphex stats                          # nodes / edges / communities / god nodes
```

### 2 · Query it

`graphex QUERY` is the default — any unrecognised first argument is treated as a
query. The graph is auto-discovered (or pass `-g PATH`).

```bash
graphex "how does session validation work" -b 2000
graphex "authorization gate" --backend local      # offline semantic recall
graphex "auth flow" --explain                      # per-node score breakdown
graphex "auth flow" --inject-code                  # include real function bodies
graphex "auth flow" --connected                    # stitch toward a connected slice
graphex "auth flow" --viz                          # interactive force-directed HTML
```

A query renders a budgeted subgraph with a header that never lies about its size:

```text
┌──────────────────────────────────────────────────────────────┐
│ Graphex subgraph for: how does session validation work       │
│ Selected 8/9314 nodes (0.1%) · 1487/2000 tokens              │
└──────────────────────────────────────────────────────────────┘
## Relevant Nodes
### validate_token (function) · score: 1.00
...
```

<details>
<summary><b>Key flags for <code>graphex query</code></b></summary>

| flag | default | meaning |
|------|---------|---------|
| `-b, --budget` | 4000 | token ceiling (never exceeded) |
| `-f, --format` | markdown | `markdown` · `json` · `yaml` |
| `--backend` | bm25 | `bm25` · `local` · `openai` · `voyage` |
| `--explain` | off | per-node BM25 / semantic / PPR / prior table |
| `--inject-code` | off | embed real source bodies (counted in the budget) |
| `--connected` | off | best-effort connected subgraph (Steiner) |
| `--min-score` | 0.05 | drop candidates below this relevance |
| `--strategy` | greedy | `greedy` (MMR) · `exact` (DP knapsack) |
| `--viz` | off | open an interactive HTML visualisation |

</details>

### 3 · Inspect & export

```bash
graphex explain <node_id>                  # a node + its neighbourhood
graphex path <a> <b>                        # shortest path between two nodes
graphex diff old.json new.json -b 2000      # change-impact subgraph
graphex export "auth flow" -f claudemd -o CONTEXT.md   # paste-ready context block
graphex benchmark -q "auth flow" -b 2000    # recall@budget + token savings
```

### 4 · Serve it to an agent (MCP)

Graphex speaks the Model Context Protocol over stdio:

```bash
graphex serve --graph graph.json
# register with Claude Code:
claude mcp add graphex -- graphex serve --graph /abs/path/to/graph.json
```

Tools exposed: `graphex_query`, `graphex_explain`, `graphex_path`, `graphex_stats`.

---

## 🛠️ Development

```bash
git clone https://github.com/alfonsomayoral/graphex && cd graphex
uv sync
uv run pytest          # 229 tests
uv run ruff check .    # lint
uv run black --check . # format
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the architecture map and
[`RELEASING.md`](RELEASING.md) for the trusted-publishing release flow.

## 📄 License

MIT © [Alfonso Mayoral](https://github.com/alfonsomayoral)
