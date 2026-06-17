# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-16

### Changed
- **Renamed the project to `apexgraph` for naming coherence.** The import package,
  the CLI command and the MCP tools are now all `apexgraph` (previously the
  command/import were `graphex` while the PyPI distribution was already
  `apexgraph`). Migration: `pip install apexgraph`, then use `apexgraph …` /
  `import apexgraph`; MCP tools are `apexgraph_query` / `_explain` / `_path` /
  `_stats`; the sidecar dir is `.apexgraph/` and the ignore file `.apexgraphignore`.

## [0.2.0] - 2026-06-16

### Added
- **Offline semantic recall** — a local embedding backend (`--backend local`,
  model2vec, behind the `[local]` extra) finds nodes the query is *about* even
  with no shared tokens. BM25 and the semantic ranking are fused with Reciprocal
  Rank Fusion to seed Personalized PageRank.
- **Stemming** in the tokenizer so morphological variants match (e.g. a query for
  "authentication" matches a node labelled "authenticate").
- **`--connected`** — stitch the result toward a connected subgraph by adding
  minimal bridge nodes (approximate Steiner), never exceeding the budget
  (best-effort: cannot bridge components already disconnected in the source graph).
- **`--strict-ids`** indexer mode for collision-free node ids (full-path module
  ids, scope-qualified symbols).
- Per-node token costs are precomputed and cached, so a query no longer
  re-tokenizes every candidate.
- A reproducible **Apexgraph-vs-slurp benchmark** under `bench/` (recall, precision
  and token cost on labeled queries).

### Changed
- The cloud `anthropic` embedding backend (which called a non-existent endpoint)
  is replaced by a correct **Voyage AI** backend (`--backend voyage`).
- Global PageRank is reconnected as a light structural tiebreak in the final
  score (it was computed and cached but no longer consumed).

## [0.1.0] - 2026-06-16

### Added
- Multi-format graph loader (graphify JSON, GraphML, Neo4j CSV) preserving
  hyperedges, edge weight/confidence, communities and god nodes.
- BM25 lexical retriever with a cached inverted index and identifier-aware
  tokenizer (camelCase / snake_case / PascalCase, compounds preserved).
- Personalized PageRank / random-walk-with-restart over weighted edges and
  hyperedge cliques, plus query-independent global PageRank.
- Scorer fusing BM25-seeded PPR with an importance/god-node prior.
- Cost-aware MMR subgraph selection with a connectivity bonus and honest token
  accounting (including injected source code); optional exact DP-knapsack mode.
- On-disk cache (`.apexgraph/`) for global PageRank and the BM25 index, invalidated
  by content fingerprint.
- Static indexer for Python (`ast`), TypeScript/JavaScript (tree-sitter with a
  regex fallback) and Go (regex), with incremental re-indexing by file hash.
- Markdown / JSON / YAML formatter and source-code injector.
- MCP stdio server exposing `apexgraph_query`, `apexgraph_explain`, `apexgraph_path`
  and `apexgraph_stats`.
- Click CLI: `query`, `index`, `serve`, `stats`, `explain`, `path`, `diff`,
  `export`, `benchmark`, `audit`, `init` — with autodiscovery, rich `--explain`,
  and UTF-8 output on Windows.
- Context export for Claude / ChatGPT / CLAUDE.md, graph diffing, `.apexgraphignore`
  filtering, a JSONL query audit log, and interactive HTML visualisation.
- Optional dense-embedding backend (OpenAI / Anthropic) behind the `[dense]` extra.
- Benchmark reporting recall@budget alongside token savings.

### Security
- Code injection (`--inject-code`) is contained to the project root: a crafted
  `source_file` in an untrusted graph can no longer read arbitrary host files via
  absolute paths or `..` traversal.
- The interactive visualisation loads vis-network from a pinned, immutable CDN
  URL with a Subresource Integrity (SRI) hash, so a compromised CDN cannot inject
  script into a generated page.

[0.3.0]: https://github.com/alfonsomayoral/apexgraph/releases/tag/v0.3.0
[0.2.0]: https://github.com/alfonsomayoral/apexgraph/releases/tag/v0.2.0
[0.1.0]: https://github.com/alfonsomayoral/apexgraph/releases/tag/v0.1.0
