# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- On-disk cache (`.graphex/`) for global PageRank and the BM25 index, invalidated
  by content fingerprint.
- Static indexer for Python (`ast`), TypeScript/JavaScript (tree-sitter with a
  regex fallback) and Go (regex), with incremental re-indexing by file hash.
- Markdown / JSON / YAML formatter and source-code injector.
- MCP stdio server exposing `graphex_query`, `graphex_explain`, `graphex_path`
  and `graphex_stats`.
- Click CLI: `query`, `index`, `serve`, `stats`, `explain`, `path`, `diff`,
  `export`, `benchmark`, `audit`, `init` — with autodiscovery, rich `--explain`,
  and UTF-8 output on Windows.
- Context export for Claude / ChatGPT / CLAUDE.md, graph diffing, `.graphexignore`
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

[0.1.0]: https://github.com/alfonsomayoral/graphex/releases/tag/v0.1.0
