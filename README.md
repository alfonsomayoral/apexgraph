<div align="center">

# Graphex

**Apex-relevance subgraph retrieval for AI agents.**

Feed your LLM the *peak* of your knowledge graph — sized to a token budget.

</div>

---

Knowledge graphs grow large. When an agent needs context about one corner of a
codebase, dumping the whole graph into the prompt wastes tokens and money.
Graphex scores every node against your query and returns the most relevant
connected subgraph that fits within a token budget.

```bash
graphex index .                            # build a graph from your code (no LLM)
graphex query "how does auth work" -b 4000 # retrieve the apex subgraph
graphex serve                              # expose it to agents over MCP
```

Graphex reads the graphs produced by [`graphify`](https://github.com/) and uses
the rich signals graphify emits — edge weights, confidence, hyperedges,
communities, and god nodes — that simpler tools throw away.

> Status: under active construction. See
> [the build plan](#) for the roadmap.

## Why Graphex

- **Principled relevance.** BM25 lexical matches seed a Personalized PageRank
  walk over the weighted graph — one unified score, not a hand-tuned linear mix.
- **Budget-aware selection.** Submodular optimisation (MMR + connectivity) picks
  a diverse, *connected* subgraph, not a bag of redundant islands.
- **Honest token accounting.** The budget counts the final rendered form,
  including injected source code — what you ask for is what you get.
- **Fast.** Query-independent work (global PageRank, the BM25 index, token costs)
  is precomputed once and cached, invalidated by content hash.
- **Local-first.** No API keys required; dense embeddings are an optional plugin.

## License

MIT © Alfonso Mayoral
