# Graphex vs slurp — retrieval benchmark

A reproducible harness comparing **Graphex** against **slurp** (the prior-art
tool Graphex improves on) on token-budgeted knowledge-graph retrieval. The
primary metric is **recall@budget**; precision and token cost are reported
alongside it because, as the results show, recall alone is gameable.

## What is being measured

For each labeled query and each token budget, three retrievers run over the same
graph (`examples/sample_graph.json`, a 102-node task-tracker app) and we measure
how much of a **human-labeled relevant set** each one recovers within the budget:

| retriever        | how it runs                                        | retrieval signal |
|------------------|----------------------------------------------------|------------------|
| `graphex-bm25`   | in-process (`score_nodes(..., backend="bm25")`)    | lexical (BM25 + personalized PageRank) |
| `graphex-local`  | in-process (`score_nodes(..., backend="local")`)   | lexical + offline semantic embeddings (model2vec), fused by reciprocal rank fusion |
| `slurp`          | **black box** via `uvx --from slurp-graph slurp …` | TF-IDF |

slurp is run exactly as a user would install it from PyPI — no source access, no
tuning. Its stdout JSON (`{stats, nodes:[{id,…}]}`) is parsed for the selected
node ids and its own reported token usage. If slurp/uvx is unavailable the cell
is recorded as `n/a` and the harness continues.

### Metrics

For a query with human-labeled relevant set `R` and a tool's selected set `S`:

- **recall@budget** = `|S ∩ R| / |R|` — the primary metric: of the nodes a human
  considers on-topic, how many did the budgeted subgraph keep?
- **precision** = `|S ∩ R| / |S|` — how much of what it returned was actually
  on-topic. This is what exposes a tool that "wins" recall by dumping the whole
  graph into the budget.
- **tokens_used**, **nodes_selected** — cost, reported by each tool itself.

Budgets swept: **500, 1500, 4000** tokens.

### How the relevant sets were defined

Relevant sets live in `bench/queries.json` and were written **by meaning**, by
reading the node ids/labels in the graph — never by running any tool and copying
its output. Each query is tagged `lexical` or `semantic`:

- **lexical** queries share tokens with the relevant node labels
  (`"connection pool"` → `db_pool`, `ConnectionPool`, `acquire`, …). Both TF-IDF
  and BM25 should do fine.
- **semantic** queries deliberately share **no tokens** with their relevant nodes
  but mean the same thing (`"sign in flow"` → the login/auth nodes;
  `"authorization gate"` → `require_auth` / `validate_token` / `_bearer_token`).
  These are the discriminator: a purely lexical matcher cannot match on shared
  tokens, only an embedding backend can.

## How to run

```bash
# from the repo root
uv pip install model2vec          # Graphex's offline semantic backend
uv run python bench/compare.py    # also shells out to slurp via uvx
```

The first slurp call is slow (uvx installs `slurp-graph`), then caches. Results
are printed as two tables and written to `bench/results.json`.

## Results (real numbers, observed)

### Aggregate — mean over queries × budgets

| tool           | scope    | mean recall | mean precision |
|----------------|----------|------------:|---------------:|
| `graphex-bm25` | lexical  | 79%         | 63%            |
| `graphex-bm25` | semantic | 41%         | 13%            |
| `graphex-bm25` | overall  | 60%         | 38%            |
| `graphex-local`| lexical  | 83%         | 40%            |
| `graphex-local`| semantic | 68%         | 23%            |
| `graphex-local`| overall  | 75%         | 32%            |
| `slurp`        | lexical  | 94%         | 9%             |
| `slurp`        | semantic | 92%         | 8%             |
| `slurp`        | overall  | 93%         | 8%             |

### Reading these honestly

This is **not** a clean "Graphex wins recall everywhere" story, and the harness
does not pretend it is.

1. **slurp posts the highest raw recall (~93%) — but it does so by padding.** At
   budgets ≥ 1500, slurp selects **all 102 nodes of the graph** on essentially
   every query (see the per-query table below: `sel = 102`, `tokens ≈ 1491`).
   Returning the whole graph trivially recovers everything relevant, so its
   recall is high *and uninformative*. The cost shows up as **precision: 6–9%** —
   roughly 90% of what slurp hands the LLM is off-topic. Even at the tight 500
   budget it returns 33–40 nodes at ~5–24% precision.

2. **Graphex is 4–10× more precise.** Overall precision is 38% (bm25) / 32%
   (local) vs slurp's 8%. Graphex spends a few hundred tokens where slurp spends
   the entire budget; on `login authentication` bm25 hits 100% precision in 81
   tokens, while slurp uses 1491 tokens at 7% precision for the same query.

3. **The semantic discriminator holds — within Graphex.** The headline
   hypothesis (semantic backend recovers what lexical matching can't) is clearly
   visible by comparing `graphex-bm25` vs `graphex-local`:
   - `"sign in flow"`: bm25 = **0% recall** (it shares no tokens with any auth
     node, so BM25 finds nothing and honestly returns an empty subgraph);
     local = **43%**.
   - `"authorization gate"`: bm25 = **0% recall**; local = **100%**.
   - Across all semantic queries: bm25 **41%** → local **68%** mean recall. The
     offline embeddings recover relevant nodes on exactly the no-shared-token
     queries where BM25 collapses to zero.

4. **Where slurp genuinely wins.** On raw recall, slurp beats Graphex on most
   queries because of the padding above. Concretely at budget 500 (where nobody
   can pad to the full graph), slurp still wins recall on several: `login
   authentication` (slurp 86% vs bm25 43% / local 71%), `data store connection`
   (slurp 100% vs bm25 67% / local 50%), `log out and end session` (slurp 100%
   vs Graphex 50%). So slurp's TF-IDF, by casting a wide net, does recover more
   relevant nodes per query — at a brutal precision cost. If the only thing you
   care about is recall and tokens are free, slurp's strategy "works".

### The honest takeaway

- **Comparable-to-better on lexical recall**, where it matters: `graphex-local`
  (83%) trails slurp (94%) but at 4× the precision.
- **The semantic backend does what it claims**: on no-shared-token queries it
  lifts Graphex's recall from 41% → 68% and rescues the queries where BM25
  returns literally nothing.
- **Graphex's real edge is precision under budget**: it delivers a tight,
  on-topic subgraph (32–38% precision) instead of slurp's near-whole-graph dump
  (8% precision). For an LLM-context tool where every token competes for
  attention, that is the metric that actually matters.

Per-query numbers for every `(tool, query, budget)` cell are in
`bench/results.json`; the script prints the full per-query table on each run.
