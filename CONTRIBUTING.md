# Contributing to Apexgraph

Thanks for your interest in improving Apexgraph.

## Setup

```bash
git clone https://github.com/alfonsomayoral/apexgraph
cd apexgraph
uv sync
```

## Before you open a PR

```bash
uv run ruff check .     # lint (must pass)
uv run black --check .  # format (must pass)
uv run pytest           # tests (must pass)
```

## Architecture at a glance

Each module has a single responsibility; the data contract lives in
`apexgraph/models.py` (`KnowledgeGraph`, `Node`, `Edge`, `Hyperedge`). The scoring
pipeline is `retrieval/bm25.py` → `retrieval/ppr.py` → `retrieval/fusion.py` →
`scorer.py`; selection is `budget.py`; the user surface is `cli.py` and `mcp.py`.

When you add a module, add a matching `tests/test_<module>.py`. Keep public
functions typed and documented, and keep new dependencies out of the default
install path — put optional features behind an extra in `pyproject.toml`.

## Guidelines

- Prefer reusing the helpers already in `models.py` and `retrieval/base.py`.
- The budget invariant is sacred: `tokens_used` must never exceed the requested
  budget. Any selection change needs a test that proves it.
- Token-saving claims must be paired with a recall metric — see `benchmark.py`.
