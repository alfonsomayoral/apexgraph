# Apexgraph examples

A small, self-contained walkthrough of Apexgraph on a realistic sample codebase.

- [`sample_project/`](sample_project/) — a tiny task-tracker web app (auth + db +
  api layers, ~10 Python files) whose modules import and call into each other.
- [`sample_graph.json`](sample_graph.json) — the knowledge graph produced by
  statically indexing `sample_project/`. It is committed so you can run every
  query below without re-indexing.

All commands are run from the repository root with `uv run apexgraph ...`. The
output shown is real output captured from this graph (numbers may shift slightly
if you re-index after editing the sample).

## The sample project

```
sample_project/
├── auth/
│   ├── service.py    # AuthService: login / logout / validate_token, password hashing
│   └── session.py    # Session model + SessionStore (create/get/destroy/purge)
├── db/
│   ├── models.py     # User, Task dataclasses (+ from_row / public_dict)
│   └── pool.py       # ConnectionPool + find_user_by_email / insert_task / list_tasks_for_user
└── api/
    ├── routes.py     # Router: login/logout/tasks/me handlers, require_auth
    └── server.py     # build_app() composition root + route table
```

The layers depend downward: `api` imports from `auth` and `db`; `auth.service`
imports from `auth.session` and `db`. That gives the indexer real
`imports_from` and `contains` edges to work with.

## 1. Index the project

```console
$ uv run apexgraph index examples/sample_project -o examples/sample_graph.json
Indexing C:\Users\amayo\apexgraph\examples\sample_project ...
  102 nodes · 94 edges · 10 files
  Saved: examples\sample_graph.json
```

The static indexer (no LLM) emits a `module` node per file, a `class`/`function`
node per definition, and an `import` node per imported name — wired together
with `contains` and `imports_from` edges.

## 2. Run a query

Ask a natural-language question and Apexgraph selects the most relevant subgraph
that fits the token budget (`-b`/`--budget`):

```console
$ uv run apexgraph "how does login authentication work" -g examples/sample_graph.json -b 1500 --no-cache --no-audit
┌──────────────────────────────────────────────────────────┐
│ Apexgraph subgraph for: how does login authentication work │
│ Selected 3/102 nodes · 81/1500 tokens (2.9%)             │
└──────────────────────────────────────────────────────────┘

## Relevant Nodes

### login (function) · score: 1.00

Function login

→ File: auth/service.py

### login_route (function) · score: 0.90

Function login_route

→ File: api/routes.py

### LoginResult (class) · score: 0.86

Class LoginResult

→ File: auth/service.py

## Key Relationships


_Tip: 99 node(s) were excluded to fit the token budget; raise the budget to see more._
```

The top hit is `AuthService.login` in `auth/service.py`, followed by the HTTP
handler `login_route` that calls it and the `LoginResult` value it returns —
exactly the slice you'd want to paste into an agent's context.

A second query, this time about sessions, pulls in the whole session subsystem:

```console
$ uv run apexgraph "session token validation and expiry" -g examples/sample_graph.json -b 1500 --no-cache --no-audit
┌───────────────────────────────────────────────────────────┐
│ Apexgraph subgraph for: session token validation and expiry │
│ Selected 11/102 nodes · 330/1500 tokens (10.8%)           │
└───────────────────────────────────────────────────────────┘
...
### validate_token (function) · score: 1.00   → auth/service.py
### Session (class) · score: 0.68             → auth/session.py
### create_session (function) · score: 0.61   → auth/session.py
### get_session (function) · score: 0.61      → auth/session.py
### is_expired (function) · score: 0.58       → auth/session.py
...
## Key Relationships

- auth_session_session → contains → auth_session_is_expired
```

## 3. Graph statistics

```console
$ uv run apexgraph stats -g examples/sample_graph.json
       Graph:
examples\sample_graph
        .json
┌─────────────┬─────┐
│ Nodes       │ 102 │
│ Edges       │  94 │
│ Hyperedges  │   0 │
│ Communities │   0 │
│ God nodes   │   0 │
└─────────────┴─────┘
```

(Hyperedges, communities, and god nodes are richer signals that the full
`graphify` producer emits; the lightweight static `index` command does not, so
they read as zero here.)

## 4. Explain a single node

`explain` shows a node and its immediate neighbourhood — what it contains and
what references it:

```console
$ uv run apexgraph explain auth_service_authservice -g examples/sample_graph.json
# AuthService  (class)
Class AuthService
→ auth/service.py L43

## Used by
- service.py → contains

## Depends on
- contains → __init__
- contains → login
- contains → _check_user
- contains → logout
- contains → validate_token
```

## 5. Shortest path between two nodes

`path` walks the directed graph from a source node to a target node. Here it
traces the containment chain from the `auth/service.py` module down to the
`login` method, through the `AuthService` class:

```console
$ uv run apexgraph path auth_service auth_service_login -g examples/sample_graph.json
service.py → AuthService → login
```

Node ids follow the indexer convention `{parent_dir}_{stem}` for modules and
`{module_id}_{symbol}` for symbols (e.g. `auth_service` for the module,
`auth_service_login` for the `login` function inside it).
