"""Append-only query audit log.

Every retrieval can be recorded as one JSON line in ``<audit_dir>/audit.jsonl``
(default ``.graphex/audit.jsonl``). The log is *best-effort*: writing it must
never break a query, so I/O errors are swallowed. Reading is tolerant of partial
writes — malformed lines are skipped rather than raised on.

Beyond the raw round-trip, :func:`top_nodes_from_audit` aggregates the most
frequently selected nodes across the whole history, a cheap proxy for "what does
this graph keep surfacing".
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

AUDIT_DIRNAME = ".graphex"
AUDIT_FILENAME = "audit.jsonl"

# Cap on how many top nodes we persist per entry.
_TOP_NODES_CAP = 10


def _audit_path(audit_dir: Path | str) -> Path:
    return Path(audit_dir) / AUDIT_FILENAME


def log_query(
    query: str,
    graph_path: Path,
    stats: dict,
    top_nodes: list[str],
    audit_dir: Path | str = AUDIT_DIRNAME,
) -> None:
    """Append one audit record for a completed query.

    The record captures the query string, the graph it ran against, selection /
    token statistics (read from ``stats`` with keys ``nodes_selected``,
    ``nodes_total``, ``tokens_used``, ``tokens_budget``), and up to the first
    ten ``top_nodes``. A UTC ISO-8601 ``timestamp`` is stamped automatically.

    This is best-effort: the audit directory is created if missing, and any
    :class:`OSError` raised while writing is swallowed so auditing never breaks
    a retrieval.
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "graph": str(graph_path),
        "nodes_selected": stats.get("nodes_selected"),
        "nodes_total": stats.get("nodes_total"),
        "tokens_used": stats.get("tokens_used"),
        "tokens_budget": stats.get("tokens_budget"),
        "top_nodes": list(top_nodes[:_TOP_NODES_CAP]),
    }
    try:
        directory = Path(audit_dir)
        directory.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str)
        with _audit_path(directory).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        # Auditing is best-effort — never propagate I/O failures to the caller.
        return


def read_audit(audit_dir: Path | str = AUDIT_DIRNAME) -> list[dict]:
    """Read and parse every audit record.

    Returns an empty list if the file does not exist. Malformed lines (e.g. from
    a partial write) are skipped rather than raised on.
    """
    path = _audit_path(audit_dir)
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def top_nodes_from_audit(
    audit_dir: Path | str = AUDIT_DIRNAME, n: int = 10
) -> list[tuple[str, int]]:
    """Rank node ids by how often they appear across all entries' ``top_nodes``.

    Returns the ``n`` most common ``(node_id, count)`` pairs in descending order
    of count.
    """
    counter: Counter[str] = Counter()
    for record in read_audit(audit_dir):
        top_nodes = record.get("top_nodes")
        if not isinstance(top_nodes, list):
            continue
        counter.update(str(node_id) for node_id in top_nodes)
    return counter.most_common(n)
