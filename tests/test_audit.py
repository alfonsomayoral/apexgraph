"""Append-only audit log round-trip and aggregation."""

from __future__ import annotations

from pathlib import Path

from graphex.audit import (
    AUDIT_FILENAME,
    log_query,
    read_audit,
    top_nodes_from_audit,
)


def _stats(selected: int = 2, total: int = 10) -> dict:
    return {
        "nodes_selected": selected,
        "nodes_total": total,
        "tokens_used": 1234,
        "tokens_budget": 8000,
    }


def test_log_then_read_round_trips_fields(tmp_path: Path):
    log_query(
        query="how does retrieval work",
        graph_path=Path("graph.json"),
        stats=_stats(),
        top_nodes=["a", "b", "c"],
        audit_dir=tmp_path,
    )
    records = read_audit(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["query"] == "how does retrieval work"
    assert rec["graph"] == "graph.json"
    assert rec["nodes_selected"] == 2
    assert rec["nodes_total"] == 10
    assert rec["tokens_used"] == 1234
    assert rec["tokens_budget"] == 8000
    assert rec["top_nodes"] == ["a", "b", "c"]
    assert "timestamp" in rec and rec["timestamp"].endswith("+00:00")


def test_log_appends_and_creates_dir(tmp_path: Path):
    audit_dir = tmp_path / "nested" / ".graphex"
    log_query("q1", Path("g.json"), _stats(), ["a"], audit_dir=audit_dir)
    log_query("q2", Path("g.json"), _stats(), ["b"], audit_dir=audit_dir)
    assert (audit_dir / AUDIT_FILENAME).exists()
    records = read_audit(audit_dir)
    assert [r["query"] for r in records] == ["q1", "q2"]


def test_top_nodes_capped_to_ten(tmp_path: Path):
    log_query(
        "q",
        Path("g.json"),
        _stats(),
        [f"n{i}" for i in range(20)],
        audit_dir=tmp_path,
    )
    rec = read_audit(tmp_path)[0]
    assert len(rec["top_nodes"]) == 10
    assert rec["top_nodes"] == [f"n{i}" for i in range(10)]


def test_read_audit_missing_file_returns_empty(tmp_path: Path):
    assert read_audit(tmp_path) == []
    assert read_audit(tmp_path / "does-not-exist") == []


def test_read_audit_skips_malformed_lines(tmp_path: Path):
    log_query("good", Path("g.json"), _stats(), ["a"], audit_dir=tmp_path)
    path = tmp_path / AUDIT_FILENAME
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{ not valid json\n")
        fh.write("\n")  # blank line
        fh.write('"a json string but not a dict"\n')
    log_query("good2", Path("g.json"), _stats(), ["b"], audit_dir=tmp_path)
    records = read_audit(tmp_path)
    assert [r["query"] for r in records] == ["good", "good2"]


def test_top_nodes_from_audit_ranks_across_entries(tmp_path: Path):
    log_query("q1", Path("g.json"), _stats(), ["a", "b"], audit_dir=tmp_path)
    log_query("q2", Path("g.json"), _stats(), ["a", "c"], audit_dir=tmp_path)
    log_query("q3", Path("g.json"), _stats(), ["a", "b"], audit_dir=tmp_path)
    ranked = top_nodes_from_audit(tmp_path)
    assert ranked[0] == ("a", 3)
    assert ("b", 2) in ranked
    assert ("c", 1) in ranked


def test_top_nodes_from_audit_respects_n(tmp_path: Path):
    log_query("q", Path("g.json"), _stats(), ["a", "b", "c", "d"], audit_dir=tmp_path)
    ranked = top_nodes_from_audit(tmp_path, n=2)
    assert len(ranked) == 2


def test_top_nodes_from_audit_empty_when_no_log(tmp_path: Path):
    assert top_nodes_from_audit(tmp_path) == []


def test_log_query_swallows_io_error(tmp_path: Path):
    # Point audit_dir at a path whose parent is an existing *file*, so mkdir
    # fails with OSError — log_query must swallow it and not raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file", encoding="utf-8")
    bad_dir = blocker / "sub"
    # Should not raise.
    log_query("q", Path("g.json"), _stats(), ["a"], audit_dir=bad_dir)
    # Nothing was written.
    assert read_audit(bad_dir) == []
