"""Cache round-trip and fingerprint-based invalidation."""

from __future__ import annotations

from pathlib import Path

from apexgraph.cache import CACHE_DIRNAME, build_artifacts, load_or_build
from apexgraph.models import Edge, KnowledgeGraph, Node


def _graph(extra: bool = False) -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node(Node(id="a", label="alpha", description="first node"))
    g.add_node(Node(id="b", label="beta", description="second node"))
    g.add_edge(Edge(source="a", target="b"))
    if extra:
        g.add_node(Node(id="c", label="gamma", description="third node"))
        g.add_edge(Edge(source="b", target="c"))
    return g


def test_build_artifacts_in_memory():
    g = _graph()
    art = build_artifacts(g)
    assert art.fingerprint == g.fingerprint()
    assert set(art.global_pagerank) == set(g.node_ids)
    assert art.bm25.scores("alpha")["a"] > 0


def test_cache_written_and_reused(tmp_path: Path):
    g = _graph()
    art1 = load_or_build(g, base_dir=tmp_path)
    cache_file = tmp_path / CACHE_DIRNAME / "cache.json"
    assert cache_file.exists()

    # Second call hits disk and returns equivalent artifacts.
    art2 = load_or_build(g, base_dir=tmp_path)
    assert art2.fingerprint == art1.fingerprint
    assert art2.bm25.scores("beta") == art1.bm25.scores("beta")


def test_fingerprint_change_invalidates(tmp_path: Path):
    g1 = _graph()
    load_or_build(g1, base_dir=tmp_path)

    g2 = _graph(extra=True)  # different structure → different fingerprint
    art = load_or_build(g2, base_dir=tmp_path)
    assert art.fingerprint == g2.fingerprint()
    assert "c" in art.global_pagerank


def test_corrupt_cache_is_rebuilt(tmp_path: Path):
    g = _graph()
    load_or_build(g, base_dir=tmp_path)
    (tmp_path / CACHE_DIRNAME / "cache.json").write_text("{ not json", encoding="utf-8")
    art = load_or_build(g, base_dir=tmp_path)  # must not raise
    assert art.fingerprint == g.fingerprint()


def test_use_cache_false_skips_disk(tmp_path: Path):
    g = _graph()
    load_or_build(g, base_dir=tmp_path, use_cache=False)
    assert not (tmp_path / CACHE_DIRNAME).exists()
