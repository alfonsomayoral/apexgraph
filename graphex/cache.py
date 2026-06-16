"""On-disk cache for the query-independent half of scoring.

Global PageRank and the BM25 inverted index depend only on the graph, not on
the query — recomputing them on every call (as naive tools do) is wasted work.
We compute them once, store them under ``.graphex/cache.json``, and invalidate
by the graph's content :meth:`~graphex.models.KnowledgeGraph.fingerprint`.

A query then costs only a BM25 lookup over the postings plus one Personalized
PageRank walk — everything heavy is already on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from graphex.models import KnowledgeGraph
from graphex.retrieval.bm25 import BM25Index
from graphex.retrieval.ppr import global_pagerank

CACHE_DIRNAME = ".graphex"
CACHE_FILENAME = "cache.json"
_CACHE_VERSION = 2


@dataclass(slots=True)
class CachedArtifacts:
    """The precomputed, query-independent scoring inputs for one graph."""

    fingerprint: str
    bm25: BM25Index
    global_pagerank: dict[str, float]


def _cache_path(base_dir: Path) -> Path:
    return base_dir / CACHE_DIRNAME / CACHE_FILENAME


def build_artifacts(graph: KnowledgeGraph) -> CachedArtifacts:
    """Compute the artifacts from scratch (no disk I/O)."""
    return CachedArtifacts(
        fingerprint=graph.fingerprint(),
        bm25=BM25Index.from_graph(graph),
        global_pagerank=global_pagerank(graph),
    )


def load_or_build(
    graph: KnowledgeGraph,
    base_dir: Path | None = None,
    *,
    use_cache: bool = True,
) -> CachedArtifacts:
    """Return cached artifacts if the fingerprint matches, else build and store.

    Args:
        graph: The graph to score against.
        base_dir: Directory whose ``.graphex/`` subdir holds the cache. Defaults
            to the current working directory. ``None`` + ``use_cache=False``
            skips disk entirely.
        use_cache: When False, always rebuild and never read or write disk.

    A corrupt or version-mismatched cache file is silently ignored and rebuilt.
    """
    if not use_cache:
        return build_artifacts(graph)

    base = base_dir or Path.cwd()
    path = _cache_path(base)
    fingerprint = graph.fingerprint()

    cached = _try_read(path, fingerprint)
    if cached is not None:
        return cached

    artifacts = build_artifacts(graph)
    _write(path, artifacts)
    return artifacts


def _try_read(path: Path, fingerprint: str) -> CachedArtifacts | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != _CACHE_VERSION:
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    try:
        bm25 = BM25Index.from_dict(data["bm25"])
        global_pr = {str(k): float(v) for k, v in data["global_pagerank"].items()}
    except (KeyError, TypeError, ValueError):
        return None
    return CachedArtifacts(fingerprint=fingerprint, bm25=bm25, global_pagerank=global_pr)


def _write(path: Path, artifacts: CachedArtifacts) -> None:
    payload = {
        "version": _CACHE_VERSION,
        "fingerprint": artifacts.fingerprint,
        "bm25": artifacts.bm25.to_dict(),
        "global_pagerank": artifacts.global_pagerank,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write: tmp then replace, so a crash never leaves a half file.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        # Cache is a performance optimisation; never fail the query over it.
        pass
