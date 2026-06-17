"""On-disk cache for the query-independent half of scoring.

Global PageRank and the BM25 inverted index depend only on the graph, not on
the query — recomputing them on every call (as naive tools do) is wasted work.
We compute them once, store them under ``.apexgraph/cache.json``, and invalidate
by the graph's content :meth:`~apexgraph.models.KnowledgeGraph.fingerprint`.

A query then costs only a BM25 lookup over the postings plus one Personalized
PageRank walk — everything heavy is already on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from apexgraph.budget import DEFAULT_TOKEN_MODEL, base_node_cost
from apexgraph.models import KnowledgeGraph
from apexgraph.retrieval.bm25 import BM25Index
from apexgraph.retrieval.ppr import global_pagerank

CACHE_DIRNAME = ".apexgraph"
CACHE_FILENAME = "cache.json"
# Bump whenever the meaning of a cached artifact changes (e.g. the tokenizer
# started stemming, which alters the BM25 index) so stale caches are discarded.
_CACHE_VERSION = 3


@dataclass(slots=True)
class CachedArtifacts:
    """The precomputed, query-independent scoring inputs for one graph."""

    fingerprint: str
    bm25: BM25Index
    global_pagerank: dict[str, float]
    # Base token cost per node (no code) for DEFAULT_TOKEN_MODEL; fed back into
    # select_subgraph to skip re-tokenizing every candidate on each query.
    token_costs: dict[str, int]
    token_model: str = DEFAULT_TOKEN_MODEL


def _cache_path(base_dir: Path) -> Path:
    return base_dir / CACHE_DIRNAME / CACHE_FILENAME


def build_artifacts(graph: KnowledgeGraph) -> CachedArtifacts:
    """Compute the artifacts from scratch (no disk I/O)."""
    return CachedArtifacts(
        fingerprint=graph.fingerprint(),
        bm25=BM25Index.from_graph(graph),
        global_pagerank=global_pagerank(graph),
        token_costs={nid: base_node_cost(graph, nid) for nid in graph.node_ids},
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
        base_dir: Directory whose ``.apexgraph/`` subdir holds the cache. Defaults
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
        token_costs = {str(k): int(v) for k, v in data["token_costs"].items()}
    except (KeyError, TypeError, ValueError):
        return None
    return CachedArtifacts(
        fingerprint=fingerprint,
        bm25=bm25,
        global_pagerank=global_pr,
        token_costs=token_costs,
        token_model=str(data.get("token_model", DEFAULT_TOKEN_MODEL)),
    )


def _write(path: Path, artifacts: CachedArtifacts) -> None:
    payload = {
        "version": _CACHE_VERSION,
        "fingerprint": artifacts.fingerprint,
        "bm25": artifacts.bm25.to_dict(),
        "global_pagerank": artifacts.global_pagerank,
        "token_costs": artifacts.token_costs,
        "token_model": artifacts.token_model,
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
