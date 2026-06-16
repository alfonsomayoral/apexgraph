"""Optional dense-embedding retriever (behind the ``[dense]`` extra).

Off the hot path by design: the default pipeline is fully local. When real
embeddings help — cross-vocabulary queries, natural-language questions that
don't share tokens with the labels — this backend scores nodes by cosine
similarity to the query embedding, and :mod:`graphex.retrieval.fusion` can blend
its ranking with BM25 via Reciprocal Rank Fusion.

Requires ``pip install 'graphex[dense]'`` and the relevant provider API key.
"""

from __future__ import annotations

import math

from graphex.models import KnowledgeGraph
from graphex.retrieval.base import normalize

_OPENAI_MODEL = "text-embedding-3-small"
_ANTHROPIC_MODEL = "voyage-3-large"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _embed_openai(texts: list[str]) -> list[list[float]]:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("The openai backend requires: pip install 'graphex[dense]'") from exc
    client = openai.OpenAI()
    resp = client.embeddings.create(model=_OPENAI_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _embed_anthropic(texts: list[str]) -> list[list[float]]:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("The anthropic backend requires: pip install 'graphex[dense]'") from exc
    client = anthropic.Anthropic()
    body = {"model": _ANTHROPIC_MODEL, "input": texts, "input_type": "document"}
    resp = client.post(path="/v1/embeddings", body=body, cast_to=object)
    return [item["embedding"] for item in resp["data"]]


_BACKENDS = {"openai": _embed_openai, "anthropic": _embed_anthropic}


class DenseRetriever:
    """Scores nodes by embedding cosine similarity. Conforms to ``Retriever``.

    Args:
        backend: ``"openai"`` or ``"anthropic"``.
        embed_fn: Inject a custom ``list[str] -> list[list[float]]`` (handy for
            tests, or to plug a local embedding model). Overrides ``backend``.
    """

    def __init__(self, backend: str = "openai", embed_fn=None) -> None:
        if embed_fn is None and backend not in _BACKENDS:
            raise ValueError(
                f"Unknown dense backend {backend!r}; expected one of {sorted(_BACKENDS)}"
            )
        self.backend = backend
        self._embed = embed_fn or _BACKENDS[backend]

    def score(self, graph: KnowledgeGraph, query: str) -> dict[str, float]:
        ids = graph.node_ids
        if not ids:
            return {}
        texts = [query] + [graph.node_text(nid) or nid for nid in ids]
        embeddings = self._embed(texts)
        q_emb = embeddings[0]
        raw = {nid: _cosine(q_emb, emb) for nid, emb in zip(ids, embeddings[1:], strict=False)}
        return normalize(raw)
