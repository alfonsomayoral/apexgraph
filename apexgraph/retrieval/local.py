"""Local static-embedding retriever (behind the ``[local]`` extra).

The default semantic path: real embeddings with **no API key and no network at
query time**. Uses ``model2vec`` static embeddings — distilled, CPU-friendly,
and fast enough to run inline — to score nodes by cosine similarity to the
query. Like :mod:`apexgraph.retrieval.dense`, its ranking can be blended with BM25
via Reciprocal Rank Fusion in :mod:`apexgraph.retrieval.fusion`.

Node embeddings are query-independent, so the caller can compute them once with
:meth:`LocalEmbeddingRetriever.node_embeddings`, cache them, and pass them back
into :meth:`LocalEmbeddingRetriever.score` on every query.

Requires ``pip install 'apexgraph[local]'``.
"""

from __future__ import annotations

import math

from apexgraph.models import KnowledgeGraph
from apexgraph.retrieval.base import normalize

_DEFAULT_MODEL = "minishlab/potion-base-8M"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


class LocalEmbeddingRetriever:
    """Scores nodes by static-embedding cosine similarity. Conforms to ``Retriever``.

    Args:
        model_name: A ``model2vec`` static model id (default
            ``"minishlab/potion-base-8M"``). Loaded lazily on first embed.
        embed_fn: Inject a custom ``list[str] -> list[list[float]]`` (handy for
            tests, or to plug a different local embedding model). When given, the
            ``model2vec`` model is never loaded.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, embed_fn=None) -> None:
        self.model_name = model_name
        self._embed_fn = embed_fn
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``. The single embedding entry point.

        Lazy-loads the ``model2vec`` model on first call unless an ``embed_fn``
        was injected.
        """
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        if self._model is None:
            try:
                from model2vec import StaticModel
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "The local backend requires: pip install 'apexgraph[local]'"
                ) from exc
            self._model = StaticModel.from_pretrained(self.model_name)
        # model2vec returns a numpy array; convert each row to a plain list[float]
        # so the rest of the pipeline works on builtin types.
        return [[float(x) for x in row] for row in self._model.encode(texts)]

    def node_embeddings(self, graph: KnowledgeGraph) -> dict[str, list[float]]:
        """Embed every node's searchable text (falling back to its id when empty).

        Query-independent — meant to be computed once and cached by the caller,
        then handed back to :meth:`score`.
        """
        ids = graph.node_ids
        if not ids:
            return {}
        texts = [graph.node_text(nid) or nid for nid in ids]
        vectors = self.embed_texts(texts)
        return dict(zip(ids, vectors, strict=False))

    def score(
        self,
        graph: KnowledgeGraph,
        query: str,
        node_embeddings: dict[str, list[float]] | None = None,
    ) -> dict[str, float]:
        """Score every node by cosine similarity to ``query``.

        Args:
            node_embeddings: Precomputed ``{node_id: vector}`` (e.g. from
                :meth:`node_embeddings`) to reuse across queries. When ``None``,
                node embeddings are computed fresh.

        Returns normalized scores in ``[0, 1]``; empty graph yields ``{}``.
        """
        ids = graph.node_ids
        if not ids:
            return {}
        if node_embeddings is None:
            node_embeddings = self.node_embeddings(graph)
        q_emb = self.embed_texts([query])[0]
        raw = {nid: _cosine(q_emb, node_embeddings[nid]) for nid in ids}
        return normalize(raw)
