"""BM25 lexical retriever over node text, backed by a cached inverted index.

This is the *lexical* leg of retrieval: it scores nodes by Okapi BM25 over the
tokenised :meth:`~apexgraph.models.KnowledgeGraph.node_text` of every node. It is
deliberately cheap to build and trivially serialisable so the index can live in
the on-disk cache alongside the graph fingerprint.

The tokenizer is identifier-aware: it splits ``camelCase`` / ``PascalCase`` /
``snake_case`` / ``kebab-case`` / ``dotted.paths`` into their parts *and* keeps
the original compound tokens, so a query for either ``player`` or the full
``recalcularPlayerStats`` lands on the same node.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import snowballstemmer

from apexgraph.models import KnowledgeGraph
from apexgraph.retrieval.base import normalize

# Default Okapi BM25 hyper-parameters.
_K1: float = 1.5
_B: float = 0.75

# A single, reusable English Snowball stemmer. It is stateless, so one shared
# instance is safe to call from anywhere. Stemming maps morphological variants
# ("authentication", "authenticate") onto a shared stem, closing a recall gap
# between query and document wording. For non-words (numbers, already-minimal
# tokens) Snowball returns the input unchanged.
_STEMMER = snowballstemmer.stemmer("english")

# Identifier-boundary substitutions: "HTTPServer" -> "HTTP Server",
# "playerStats" -> "player Stats". Applied before lowercasing.
_BOUNDARY_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_BOUNDARY_LOWER_UPPER = re.compile(r"([a-z0-9])([A-Z])")
# A run of alphanumerics, the atomic token unit.
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Split ``text`` into lexical tokens, preserving compound identifiers.

    Splits ``camelCase`` / ``PascalCase`` / ``snake_case`` / ``kebab-case`` /
    ``dotted.paths`` into their parts by inserting spaces at the
    ``([A-Z]+)([A-Z][a-z])`` and ``([a-z0-9])([A-Z])`` boundaries, lowercasing,
    and extracting ``[a-z0-9]+`` runs. It then also appends any lowercase
    ``[a-z0-9]+`` run from the *original* text that isn't already present, so the
    original compound token (e.g. ``recalcularplayerstats``) survives alongside
    its split parts (``recalcular``, ``player``, ``stats``).

    When ``stem`` is true (the default) each token — split parts and preserved
    compounds alike — is reduced to its English Snowball stem, so morphological
    variants match across queries and documents (e.g. ``authentication`` and
    ``authenticate`` collapse onto a shared stem). Stemming is applied
    identically here for both sides, so the BM25 ``df``/IDF are computed over
    stems, which is intended. Numeric and other non-word tokens are returned
    unchanged by Snowball.

    The returned order is deterministic: split parts first (in reading order),
    then the extra compound tokens (in reading order). Consecutive duplicates
    that stemming may introduce (e.g. a split part and its stem collapsing into
    its neighbour) are dropped to keep the sequence tidy.

    Args:
        text: Arbitrary free text or an identifier.
        stem: When true, reduce every token to its English Snowball stem.

    Returns:
        The list of tokens. May contain duplicates among the split parts, but no
        appended compound duplicates a token already produced by the split.
    """
    spaced = _BOUNDARY_ACRONYM.sub(r"\1 \2", text)
    spaced = _BOUNDARY_LOWER_UPPER.sub(r"\1 \2", spaced)
    parts = _WORD.findall(spaced.lower())

    seen = set(parts)
    for compound in _WORD.findall(text.lower()):
        if compound not in seen:
            parts.append(compound)
            seen.add(compound)

    if not stem:
        return parts

    stemmed: list[str] = []
    for tok in parts:
        if not tok:
            continue
        stem_tok = _STEMMER.stemWord(tok) or tok
        # Drop only consecutive duplicates stemming may create; non-adjacent
        # repeats (genuine term frequency) are preserved for BM25.
        if stemmed and stemmed[-1] == stem_tok:
            continue
        stemmed.append(stem_tok)
    return stemmed


class BM25Index:
    """An inverted-index BM25 model over a knowledge graph's node text.

    The index stores, per node, the token :class:`~collections.Counter` and the
    document length, plus corpus-level document frequencies, the document count
    ``N`` and the average document length ``avgdl``. Everything is plain Python
    so :meth:`to_dict` / :meth:`from_dict` round-trip it through JSON.

    Attributes:
        doc_tokens: ``{node_id: Counter(term -> term_frequency)}``.
        doc_len: ``{node_id: token_count}``.
        df: ``{term: number_of_docs_containing_term}``.
        N: Number of documents (nodes) in the corpus.
        avgdl: Mean document length; ``0.0`` for an empty corpus.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalisation parameter.
    """

    def __init__(
        self,
        doc_tokens: dict[str, Counter[str]],
        doc_len: dict[str, int],
        df: dict[str, int],
        N: int,
        avgdl: float,
        k1: float = _K1,
        b: float = _B,
    ) -> None:
        self.doc_tokens = doc_tokens
        self.doc_len = doc_len
        self.df = df
        self.N = N
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b

    # -- construction --------------------------------------------------------

    @classmethod
    def from_graph(cls, graph: KnowledgeGraph, k1: float = _K1, b: float = _B) -> BM25Index:
        """Build an index by tokenising every node's :meth:`node_text`."""
        doc_tokens: dict[str, Counter[str]] = {}
        doc_len: dict[str, int] = {}
        df: dict[str, int] = {}

        for node_id in graph.node_ids:
            tokens = tokenize(graph.node_text(node_id))
            counts: Counter[str] = Counter(tokens)
            doc_tokens[node_id] = counts
            doc_len[node_id] = sum(counts.values())
            for term in counts:
                df[term] = df.get(term, 0) + 1

        N = len(doc_tokens)
        total_len = sum(doc_len.values())
        avgdl = total_len / N if N else 0.0
        return cls(doc_tokens, doc_len, df, N, avgdl, k1=k1, b=b)

    # -- scoring -------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Okapi BM25 IDF with the ``ln(1 + ...)`` smoothing (always >= 0)."""
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[str, float]:
        """Return the raw BM25 score for every node id (``0.0`` if no overlap).

        Only query terms that appear in the index contribute; nodes with no
        query-term overlap score exactly ``0.0``.
        """
        result: dict[str, float] = {node_id: 0.0 for node_id in self.doc_tokens}
        if self.N == 0 or self.avgdl == 0.0:
            return result

        # Distinct query terms that actually exist in the corpus.
        query_terms = {t for t in tokenize(query) if self.df.get(t, 0) > 0}
        for term in query_terms:
            idf = self._idf(term)
            for node_id, counts in self.doc_tokens.items():
                f = counts.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[node_id] / self.avgdl)
                result[node_id] += idf * (f * (self.k1 + 1.0)) / denom
        return result

    def normalized_scores(self, query: str) -> dict[str, float]:
        """:meth:`scores` rescaled to ``[0, 1]`` via :func:`normalize`."""
        return normalize(self.scores(query))

    def seeds(self, query: str, k: int = 10) -> dict[str, float]:
        """Top-``k`` matching nodes as a probability distribution.

        Takes the ``k`` highest raw-BM25 nodes with a strictly positive score and
        normalises them to sum to ``1.0`` — a seed distribution for a random walk.
        Returns an empty dict when nothing matches.
        """
        raw = self.scores(query)
        ranked = sorted(
            ((nid, s) for nid, s in raw.items() if s > 0.0),
            key=lambda kv: (-kv[1], kv[0]),
        )[:k]
        total = sum(s for _, s in ranked)
        if total <= 0.0:
            return {}
        return {nid: s / total for nid, s in ranked}

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for caching."""
        return {
            "doc_tokens": {nid: dict(counts) for nid, counts in self.doc_tokens.items()},
            "doc_len": dict(self.doc_len),
            "df": dict(self.df),
            "N": self.N,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BM25Index:
        """Reconstruct an index from :meth:`to_dict` output."""
        doc_tokens = {nid: Counter(counts) for nid, counts in data["doc_tokens"].items()}
        doc_len = {nid: int(v) for nid, v in data["doc_len"].items()}
        df = {term: int(v) for term, v in data["df"].items()}
        return cls(
            doc_tokens=doc_tokens,
            doc_len=doc_len,
            df=df,
            N=int(data["N"]),
            avgdl=float(data["avgdl"]),
            k1=float(data.get("k1", _K1)),
            b=float(data.get("b", _B)),
        )


class BM25Retriever:
    """A :class:`~apexgraph.retrieval.base.Retriever` backed by :class:`BM25Index`.

    Builds a fresh index per call and returns length-normalised scores. Use
    :class:`BM25Index` directly when you want to reuse a cached index across
    queries.
    """

    def __init__(self, k1: float = _K1, b: float = _B) -> None:
        self.k1 = k1
        self.b = b

    def score(self, graph: KnowledgeGraph, query: str) -> dict[str, float]:
        """Build an index over ``graph`` and return normalised BM25 scores."""
        index = BM25Index.from_graph(graph, k1=self.k1, b=self.b)
        return index.normalized_scores(query)
