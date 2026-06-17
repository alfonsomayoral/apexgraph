"""Retrieval backends — turn a query into a per-node relevance score.

The default pipeline is fully local and dependency-light: BM25 lexical matching
seeds a Personalized PageRank walk over the weighted graph. Dense embeddings are
an optional plugin (see :mod:`apexgraph.retrieval.dense`).
"""
