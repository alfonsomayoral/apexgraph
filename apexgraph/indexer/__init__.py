"""Static source indexer — build a graph.json directly from code, no LLM needed.

Supports Python (stdlib ``ast``), TypeScript/JavaScript (tree-sitter with a regex
fallback), and Go (regex). Output is graphify-compatible so the rest of Apexgraph
treats indexed graphs and graphify graphs identically.
"""
