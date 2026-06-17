"""Tests for :mod:`apexgraph.loader` across JSON, GraphML and Neo4j CSV formats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apexgraph.loader import (
    ApexgraphLoadError,
    convert_graph,
    detect_format,
    load_graph,
    load_graph_neo4j,
)
from apexgraph.models import KnowledgeGraph

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _graphify_payload() -> dict:
    """A rich graphify document with links, weights, communities and hyperedges."""
    return {
        "nodes": [
            {
                "id": "a",
                "label": "Alpha",
                "type": "function",
                "file_type": "code",
                "description": "entry point",
                "importance": 0.9,
                "community": 0,
                "is_god": True,
                "source_file": "a.py",
                "custom_key": "kept",
            },
            {
                "id": "b",
                "label": "Beta",
                "type": "class",
                "file_type": "code",
                "importance": 0.4,
                "community": 1,
                "god": True,  # alias for is_god
            },
            {"id": "c", "file_type": "weird", "community": 1},
        ],
        "links": [
            {
                "source": "a",
                "target": "b",
                "relation": "calls",
                "weight": 2.5,
                "confidence": "INFERRED",
                "confidence_score": 0.5,
                "note": "extra-edge-key",
            },
            {"source": "b", "target": "c", "relation": "owns", "weight": 1.0},
            # dangling edge: endpoint 'z' is not a node -> skipped
            {"source": "a", "target": "z", "relation": "ghost"},
        ],
        "hyperedges": [
            {
                "id": "he1",
                "label": "flow",
                "nodes": ["a", "b", "c"],
                "relation": "pipeline",
                "confidence_score": 0.8,
            }
        ],
    }


def _write_json(tmp_path: Path, payload: dict, name: str = "g.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# JSON: graphify
# ---------------------------------------------------------------------------


def test_graphify_json_loads_rich_signals(tmp_path: Path) -> None:
    path = _write_json(tmp_path, _graphify_payload())
    kg = load_graph(path)

    assert isinstance(kg, KnowledgeGraph)
    assert len(kg) == 3

    # weights & confidence preserved on the edge
    edge_ab = kg.digraph.edges["a", "b"]
    assert edge_ab["weight"] == 2.5
    assert edge_ab["confidence"] == "INFERRED"
    assert edge_ab["confidence_score"] == 0.5
    assert edge_ab["relation"] == "calls"
    assert edge_ab["note"] == "extra-edge-key"  # edge extras preserved

    # dangling edge to 'z' was skipped
    assert not kg.digraph.has_edge("a", "z")
    assert kg.digraph.number_of_edges() == 2

    # communities preserved
    assert kg.communities == {"a": 0, "b": 1, "c": 1}

    # god nodes (is_god + god alias)
    assert kg.god_nodes == {"a", "b"}

    # node extras preserved
    assert kg.node("a")["custom_key"] == "kept"

    # invalid file_type coerced to "concept"
    assert kg.node("c")["file_type"] == "concept"

    # hyperedges preserved
    assert len(kg.hyperedges) == 1
    he = kg.hyperedges[0]
    assert he.id == "he1"
    assert he.nodes == ["a", "b", "c"]
    assert he.confidence_score == 0.8


def test_detect_format_graphify_vs_generic() -> None:
    assert detect_format(_graphify_payload()) == "graphify"
    # generic: plain edges, no rich node keys
    generic = {
        "nodes": [{"id": "x"}, {"id": "y"}],
        "edges": [{"source": "x", "target": "y"}],
    }
    assert detect_format(generic) == "generic"
    # presence of a rich node key flips to graphify
    rich = {"nodes": [{"id": "x", "importance": 0.5}], "edges": []}
    assert detect_format(rich) == "graphify"


# ---------------------------------------------------------------------------
# JSON: generic
# ---------------------------------------------------------------------------


def test_generic_json_with_edges(tmp_path: Path) -> None:
    payload = {
        "nodes": [{"id": "x", "label": "X"}, {"id": "y"}],
        "edges": [{"source": "x", "target": "y", "relation": "near"}],
    }
    path = _write_json(tmp_path, payload, name="generic.json")
    kg = load_graph(path)

    assert len(kg) == 2
    assert kg.digraph.has_edge("x", "y")
    assert kg.digraph.edges["x", "y"]["relation"] == "near"
    # label defaults to id when missing
    assert kg.node("y")["label"] == "y"


# ---------------------------------------------------------------------------
# JSON: error cases
# ---------------------------------------------------------------------------


def test_missing_node_id_raises(tmp_path: Path) -> None:
    path = _write_json(tmp_path, {"nodes": [{"label": "no id"}]}, name="bad.json")
    with pytest.raises(ApexgraphLoadError, match="id"):
        load_graph(path)


def test_empty_nodes_raises(tmp_path: Path) -> None:
    path = _write_json(tmp_path, {"nodes": []}, name="empty.json")
    with pytest.raises(ApexgraphLoadError, match="no nodes"):
        load_graph(path)


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ApexgraphLoadError, match="invalid JSON"):
        load_graph(path)


def test_non_object_toplevel_raises(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ApexgraphLoadError, match="object"):
        load_graph(path)


def test_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(ApexgraphLoadError, match="not found"):
        load_graph(tmp_path / "missing.json")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "thing.txt"
    path.write_text("hi", encoding="utf-8")
    with pytest.raises(ApexgraphLoadError, match="unsupported"):
        load_graph(path)


# ---------------------------------------------------------------------------
# GraphML round-trip
# ---------------------------------------------------------------------------


def test_graphml_roundtrip(tmp_path: Path) -> None:
    path = _write_json(tmp_path, _graphify_payload())
    kg = load_graph(path)

    out = convert_graph(kg, tmp_path / "exported", format="graphml")
    assert len(out) == 1
    assert out[0].exists()
    assert out[0].suffix == ".graphml"

    reloaded = load_graph(out[0])
    assert len(reloaded) == 3
    assert reloaded.digraph.has_edge("a", "b")
    # numeric weight/confidence survived the round-trip
    edge = reloaded.digraph.edges["a", "b"]
    assert edge["weight"] == 2.5
    assert edge["confidence_score"] == 0.5
    assert edge["relation"] == "calls"
    # label / type / description canonicalized
    assert reloaded.node("a")["label"] == "Alpha"
    assert reloaded.node("a")["type"] == "function"
    assert reloaded.node("a")["description"] == "entry point"


def test_graphml_canonicalizes_aliases(tmp_path: Path) -> None:
    import networkx as nx

    g = nx.DiGraph()
    g.add_node("n1", name="Named", kind="widget", desc="a thing")
    g.add_node("n2", label="Explicit")
    g.add_edge("n1", "n2", label="links-to", weight="3")
    gpath = tmp_path / "raw.graphml"
    nx.write_graphml(g, gpath)

    kg = load_graph(gpath)
    # name -> label, kind -> type, desc -> description
    assert kg.node("n1")["label"] == "Named"
    assert kg.node("n1")["type"] == "widget"
    assert kg.node("n1")["description"] == "a thing"
    # edge label -> relation, weight coerced to float
    edge = kg.digraph.edges["n1", "n2"]
    assert edge["relation"] == "links-to"
    assert edge["weight"] == 3.0


# ---------------------------------------------------------------------------
# Neo4j CSV
# ---------------------------------------------------------------------------


def test_neo4j_native_columns(tmp_path: Path) -> None:
    nodes = tmp_path / "graph_nodes.csv"
    nodes.write_text(
        ":ID,name,:LABEL,importance,extra_col\n"
        "n1,First,Concept,0.7,foo\n"
        "n2,Second,Concept,0.2,bar\n",
        encoding="utf-8",
    )
    rels = tmp_path / "graph_relationships.csv"
    rels.write_text(
        ":START_ID,:END_ID,:TYPE,weight,confidence_score\n" "n1,n2,RELATES,1.5,0.9\n"
        # malformed row (missing end) -> skipped silently
        "n1,,BROKEN,1.0,1.0\n",
        encoding="utf-8",
    )

    kg = load_graph_neo4j(nodes, rels)
    assert len(kg) == 2
    assert kg.node("n1")["label"] == "First"
    assert kg.node("n1")["importance"] == 0.7
    assert kg.node("n1")["extra_col"] == "foo"  # unknown column -> extras

    assert kg.digraph.number_of_edges() == 1
    edge = kg.digraph.edges["n1", "n2"]
    assert edge["relation"] == "RELATES"
    assert edge["weight"] == 1.5
    assert edge["confidence_score"] == 0.9


def test_neo4j_simplified_columns_autodiscovery(tmp_path: Path) -> None:
    # load_graph(.csv) should auto-discover the sibling relationships file
    nodes = tmp_path / "mygraph_nodes.csv"
    nodes.write_text(
        "nodeId,name,labels\nn1,First,Concept\nn2,Second,Concept\n",
        encoding="utf-8",
    )
    rels = tmp_path / "mygraph_relationships.csv"
    rels.write_text(
        "startNodeId,endNodeId,type\nn1,n2,LINKS\n",
        encoding="utf-8",
    )

    kg = load_graph(nodes)
    assert len(kg) == 2
    assert kg.digraph.has_edge("n1", "n2")
    assert kg.digraph.edges["n1", "n2"]["relation"] == "LINKS"


def test_neo4j_missing_id_column_raises(tmp_path: Path) -> None:
    nodes = tmp_path / "bad_nodes.csv"
    nodes.write_text("name,labels\nFirst,Concept\n", encoding="utf-8")
    with pytest.raises(ApexgraphLoadError, match="node-id column"):
        load_graph_neo4j(nodes)


def test_neo4j_no_data_rows_raises(tmp_path: Path) -> None:
    nodes = tmp_path / "header_only_nodes.csv"
    nodes.write_text(":ID,name\n", encoding="utf-8")
    with pytest.raises(ApexgraphLoadError, match="no data rows"):
        load_graph_neo4j(nodes)


def test_neo4j_nodes_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ApexgraphLoadError, match="not found"):
        load_graph_neo4j(tmp_path / "nope_nodes.csv")


# ---------------------------------------------------------------------------
# Neo4j export round-trip
# ---------------------------------------------------------------------------


def test_convert_neo4j_writes_two_csvs(tmp_path: Path) -> None:
    path = _write_json(tmp_path, _graphify_payload())
    kg = load_graph(path)

    out = convert_graph(kg, tmp_path / "exp", format="neo4j")
    assert len(out) == 2
    assert all(p.exists() for p in out)
    assert out[0].name == "exp_nodes.csv"
    assert out[1].name == "exp_relationships.csv"

    # round-trip back in
    reloaded = load_graph_neo4j(out[0], out[1])
    assert len(reloaded) == 3
    assert reloaded.digraph.has_edge("a", "b")
    assert reloaded.digraph.edges["a", "b"]["weight"] == 2.5


def test_convert_unknown_format_raises(tmp_path: Path) -> None:
    kg = KnowledgeGraph()
    with pytest.raises(ApexgraphLoadError, match="unknown export format"):
        convert_graph(kg, tmp_path / "x", format="bogus")
