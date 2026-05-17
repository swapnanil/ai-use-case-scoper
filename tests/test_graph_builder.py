"""Tests for ingestion/graph_builder.py — entity extraction and graph merging."""

import json
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from ingestion.graph_builder import _empty_extraction, _merge_into_graph, build_graph, graph_to_dict
from ingestion.loader import DocumentChunk


def _make_chunks(texts: list[str]) -> list[DocumentChunk]:
    return [DocumentChunk(text=t, source="test.pdf", page=i + 1, chunk_index=i) for i, t in enumerate(texts)]


def _mock_extraction(entities=None, relationships=None) -> dict:
    return {
        "entities": entities or {
            "systems": [], "teams": [], "data_sources": [],
            "compliance_requirements": [], "tech_stack": [],
        },
        "relationships": relationships or [],
    }


class TestMergeIntoGraph:
    def test_adds_system_nodes(self):
        G = nx.DiGraph()
        extracted = _mock_extraction(entities={
            "systems": [{"name": "PaymentService", "owner": "FinTeam", "documented": True, "legacy": False, "description": "Handles payments"}],
            "teams": [], "data_sources": [], "compliance_requirements": [], "tech_stack": [],
        })
        _merge_into_graph(G, extracted)
        assert "PaymentService" in G.nodes
        assert G.nodes["PaymentService"]["node_type"] == "System"

    def test_adds_compliance_nodes(self):
        G = nx.DiGraph()
        extracted = _mock_extraction(entities={
            "systems": [], "teams": [], "data_sources": [],
            "compliance_requirements": [{"name": "GDPR", "applies_to": ["PaymentService"]}],
            "tech_stack": [],
        })
        _merge_into_graph(G, extracted)
        assert "GDPR" in G.nodes
        assert G.nodes["GDPR"]["node_type"] == "ComplianceRequirement"

    def test_adds_tech_stack_nodes(self):
        G = nx.DiGraph()
        extracted = _mock_extraction(entities={
            "systems": [], "teams": [], "data_sources": [],
            "compliance_requirements": [],
            "tech_stack": [
                {"item": "AWS", "category": "cloud"},
                {"item": "Postgres", "category": "database"},
            ],
        })
        _merge_into_graph(G, extracted)
        assert "AWS" in G.nodes
        assert "Postgres" in G.nodes

    def test_adds_relationships(self):
        G = nx.DiGraph()
        extracted = _mock_extraction(
            entities={
                "systems": [
                    {"name": "SystemA", "owner": None, "documented": True, "legacy": False, "description": ""},
                    {"name": "SystemB", "owner": None, "documented": True, "legacy": False, "description": ""},
                ],
                "teams": [], "data_sources": [], "compliance_requirements": [], "tech_stack": [],
            },
            relationships=[{"type": "DEPENDS_ON", "from": "SystemA", "to": "SystemB"}],
        )
        _merge_into_graph(G, extracted)
        assert G.has_edge("SystemA", "SystemB")
        assert G.edges["SystemA", "SystemB"]["relationship"] == "DEPENDS_ON"

    def test_skips_relationship_if_nodes_missing(self):
        G = nx.DiGraph()
        extracted = _mock_extraction(
            relationships=[{"type": "DEPENDS_ON", "from": "Ghost", "to": "Also_Ghost"}],
        )
        _merge_into_graph(G, extracted)
        assert G.number_of_edges() == 0

    def test_merges_legacy_signals_across_batches(self):
        G = nx.DiGraph()
        first = _mock_extraction(entities={
            "systems": [{"name": "CoreSystem", "owner": None, "documented": True, "legacy": False, "description": ""}],
            "teams": [], "data_sources": [], "compliance_requirements": [], "tech_stack": [],
        })
        second = _mock_extraction(entities={
            "systems": [{"name": "CoreSystem", "owner": None, "documented": False, "legacy": True, "description": "old"}],
            "teams": [], "data_sources": [], "compliance_requirements": [], "tech_stack": [],
        })
        _merge_into_graph(G, first)
        _merge_into_graph(G, second)
        # legacy=True should win (OR), documented=False should win (AND)
        assert G.nodes["CoreSystem"]["legacy"] is True
        assert G.nodes["CoreSystem"]["documented"] is False

    def test_graph_from_two_docs_larger_than_either(self):
        G1 = nx.DiGraph()
        G2 = nx.DiGraph()
        doc1 = _mock_extraction(entities={
            "systems": [{"name": "AuthService", "owner": None, "documented": True, "legacy": False, "description": ""}],
            "teams": [], "data_sources": [], "compliance_requirements": [], "tech_stack": [{"item": "AWS", "category": "cloud"}],
        })
        doc2 = _mock_extraction(entities={
            "systems": [{"name": "BillingService", "owner": None, "documented": True, "legacy": False, "description": ""}],
            "teams": [], "data_sources": [], "compliance_requirements": [{"name": "GDPR", "applies_to": []}], "tech_stack": [],
        })
        _merge_into_graph(G1, doc1)
        _merge_into_graph(G2, doc2)

        G_combined = nx.DiGraph()
        _merge_into_graph(G_combined, doc1)
        _merge_into_graph(G_combined, doc2)

        assert G_combined.number_of_nodes() > G1.number_of_nodes()
        assert G_combined.number_of_nodes() > G2.number_of_nodes()


class TestBuildGraph:
    @patch("ingestion.graph_builder._get_client")
    def test_calls_llm_per_batch(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(_empty_extraction()))]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunks = _make_chunks(["text"] * 7)  # 7 chunks → 2 batches (BATCH_SIZE=5)
        G = build_graph(chunks, "claude-sonnet-4-6")
        assert mock_client.messages.create.call_count == 2

    @patch("ingestion.graph_builder._get_client")
    def test_handles_malformed_llm_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="this is not json")]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunks = _make_chunks(["text"])
        G = build_graph(chunks, "claude-sonnet-4-6")
        assert isinstance(G, nx.DiGraph)  # doesn't raise, returns empty graph


class TestGraphToDict:
    def test_serializes_nodes_and_edges(self):
        G = nx.DiGraph()
        G.add_node("A", node_type="System")
        G.add_node("B", node_type="Team")
        G.add_edge("A", "B", relationship="OWNED_BY")
        result = graph_to_dict(G)
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["edges"][0]["relationship"] == "OWNED_BY"
