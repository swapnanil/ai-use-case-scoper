"""Tests for ingestion/extractor.py — graph → CompanyProfile mapping."""

import networkx as nx
import pytest

from ingestion.extractor import CONFIDENCE_THRESHOLD, _infer_ai_experience, _infer_data_maturity, extract_profile


def _make_graph(**nodes) -> nx.DiGraph:
    G = nx.DiGraph()
    for name, attrs in nodes.items():
        G.add_node(name, **attrs)
    return G


class TestInferDataMaturity:
    def test_clean_majority_returns_high(self):
        ds_nodes = [
            {"governance_status": "clean"},
            {"governance_status": "clean"},
            {"governance_status": "clean"},
            {"governance_status": "messy"},
        ]
        result, conf = _infer_data_maturity(ds_nodes)
        assert result == "high"
        assert conf >= 0.7

    def test_messy_majority_returns_low(self):
        ds_nodes = [
            {"governance_status": "messy"},
            {"governance_status": "messy"},
            {"governance_status": "messy"},
            {"governance_status": "clean"},
        ]
        result, conf = _infer_data_maturity(ds_nodes)
        assert result == "low"

    def test_unknown_majority_returns_low(self):
        ds_nodes = [{"governance_status": "unknown"}] * 5
        result, conf = _infer_data_maturity(ds_nodes)
        assert result == "low"

    def test_no_data_sources_returns_low_with_low_confidence(self):
        result, conf = _infer_data_maturity([])
        assert result == "low"
        assert conf < CONFIDENCE_THRESHOLD

    def test_mixed_returns_medium(self):
        ds_nodes = [
            {"governance_status": "clean"},
            {"governance_status": "messy"},
            {"governance_status": "unknown"},
        ]
        result, conf = _infer_data_maturity(ds_nodes)
        assert result == "medium"


class TestInferAiExperience:
    def test_has_ml_engineers_returns_basic(self):
        result, conf = _infer_ai_experience([], True, 0, 0)
        assert result == "basic"
        assert conf > 0.6

    def test_many_legacy_systems_returns_none(self):
        systems = [{"legacy": True}] * 6
        result, conf = _infer_ai_experience(systems, False, 6, 0)
        assert result == "none"

    def test_many_undocumented_returns_none(self):
        systems = [{"documented": False}] * 4
        result, conf = _infer_ai_experience(systems, False, 0, 4)
        assert result == "none"

    def test_no_signals_returns_none_low_confidence(self):
        result, conf = _infer_ai_experience([], False, 0, 0)
        assert result == "none"
        assert conf < CONFIDENCE_THRESHOLD


class TestExtractProfile:
    def test_tech_stack_extracted_from_graph(self):
        G = _make_graph(
            AWS={"node_type": "TechStack", "item": "AWS", "category": "cloud"},
            Postgres={"node_type": "TechStack", "item": "Postgres", "category": "database"},
        )
        result = extract_profile(G, overrides={
            "industry": "FinTech", "company_size": "enterprise",
            "budget_tier": "enterprise", "timeline_pressure": "production_in_6_months",
            "ai_ambition": "Automate customer support and compliance monitoring",
            "pain_points": ["Manual compliance checks", "Slow customer support"],
        })
        assert "AWS" in result.extracted_profile.tech_stack
        assert "Postgres" in result.extracted_profile.tech_stack
        assert result.confidence_scores["tech_stack"] >= 0.9

    def test_compliance_extracted_from_graph(self):
        G = _make_graph(
            GDPR={"node_type": "ComplianceRequirement", "name": "GDPR", "applies_to": []},
        )
        result = extract_profile(G, overrides={
            "industry": "FinTech", "company_size": "enterprise",
            "budget_tier": "enterprise", "timeline_pressure": "experimental",
            "ai_ambition": "Improve risk management with AI tools and monitoring",
            "pain_points": ["Manual compliance"],
        })
        assert "GDPR" in result.extracted_profile.compliance_requirements
        assert result.confidence_scores["compliance_requirements"] >= 0.9

    def test_low_confidence_compliance_when_none_found(self):
        G = nx.DiGraph()  # empty graph
        result = extract_profile(G, overrides={
            "industry": "Retail", "company_size": "smb",
            "budget_tier": "growth", "timeline_pressure": "experimental",
            "ai_ambition": "Automate product recommendations for our e-commerce platform",
            "pain_points": ["Low conversion rate"],
        })
        assert result.confidence_scores["compliance_requirements"] < CONFIDENCE_THRESHOLD
        assert "compliance_requirements" in result.low_confidence_fields
        assert any("compliance" in w.lower() for w in result.extraction_warnings)

    def test_low_confidence_fields_identified(self):
        G = nx.DiGraph()
        result = extract_profile(G, overrides={
            "industry": "FinTech", "company_size": "enterprise",
            "budget_tier": "enterprise", "timeline_pressure": "experimental",
            "ai_ambition": "Automate our underwriting and risk scoring pipeline for SME loans",
            "pain_points": ["Manual underwriting"],
        })
        # Without doc context, data_maturity should be low confidence
        assert "data_maturity" in result.low_confidence_fields

    def test_overrides_applied_with_full_confidence(self):
        G = nx.DiGraph()
        result = extract_profile(G, overrides={
            "industry": "Banking", "company_size": "enterprise",
            "budget_tier": "enterprise", "timeline_pressure": "production_in_6_months",
            "ai_ambition": "Automate KYC document verification and customer onboarding workflows",
            "pain_points": ["Slow KYC process", "Manual document review"],
            "data_maturity": "high",
        })
        assert result.extracted_profile.data_maturity == "high"
        assert result.confidence_scores["data_maturity"] == 1.0

    def test_extraction_warnings_present_for_missing_signals(self):
        G = nx.DiGraph()
        result = extract_profile(G, overrides={
            "industry": "Healthcare", "company_size": "mid_market",
            "budget_tier": "growth", "timeline_pressure": "experimental",
            "ai_ambition": "Build patient communication AI for appointment reminders and follow-ups",
            "pain_points": ["High no-show rate", "Staff time on phone calls"],
        })
        assert len(result.extraction_warnings) > 0

    def test_result_is_valid_company_profile(self):
        G = _make_graph(
            AWS={"node_type": "TechStack", "item": "AWS", "category": "cloud"},
            HIPAA={"node_type": "ComplianceRequirement", "name": "HIPAA", "applies_to": []},
        )
        result = extract_profile(G, overrides={
            "industry": "Healthcare", "company_size": "enterprise",
            "budget_tier": "enterprise", "timeline_pressure": "production_in_6_months",
            "ai_ambition": "Automate clinical documentation and reduce physician admin burden",
            "pain_points": ["Physician burnout", "Documentation overhead"],
        })
        # Should construct without raising
        from agent.models import CompanyProfile
        assert isinstance(result.extracted_profile, CompanyProfile)
