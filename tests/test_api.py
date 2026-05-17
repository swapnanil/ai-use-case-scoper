"""Tests for api.py — FastAPI endpoints."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import app
from agent.models import AIRoadmap

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

MOCK_ROADMAP_DATA = {
    "executive_summary": "API test executive summary.",
    "strategic_assessment": "API test strategic assessment.",
    "use_cases": [
        {
            "title": "Test Use Case",
            "description": "Test description",
            "business_value": "Test value",
            "feasibility_score": 8,
            "roi_score": 8,
            "risk_score": 2,
            "priority_score": 8,
            "ai_approach": ["LLM API"],
            "estimated_complexity": "low",
            "estimated_timeline": "3 weeks",
            "estimated_cost_tier": "$",
            "data_requirements": "Basic data",
            "integration_requirements": ["API"],
            "team_requirements": "1 engineer",
            "key_risks": ["Risk"],
            "mitigation": ["Mitigation"],
        }
    ],
    "recommended_first_project": "Test Use Case — low risk",
    "roadmap_90_day": [
        {
            "week": "Week 1-2",
            "phase": "Discovery",
            "deliverables": ["Audit"],
            "success_criteria": "Pipeline works",
        },
        {
            "week": "Week 3-6",
            "phase": "Build",
            "deliverables": ["MVP"],
            "success_criteria": "MVP done",
        },
        {
            "week": "Week 7-10",
            "phase": "Pilot",
            "deliverables": ["Pilot"],
            "success_criteria": "Users active",
        },
        {
            "week": "Week 11-13",
            "phase": "Rollout",
            "deliverables": ["Rollout"],
            "success_criteria": "ROI documented",
        },
    ],
    "quick_wins": ["Quick win"],
    "things_to_avoid": ["Avoid this"],
    "questions_to_answer_first": ["Question?"],
    "readiness_score": 70,
    "readiness_blockers": ["Blocker"],
    "readiness_accelerators": ["Accelerator"],
}

VALID_PROFILE = {
    "industry": "E-commerce",
    "company_size": "smb",
    "data_maturity": "high",
    "ai_experience": "none",
    "engineering_team_size": 6,
    "has_ml_engineers": False,
    "budget_tier": "bootstrap",
    "ai_ambition": "We want AI to automate our customer support and improve response times significantly",
    "pain_points": ["Support is overwhelmed"],
    "compliance_requirements": [],
    "timeline_pressure": "urgent",
}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_scope():
    mock_roadmap = AIRoadmap(**MOCK_ROADMAP_DATA)
    with patch("api.scope_company", return_value=mock_roadmap) as mock:
        yield mock


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_model_name(self, client):
        response = client.get("/health")
        data = response.json()
        assert "model" in data
        assert "claude" in data["model"].lower()


class TestHeuristicsEndpoint:
    def test_heuristics_returns_200(self, client):
        response = client.get("/heuristics")
        assert response.status_code == 200

    def test_heuristics_contains_universal_heuristics(self, client):
        response = client.get("/heuristics")
        data = response.json()
        assert "universal_heuristics" in data
        assert len(data["universal_heuristics"]) > 0

    def test_heuristics_contains_industries(self, client):
        response = client.get("/heuristics")
        data = response.json()
        assert "industries" in data


class TestScopeEndpoint:
    def test_scope_valid_profile_returns_roadmap(self, client, mock_scope):
        response = client.post("/scope", json=VALID_PROFILE)
        assert response.status_code == 200
        data = response.json()
        assert "executive_summary" in data
        assert "use_cases" in data
        assert "roadmap_90_day" in data

    def test_scope_calls_scope_company(self, client, mock_scope):
        client.post("/scope", json=VALID_PROFILE)
        assert mock_scope.called

    def test_scope_invalid_profile_returns_422(self, client):
        bad_profile = {"industry": "test"}  # missing required fields
        response = client.post("/scope", json=bad_profile)
        assert response.status_code == 422

    def test_scope_short_ambition_returns_422(self, client):
        bad_profile = {**VALID_PROFILE, "ai_ambition": "do AI"}
        response = client.post("/scope", json=bad_profile)
        assert response.status_code == 422

    def test_scope_use_cases_ordered_by_priority(self, client, mock_scope):
        response = client.post("/scope", json=VALID_PROFILE)
        data = response.json()
        scores = [uc["priority_score"] for uc in data["use_cases"]]
        assert scores == sorted(scores, reverse=True)

    def test_scope_adtech_example(self, client, mock_scope):
        adtech = json.loads((EXAMPLES_DIR / "company_adtech.json").read_text())
        response = client.post("/scope", json=adtech)
        assert response.status_code == 200

    def test_scope_legal_example(self, client, mock_scope):
        legal = json.loads((EXAMPLES_DIR / "company_legal.json").read_text())
        response = client.post("/scope", json=legal)
        assert response.status_code == 200

    def test_scope_ecommerce_example(self, client, mock_scope):
        ecommerce = json.loads((EXAMPLES_DIR / "company_ecommerce.json").read_text())
        response = client.post("/scope", json=ecommerce)
        assert response.status_code == 200


class TestQuickScopeEndpoint:
    def test_quick_scope_minimal_fields(self, client, mock_scope):
        payload = {
            "industry": "E-commerce",
            "company_size": "smb",
            "ai_ambition": "We want AI to automate customer support and product recommendations for our store",
            "data_maturity": "high",
        }
        response = client.post("/scope/quick", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "executive_summary" in data

    def test_quick_scope_applies_defaults(self, client, mock_scope):
        payload = {
            "industry": "Legal Services",
            "company_size": "enterprise",
            "ai_ambition": "Help lawyers review contracts faster and find relevant case precedents",
            "data_maturity": "medium",
        }
        client.post("/scope/quick", json=payload)
        assert mock_scope.called
        profile_arg = mock_scope.call_args[0][0]
        assert profile_arg.timeline_pressure == "experimental"
        assert profile_arg.has_ml_engineers is False
        assert profile_arg.budget_tier == "growth"

    def test_quick_scope_missing_field_returns_422(self, client):
        payload = {"industry": "E-commerce", "company_size": "smb"}
        response = client.post("/scope/quick", json=payload)
        assert response.status_code == 422

    def test_quick_scope_short_ambition_returns_422(self, client):
        payload = {
            "industry": "E-commerce",
            "company_size": "smb",
            "ai_ambition": "do AI",
            "data_maturity": "high",
        }
        response = client.post("/scope/quick", json=payload)
        assert response.status_code == 422
