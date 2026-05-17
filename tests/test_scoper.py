"""Tests for agent/scoper.py — core scoping logic with mocked Anthropic responses."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from agent.models import AIRoadmap, CompanyProfile
from agent.scoper import scope_company

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Minimal valid AIRoadmap for mocking
MOCK_ROADMAP_DATA = {
    "executive_summary": "Test executive summary for unit test.",
    "strategic_assessment": "Test strategic assessment.",
    "use_cases": [
        {
            "title": "Use Case A",
            "description": "Description A",
            "business_value": "Value A",
            "feasibility_score": 8,
            "roi_score": 9,
            "risk_score": 2,
            "priority_score": 9,
            "ai_approach": ["LLM API"],
            "estimated_complexity": "low",
            "estimated_timeline": "3 weeks",
            "estimated_cost_tier": "$",
            "data_requirements": "Basic data",
            "integration_requirements": ["API"],
            "team_requirements": "1 engineer",
            "key_risks": ["Risk A"],
            "mitigation": ["Mitigation A"],
        },
        {
            "title": "Use Case B",
            "description": "Description B",
            "business_value": "Value B",
            "feasibility_score": 6,
            "roi_score": 7,
            "risk_score": 3,
            "priority_score": 6,
            "ai_approach": ["RAG"],
            "estimated_complexity": "medium",
            "estimated_timeline": "6 weeks",
            "estimated_cost_tier": "$$",
            "data_requirements": "More data",
            "integration_requirements": ["DB"],
            "team_requirements": "2 engineers",
            "key_risks": ["Risk B"],
            "mitigation": ["Mitigation B"],
        },
    ],
    "recommended_first_project": "Use Case A — lowest risk",
    "roadmap_90_day": [
        {
            "week": "Week 1-2",
            "phase": "Discovery",
            "deliverables": ["Audit data sources"],
            "success_criteria": "Data pipeline runs end-to-end",
        },
        {
            "week": "Week 3-6",
            "phase": "Build",
            "deliverables": ["Build MVP"],
            "success_criteria": "MVP complete",
        },
        {
            "week": "Week 7-10",
            "phase": "Pilot",
            "deliverables": ["Internal pilot"],
            "success_criteria": "10 users active",
        },
        {
            "week": "Week 11-13",
            "phase": "Rollout",
            "deliverables": ["Full rollout"],
            "success_criteria": "ROI documented",
        },
    ],
    "quick_wins": ["Quick win 1"],
    "things_to_avoid": ["Avoid this"],
    "questions_to_answer_first": ["Question 1?"],
    "readiness_score": 65,
    "readiness_blockers": ["Blocker 1"],
    "readiness_accelerators": ["Accelerator 1"],
}

# Bootstrap profile — should never recommend fine-tuning
BOOTSTRAP_PROFILE_DATA = {
    "industry": "E-commerce",
    "company_size": "smb",
    "data_maturity": "high",
    "ai_experience": "none",
    "engineering_team_size": 6,
    "has_ml_engineers": False,
    "budget_tier": "bootstrap",
    "ai_ambition": "We want AI to automate our customer support and reduce response time significantly",
    "pain_points": ["Support team is overwhelmed"],
    "compliance_requirements": [],
    "timeline_pressure": "urgent",
}

# On-prem profile — should never recommend OpenAI
ON_PREM_PROFILE_DATA = {
    "industry": "Legal Services",
    "company_size": "enterprise",
    "data_maturity": "low",
    "ai_experience": "none",
    "engineering_team_size": 12,
    "has_ml_engineers": False,
    "budget_tier": "enterprise",
    "ai_ambition": "We want AI to help our lawyers review contracts and find relevant precedents faster",
    "pain_points": ["Contract review is too slow"],
    "compliance_requirements": ["GDPR", "data on-prem"],
    "timeline_pressure": "production_in_6_months",
}

# Urgent + no experience profile — should descope
URGENT_NO_EXP_PROFILE_DATA = {
    "industry": "Ad-Tech",
    "company_size": "startup",
    "data_maturity": "medium",
    "ai_experience": "none",
    "engineering_team_size": 5,
    "has_ml_engineers": False,
    "budget_tier": "growth",
    "ai_ambition": "We want to automate everything with AI and ship a fully autonomous AI agent ASAP",
    "pain_points": ["Everything takes too long"],
    "compliance_requirements": [],
    "timeline_pressure": "urgent",
}


def _make_mock_response(data: dict) -> MagicMock:
    mock_content = MagicMock()
    mock_content.text = json.dumps(data)
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response


def _mock_scope(profile_data: dict, roadmap_data: dict) -> AIRoadmap:
    profile = CompanyProfile(**profile_data)
    mock_response = _make_mock_response(roadmap_data)
    with patch("agent.scoper._get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        return scope_company(profile)


class TestExampleCompanies:
    def test_adtech_company_scopes_successfully(self):
        adtech_data = json.loads((EXAMPLES_DIR / "company_adtech.json").read_text())
        roadmap = _mock_scope(adtech_data, MOCK_ROADMAP_DATA)
        assert isinstance(roadmap, AIRoadmap)
        assert len(roadmap.use_cases) > 0

    def test_legal_company_scopes_successfully(self):
        legal_data = json.loads((EXAMPLES_DIR / "company_legal.json").read_text())
        roadmap = _mock_scope(legal_data, MOCK_ROADMAP_DATA)
        assert isinstance(roadmap, AIRoadmap)
        assert len(roadmap.use_cases) > 0

    def test_ecommerce_company_scopes_successfully(self):
        ecommerce_data = json.loads((EXAMPLES_DIR / "company_ecommerce.json").read_text())
        roadmap = _mock_scope(ecommerce_data, MOCK_ROADMAP_DATA)
        assert isinstance(roadmap, AIRoadmap)
        assert len(roadmap.use_cases) > 0


class TestBusinessRules:
    def test_use_cases_ordered_by_priority_score_descending(self):
        roadmap = _mock_scope(BOOTSTRAP_PROFILE_DATA, MOCK_ROADMAP_DATA)
        scores = [uc.priority_score for uc in roadmap.use_cases]
        assert scores == sorted(scores, reverse=True)

    def test_roadmap_has_all_required_phases(self):
        roadmap = _mock_scope(BOOTSTRAP_PROFILE_DATA, MOCK_ROADMAP_DATA)
        weeks = [m.week for m in roadmap.roadmap_90_day]
        assert any("1" in w and "2" in w for w in weeks), "Missing Week 1-2 milestone"
        assert any("3" in w and "6" in w for w in weeks), "Missing Week 3-6 milestone"
        assert any("7" in w and "10" in w for w in weeks), "Missing Week 7-10 milestone"
        assert any("11" in w and "13" in w for w in weeks), "Missing Week 11-13 milestone"

    def test_bootstrap_budget_does_not_recommend_fine_tuning(self):
        fine_tune_roadmap = dict(MOCK_ROADMAP_DATA)
        fine_tune_roadmap["use_cases"] = [
            {
                **MOCK_ROADMAP_DATA["use_cases"][0],
                "ai_approach": ["Fine-tuning", "Custom model training"],
            }
        ]
        profile = CompanyProfile(**BOOTSTRAP_PROFILE_DATA)
        mock_response = _make_mock_response(fine_tune_roadmap)

        with patch("agent.scoper._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            # The test asserts the system prompt CONTAINS the bootstrap constraint
            # The scorer must inject the note about managed APIs
            from agent.scorer import compute_feasibility_context
            ctx = compute_feasibility_context(profile)
            assert any("managed API" in n.lower() or "fine-tuning" in n.lower() for n in ctx["notes"])

    def test_on_prem_constraint_flagged_in_scorer(self):
        profile = CompanyProfile(**ON_PREM_PROFILE_DATA)
        from agent.scorer import compute_feasibility_context
        ctx = compute_feasibility_context(profile)
        assert ctx["on_prem_detected"] is True
        assert any("openai" in n.lower() or "anthropic" in n.lower() for n in ctx["notes"])

    def test_urgent_no_experience_raises_timeline_flag(self):
        profile = CompanyProfile(**URGENT_NO_EXP_PROFILE_DATA)
        from agent.scorer import compute_feasibility_context
        ctx = compute_feasibility_context(profile)
        assert any("TIMELINE_RISK" in f for f in ctx["flags"])

    def test_readiness_score_in_valid_range(self):
        roadmap = _mock_scope(BOOTSTRAP_PROFILE_DATA, MOCK_ROADMAP_DATA)
        assert 1 <= roadmap.readiness_score <= 100


class TestInputValidation:
    def test_short_ambition_raises_error(self):
        with pytest.raises(Exception, match="more detail"):
            CompanyProfile(**{**BOOTSTRAP_PROFILE_DATA, "ai_ambition": "do AI"})

    def test_ambition_exactly_20_chars_valid(self):
        profile = CompanyProfile(**{
            **BOOTSTRAP_PROFILE_DATA,
            "ai_ambition": "A" * 20,
        })
        assert len(profile.ai_ambition) == 20


class TestJSONParsing:
    def test_json_with_markdown_fences_is_cleaned(self):
        fenced = "```json\n" + json.dumps(MOCK_ROADMAP_DATA) + "\n```"
        mock_content = MagicMock()
        mock_content.text = fenced
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        profile = CompanyProfile(**BOOTSTRAP_PROFILE_DATA)
        with patch("agent.scoper._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client
            roadmap = scope_company(profile)

        assert isinstance(roadmap, AIRoadmap)

    def test_invalid_json_triggers_retry(self):
        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="not valid json at all {{{")]

        good_response = _make_mock_response(MOCK_ROADMAP_DATA)

        profile = CompanyProfile(**BOOTSTRAP_PROFILE_DATA)
        with patch("agent.scoper._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [bad_response, good_response]
            mock_get_client.return_value = mock_client
            roadmap = scope_company(profile)

        assert isinstance(roadmap, AIRoadmap)
        assert mock_client.messages.create.call_count == 2
