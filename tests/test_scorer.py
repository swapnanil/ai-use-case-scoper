"""Tests for agent/scorer.py — pre-LLM feasibility pre-scoring."""

import pytest
from agent.models import CompanyProfile
from agent.scorer import compute_feasibility_context


def _make_profile(**overrides) -> CompanyProfile:
    defaults = {
        "industry": "E-commerce",
        "company_size": "smb",
        "data_maturity": "medium",
        "ai_experience": "none",
        "engineering_team_size": 10,
        "has_ml_engineers": False,
        "budget_tier": "growth",
        "ai_ambition": "We want AI to help our team work more efficiently",
        "pain_points": ["Manual reporting takes too long"],
        "compliance_requirements": [],
        "timeline_pressure": "experimental",
    }
    defaults.update(overrides)
    return CompanyProfile(**defaults)


class TestDataModifier:
    def test_low_data_maturity_gives_negative_modifier(self):
        profile = _make_profile(data_maturity="low")
        ctx = compute_feasibility_context(profile)
        assert ctx["data_modifier"] == -2

    def test_medium_data_maturity_gives_zero_modifier(self):
        profile = _make_profile(data_maturity="medium")
        ctx = compute_feasibility_context(profile)
        assert ctx["data_modifier"] == 0

    def test_high_data_maturity_gives_positive_modifier(self):
        profile = _make_profile(data_maturity="high")
        ctx = compute_feasibility_context(profile)
        assert ctx["data_modifier"] == 2

    def test_low_data_maturity_adds_cleanup_note(self):
        profile = _make_profile(data_maturity="low")
        ctx = compute_feasibility_context(profile)
        assert any("cleanup" in n.lower() or "data" in n.lower() for n in ctx["notes"])


class TestTeamModifier:
    def test_ml_engineers_gives_positive_modifier(self):
        profile = _make_profile(has_ml_engineers=True, ai_experience="basic")
        ctx = compute_feasibility_context(profile)
        assert ctx["team_modifier"] == 2

    def test_no_ai_experience_gives_negative_modifier(self):
        profile = _make_profile(has_ml_engineers=False, ai_experience="none")
        ctx = compute_feasibility_context(profile)
        assert ctx["team_modifier"] == -1

    def test_ml_engineers_plus_no_experience(self):
        profile = _make_profile(has_ml_engineers=True, ai_experience="none")
        ctx = compute_feasibility_context(profile)
        assert ctx["team_modifier"] == 1  # +2 - 1

    def test_no_ml_engineers_basic_experience(self):
        profile = _make_profile(has_ml_engineers=False, ai_experience="basic")
        ctx = compute_feasibility_context(profile)
        assert ctx["team_modifier"] == 0


class TestCompliancePenalty:
    def test_no_compliance_zero_penalty(self):
        profile = _make_profile(compliance_requirements=[])
        ctx = compute_feasibility_context(profile)
        assert ctx["compliance_penalty"] == 0.0

    def test_one_requirement_half_penalty(self):
        profile = _make_profile(compliance_requirements=["GDPR"])
        ctx = compute_feasibility_context(profile)
        assert ctx["compliance_penalty"] == -0.5

    def test_two_requirements_full_penalty(self):
        profile = _make_profile(compliance_requirements=["GDPR", "SOC2"])
        ctx = compute_feasibility_context(profile)
        assert ctx["compliance_penalty"] == -1.0

    def test_four_requirements(self):
        profile = _make_profile(compliance_requirements=["GDPR", "SOC2", "HIPAA", "data on-prem"])
        ctx = compute_feasibility_context(profile)
        assert ctx["compliance_penalty"] == -2.0


class TestTimelineFlags:
    def test_urgent_with_no_experience_raises_flag(self):
        profile = _make_profile(timeline_pressure="urgent", ai_experience="none")
        ctx = compute_feasibility_context(profile)
        assert any("TIMELINE_RISK" in f for f in ctx["flags"])

    def test_urgent_with_advanced_experience_no_flag(self):
        profile = _make_profile(timeline_pressure="urgent", ai_experience="advanced")
        ctx = compute_feasibility_context(profile)
        assert not any("TIMELINE_RISK" in f for f in ctx["flags"])

    def test_experimental_with_no_experience_no_flag(self):
        profile = _make_profile(timeline_pressure="experimental", ai_experience="none")
        ctx = compute_feasibility_context(profile)
        assert not any("TIMELINE_RISK" in f for f in ctx["flags"])


class TestBudgetNotes:
    def test_bootstrap_budget_adds_api_only_note(self):
        profile = _make_profile(budget_tier="bootstrap")
        ctx = compute_feasibility_context(profile)
        assert any("managed API" in n.lower() or "fine-tuning" in n.lower() for n in ctx["notes"])

    def test_growth_budget_no_budget_note(self):
        profile = _make_profile(budget_tier="growth")
        ctx = compute_feasibility_context(profile)
        assert not any("fine-tuning" in n.lower() for n in ctx["notes"])


class TestGDPROnPrem:
    def test_gdpr_adds_anthropic_note(self):
        profile = _make_profile(compliance_requirements=["GDPR"])
        ctx = compute_feasibility_context(profile)
        assert any("anthropic" in n.lower() or "openai" in n.lower() for n in ctx["notes"])

    def test_on_prem_detected(self):
        profile = _make_profile(compliance_requirements=["data on-prem"])
        ctx = compute_feasibility_context(profile)
        assert ctx["on_prem_detected"] is True

    def test_no_on_prem(self):
        profile = _make_profile(compliance_requirements=["SOC2"])
        ctx = compute_feasibility_context(profile)
        assert ctx["on_prem_detected"] is False

    def test_conflict_flag_urgent_no_experience_on_prem(self):
        profile = _make_profile(
            timeline_pressure="urgent",
            ai_experience="none",
            compliance_requirements=["data on-prem"],
        )
        ctx = compute_feasibility_context(profile)
        assert any("CONFLICT" in f for f in ctx["flags"])


class TestCompositeModifier:
    def test_best_case(self):
        profile = _make_profile(
            data_maturity="high",
            has_ml_engineers=True,
            ai_experience="advanced",
            compliance_requirements=[],
        )
        ctx = compute_feasibility_context(profile)
        assert ctx["composite_modifier"] == 4.0  # +2 + 2 + 0

    def test_worst_case(self):
        profile = _make_profile(
            data_maturity="low",
            has_ml_engineers=False,
            ai_experience="none",
            compliance_requirements=["GDPR", "SOC2", "HIPAA", "on-prem"],
        )
        ctx = compute_feasibility_context(profile)
        # -2 (data) + -1 (no exp) + -2 (4 compliance)
        assert ctx["composite_modifier"] == -5.0
