"""Tests for agent/memory.py — SQLAlchemy persistence layer."""

import os
import pytest

# Use a fresh in-memory SQLite DB for tests
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_scoper.db")

from agent import memory as mem
from agent.models import AIRoadmap, CompanyProfile, Milestone, PlanOutcome, UseCase


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # Re-create engine with test DB
    import sqlalchemy
    from agent.memory import Base
    test_engine = sqlalchemy.create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    monkeypatch.setattr(mem, "engine", test_engine)
    Base.metadata.create_all(test_engine)
    yield
    Base.metadata.drop_all(test_engine)


def _make_profile() -> CompanyProfile:
    return CompanyProfile(
        industry="Ad-Tech",
        company_size="smb",
        data_maturity="medium",
        ai_experience="none",
        engineering_team_size=25,
        has_ml_engineers=False,
        budget_tier="growth",
        ai_ambition="Use AI to make our account managers more productive",
        pain_points=["Manual reporting takes 4-6 hrs/week"],
        timeline_pressure="pilot_in_90_days",
    )


def _make_roadmap(readiness_score: int = 62) -> AIRoadmap:
    uc = UseCase(
        title="Automated Report Generator",
        description="Auto-generate weekly campaign reports",
        business_value="Save 4 hrs/AM/week",
        feasibility_score=9, roi_score=9, risk_score=2, priority_score=9,
        ai_approach=["LLM API"],
        estimated_complexity="low",
        estimated_timeline="2-3 weeks",
        estimated_cost_tier="$",
        data_requirements="Campaign performance data in Postgres",
        integration_requirements=["Google Ads API", "Postgres"],
        team_requirements="1 backend engineer",
        key_risks=["Data quality gaps"],
        mitigation=["Add data validation step"],
    )
    milestone = Milestone(
        week="Week 1-2",
        phase="Discovery",
        deliverables=["Audit campaign data", "Set up Claude API integration"],
        success_criteria="Data pipeline validated, first report draft generated",
    )
    return AIRoadmap(
        executive_summary="This company is a good fit for AI adoption.",
        strategic_assessment="Start small, prove value, then scale.",
        use_cases=[uc],
        recommended_first_project="Automated Report Generator — low risk, high visibility",
        roadmap_90_day=[milestone],
        quick_wins=["Automate one client report manually first"],
        things_to_avoid=["Don't build RAG before data is clean"],
        questions_to_answer_first=["Do you have API access to all ad platforms?"],
        readiness_score=readiness_score,
        readiness_blockers=["No ML engineers on team"],
        readiness_accelerators=["Clean Postgres data for top 10 clients"],
    )


class TestCompanyCRUD:
    def test_create_and_get_company(self):
        company = mem.create_company("SwapBank")
        assert company["id"]
        assert company["name"] == "SwapBank"

        fetched = mem.get_company(company["id"])
        assert fetched["name"] == "SwapBank"

    def test_get_missing_company_returns_none(self):
        assert mem.get_company("nonexistent-id") is None

    def test_company_id_is_unique(self):
        c1 = mem.create_company("Acme")
        c2 = mem.create_company("Acme")
        assert c1["id"] != c2["id"]


class TestSessionCRUD:
    def test_save_and_get_session(self):
        company = mem.create_company("TestCo")
        profile = _make_profile()
        roadmap = _make_roadmap()

        session_id = mem.save_session(
            company_id=company["id"],
            profile=profile,
            roadmap=roadmap,
            session_type="initial",
        )
        assert session_id

        session = mem.get_session(session_id)
        assert session["company_id"] == company["id"]
        assert session["session_type"] == "initial"
        assert session["roadmap"]["readiness_score"] == 62

    def test_get_missing_session_returns_none(self):
        assert mem.get_session("nonexistent-session") is None

    def test_sessions_ordered_by_date_descending(self):
        company = mem.create_company("OrderTest")
        profile = _make_profile()
        roadmap = _make_roadmap()

        id1 = mem.save_session(company["id"], profile, roadmap, "initial")
        id2 = mem.save_session(company["id"], profile, roadmap, "evolved", parent_session_id=id1)

        sessions = mem.get_company_sessions(company["id"])
        assert sessions[0]["id"] == id2  # most recent first

    def test_parent_session_linkage(self):
        company = mem.create_company("LinkTest")
        profile = _make_profile()
        roadmap = _make_roadmap()

        id1 = mem.save_session(company["id"], profile, roadmap, "initial")
        id2 = mem.save_session(company["id"], profile, roadmap, "evolved", parent_session_id=id1)

        child = mem.get_session(id2)
        assert child["parent_session_id"] == id1

    def test_session_summary_includes_readiness_score(self):
        company = mem.create_company("ScoreTest")
        profile = _make_profile()
        roadmap = _make_roadmap(readiness_score=75)

        mem.save_session(company["id"], profile, roadmap, "initial")
        sessions = mem.get_company_sessions(company["id"])
        assert sessions[0]["readiness_score"] == 75


class TestOutcomeRecording:
    def test_save_outcome_does_not_raise(self):
        company = mem.create_company("OutcomeTest")
        profile = _make_profile()
        roadmap = _make_roadmap()
        session_id = mem.save_session(company["id"], profile, roadmap, "initial")

        outcome = PlanOutcome(
            session_id=session_id,
            use_case_title="Automated Report Generator",
            status="shipped",
            actual_timeline="4 weeks",
            notes="Delivered on time",
        )
        mem.save_outcome(outcome)  # should not raise


class TestMemoryContext:
    def test_no_sessions_returns_empty_string(self):
        company = mem.create_company("NoSessions")
        ctx = mem.build_memory_context(company["id"])
        assert ctx == ""

    def test_with_sessions_returns_context_string(self):
        company = mem.create_company("WithSessions")
        profile = _make_profile()
        roadmap = _make_roadmap(readiness_score=55)
        mem.save_session(company["id"], profile, roadmap, "initial")

        ctx = mem.build_memory_context(company["id"])
        assert "COMPANY MEMORY" in ctx
        assert "55" in ctx
        assert "Automated Report Generator" in ctx
