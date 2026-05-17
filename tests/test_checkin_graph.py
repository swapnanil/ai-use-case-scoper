"""Tests for pipeline/checkin_graph.py — LangGraph pipeline logic."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.models import (
    AIRoadmap,
    CheckinInput,
    Milestone,
    PlanOutcome,
    UseCase,
)
from pipeline.nodes import (
    classify_changes,
    reassess_use_cases,
    update_milestones,
)


def _make_use_case(title: str, priority: int = 7, feasibility: int = 7, roi: int = 7, risk: int = 3) -> UseCase:
    return UseCase(
        title=title,
        description=f"Description of {title}",
        business_value="High business value",
        feasibility_score=feasibility,
        roi_score=roi,
        risk_score=risk,
        priority_score=priority,
        ai_approach=["LLM API"],
        estimated_complexity="medium",
        estimated_timeline="4-6 weeks",
        estimated_cost_tier="$$",
        data_requirements="Structured data in DB",
        integration_requirements=["Internal API"],
        team_requirements="1 engineer",
        key_risks=["Data quality"],
        mitigation=["Data validation sprint"],
    )


def _make_milestone(week: str, phase: str, deliverables: list[str] | None = None) -> Milestone:
    return Milestone(
        week=week,
        phase=phase,
        deliverables=deliverables or [f"Deliverable for {phase}"],
        success_criteria=f"{phase} complete",
    )


def _make_roadmap(use_cases: list[UseCase], milestones: list[Milestone], score: int = 62) -> AIRoadmap:
    return AIRoadmap(
        executive_summary="Company is on track.",
        strategic_assessment="Good readiness.",
        use_cases=use_cases,
        recommended_first_project=use_cases[0].title if use_cases else "TBD",
        roadmap_90_day=milestones,
        quick_wins=["Quick win"],
        things_to_avoid=["Avoid X"],
        questions_to_answer_first=["Question?"],
        readiness_score=score,
        readiness_blockers=["No ML engineers"],
        readiness_accelerators=["Clean data"],
    )


def _make_checkin(notes: str = "Good progress.", shipped: list[str] | None = None) -> CheckinInput:
    return CheckinInput(
        company_id="test-company",
        parent_session_id="test-session",
        notes=notes,
        outcome_updates=[
            PlanOutcome(
                session_id="test-session",
                use_case_title=title,
                status="shipped",
            )
            for title in (shipped or [])
        ],
        new_constraints=[],
        profile_changes={},
    )


class TestClassifyChanges:
    def test_shipped_work_detected(self):
        state = {
            "deltas": {"shipped_use_cases": ["Report Generator"], "abandoned_use_cases": [],
                       "resolved_blockers": [], "new_blockers": [], "urgency_change": "unchanged",
                       "overall_progress": "on_track"},
            "checkin_input": _make_checkin(),
        }
        result = classify_changes(state)
        assert result["classified_changes"]["has_shipped_work"] is True

    def test_new_blockers_detected(self):
        state = {
            "deltas": {"new_blockers": ["Data schema inconsistency"], "shipped_use_cases": [],
                       "abandoned_use_cases": [], "resolved_blockers": [],
                       "urgency_change": "unchanged", "overall_progress": "behind"},
            "checkin_input": _make_checkin(),
        }
        result = classify_changes(state)
        assert result["classified_changes"]["has_new_blockers"] is True
        assert result["classified_changes"]["progress"] == "behind"

    def test_urgency_increased_detected(self):
        state = {
            "deltas": {"urgency_change": "increased", "shipped_use_cases": [],
                       "abandoned_use_cases": [], "resolved_blockers": [],
                       "new_blockers": [], "overall_progress": "on_track"},
            "checkin_input": _make_checkin(),
        }
        result = classify_changes(state)
        assert result["classified_changes"]["urgency_increased"] is True


class TestUpdateMilestones:
    def test_shipped_use_case_removes_related_milestone(self):
        uc = _make_use_case("Report Generator")
        milestones = [
            _make_milestone("Week 1-2", "Discovery"),
            _make_milestone("Week 3-6", "Report Generator", ["Build report generator"]),
            _make_milestone("Week 7-10", "Pilot"),
        ]
        roadmap = _make_roadmap([uc], milestones)

        state = {
            "previous_roadmap": roadmap,
            "classified_changes": {"progress": "on_track", "has_new_blockers": False},
            "shipped_use_cases": ["Report Generator"],
        }
        result = update_milestones(state)
        remaining_phases = [m.phase for m in result["updated_milestones"]]
        assert "Report Generator" not in remaining_phases

    def test_behind_schedule_adds_shift_record(self):
        roadmap = _make_roadmap([_make_use_case("X")], [_make_milestone("Week 1-2", "Discovery")])
        state = {
            "previous_roadmap": roadmap,
            "classified_changes": {"progress": "behind", "has_new_blockers": False},
            "shipped_use_cases": [],
        }
        result = update_milestones(state)
        assert any(s["shift"] == "+2 weeks" for s in result["milestone_shifts"])

    def test_stalled_adds_reset_record(self):
        roadmap = _make_roadmap([_make_use_case("X")], [_make_milestone("Week 1-2", "Discovery")])
        state = {
            "previous_roadmap": roadmap,
            "classified_changes": {"progress": "stalled", "has_new_blockers": False},
            "shipped_use_cases": [],
        }
        result = update_milestones(state)
        assert any(s["shift"] == "reset" for s in result["milestone_shifts"])


class TestReassessUseCases:
    @patch("pipeline.nodes._get_client")
    def test_shipped_use_cases_removed_from_output(self, mock_get_client):
        shipped_uc = _make_use_case("Shipped Feature", priority=9)
        remaining_uc = _make_use_case("Remaining Feature", priority=7)
        roadmap = _make_roadmap([shipped_uc, remaining_uc], [])

        score_update = [{"title": "Remaining Feature", "feasibility_score": 8, "roi_score": 8, "risk_score": 2, "priority_score": 8, "score_change_reason": "Blocker resolved"}]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(score_update))]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        state = {
            "previous_roadmap": roadmap,
            "deltas": {"shipped_use_cases": ["Shipped Feature"], "abandoned_use_cases": [],
                       "resolved_blockers": [], "new_blockers": []},
            "classified_changes": {"progress": "on_track", "urgency_increased": False},
            "checkin_input": _make_checkin(shipped=["Shipped Feature"]),
        }
        result = reassess_use_cases(state)
        titles = [uc.title for uc in result["reassessed_use_cases"]]
        assert "Shipped Feature" not in titles
        assert "Remaining Feature" in titles

    @patch("pipeline.nodes._get_client")
    def test_use_cases_ordered_by_priority_descending(self, mock_get_client):
        uc_a = _make_use_case("Feature A", priority=5)
        uc_b = _make_use_case("Feature B", priority=8)
        roadmap = _make_roadmap([uc_a, uc_b], [])

        score_update = [
            {"title": "Feature A", "feasibility_score": 5, "roi_score": 5, "risk_score": 5, "priority_score": 5, "score_change_reason": "No change"},
            {"title": "Feature B", "feasibility_score": 8, "roi_score": 9, "risk_score": 2, "priority_score": 9, "score_change_reason": "Blocker resolved"},
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(score_update))]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        state = {
            "previous_roadmap": roadmap,
            "deltas": {"shipped_use_cases": [], "abandoned_use_cases": [],
                       "resolved_blockers": ["ML engineer hired"], "new_blockers": []},
            "classified_changes": {"progress": "on_track", "urgency_increased": False},
            "checkin_input": _make_checkin(),
        }
        result = reassess_use_cases(state)
        priorities = [uc.priority_score for uc in result["reassessed_use_cases"]]
        assert priorities == sorted(priorities, reverse=True)

    @patch("pipeline.nodes._get_client")
    def test_resolved_blocker_can_increase_readiness_reflected_in_scores(self, mock_get_client):
        uc = _make_use_case("Campaign QA", priority=6, feasibility=5)
        roadmap = _make_roadmap([uc], [])

        # LLM returns higher scores after blocker resolved
        score_update = [{"title": "Campaign QA", "feasibility_score": 8, "roi_score": 8, "risk_score": 3, "priority_score": 8, "score_change_reason": "Data schema fixed"}]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(score_update))]
        mock_client.messages.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        state = {
            "previous_roadmap": roadmap,
            "deltas": {"shipped_use_cases": [], "abandoned_use_cases": [],
                       "resolved_blockers": ["Data schema inconsistency fixed"], "new_blockers": []},
            "classified_changes": {"progress": "ahead", "urgency_increased": False},
            "checkin_input": _make_checkin(),
        }
        result = reassess_use_cases(state)
        updated = result["reassessed_use_cases"][0]
        assert updated.feasibility_score > uc.feasibility_score
