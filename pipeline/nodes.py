import json
import logging
import os
from typing import Any

from anthropic import Anthropic

from agent import memory as mem
from agent.models import AIRoadmap, CheckinInput, EvolvedAIRoadmap, UseCase
from agent.prompts import CHECKIN_DELTA_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

_client: Anthropic | None = None
MODEL = os.getenv("MODEL", "claude-sonnet-4-6")

REASSESS_USE_CASES_SCHEMA = """[
  {
    "title": "string (must match an existing use case title exactly)",
    "feasibility_score": "integer 1-10",
    "roi_score": "integer 1-10",
    "risk_score": "integer 1-10",
    "priority_score": "integer 1-10 (composite)",
    "score_change_reason": "string — one sentence explaining what changed"
  }
]"""

EVOLVED_PLAN_SCHEMA = """{
  "executive_summary": "string (3-4 sentences for CTO/CEO — where they are now, what's next)",
  "strategic_assessment": "string (honest assessment given progress so far)",
  "use_cases": [/* full UseCase objects — include all reassessed use cases */],
  "recommended_first_project": "string (updated recommendation + one-line reason)",
  "roadmap_90_day": [/* updated milestones — full Milestone objects */],
  "quick_wins": ["list of updated quick wins"],
  "things_to_avoid": ["updated list"],
  "questions_to_answer_first": ["updated questions"],
  "readiness_score": "integer 1-100",
  "readiness_blockers": ["updated blockers"],
  "readiness_accelerators": ["updated accelerators"],
  "evolution_summary": "string (3-4 sentences: what specifically changed and why the plan evolved)"
}"""


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def _call_llm(prompt: str, system: str | None = None, max_tokens: int = 2000) -> str:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = _get_client().messages.create(**kwargs)
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return raw


# ---------------------------------------------------------------------------
# Node functions — each takes state dict, returns updated state dict
# ---------------------------------------------------------------------------

def load_previous_plan(state: dict) -> dict:
    checkin: CheckinInput = state["checkin_input"]
    session_data = mem.get_session(checkin.parent_session_id)
    if not session_data:
        raise ValueError(f"Session '{checkin.parent_session_id}' not found in database")
    state["previous_roadmap"] = AIRoadmap(**session_data["roadmap"])
    state["previous_profile"] = session_data["input_profile"]
    logger.info(f"Loaded previous plan — readiness score: {state['previous_roadmap'].readiness_score}")
    return state


def extract_checkin_deltas(state: dict) -> dict:
    checkin: CheckinInput = state["checkin_input"]
    previous: AIRoadmap = state["previous_roadmap"]

    prompt = f"""Analyze this enterprise AI adoption check-in and extract what has changed.

PREVIOUS ROADMAP SUMMARY:
- Recommended first project: {previous.recommended_first_project}
- Readiness score: {previous.readiness_score}/100
- Top use cases: {", ".join(uc.title for uc in previous.use_cases[:5])}
- Key blockers: {", ".join(previous.readiness_blockers[:3])}

CHECK-IN NOTES:
{checkin.notes}

OUTCOME UPDATES:
{json.dumps([o.model_dump(mode="json") for o in checkin.outcome_updates], indent=2)}

NEW CONSTRAINTS: {checkin.new_constraints}
PROFILE CHANGES REPORTED: {checkin.profile_changes}

Return ONLY valid JSON matching this schema exactly:
{CHECKIN_DELTA_EXTRACTION_PROMPT.split("Return ONLY valid JSON:")[1]}"""

    raw = _call_llm(prompt, system=CHECKIN_DELTA_EXTRACTION_PROMPT.split("Return ONLY valid JSON:")[0])
    state["deltas"] = json.loads(raw)
    logger.info(f"Deltas extracted — progress: {state['deltas'].get('overall_progress')}")
    return state


def classify_changes(state: dict) -> dict:
    deltas = state["deltas"]
    checkin: CheckinInput = state["checkin_input"]

    state["classified_changes"] = {
        "has_resolved_blockers": len(deltas.get("resolved_blockers", [])) > 0,
        "has_new_blockers": len(deltas.get("new_blockers", [])) > 0,
        "has_shipped_work": len(deltas.get("shipped_use_cases", [])) > 0,
        "has_profile_changes": bool(checkin.profile_changes),
        "has_new_constraints": len(checkin.new_constraints) > 0,
        "urgency_increased": deltas.get("urgency_change") == "increased",
        "progress": deltas.get("overall_progress", "on_track"),
    }
    return state


def reassess_use_cases(state: dict) -> dict:
    previous: AIRoadmap = state["previous_roadmap"]
    deltas = state["deltas"]
    classified = state["classified_changes"]
    checkin: CheckinInput = state["checkin_input"]

    shipped = set(deltas.get("shipped_use_cases", []))
    abandoned = set(uc.split(" — ")[0] for uc in deltas.get("abandoned_use_cases", []))

    remaining = [uc for uc in previous.use_cases if uc.title not in shipped and uc.title not in abandoned]

    prompt = f"""Re-score these AI use cases based on what changed since the last plan.

WHAT CHANGED:
- Resolved blockers: {deltas.get("resolved_blockers", [])}
- New blockers: {deltas.get("new_blockers", [])}
- New constraints: {checkin.new_constraints}
- Profile changes: {checkin.profile_changes}
- Timeline pressure change: {checkin.timeline_pressure or "unchanged"}
- Overall progress: {classified["progress"]}

USE CASES TO RE-SCORE:
{json.dumps([uc.model_dump() for uc in remaining], indent=2)}

Return ONLY valid JSON — the same list with updated scores:
{REASSESS_USE_CASES_SCHEMA}"""

    raw = _call_llm(prompt, max_tokens=2000)
    score_updates = {item["title"]: item for item in json.loads(raw)}

    reassessed: list[UseCase] = []
    for uc in remaining:
        update = score_updates.get(uc.title, {})
        reassessed.append(uc.model_copy(update={
            "feasibility_score": update.get("feasibility_score", uc.feasibility_score),
            "roi_score": update.get("roi_score", uc.roi_score),
            "risk_score": update.get("risk_score", uc.risk_score),
            "priority_score": update.get("priority_score", uc.priority_score),
        }))

    reassessed.sort(key=lambda uc: uc.priority_score, reverse=True)
    state["reassessed_use_cases"] = reassessed
    state["dropped_use_cases"] = list(abandoned)
    state["shipped_use_cases"] = list(shipped)
    return state


def update_milestones(state: dict) -> dict:
    previous: AIRoadmap = state["previous_roadmap"]
    classified = state["classified_changes"]
    shipped: set[str] = set(state.get("shipped_use_cases", []))

    milestones = list(previous.roadmap_90_day)
    shifts: list[dict] = []

    if classified["progress"] == "behind":
        shifts.append({"milestone": "all", "shift": "+2 weeks", "reason": "overall progress behind schedule"})
    elif classified["progress"] == "stalled":
        shifts.append({"milestone": "all", "shift": "reset", "reason": "progress stalled — milestones reset"})

    if shipped:
        before_count = len(milestones)
        milestones = [m for m in milestones if not any(s.lower() in m.phase.lower() for s in shipped)]
        removed = before_count - len(milestones)
        if removed:
            shifts.append({"milestone": f"{removed} milestone(s)", "shift": "removed", "reason": "use case shipped"})

    state["updated_milestones"] = milestones
    state["milestone_shifts"] = shifts
    return state


def generate_evolved_plan(state: dict) -> dict:
    previous: AIRoadmap = state["previous_roadmap"]
    checkin: CheckinInput = state["checkin_input"]
    deltas = state["deltas"]
    reassessed: list[UseCase] = state["reassessed_use_cases"]
    updated_milestones = state["updated_milestones"]

    prompt = f"""You are a senior AI deployment consultant generating an evolved AI adoption roadmap.

WHAT CHANGED SINCE LAST PLAN:
{json.dumps(deltas, indent=2)}

PREVIOUS READINESS SCORE: {previous.readiness_score}/100
PREVIOUS BLOCKERS: {previous.readiness_blockers}
NEW CONSTRAINTS: {checkin.new_constraints}
PROFILE CHANGES: {checkin.profile_changes}

REASSESSED USE CASES (ordered by priority):
{json.dumps([uc.model_dump() for uc in reassessed[:6]], indent=2)}

UPDATED MILESTONES:
{json.dumps([m.model_dump() for m in updated_milestones], indent=2)}

Generate an evolved roadmap. Return ONLY valid JSON:
{EVOLVED_PLAN_SCHEMA}

IMPORTANT:
- use_cases must be ordered by priority_score descending
- roadmap_90_day must cover weeks 1-2, 3-6, 7-10, and 11-13 (rebuild if needed)
- readiness_score must reflect actual progress — increase if blockers resolved, decrease if new blockers added
- evolution_summary must be specific: what changed, why the plan evolved, what the team accomplished"""

    raw = _call_llm(prompt, max_tokens=4000)
    data = json.loads(raw)

    previous_score = previous.readiness_score
    new_score = data.get("readiness_score", previous_score)

    # Build EvolvedAIRoadmap — separate evolution-only fields from base AIRoadmap fields
    base_fields = {k: v for k, v in data.items() if k not in ("evolution_summary",)}
    evolved = EvolvedAIRoadmap(
        **base_fields,
        evolution_summary=data.get("evolution_summary", "Plan evolved based on check-in."),
        dropped_use_cases=state.get("dropped_use_cases", []),
        added_use_cases=[],
        milestone_shifts=state.get("milestone_shifts", []),
        readiness_score_delta=new_score - previous_score,
    )
    evolved.use_cases.sort(key=lambda uc: uc.priority_score, reverse=True)

    state["evolved_roadmap"] = evolved
    logger.info(f"Evolved plan generated — new readiness score: {new_score} (delta: {new_score - previous_score:+d})")
    return state
