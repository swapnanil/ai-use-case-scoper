from typing import Any, Literal
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class CompanyProfile(BaseModel):
    # Company basics
    industry: str
    company_size: Literal["startup", "smb", "mid_market", "enterprise"]
    geography: str | None = None

    # Current state
    tech_stack: list[str] = []
    data_maturity: Literal["low", "medium", "high"]
    ai_experience: Literal["none", "basic", "advanced"]

    # Team
    engineering_team_size: int
    has_ml_engineers: bool
    budget_tier: Literal["bootstrap", "growth", "enterprise"]

    # The ask
    ai_ambition: str
    pain_points: list[str]

    # Constraints
    compliance_requirements: list[str] = []
    timeline_pressure: Literal[
        "experimental",
        "pilot_in_90_days",
        "production_in_6_months",
        "urgent",
    ]

    # Optional context
    existing_ai_initiatives: str | None = None
    competitors_doing: str | None = None

    @field_validator("ai_ambition")
    @classmethod
    def ambition_min_length(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("please describe your goal in more detail (minimum 20 characters)")
        return v


class UseCase(BaseModel):
    title: str
    description: str
    business_value: str

    # Scoring
    feasibility_score: int
    roi_score: int
    risk_score: int
    priority_score: int

    # Classification
    ai_approach: list[str]
    estimated_complexity: Literal["low", "medium", "high"]
    estimated_timeline: str
    estimated_cost_tier: Literal["$", "$$", "$$$"]

    # Requirements
    data_requirements: str
    integration_requirements: list[str]
    team_requirements: str

    # Risks
    key_risks: list[str]
    mitigation: list[str]


class Milestone(BaseModel):
    week: str
    phase: str
    deliverables: list[str]
    success_criteria: str


class AIRoadmap(BaseModel):
    # Executive summary
    executive_summary: str
    strategic_assessment: str

    # Use case portfolio
    use_cases: list[UseCase]
    recommended_first_project: str

    # 90-day roadmap
    roadmap_90_day: list[Milestone]

    # Strategic guidance
    quick_wins: list[str]
    things_to_avoid: list[str]
    questions_to_answer_first: list[str]

    # Readiness assessment
    readiness_score: int
    readiness_blockers: list[str]
    readiness_accelerators: list[str]


# ---------------------------------------------------------------------------
# v2: document ingestion
# ---------------------------------------------------------------------------

class GraphExtractionResult(BaseModel):
    extracted_profile: CompanyProfile
    confidence_scores: dict[str, float]
    low_confidence_fields: list[str]
    extracted_entities: list[dict]
    extraction_warnings: list[str]


# ---------------------------------------------------------------------------
# v2: company memory & session persistence
# ---------------------------------------------------------------------------

class Company(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    profile: CompanyProfile | None = None


class ScopingSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    input_profile: CompanyProfile
    roadmap: AIRoadmap
    session_type: Literal["initial", "checkin", "evolved"]
    checkin_notes: str | None = None
    parent_session_id: str | None = None


class PlanOutcome(BaseModel):
    session_id: str
    use_case_title: str
    status: Literal["not_started", "in_progress", "shipped", "abandoned", "blocked"]
    blocker: str | None = None
    actual_timeline: str | None = None
    notes: str | None = None
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# v2: iterative plan evolution
# ---------------------------------------------------------------------------

class CheckinInput(BaseModel):
    company_id: str
    parent_session_id: str
    notes: str
    outcome_updates: list[PlanOutcome] = []
    profile_changes: dict[str, Any] = {}
    new_constraints: list[str] = []
    timeline_pressure: Literal[
        "experimental", "pilot_in_90_days", "production_in_6_months", "urgent"
    ] | None = None


class EvolvedAIRoadmap(AIRoadmap):
    evolution_summary: str
    dropped_use_cases: list[str] = []
    added_use_cases: list[UseCase] = []
    milestone_shifts: list[dict] = []
    readiness_score_delta: int = 0
