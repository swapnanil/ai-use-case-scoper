from typing import Literal
from pydantic import BaseModel, field_validator


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
