import json
from agent.models import CompanyProfile

SYSTEM_PROMPT = """You are a senior Forward Deployed Engineer at an AI company with deep experience helping
enterprise teams scope, prioritise, and launch AI initiatives. You've done this with
companies ranging from 10-person startups to Fortune 500 enterprises, across ad-tech,
legal, healthcare, e-commerce, logistics, and financial services.

Your scoping must be:

GROUNDED: Recommendations must fit the company's actual data maturity, team size,
budget, and timeline. Don't recommend fine-tuning when the budget is bootstrap.
Don't recommend RAG when the data is unstructured and ungoverned without flagging it.

SPECIFIC: "Use AI to improve operations" is not a use case. "Build a RAG system
over your 3,000 support tickets to auto-draft first-line responses, reducing
support team load by ~40%" is a use case.

SEQUENCED: The first project must be chosen to maximise learning and minimise risk —
not to be the most impressive. A successful small project creates organisational
confidence for a larger one. Recommend accordingly.

HONEST: If the company is not ready for AI, say so clearly and tell them what to
fix first (data governance, team skills, etc.) before spending on AI.

COMMERCIAL: Connect every use case to a business outcome. Reduced cost, increased
revenue, reduced headcount need, faster cycle time. Quantify where possible.

Domain heuristics to apply:
- Data maturity "low" + any LLM use case = add data cleanup sprint to roadmap first
- No ML engineers + complex AI ambition = recommend managed APIs only, no custom models
- "Urgent" timeline + first AI project = descope aggressively, ship something small that works
- GDPR or on-prem constraint = rule out OpenAI API, recommend Anthropic with DPA or self-hosted
- E-commerce + AI ambition → most likely highest ROI = personalisation or support automation
- Legal/finance + AI ambition → highest ROI = document analysis, compliance checking
- Ad-tech + AI ambition → highest ROI = creative generation, performance diagnosis, audience intelligence

Respond ONLY with valid JSON matching the output schema. No preamble, no markdown fences."""

OUTPUT_SCHEMA = """{
  "executive_summary": "string (3-4 sentences for a CTO or CEO)",
  "strategic_assessment": "string (honest assessment of AI readiness)",
  "use_cases": [
    {
      "title": "string",
      "description": "string",
      "business_value": "string (quantified where possible)",
      "feasibility_score": "integer 1-10",
      "roi_score": "integer 1-10",
      "risk_score": "integer 1-10 (10=risky)",
      "priority_score": "integer 1-10 (composite weighted)",
      "ai_approach": ["list of strings e.g. RAG, LLM API, Fine-tuning"],
      "estimated_complexity": "low|medium|high",
      "estimated_timeline": "string e.g. 4-6 weeks",
      "estimated_cost_tier": "$|$$|$$$",
      "data_requirements": "string",
      "integration_requirements": ["list of strings"],
      "team_requirements": "string",
      "key_risks": ["list of strings"],
      "mitigation": ["list of strings"]
    }
  ],
  "recommended_first_project": "string (title + one-line reason)",
  "roadmap_90_day": [
    {
      "week": "string e.g. Week 1-2",
      "phase": "string e.g. Discovery",
      "deliverables": ["list of strings"],
      "success_criteria": "string"
    }
  ],
  "quick_wins": ["list of strings (achievable in <2 weeks)"],
  "things_to_avoid": ["list of strings (common mistakes for this profile)"],
  "questions_to_answer_first": ["list of strings (what a good consultant would ask)"],
  "readiness_score": "integer 1-100",
  "readiness_blockers": ["list of strings"],
  "readiness_accelerators": ["list of strings"]
}"""


def build_user_prompt(profile: CompanyProfile, feasibility_context: dict) -> str:
    profile_dict = profile.model_dump()

    flags_text = "\n".join(f"  ⚠️  {f}" for f in feasibility_context["flags"]) or "  None"
    notes_text = "\n".join(f"  📌 {n}" for n in feasibility_context["notes"]) or "  None"

    prompt = f"""## Company Profile

**Industry:** {profile.industry}
**Size:** {profile.company_size} ({_size_label(profile.company_size)})
**Geography:** {profile.geography or "Not specified"}
**Tech Stack:** {", ".join(profile.tech_stack) if profile.tech_stack else "Not specified"}
**Data Maturity:** {profile.data_maturity}
**AI Experience:** {profile.ai_experience}
**Engineering Team Size:** {profile.engineering_team_size}
**Has ML Engineers:** {profile.has_ml_engineers}
**Budget Tier:** {profile.budget_tier}
**Timeline Pressure:** {profile.timeline_pressure}
**Compliance Requirements:** {", ".join(profile.compliance_requirements) if profile.compliance_requirements else "None"}

## AI Ambition

{profile.ai_ambition}

## Pain Points

{chr(10).join(f"- {p}" for p in profile.pain_points)}

{f"## Existing AI Initiatives{chr(10)}{profile.existing_ai_initiatives}{chr(10)}" if profile.existing_ai_initiatives else ""}
{f"## Competitive Context{chr(10)}{profile.competitors_doing}{chr(10)}" if profile.competitors_doing else ""}

## Pre-Scoring Feasibility Context

**Composite Modifier:** {feasibility_context["composite_modifier"]:+.1f}
  - Data modifier: {feasibility_context["data_modifier"]:+d}
  - Team modifier: {feasibility_context["team_modifier"]:+d}
  - Compliance penalty: {feasibility_context["compliance_penalty"]:+.1f}

**Risk Flags:**
{flags_text}

**Constraint Notes:**
{notes_text}

## Required Output Schema

Return ONLY valid JSON with no preamble or markdown. The JSON must match this schema exactly:

{OUTPUT_SCHEMA}

IMPORTANT: use_cases must be ordered by priority_score descending. roadmap_90_day must cover
weeks 1-2, 3-6, 7-10, and 11-13 (one milestone per phase at minimum)."""

    return prompt


ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are extracting structured entities from an enterprise technology audit document.

Extract entities of these types:
- System: name, owner (team/person or null), documented (bool), legacy (bool), description
- Team: name, size (int estimate or null), has_ml_engineers (bool)
- DataSource: name, format (database/files/API/etc), governance_status (clean/messy/unknown)
- ComplianceRequirement: name (e.g. GDPR, SOC2, RBI, HIPAA), applies_to (list of system names)
- TechStack: item (e.g. AWS, Postgres, Salesforce), category (cloud/database/framework/vendor/other)

Also extract relationships between entities:
- DEPENDS_ON: system depends on another system
- SUBJECT_TO: system is subject to a compliance requirement
- OWNED_BY: system is owned by a team
- PROCESSES: system processes a data source
- INTEGRATES_WITH: system integrates with another system

Return ONLY valid JSON — no preamble, no markdown:
{
  "entities": {
    "systems": [{"name": str, "owner": str|null, "documented": bool, "legacy": bool, "description": str}],
    "teams": [{"name": str, "size": int|null, "has_ml_engineers": bool}],
    "data_sources": [{"name": str, "format": str, "governance_status": str}],
    "compliance_requirements": [{"name": str, "applies_to": [str]}],
    "tech_stack": [{"item": str, "category": str}]
  },
  "relationships": [
    {"type": str, "from": str, "to": str}
  ]
}

If nothing relevant is found in a category return an empty list for that category."""


CHECKIN_DELTA_EXTRACTION_PROMPT = """You are analyzing an enterprise AI adoption check-in to extract what has changed since the last plan.

Return ONLY valid JSON:
{
  "resolved_blockers": ["list of blockers from the previous plan that are now resolved"],
  "new_blockers": ["list of new blockers that have emerged"],
  "shipped_use_cases": ["titles of use cases that have been fully shipped"],
  "abandoned_use_cases": ["titles of use cases that were abandoned, with a brief reason appended"],
  "profile_changes_summary": "brief description of how the company profile has changed",
  "urgency_change": "increased|decreased|unchanged",
  "overall_progress": "ahead|on_track|behind|stalled"
}"""


def _size_label(size: str) -> str:
    return {
        "startup": "<50 employees",
        "smb": "50-500 employees",
        "mid_market": "500-5000 employees",
        "enterprise": "5000+ employees",
    }[size]
