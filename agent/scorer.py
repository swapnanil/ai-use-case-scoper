from agent.models import CompanyProfile


def compute_feasibility_context(profile: CompanyProfile) -> dict:
    flags = []
    notes = []

    # Data readiness modifier
    data_mod = {"low": -2, "medium": 0, "high": 2}[profile.data_maturity]
    if profile.data_maturity == "low":
        notes.append("Data maturity is low — a data cleanup sprint must precede any LLM use case.")

    # Team modifier
    team_mod = 0
    if profile.has_ml_engineers:
        team_mod += 2
    if profile.ai_experience == "none":
        team_mod -= 1

    # Compliance complexity
    compliance_penalty = len(profile.compliance_requirements) * -0.5

    # Timeline realism flag
    if profile.timeline_pressure == "urgent" and profile.ai_experience == "none":
        flags.append(
            "TIMELINE_RISK: urgent timeline with no AI experience is high-risk — "
            "descope aggressively, ship something small that works"
        )

    # Budget ceiling
    if profile.budget_tier == "bootstrap":
        notes.append(
            "Limit recommendations to managed API approaches — "
            "no fine-tuning, no self-hosted models"
        )

    # Conflicting constraint: urgent + no AI experience + on-prem
    on_prem = any(
        r.lower() in ("on-prem", "on_prem", "data on-prem", "data residency")
        for r in profile.compliance_requirements
    )
    if profile.timeline_pressure == "urgent" and profile.ai_experience == "none" and on_prem:
        flags.append(
            "CONFLICT: urgent timeline + no AI experience + on-prem requirement — "
            "this combination is extremely high-risk; reflect in readiness_blockers"
        )

    # GDPR / on-prem: rule out OpenAI, recommend Anthropic with DPA or self-hosted
    if on_prem or "GDPR" in [r.upper() for r in profile.compliance_requirements]:
        notes.append(
            "GDPR or on-prem constraint detected — rule out OpenAI API; "
            "recommend Anthropic with DPA or self-hosted model"
        )

    composite = data_mod + team_mod + compliance_penalty

    return {
        "data_modifier": data_mod,
        "team_modifier": team_mod,
        "compliance_penalty": compliance_penalty,
        "composite_modifier": composite,
        "flags": flags,
        "notes": notes,
        "on_prem_detected": on_prem,
    }
