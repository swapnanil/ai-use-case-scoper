import networkx as nx

from agent.models import CompanyProfile, GraphExtractionResult

CONFIDENCE_THRESHOLD = 0.6


def extract_profile(G: nx.DiGraph, overrides: dict | None = None) -> GraphExtractionResult:
    """Map a NetworkX entity graph to a CompanyProfile with per-field confidence scores."""
    overrides = overrides or {}
    profile_data: dict = {}
    confidence: dict[str, float] = {}
    warnings: list[str] = []

    nodes_by_type = _group_by_type(G)

    # tech_stack
    tech_items = [n for n, d in G.nodes(data=True) if d.get("node_type") == "TechStack"]
    profile_data["tech_stack"] = tech_items
    confidence["tech_stack"] = 0.9 if tech_items else 0.2

    # compliance_requirements
    compliance_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "ComplianceRequirement"]
    profile_data["compliance_requirements"] = compliance_nodes
    confidence["compliance_requirements"] = 0.9 if compliance_nodes else 0.3
    if not compliance_nodes:
        warnings.append("No compliance requirements detected in document — verify manually")

    # data_maturity — inferred from DataSource governance_status signals
    ds_nodes = [d for _, d in G.nodes(data=True) if d.get("node_type") == "DataSource"]
    data_maturity, dm_conf = _infer_data_maturity(ds_nodes)
    profile_data["data_maturity"] = data_maturity
    confidence["data_maturity"] = dm_conf
    if dm_conf < CONFIDENCE_THRESHOLD:
        warnings.append(
            f"data_maturity inferred as '{data_maturity}' from {len(ds_nodes)} signals — verify manually"
        )

    # has_ml_engineers
    team_nodes = [d for _, d in G.nodes(data=True) if d.get("node_type") == "Team"]
    has_ml = any(t.get("has_ml_engineers", False) for t in team_nodes)
    profile_data["has_ml_engineers"] = has_ml
    confidence["has_ml_engineers"] = 0.8 if team_nodes else 0.3

    # engineering_team_size — sum of team sizes found
    sizes = [t["size"] for t in team_nodes if t.get("size")]
    profile_data["engineering_team_size"] = sum(sizes) if sizes else 10
    confidence["engineering_team_size"] = 0.7 if sizes else 0.2
    if not sizes:
        warnings.append("engineering_team_size not found in document — defaulting to 10")

    # ai_experience — inferred from legacy system ratio and team ML signals
    system_nodes = [d for _, d in G.nodes(data=True) if d.get("node_type") == "System"]
    legacy_count = sum(1 for s in system_nodes if s.get("legacy", False))
    undocumented_count = sum(1 for s in system_nodes if not s.get("documented", True))
    ai_experience, ai_conf = _infer_ai_experience(system_nodes, has_ml, legacy_count, undocumented_count)
    profile_data["ai_experience"] = ai_experience
    confidence["ai_experience"] = ai_conf
    if ai_conf < CONFIDENCE_THRESHOLD:
        warnings.append("ai_experience inferred from system and team signals — verify manually")

    # Any field provided in overrides is fully trusted — bump its confidence to 1.0
    for field in overrides:
        if field in confidence:
            confidence[field] = 1.0

    # Fields that require human input — mark with zero confidence unless overridden
    human_input_fields = [
        "industry", "company_size", "budget_tier", "timeline_pressure",
        "ai_ambition", "pain_points", "geography",
    ]
    for field in human_input_fields:
        if field in overrides:
            profile_data[field] = overrides[field]
            confidence[field] = 1.0
        else:
            confidence[field] = 0.0

    low_confidence_fields = [f for f, c in confidence.items() if c < CONFIDENCE_THRESHOLD]

    extracted_entities = [
        {"id": n, **{k: v for k, v in d.items()}}
        for n, d in G.nodes(data=True)
    ]

    return GraphExtractionResult(
        extracted_profile=_build_profile(profile_data, overrides),
        confidence_scores=confidence,
        low_confidence_fields=low_confidence_fields,
        extracted_entities=extracted_entities,
        extraction_warnings=warnings,
    )


def _infer_data_maturity(ds_nodes: list[dict]) -> tuple[str, float]:
    if not ds_nodes:
        return "low", 0.3

    statuses = [d.get("governance_status", "unknown") for d in ds_nodes]
    total = len(statuses)
    clean = statuses.count("clean")
    messy = statuses.count("messy")
    unknown = statuses.count("unknown")

    if clean / total >= 0.7:
        return "high", 0.75
    elif messy / total >= 0.5 or unknown / total >= 0.6:
        return "low", 0.65
    else:
        return "medium", 0.6


def _infer_ai_experience(
    system_nodes: list[dict],
    has_ml: bool,
    legacy_count: int,
    undocumented_count: int,
) -> tuple[str, float]:
    if has_ml:
        return "basic", 0.65
    if legacy_count > 5 or undocumented_count > 3:
        return "none", 0.55
    return "none", 0.45


def _group_by_type(G: nx.DiGraph) -> dict[str, list]:
    result: dict[str, list] = {}
    for n, d in G.nodes(data=True):
        t = d.get("node_type", "Unknown")
        result.setdefault(t, []).append((n, d))
    return result


def _build_profile(profile_data: dict, overrides: dict) -> CompanyProfile:
    defaults = {
        "industry": "Unknown",
        "company_size": "enterprise",
        "geography": None,
        "tech_stack": [],
        "data_maturity": "low",
        "ai_experience": "none",
        "engineering_team_size": 10,
        "has_ml_engineers": False,
        "budget_tier": "growth",
        "ai_ambition": "Improve operational efficiency with AI",
        "pain_points": ["Not yet specified — provide during profile review"],
        "compliance_requirements": [],
        "timeline_pressure": "experimental",
        "existing_ai_initiatives": None,
        "competitors_doing": None,
    }
    merged = {**defaults, **profile_data, **overrides}
    return CompanyProfile(**merged)
