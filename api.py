"""FastAPI REST API for the Enterprise AI Use Case Scoper."""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.models import AIRoadmap, CompanyProfile
from agent.scoper import scope_company

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise AI Use Case Scoper",
    description="Transforms a vague enterprise AI ambition into a structured, prioritised AI adoption roadmap.",
    version="1.0.0",
)

MODEL = os.getenv("MODEL", "claude-sonnet-4-6")


class QuickScopeRequest(BaseModel):
    industry: str
    company_size: str
    ai_ambition: str
    data_maturity: str


INDUSTRY_HEURISTICS = {
    "ad-tech": {
        "highest_roi_use_cases": [
            "Creative generation and testing automation",
            "Campaign performance diagnosis",
            "Audience intelligence and segmentation",
        ],
        "common_mistakes": [
            "Starting with brand safety AI before fixing data pipelines",
            "Over-investing in custom models when LLM APIs suffice",
        ],
        "readiness_prerequisites": ["Clean campaign performance data", "API access to ad platforms"],
    },
    "legal": {
        "highest_roi_use_cases": [
            "Document analysis and contract review",
            "Compliance checking automation",
            "Internal knowledge base Q&A",
        ],
        "common_mistakes": [
            "Using contract review AI without human review layer (liability risk)",
            "Ignoring data residency requirements",
        ],
        "readiness_prerequisites": ["Document digitisation", "Data residency compliance"],
    },
    "e-commerce": {
        "highest_roi_use_cases": [
            "Support automation (chatbot + ticket deflection)",
            "Product description generation",
            "Personalisation engine",
        ],
        "common_mistakes": [
            "Building personalisation before support automation (lower ROI)",
            "Skipping A/B testing on AI-generated content",
        ],
        "readiness_prerequisites": ["Clean order and customer data", "Shopify/platform API access"],
    },
    "healthcare": {
        "highest_roi_use_cases": [
            "Clinical documentation assistance",
            "Prior authorisation automation",
            "Patient communication personalisation",
        ],
        "common_mistakes": [
            "Ignoring HIPAA/PHI constraints on third-party LLM APIs",
            "Deploying without clinician review workflows",
        ],
        "readiness_prerequisites": ["HIPAA compliance framework", "EHR integration capability"],
    },
    "financial-services": {
        "highest_roi_use_cases": [
            "Document intelligence (contracts, reports)",
            "Compliance monitoring",
            "Customer support automation",
        ],
        "common_mistakes": [
            "Hallucination risk in financial advice contexts without guardrails",
            "Ignoring model explainability requirements",
        ],
        "readiness_prerequisites": ["Regulatory approval process", "Data governance framework"],
    },
    "general": {
        "highest_roi_use_cases": [
            "Internal knowledge base Q&A",
            "Document summarisation",
            "Customer support automation",
        ],
        "common_mistakes": [
            "Starting with the most complex use case",
            "Underestimating data preparation effort",
        ],
        "readiness_prerequisites": ["Centralised data access", "Clear success metrics"],
    },
}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL}


@app.get("/heuristics")
def heuristics() -> dict:
    return {
        "description": "Industry-specific scoping heuristics used by the AI Use Case Scoper",
        "industries": INDUSTRY_HEURISTICS,
        "universal_heuristics": [
            "Data maturity 'low' + any LLM use case = add data cleanup sprint to roadmap first",
            "No ML engineers + complex AI ambition = recommend managed APIs only, no custom models",
            "'Urgent' timeline + first AI project = descope aggressively, ship something small that works",
            "GDPR or on-prem constraint = rule out OpenAI API, recommend Anthropic with DPA or self-hosted",
        ],
    }


@app.post("/scope")
def scope(profile_data: dict) -> dict:
    try:
        profile = CompanyProfile(**profile_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        roadmap = scope_company(profile)
        return roadmap.model_dump()
    except Exception as e:
        logger.exception("Error during scoping")
        raise HTTPException(status_code=500, detail=f"Scoping failed: {str(e)}")


@app.post("/scope/quick")
def scope_quick(request: QuickScopeRequest) -> dict:
    defaults = {
        "geography": None,
        "tech_stack": [],
        "ai_experience": "none",
        "engineering_team_size": 10,
        "has_ml_engineers": False,
        "budget_tier": "growth",
        "pain_points": [],
        "compliance_requirements": [],
        "timeline_pressure": "experimental",
        "existing_ai_initiatives": None,
        "competitors_doing": None,
    }

    profile_data = {
        "industry": request.industry,
        "company_size": request.company_size,
        "ai_ambition": request.ai_ambition,
        "data_maturity": request.data_maturity,
        **defaults,
    }

    try:
        profile = CompanyProfile(**profile_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        roadmap = scope_company(profile)
        return roadmap.model_dump()
    except Exception as e:
        logger.exception("Error during quick scoping")
        raise HTTPException(status_code=500, detail=f"Scoping failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
