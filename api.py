"""FastAPI REST API for the Enterprise AI Use Case Scoper."""

import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent import memory as mem
from agent.models import (
    AIRoadmap,
    CheckinInput,
    CompanyProfile,
    EvolvedAIRoadmap,
    PlanOutcome,
)
from agent.scoper import scope_company
from ingestion import graph_builder, store
from ingestion.extractor import extract_profile
from ingestion.loader import SUPPORTED_EXTENSIONS, load_file

load_dotenv()
mem.init_db()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))

app = FastAPI(
    title="Enterprise AI Use Case Scoper",
    description="Transforms a vague enterprise AI ambition into a structured, prioritised AI adoption roadmap.",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QuickScopeRequest(BaseModel):
    industry: str
    company_size: str
    ai_ambition: str
    data_maturity: str


class IngestConfirmRequest(BaseModel):
    company_id: str
    confirmed_profile: dict


class CreateCompanyRequest(BaseModel):
    name: str


class JiraExportRequest(BaseModel):
    jira_base_url: str
    project_key: str
    api_token: str
    user_email: str


class ClickUpExportRequest(BaseModel):
    clickup_list_id: str
    api_token: str


# ---------------------------------------------------------------------------
# Industry heuristics (v1, kept for /heuristics endpoint)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# v1 endpoints (unchanged)
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL, "version": "2.0.0"}


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


# ---------------------------------------------------------------------------
# v2: company memory
# ---------------------------------------------------------------------------

@app.post("/companies")
def create_company(request: CreateCompanyRequest) -> dict:
    return mem.create_company(request.name)


@app.get("/companies/{company_id}")
def get_company(company_id: str) -> dict:
    company = mem.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company


@app.get("/companies/{company_id}/sessions")
def get_sessions(company_id: str) -> list:
    if not mem.get_company(company_id):
        raise HTTPException(status_code=404, detail="Company not found")
    return mem.get_company_sessions(company_id)


@app.post("/companies/{company_id}/outcomes")
def record_outcome(company_id: str, outcome: PlanOutcome) -> dict:
    if not mem.get_company(company_id):
        raise HTTPException(status_code=404, detail="Company not found")
    mem.save_outcome(outcome)
    return {"status": "recorded"}


# ---------------------------------------------------------------------------
# v2: document ingestion
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest_documents(
    files: list[UploadFile] = File(...),
    company_id: str = Form(None),
) -> dict:
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_UPLOAD_FILES} files per session")

    all_chunks = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported file type '{suffix}'. Supported: {SUPPORTED_EXTENSIONS}",
                )
            content = await upload.read()
            if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File '{upload.filename}' exceeds {MAX_UPLOAD_SIZE_MB}MB limit")

            tmp_path = Path(tmpdir) / (upload.filename or "upload")
            tmp_path.write_bytes(content)
            try:
                chunks = load_file(tmp_path)
                all_chunks.extend(chunks)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Failed to parse '{upload.filename}': {e}")

    if not all_chunks:
        raise HTTPException(status_code=422, detail="No text content extracted from uploaded files")

    G = graph_builder.build_graph(all_chunks, MODEL)
    result = extract_profile(G)

    if company_id:
        store.store_chunks(company_id, all_chunks)

    return {
        "extracted_profile": result.extracted_profile.model_dump(),
        "confidence_scores": result.confidence_scores,
        "low_confidence_fields": result.low_confidence_fields,
        "extraction_warnings": result.extraction_warnings,
        "entity_count": len(result.extracted_entities),
        "chunk_count": len(all_chunks),
    }


@app.post("/ingest/confirm")
def ingest_confirm(request: IngestConfirmRequest) -> dict:
    """Confirm an extracted profile and run scoping. Saves session if company exists."""
    try:
        profile = CompanyProfile(**request.confirmed_profile)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid profile: {e}")
    try:
        roadmap = scope_company(profile)
    except Exception as e:
        logger.exception("Scoping failed after ingest confirm")
        raise HTTPException(status_code=500, detail=f"Scoping failed: {e}")

    session_id = None
    if mem.get_company(request.company_id):
        session_id = mem.save_session(
            company_id=request.company_id,
            profile=profile,
            roadmap=roadmap,
            session_type="initial",
        )

    response = roadmap.model_dump()
    if session_id:
        response["session_id"] = session_id
    return response


@app.get("/ingest/{company_id}/graph")
def get_entity_graph(company_id: str) -> dict:
    """Return the stored document chunks for a company (graph transparency endpoint)."""
    chunks = store.query_chunks(company_id, "systems compliance data", n_results=20)
    return {
        "company_id": company_id,
        "chunk_count": len(chunks),
        "sample_chunks": chunks[:3],
    }


# ---------------------------------------------------------------------------
# v2: iterative plan evolution (checkin)
# ---------------------------------------------------------------------------

@app.post("/companies/{company_id}/checkin")
def checkin(company_id: str, checkin_input: CheckinInput) -> dict:
    if not mem.get_company(company_id):
        raise HTTPException(status_code=404, detail="Company not found")
    if checkin_input.company_id != company_id:
        raise HTTPException(status_code=422, detail="company_id mismatch in body vs path")

    parent = mem.get_session(checkin_input.parent_session_id)
    if not parent:
        raise HTTPException(status_code=404, detail="Parent session not found")

    try:
        from pipeline.checkin_graph import run_checkin
        evolved = run_checkin(checkin_input)
    except Exception as e:
        logger.exception("Checkin pipeline failed")
        raise HTTPException(status_code=500, detail=f"Checkin failed: {e}")

    profile = CompanyProfile(**parent["input_profile"])
    if checkin_input.profile_changes:
        profile = CompanyProfile(**{**parent["input_profile"], **checkin_input.profile_changes})

    session_id = mem.save_session(
        company_id=company_id,
        profile=profile,
        roadmap=evolved,
        session_type="evolved",
        checkin_notes=checkin_input.notes,
        parent_session_id=checkin_input.parent_session_id,
    )

    result = evolved.model_dump()
    result["session_id"] = session_id
    return result


# ---------------------------------------------------------------------------
# v2: export integrations
# ---------------------------------------------------------------------------

@app.post("/sessions/{session_id}/export/jira")
def export_jira(session_id: str, request: JiraExportRequest) -> dict:
    session = mem.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    roadmap = AIRoadmap(**session["roadmap"])
    try:
        from integrations.jira import export_to_jira
        return export_to_jira(
            roadmap=roadmap,
            jira_base_url=request.jira_base_url,
            project_key=request.project_key,
            api_token=request.api_token,
            user_email=request.user_email,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Jira export failed: {e}")


@app.post("/sessions/{session_id}/export/clickup")
def export_clickup(session_id: str, request: ClickUpExportRequest) -> dict:
    session = mem.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    roadmap = AIRoadmap(**session["roadmap"])
    try:
        from integrations.clickup import export_to_clickup
        return export_to_clickup(roadmap=roadmap, list_id=request.clickup_list_id, api_token=request.api_token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ClickUp export failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
