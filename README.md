# ai-use-case-scoper
> Turn "we want to do AI" into a prioritised roadmap with a recommended first project and a 90-day plan — before you write a line of code.

Part of the [llm-tools suite](https://github.com/swapnanil) by [Swapnanil Saha](https://swapnanilsaha.com)

## What it does

Every enterprise AI engagement starts the same way: leadership wants AI, nobody agrees on what for, and the first project gets chosen based on whoever argues loudest in a workshop. AI Use Case Scoper scores every candidate use case on feasibility, ROI, and risk — then returns a ranked list with one recommended first project and a milestone-by-milestone 90-day plan. If the company isn't ready, it says so and explains what to fix first.

**v2 adds three capabilities surfaced from enterprise customer discovery:**

- **Hybrid document ingestion** — the guided 8-question flow remains primary. At the start, the CLI optionally asks: *"Do you have a tech audit doc to share?"*. If yes, a Graph RAG pipeline (NetworkX + ChromaDB) extracts your tech profile from the PDF/DOCX, pre-fills technical fields where confidence ≥ 70%, and you still answer the qualitative questions (AI ambition, pain points, timeline) yourself. Docs enrich the context; they don't replace the conversation.
- **Company memory** — every scoping session is persisted (SQLite locally, Postgres in Docker). The tool remembers previous plans and injects that context into future recommendations.
- **Iterative plan evolution** — run a check-in 90 days later. A LangGraph pipeline compares what happened against the original roadmap and generates an evolved plan: updated use case scores, shifted milestones, and an explicit delta summary.

## Quick start

```bash
git clone https://github.com/swapnanil/ai-use-case-scoper
cd ai-use-case-scoper
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker-compose up api  # starts API + Postgres
```

Local dev without Docker (uses SQLite):
```bash
pip install -r requirements.txt
python api.py
```

## CLI usage

```bash
# Interactive hybrid mode: 8 questions + optional doc enrichment at the start
docker-compose run --rm -it cli scope --interactive

# Skip the doc ingestion prompt (pure 8-question flow)
docker-compose run --rm -it cli scope --interactive --no-ingest

# Scope from a JSON company profile (non-interactive)
docker-compose run cli scope \
  --file examples/company_adtech.json \
  --format markdown

# Run a check-in on a previous plan (LangGraph pipeline)
python main.py checkin <company-id> --session-id <session-id> --interactive

# View all scoping sessions for a company
python main.py history <company-id>
```

**What the hybrid interactive flow looks like:**

```
[Optional] Do you have internal tech docs or audit reports to share?
  (PDF, DOCX, TXT — we'll extract your tech profile and pre-fill where confident)
> yes

  Enter file paths (one per line, blank line to finish):
  > /path/to/tech_audit.pdf
  ✓ Loaded tech_audit.pdf (42 chunks)

  Building entity graph from documents...

  Document Extraction Summary
  ──────────────────────────────────────────────────────
  Field                     Extracted Value      Confidence
  tech_stack                Oracle DB, Java EE   90%
  compliance_requirements   RBI, PCI-DSS         90%
  has_ml_engineers          False                80%
  data_maturity             low                  65%
  engineering_team_size     120                  70%
  ──────────────────────────────────────────────────────
  Fields in green (≥70%) will be offered as pre-fills below.

[1/8] What industry is your company in?
  Pre-filled from docs: Financial Services (87% confidence) — press Enter to accept
  > _                   ← user presses Enter or types override

[3/8] Describe what you want AI to do for you:
  > Automate credit decisioning and reduce manual review time   ← always asked
```

## API usage

```bash
# Register a company
curl -X POST http://localhost:8000/companies \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}'

# Scope from a company profile JSON
curl -X POST http://localhost:8000/scope \
  -H "Content-Type: application/json" \
  -d @examples/company_adtech.json

# Ingest a document and extract profile via Graph RAG
curl -X POST http://localhost:8000/ingest \
  -F "files=@tech_audit.pdf" \
  -F "company_id=<company-id>"

# Confirm extracted profile and run scoping
curl -X POST http://localhost:8000/ingest/confirm \
  -H "Content-Type: application/json" \
  -d '{"company_id": "<id>", "confirmed_profile": {...}}'

# Run a check-in (LangGraph evolved plan)
curl -X POST http://localhost:8000/companies/<company-id>/checkin \
  -H "Content-Type: application/json" \
  -d @examples/sample_checkin.json

# Export roadmap to Jira
curl -X POST http://localhost:8000/sessions/<session-id>/export/jira \
  -H "Content-Type: application/json" \
  -d '{"jira_base_url": "https://yourorg.atlassian.net", "project_key": "AI", "api_token": "...", "user_email": "..."}'

# Get scoping heuristics (transparency endpoint)
curl http://localhost:8000/heuristics
```

## Input / Output

**Input (v1 — structured JSON):**
```json
{
  "industry": "Ad-Tech",
  "company_size": "smb",
  "ai_ambition": "Use AI to help account managers work better and automate weekly reporting",
  "pain_points": ["Manual reporting takes 4–6 hrs/week per AM", "Campaign insights delayed"],
  "data_maturity": "medium",
  "has_ml_engineers": false,
  "timeline_pressure": "pilot_in_90_days"
}
```

**Input (v2 — document upload):**
```bash
# POST /ingest — returns extracted profile + confidence scores per field
curl -X POST http://localhost:8000/ingest -F "files=@tech_audit.pdf"
# → {"extracted_profile": {...}, "confidence_scores": {"tech_stack": 0.9, "data_maturity": 0.55, ...},
#    "low_confidence_fields": ["data_maturity", "ai_experience"], "extraction_warnings": [...]}
```

**Output excerpt:**
```json
{
  "readiness_score": 62,
  "recommended_first_project": "Automated Weekly Campaign Report Generator",
  "roadmap_90_day": [
    {"week": "Week 1–2", "phase": "Discovery", "deliverables": ["..."], "success_criteria": "..."}
  ],
  "readiness_blockers": ["No ML engineers on team"],
  "readiness_accelerators": ["Clean campaign data in Postgres"]
}
```

**Check-in output (v2 — evolved plan):**
```json
{
  "readiness_score": 74,
  "readiness_score_delta": +12,
  "evolution_summary": "Report generator shipped in Week 4. Data schema blocker resolved. Q&A agent now feasible.",
  "dropped_use_cases": [],
  "milestone_shifts": [{"milestone": "Week 3–6", "shift": "removed", "reason": "use case shipped"}]
}
```

## Architecture

**v1 — single-pass scoping:**
```
CompanyProfile → Pre-scorer → Claude (claude-sonnet-4-6) → AIRoadmap
```

**v2 — hybrid interactive + iterative:**
```
[Optional] Document (PDF/DOCX/CSV)
  → Graph RAG (NetworkX + ChromaDB)
  → Confidence-scored field extraction (PRE_FILL_THRESHOLD = 0.7)
  ↘ Pre-fills: tech_stack, compliance, data_maturity, has_ml_engineers
  ↘ Always asked: ai_ambition, pain_points, timeline_pressure, budget_tier

8-question guided flow → CompanyProfile (enriched)
  → Pre-scorer → Claude (claude-sonnet-4-6) → AIRoadmap
  → SQLite / Postgres (session persisted)

Check-in (90 days later):
  → LangGraph pipeline:
     load_previous_plan → extract_deltas → classify_changes
     → reassess_use_cases → update_milestones → generate_evolved_plan
  → EvolvedAIRoadmap (with delta summary)
```

## Built with

| Component | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| Anthropic SDK (`claude-sonnet-4-6`) | LLM — scoping, entity extraction, plan evolution |
| LangGraph | Stateful agentic pipeline for check-in / plan evolution |
| NetworkX | In-process entity relationship graph (Graph RAG) |
| ChromaDB | Document chunk vector store (per-company, namespaced) |
| SQLAlchemy + SQLite/Postgres | Company, session, and outcome persistence |
| FastAPI + uvicorn | REST API |
| Typer + Rich | CLI |
| Docker + docker-compose | Containerisation (includes pgvector/pg16 for Postgres) |
| pytest | 119 tests |

## Author

Swapnanil Saha · [swapnanilsaha.com](https://swapnanilsaha.com) · [LinkedIn](https://linkedin.com/in/swapnanil)
