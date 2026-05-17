# Enterprise AI Use Case Scoper

**Tool 5 of 5 — llm-tools suite by [Swapnanil Saha](https://swapnanilsaha.com)**

A production-grade Python CLI + REST API that transforms a vague enterprise AI ambition into a structured, prioritised, and actionable AI adoption roadmap.

Input your company profile and a rough idea of where you want AI — the tool outputs: a scoped list of use cases ranked by feasibility and ROI, a recommended first project, a 90-day implementation roadmap, risk flags, and the questions a good consultant would ask before starting.

Built with the Anthropic Python SDK. Fully containerised with Docker.

---

## What It Does

Every enterprise team wants to adopt AI but most don't know where to start. They've heard about RAG, agents, and LLMs but can't connect these technologies to their specific workflows and constraints. The result: expensive pilots on the wrong use case, failed deployments, and leadership losing faith in AI investment.

This tool encodes the strategic scoping judgment of a senior AI deployment practitioner — asking the right questions, identifying the highest-ROI starting point, and de-risking the roadmap before a single line of code is written.

**Input:** Company profile (industry, size, tech stack, data maturity, team, budget, compliance, pain points, AI ambition)

**Output:**
- 4–6 scoped AI use cases ranked by feasibility, ROI, and risk
- A recommended first project with clear justification
- A 90-day implementation roadmap with milestones and success criteria
- Quick wins (< 2 weeks), things to avoid, and questions to answer first
- An honest AI readiness score (1–100) with blockers and accelerators

---

## Who It's For

- **Enterprise teams** evaluating where to start with AI
- **AI practitioners and consultants** who want a structured first-engagement output
- **Technical leads** who need to present a credible AI roadmap to leadership

---

## The Scoping Framework

The tool applies a set of industry-tested heuristics before and during the LLM call:

**Pre-LLM feasibility scoring** adjusts recommendations based on:
- Data maturity modifier (low: −2, medium: 0, high: +2)
- Team modifier (ML engineers: +2, no AI experience: −1)
- Compliance penalty (−0.5 per requirement)
- Timeline risk flag (urgent + no AI experience = high-risk signal)
- Budget ceiling (bootstrap = managed APIs only, no fine-tuning)

**Domain heuristics (injected into the LLM prompt):**
- Data maturity "low" + any LLM use case → data cleanup sprint required first
- No ML engineers + complex AI ambition → managed APIs only, no custom models
- "Urgent" timeline + first AI project → descope aggressively, ship something small
- GDPR or on-prem → rule out OpenAI, recommend Anthropic with DPA or self-hosted
- E-commerce → highest ROI: support automation or personalisation
- Legal/finance → highest ROI: document analysis, compliance checking
- Ad-tech → highest ROI: creative generation, performance diagnosis, audience intelligence

---

## Quick Start with Docker

**1. Clone and configure:**

```bash
git clone <repo>
cd ai-use-case-scoper
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

**2. Run the interactive CLI:**

```bash
docker-compose run cli scope --interactive
```

**3. Run the API:**

```bash
docker-compose up api
# API available at http://localhost:8000
```

---

## CLI Usage

**Interactive mode (recommended for first use):**

```bash
python main.py scope --interactive
```

**From a JSON company profile file:**

```bash
python main.py scope --file examples/company_adtech.json
python main.py scope --file examples/company_legal.json --format markdown
python main.py scope --file examples/company_ecommerce.json --format html --output roadmap.html
```

**Quick mode (minimal required fields):**

```bash
python main.py scope \
  --industry "Legal Services" \
  --size mid_market \
  --ambition "We want AI to help our lawyers review contracts faster and find relevant precedents" \
  --data-maturity medium \
  --ai-experience none
```

**Output formats:**

```bash
python main.py scope --file company.json --format json
python main.py scope --file company.json --format markdown
python main.py scope --file company.json --format html --output roadmap.html
```

**Verbose mode (shows pre-scoring context before LLM call):**

```bash
python main.py scope --file company.json --verbose
```

**Docker variants:**

```bash
docker-compose run cli scope --interactive
docker-compose run cli scope --file examples/company_legal.json --format markdown
```

---

## API Usage

**Start the API:**

```bash
docker-compose up api
# or: uvicorn api:app --reload
```

**Full scope (POST /scope):**

```bash
curl -X POST http://localhost:8000/scope \
  -H "Content-Type: application/json" \
  -d @examples/company_adtech.json | jq .
```

**Quick scope (POST /scope/quick):**

```bash
curl -X POST http://localhost:8000/scope/quick \
  -H "Content-Type: application/json" \
  -d '{
    "industry": "E-commerce",
    "company_size": "smb",
    "ai_ambition": "We want AI to automate our customer support and reduce response times significantly",
    "data_maturity": "high"
  }' | jq .
```

**Health check:**

```bash
curl http://localhost:8000/health
# {"status":"ok","model":"claude-sonnet-4-6"}
```

**Scoping heuristics (transparency endpoint):**

```bash
curl http://localhost:8000/heuristics | jq .
```

**API docs:**

```
http://localhost:8000/docs
```

---

## Sample Input → Sample Output

**Input** (`examples/company_adtech.json`): Mid-market ad-tech company, 300 employees, Python + AWS, medium data maturity, no ML engineers, basic AI experience (vendor APIs), GDPR, growth budget, pilot in 90 days. Pain points: manual reporting (4-6h/AM/week), delayed campaign insights, slow onboarding.

**Output summary** (`examples/sample_output.json`):

| # | Use Case | Priority | ROI | Feasibility | Timeline |
|---|----------|----------|-----|-------------|----------|
| 1 | Automated Weekly Campaign Report Generation | 9 | 9/10 | 9/10 | 3 weeks |
| 2 | Campaign Q&A Agent for Account Managers | 8 | 8/10 | 7/10 | 6–8 weeks |
| 3 | New Account Manager Onboarding Knowledge Base | 7 | 7/10 | 7/10 | 5–7 weeks |

**Recommended first project:** Automated Weekly Campaign Report Generation — lowest risk, fastest ROI, zero new infrastructure. Ship in 3 weeks and prove AI value before committing to more complex projects.

**AI Readiness Score:** 72/100

**Quick wins:**
- Use Claude API to generate one-paragraph campaign summaries for top 5 clients this week (no infrastructure needed, just a script)
- Ask AMs to document their top 10 most-asked client questions
- Sign Anthropic GDPR DPA in week 1 — unblocks everything, takes 1 hour

---

## Why the First Project Recommendation Matters

The most common failure mode in enterprise AI adoption is not technical — it's organisational. Teams pick the most impressive-sounding use case (a full autonomous agent, a complex RAG system over unstructured data) and spend 6 months on it. When it doesn't ship on time or doesn't work as expected, leadership loses confidence in AI investment and the entire programme stalls.

The right first project is not the most impressive one. It's the one that:

1. **Solves a real, measurable pain point** — so ROI is visible and undeniable
2. **Has manageable data requirements** — so the team isn't blocked by data cleanup for months
3. **Fits the team's current skills** — so it actually ships
4. **Creates organisational confidence** — so there's budget and buy-in for the harder projects next

This tool is opinionated about sequencing. The recommended first project is chosen to maximise learning and minimise risk — not to be the most technically interesting. A successful small project opens the door to everything else.

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Project Structure

```
ai-use-case-scoper/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── main.py                       # CLI entry point
├── api.py                        # FastAPI app
├── agent/
│   ├── scoper.py                 # Core scoping logic
│   ├── prompts.py                # System + user prompts
│   ├── models.py                 # Pydantic input/output models
│   └── scorer.py                 # Pre-LLM feasibility pre-scoring
├── examples/
│   ├── company_adtech.json
│   ├── company_legal.json
│   ├── company_ecommerce.json
│   └── sample_output.json
└── tests/
    ├── test_scoper.py
    ├── test_scorer.py
    └── test_api.py
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Your Anthropic API key |
| `MODEL` | `claude-sonnet-4-6` | Model to use |
| `MAX_TOKENS` | `4000` | Max output tokens |
| `LOG_LEVEL` | `INFO` | Logging level |

---

Built by [Swapnanil Saha](https://swapnanilsaha.com) — Tool 5 of 5 in the llm-tools suite.
