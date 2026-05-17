# ai-use-case-scoper
> Turn "we want to do AI" into a prioritised roadmap with a recommended first project and a 90-day plan — before you write a line of code.

Part of the [llm-tools suite](https://github.com/swapnanil) by [Swapnanil Saha](https://swapnanilsaha.com)

## What it does

Every enterprise AI engagement starts the same way: leadership wants AI, nobody agrees on what for, and the first project gets chosen based on whoever argues loudest in a workshop. AI Use Case Scoper scores every candidate use case on feasibility, ROI, and risk — then returns a ranked list with one recommended first project and a milestone-by-milestone 90-day plan. If the company isn't ready, it says so and explains what to fix first.

## Quick start

```bash
git clone https://github.com/swapnanil/ai-use-case-scoper
cd ai-use-case-scoper
cp .env.example .env   # add your ANTHROPIC_API_KEY
docker-compose up api
```

## CLI usage

```bash
# Interactive 8-question guided scoping
docker-compose run --rm -it cli scope --interactive

# Scope from a JSON company profile
docker-compose run cli scope \
  --file examples/adtech_smb_profile.json \
  --format markdown

# Output JSON for downstream processing
docker-compose run cli scope \
  --file examples/adtech_smb_profile.json \
  --format json --output outputs/scoping_result.json
```

## API usage

```bash
# Run a scoping session
curl -X POST http://localhost:8000/scope \
  -H "Content-Type: application/json" \
  -d '{"industry": "Ad-Tech", "company_size": "smb", "ai_ambition": "Use AI to help account managers work better and automate weekly reporting", "pain_points": ["Manual reporting takes 4-6 hrs/week per AM"], "data_maturity": "medium", "has_ml_engineers": false, "timeline_pressure": "pilot_in_90_days"}'

# Get scoping heuristics
curl http://localhost:8000/heuristics
```

## Input / Output

**Input:**
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

**Output excerpt:**
```json
{
  "readiness_score": 62,
  "recommended_first_project": "Automated Weekly Campaign Report Generator",
  "feasibility_score": 9,
  "roi_score": 9,
  "estimated_timeline": "2–3 weeks",
  "estimated_cost_tier": "$",
  "quick_win": "Use Claude API for product description drafts — zero infrastructure, value in one day",
  "roadmap": {
    "week_1": "Integrate Claude API, connect to campaign data source",
    "week_4": "Automated report generation live for 2 AMs in pilot",
    "week_8": "Rolled out to full AM team, feedback incorporated",
    "week_12": "Measure time saved, build case for next use case"
  }
}
```

## Built with

- Python 3.11
- Anthropic SDK (claude-sonnet-4-6)
- FastAPI + uvicorn
- Docker + docker-compose
- pytest

## Author

Swapnanil Saha · [swapnanilsaha.com](https://swapnanilsaha.com) · [LinkedIn](https://linkedin.com/in/swapnanil)
