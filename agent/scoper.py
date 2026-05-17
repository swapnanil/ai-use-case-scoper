import json
import logging
import os
import time

import anthropic
from dotenv import load_dotenv

from agent.models import AIRoadmap, CompanyProfile
from agent.prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from agent.scorer import compute_feasibility_context

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = os.getenv("MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def scope_company(profile: CompanyProfile, verbose: bool = False) -> AIRoadmap:
    feasibility_ctx = compute_feasibility_context(profile)

    if verbose:
        _log_feasibility(feasibility_ctx)

    user_prompt = build_user_prompt(profile, feasibility_ctx)

    raw_json = _call_llm_with_retry(user_prompt, verbose=verbose)
    return _parse_roadmap(raw_json)


def _call_llm_with_retry(user_prompt: str, verbose: bool = False) -> str:
    client = _get_client()
    max_retries = 3
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            if verbose and attempt == 0:
                logger.info(f"Calling {MODEL} (max_tokens={MAX_TOKENS})")

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        except anthropic.RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff * (2**attempt)
            logger.warning(f"Rate limit hit, retrying in {wait:.1f}s (attempt {attempt + 1})")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff * (2**attempt)
            logger.warning(f"API error {e.status_code}, retrying in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError("Exhausted all retries")


def _parse_roadmap(raw: str) -> AIRoadmap:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(cleaned)
        roadmap = AIRoadmap(**data)
        # Enforce ordering by priority_score descending
        roadmap.use_cases.sort(key=lambda uc: uc.priority_score, reverse=True)
        return roadmap
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Initial parse failed: {e}. Retrying with schema correction.")
        return _retry_with_schema_correction(cleaned, str(e))


def _retry_with_schema_correction(bad_response: str, error: str) -> AIRoadmap:
    client = _get_client()
    correction_prompt = f"""Your previous response could not be parsed as valid JSON.
Error: {error}

Previous response (first 500 chars):
{bad_response[:500]}

Please return ONLY valid JSON matching this exact schema — no preamble, no markdown fences:

{OUTPUT_SCHEMA}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": correction_prompt}],
    )
    raw = response.content[0].text.strip()
    data = json.loads(raw)
    roadmap = AIRoadmap(**data)
    roadmap.use_cases.sort(key=lambda uc: uc.priority_score, reverse=True)
    return roadmap


def _log_feasibility(ctx: dict) -> None:
    logger.info("=== Pre-Scoring Feasibility Context ===")
    logger.info(f"  Composite modifier: {ctx['composite_modifier']:+.1f}")
    logger.info(f"  Data modifier:      {ctx['data_modifier']:+d}")
    logger.info(f"  Team modifier:      {ctx['team_modifier']:+d}")
    logger.info(f"  Compliance penalty: {ctx['compliance_penalty']:+.1f}")
    for flag in ctx["flags"]:
        logger.warning(f"  FLAG: {flag}")
    for note in ctx["notes"]:
        logger.info(f"  NOTE: {note}")
    logger.info("=======================================")
