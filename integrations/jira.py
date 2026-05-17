import httpx

from agent.models import AIRoadmap


def export_to_jira(
    roadmap: AIRoadmap,
    jira_base_url: str,
    project_key: str,
    api_token: str,
    user_email: str,
) -> dict:
    """Export a roadmap to Jira: one Epic per phase, one Story per deliverable."""
    auth = (user_email, api_token)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    epic_keys: list[str] = []
    story_keys: list[str] = []

    # Group milestones by phase
    phases: dict[str, list] = {}
    for milestone in roadmap.roadmap_90_day:
        phases.setdefault(milestone.phase, []).append(milestone)

    with httpx.Client(base_url=jira_base_url.rstrip("/"), auth=auth, headers=headers, timeout=30) as client:
        for phase, milestones in phases.items():
            epic_key = _create_epic(client, project_key, phase)
            epic_keys.append(epic_key)

            for milestone in milestones:
                for deliverable in milestone.deliverables:
                    story_key = _create_story(
                        client, project_key, epic_key, deliverable, milestone
                    )
                    story_keys.append(story_key)

    return {"epic_keys": epic_keys, "story_keys": story_keys}


def _create_epic(client: httpx.Client, project_key: str, phase: str) -> str:
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": f"AI Roadmap: {phase}",
            "issuetype": {"name": "Epic"},
            "description": _adf_paragraph(f"AI adoption roadmap phase: {phase}"),
        }
    }
    resp = client.post("/rest/api/3/issue", json=payload)
    resp.raise_for_status()
    return resp.json()["key"]


def _create_story(client: httpx.Client, project_key: str, epic_key: str, deliverable: str, milestone) -> str:
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": deliverable[:255],
            "issuetype": {"name": "Story"},
            "description": _adf_paragraph(
                f"{milestone.week} | Success criteria: {milestone.success_criteria}"
            ),
            "customfield_10014": epic_key,  # Epic Link field (standard Jira field)
        }
    }
    resp = client.post("/rest/api/3/issue", json=payload)
    resp.raise_for_status()
    return resp.json()["key"]


def _adf_paragraph(text: str) -> dict:
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}],
    }
