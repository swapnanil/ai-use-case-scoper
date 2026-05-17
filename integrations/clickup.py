import httpx

from agent.models import AIRoadmap

CLICKUP_API_BASE = "https://api.clickup.com/api/v2"


def export_to_clickup(roadmap: AIRoadmap, list_id: str, api_token: str) -> dict:
    """Export a roadmap to ClickUp: one task per deliverable, tagged by phase."""
    headers = {"Authorization": api_token, "Content-Type": "application/json"}
    task_ids: list[str] = []

    with httpx.Client(base_url=CLICKUP_API_BASE, headers=headers, timeout=30) as client:
        for milestone in roadmap.roadmap_90_day:
            for deliverable in milestone.deliverables:
                task_id = _create_task(client, list_id, deliverable, milestone)
                task_ids.append(task_id)

    return {"task_ids": task_ids}


def _create_task(client: httpx.Client, list_id: str, deliverable: str, milestone) -> str:
    payload = {
        "name": deliverable[:255],
        "description": (
            f"Phase: {milestone.phase} | {milestone.week}\n\n"
            f"Success criteria: {milestone.success_criteria}"
        ),
        "tags": [milestone.phase.lower().replace(" ", "-")[:50]],
    }
    resp = client.post(f"/list/{list_id}/task", json=payload)
    resp.raise_for_status()
    return resp.json()["id"]
