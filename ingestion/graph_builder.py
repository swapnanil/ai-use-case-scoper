import json
import logging
import os

import networkx as nx
from anthropic import Anthropic

from agent.prompts import ENTITY_EXTRACTION_SYSTEM_PROMPT
from ingestion.loader import DocumentChunk

logger = logging.getLogger(__name__)

_client: Anthropic | None = None
BATCH_SIZE = 5  # chunks per LLM call
MAX_CHARS_PER_BATCH = 6000


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def build_graph(chunks: list[DocumentChunk], model: str) -> nx.DiGraph:
    G = nx.DiGraph()
    client = _get_client()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        combined = "\n\n---\n\n".join(c.text for c in batch)[:MAX_CHARS_PER_BATCH]
        extracted = _extract_entities(combined, client, model)
        _merge_into_graph(G, extracted)
        logger.debug(f"Processed chunks {i}–{i + len(batch)}, graph now has {G.number_of_nodes()} nodes")

    return G


def _extract_entities(text: str, client: Anthropic, model: str) -> dict:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=ENTITY_EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Extract entities from this document excerpt:\n\n{text}"}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Entity extraction failed for batch: {e}")
        return _empty_extraction()


def _merge_into_graph(G: nx.DiGraph, extracted: dict) -> None:
    entities = extracted.get("entities", {})

    for system in entities.get("systems", []):
        name = system.get("name")
        if name:
            if G.has_node(name):
                # Merge: prefer documented=True and legacy=True signals
                existing = G.nodes[name]
                G.nodes[name]["documented"] = existing.get("documented", True) and system.get("documented", True)
                G.nodes[name]["legacy"] = existing.get("legacy", False) or system.get("legacy", False)
            else:
                G.add_node(name, node_type="System", **system)

    for team in entities.get("teams", []):
        name = team.get("name")
        if name:
            G.add_node(name, node_type="Team", **team)

    for ds in entities.get("data_sources", []):
        name = ds.get("name")
        if name:
            G.add_node(name, node_type="DataSource", **ds)

    for comp in entities.get("compliance_requirements", []):
        name = comp.get("name")
        if name:
            G.add_node(name, node_type="ComplianceRequirement", **comp)

    for ts in entities.get("tech_stack", []):
        item = ts.get("item")
        if item:
            G.add_node(item, node_type="TechStack", **ts)

    for rel in extracted.get("relationships", []):
        src, dst = rel.get("from"), rel.get("to")
        if src and dst and G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst, relationship=rel.get("type", "RELATED_TO"))


def _empty_extraction() -> dict:
    return {
        "entities": {
            "systems": [], "teams": [], "data_sources": [],
            "compliance_requirements": [], "tech_stack": [],
        },
        "relationships": [],
    }


def graph_to_dict(G: nx.DiGraph) -> dict:
    return {
        "nodes": [{"id": n, **d} for n, d in G.nodes(data=True)],
        "edges": [{"from": u, "to": v, **d} for u, v, d in G.edges(data=True)],
    }
