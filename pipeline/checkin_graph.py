from typing import Any

from langgraph.graph import END, StateGraph

from agent.models import CheckinInput, EvolvedAIRoadmap
from pipeline.nodes import (
    classify_changes,
    extract_checkin_deltas,
    generate_evolved_plan,
    load_previous_plan,
    reassess_use_cases,
    update_milestones,
)


# State dict keys — typed for clarity
class CheckinState(dict):
    """Mutable state dict passed through the LangGraph pipeline."""


def build_checkin_graph() -> Any:
    graph: StateGraph = StateGraph(dict)

    graph.add_node("load_previous_plan", load_previous_plan)
    graph.add_node("extract_checkin_deltas", extract_checkin_deltas)
    graph.add_node("classify_changes", classify_changes)
    graph.add_node("reassess_use_cases", reassess_use_cases)
    graph.add_node("update_milestones", update_milestones)
    graph.add_node("generate_evolved_plan", generate_evolved_plan)

    graph.set_entry_point("load_previous_plan")
    graph.add_edge("load_previous_plan", "extract_checkin_deltas")
    graph.add_edge("extract_checkin_deltas", "classify_changes")
    graph.add_edge("classify_changes", "reassess_use_cases")
    graph.add_edge("reassess_use_cases", "update_milestones")
    graph.add_edge("update_milestones", "generate_evolved_plan")
    graph.add_edge("generate_evolved_plan", END)

    return graph.compile()


def run_checkin(checkin_input: CheckinInput) -> EvolvedAIRoadmap:
    compiled = build_checkin_graph()
    initial_state: dict = {
        "checkin_input": checkin_input,
        "previous_roadmap": None,
        "previous_profile": {},
        "deltas": {},
        "classified_changes": {},
        "reassessed_use_cases": [],
        "dropped_use_cases": [],
        "shipped_use_cases": [],
        "updated_milestones": [],
        "milestone_shifts": [],
        "evolved_roadmap": None,
    }
    final_state = compiled.invoke(initial_state)
    return final_state["evolved_roadmap"]
