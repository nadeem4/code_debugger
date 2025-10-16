"""LangGraph workflow that coordinates the debugging agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, TypedDict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from src.agents import ErrorExplainerAgent, SolutionDesignerAgent


class DebuggerState(TypedDict, total=False):
    """Shared state passed between workflow nodes."""

    error_description: str
    context_documents: List[Document]
    explanation: str
    solution_suggestions: str
    report: Dict[str, object]


@dataclass
class DebuggerWorkflow:
    """Coordinate retrieval, diagnosis, and remediation suggestions."""

    retriever: BaseRetriever
    explainer: ErrorExplainerAgent
    designer: SolutionDesignerAgent

    def __post_init__(self) -> None:
        graph = StateGraph(DebuggerState)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("generate_explanation", self._generate_explanation)
        graph.add_node("design_solutions", self._design_solutions)
        graph.add_node("finalize_report", self._finalize_report)

        graph.add_edge(START, "retrieve_context")
        graph.add_edge("retrieve_context", "generate_explanation")
        graph.add_edge("generate_explanation", "design_solutions")
        graph.add_edge("design_solutions", "finalize_report")
        graph.add_edge("finalize_report", END)

        self._app = graph.compile()

    def run(self, error_description: str) -> Dict[str, object]:
        """Execute the debugger workflow and return the structured report."""

        state: DebuggerState = {"error_description": error_description.strip()}
        final_state: DebuggerState = self._app.invoke(state)
        return final_state["report"]

    def stream(self, error_description: str):
        """Yield intermediate updates while running the workflow."""

        state: DebuggerState = {"error_description": error_description.strip()}
        yield from self._app.stream(state)

    # Node implementations -------------------------------------------------

    def _retrieve_context(self, state: DebuggerState) -> DebuggerState:
        documents = self.retriever.get_relevant_documents(state["error_description"])
        return {"context_documents": documents}

    def _generate_explanation(self, state: DebuggerState) -> DebuggerState:
        explanation = self.explainer.explain(
            state["error_description"], state.get("context_documents", [])
        )
        return {"explanation": explanation}

    def _design_solutions(self, state: DebuggerState) -> DebuggerState:
        solutions = self.designer.propose_solutions(
            state["error_description"],
            state["explanation"],
            state.get("context_documents", []),
        )
        return {"solution_suggestions": solutions}

    def _finalize_report(self, state: DebuggerState) -> DebuggerState:
        suggestions = self._parse_numbered_list(state["solution_suggestions"])
        report = {
            "error": state["error_description"],
            "explanation": state["explanation"],
            "solutions": suggestions,
            "raw_solution_text": state["solution_suggestions"],
        }
        return {"report": report}

    @staticmethod
    def _parse_numbered_list(output: str) -> List[str]:
        """Extract numbered list items from the LLM output."""

        pattern = re.compile(r"^\s*\d+\.\s+(.*)$")
        items: List[str] = []
        for line in output.splitlines():
            match = pattern.match(line)
            if match:
                items.append(match.group(1).strip())
        # Fall back to entire response if parsing failed.
        if not items and output.strip():
            items.append(output.strip())
        return items
