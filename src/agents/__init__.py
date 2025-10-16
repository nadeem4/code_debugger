"""Agent definitions for the LangGraph code debugger."""

from .error_explainer import ErrorExplainerAgent
from .solution_designer import SolutionDesignerAgent

__all__ = ["ErrorExplainerAgent", "SolutionDesignerAgent"]

