"""Agent that proposes remediation options for the diagnosed issue."""

from __future__ import annotations

from textwrap import dedent
from typing import Sequence

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser


class SolutionDesignerAgent:
    """Generate multiple, distinct remediation strategies for the error."""

    def __init__(self, llm: BaseChatModel) -> None:
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    dedent(
                        """
                        You are an expert software engineer helping a teammate fix a bug.
                        Suggest at least two concrete, distinct strategies to resolve the issue.
                        Each strategy should include a short rationale and point to relevant code when possible.
                        """
                    ).strip(),
                ),
                (
                    "human",
                    dedent(
                        """
                        Error details:
                        {error_description}

                        Diagnostic summary:
                        {explanation}

                        Code context:
                        {context}

                        Produce your answer as a numbered list where each item is a candidate solution.
                        """
                    ).strip(),
                ),
            ]
        )
        self._chain = template | llm | StrOutputParser()

    def propose_solutions(
        self,
        error_description: str,
        explanation: str,
        documents: Sequence[Document],
    ) -> str:
        """Return candidate fixes for the identified issue."""

        context = ErrorContextFormatter.format_documents(documents)
        return self._chain.invoke(
            {
                "error_description": error_description.strip(),
                "explanation": explanation.strip(),
                "context": context,
            }
        )


class ErrorContextFormatter:
    """Helpers for formatting retrieved documents."""

    @staticmethod
    def format_documents(documents: Sequence[Document]) -> str:
        if not documents:
            return "No supporting context retrieved."

        formatted = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown file")
            formatted.append(f"[{source}]\n{doc.page_content}")
        return "\n\n".join(formatted)

