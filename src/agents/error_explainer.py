"""Agent responsible for translating technical errors into plain English explanations."""

from __future__ import annotations

from textwrap import dedent
from typing import Sequence

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser


class ErrorExplainerAgent:
    """Use an LLM to craft an accessible explanation of the failure."""

    def __init__(self, llm: BaseChatModel) -> None:
        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    dedent(
                        """
                        You are a senior software engineer.
                        Explain technical issues to developers of varying experience in concise, plain English.
                        Avoid restating the error verbatim unless needed for clarity.
                        """
                    ).strip(),
                ),
                (
                    "human",
                    dedent(
                        """
                        Error details:
                        {error_description}

                        Relevant code context:
                        {context}

                        Provide a brief explanation (3-5 sentences) that describes why this error happens.
                        """
                    ).strip(),
                ),
            ]
        )
        self._chain = template | llm | StrOutputParser()

    def explain(self, error_description: str, documents: Sequence[Document]) -> str:
        """Produce a natural-language explanation for the observed error."""

        context = self._summarize_documents(documents)
        return self._chain.invoke(
            {"error_description": error_description.strip(), "context": context}
        )

    @staticmethod
    def _summarize_documents(documents: Sequence[Document]) -> str:
        """Format retrieved documents into a compact prompt context."""

        if not documents:
            return "No supporting context retrieved."

        formatted = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown file")
            formatted.append(f"[{source}]\n{doc.page_content}")
        return "\n\n".join(formatted)

