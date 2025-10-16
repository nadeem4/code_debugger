"""CLI entry point for the LangGraph-powered code debugger."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.agents import ErrorExplainerAgent, SolutionDesignerAgent
from src.config import EmbeddingConfig, ModelConfig, create_chat_model, create_embeddings
from src.retrieval import CodebaseVectorizer
from src.workflows import DebuggerWorkflow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("debugger")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph-powered code debugger.")
    parser.add_argument(
        "--codebase",
        type=Path,
        required=True,
        help="Path to the repository or directory that should be indexed.",
    )
    parser.add_argument(
        "--error",
        required=True,
        help="Error message, stack trace, or failure description to analyze.",
    )
    parser.add_argument(
        "--persist",
        type=Path,
        default=Path("./vector_store"),
        help="Directory used to persist the vector store.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force a rebuild of the vector store even if a persisted copy exists.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of context chunks to retrieve for each query.",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Override the default chat model identifier.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Override the default embedding model identifier.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the chat model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    default_model_cfg = ModelConfig()
    model_cfg = ModelConfig(
        model=args.chat_model or default_model_cfg.model,
        temperature=args.temperature,
    )
    default_embedding_cfg = EmbeddingConfig()
    embedding_cfg = EmbeddingConfig(
        model=args.embedding_model or default_embedding_cfg.model,
    )

    LOGGER.info("Loading models...")
    llm = create_chat_model(model_cfg)
    embeddings = create_embeddings(embedding_cfg)

    LOGGER.info("Indexing codebase at %s", args.codebase)
    vectorizer = CodebaseVectorizer(
        embeddings=embeddings,
        persist_directory=args.persist,
        retriever_k=args.top_k,
    )
    retrieval_bundle = vectorizer.index_codebase(
        root=args.codebase, reindex=args.reindex
    )

    LOGGER.info("Building multi-agent workflow...")
    explainer = ErrorExplainerAgent(llm)
    designer = SolutionDesignerAgent(llm)
    workflow = DebuggerWorkflow(
        retriever=retrieval_bundle.retriever,
        explainer=explainer,
        designer=designer,
    )

    LOGGER.info("Running debugger workflow...")
    report = workflow.run(error_description=args.error)

    print("\n=== Debugger Report ===")
    print(f"Error:\n{report['error']}\n")
    print("Explanation:")
    print(report["explanation"])
    print("\nCandidate Solutions:")
    for idx, suggestion in enumerate(report["solutions"], start=1):
        print(f"{idx}. {suggestion}")


if __name__ == "__main__":
    main()
