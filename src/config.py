"""Configuration helpers for the LangGraph code debugger."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()


@dataclass
class ModelConfig:
    """LLM configuration details."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 2


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = "text-embedding-3-large"
    chunk_size: int = 1000
    chunk_overlap: int = 200


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch an environment variable with an optional default."""

    value = os.getenv(key, default)
    if value is None:
        raise RuntimeError(
            f"Environment variable '{key}' is required but was not provided."
        )
    return value


def create_chat_model(config: Optional[ModelConfig] = None) -> ChatOpenAI:
    """Instantiate the chat model used by agents."""

    cfg = config or ModelConfig()
    return ChatOpenAI(
        model=cfg.model,
        temperature=cfg.temperature,
        max_retries=cfg.max_retries,
    )


def create_embeddings(config: Optional[EmbeddingConfig] = None) -> OpenAIEmbeddings:
    """Instantiate the embedding model used for code vectorization."""

    cfg = config or EmbeddingConfig()
    return OpenAIEmbeddings(model=cfg.model)

