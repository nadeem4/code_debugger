"""Utilities for vectorizing a codebase for retrieval-augmented debugging."""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


LOGGER = logging.getLogger(__name__)


@dataclass
class RetrievalBundle:
    """Container for the vector store and retriever."""

    vector_store: Chroma
    retriever: BaseRetriever


@dataclass
class CodebaseVectorizer:
    """Builds and maintains a vector store over a repository's source code."""

    embeddings: Embeddings
    persist_directory: Path
    include_globs: Sequence[str] = field(
        default_factory=lambda: ("**/*.py", "**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx")
    )
    exclude_globs: Sequence[str] = field(
        default_factory=lambda: ("**/__pycache__/**", "**/node_modules/**", "**/.git/**")
    )
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 6

    def index_codebase(self, root: Path, reindex: bool = False) -> RetrievalBundle:
        """Index the target codebase and return a retriever bundle."""

        if not root.exists():
            raise FileNotFoundError(f"Codebase path '{root}' does not exist.")

        root = root.resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        vector_store = None
        if not reindex and (self.persist_directory / "chroma.sqlite3").exists():
            LOGGER.info("Loading existing Chroma vector store from %s", self.persist_directory)
            vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
            )

        if vector_store is None:
            LOGGER.info("Rebuilding vector store from source files in %s", root)
            documents = self._load_documents(root)
            LOGGER.debug("Loaded %s source documents", len(documents))
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            LOGGER.debug("Split into %s chunks", len(chunks))
            vector_store = Chroma.from_documents(
                chunks,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
        vector_store.persist()

        retriever = vector_store.as_retriever(search_kwargs={"k": self.retriever_k})
        return RetrievalBundle(vector_store=vector_store, retriever=retriever)

    def _load_documents(self, root: Path) -> List[Document]:
        """Collect and load source documents under the root directory."""

        documents: List[Document] = []
        for file_path in self._iter_source_files(root):
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                for doc in loader.load():
                    doc.metadata["source"] = str(file_path.relative_to(root))
                    documents.append(doc)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Skipping %s due to error: %s", file_path, exc)
        return documents

    def _iter_source_files(self, root: Path) -> Iterable[Path]:
        """Yield file paths that match the include patterns while respecting exclusions."""

        all_files = root.rglob("*")
        included: List[Path] = []
        for path in all_files:
            if path.is_dir():
                continue
            rel = path.relative_to(root)
            if not self._matches_any(rel, self.include_globs):
                continue
            if self._matches_any(rel, self.exclude_globs):
                continue
            included.append(path)
        return included

    @staticmethod
    def _matches_any(path: Path, patterns: Sequence[str]) -> bool:
        """Check if a path matches any glob patterns."""

        path_str = str(path).replace("\\", "/")
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in patterns)

