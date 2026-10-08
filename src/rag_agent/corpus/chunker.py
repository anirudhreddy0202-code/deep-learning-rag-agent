"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports Markdown files for the workshop. Chunking strategy uses
    Markdown header splitting followed by recursive character splitting.
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.
        """

        file_path = Path(file_path)

        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        # Validate chunk settings
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        # Route Markdown files to the Markdown chunker
        suffix = file_path.suffix.lower()

        if suffix in {".md", ".markdown"}:
            raw_chunks = self._chunk_markdown(
                file_path,
                chunk_size,
                chunk_overlap,
            )
        elif suffix == ".pdf":
            raise ValueError(
                "PDF ingestion is not required for this workshop. "
                "Please use a Markdown (.md) file."
            )
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                "Please use a Markdown (.md) file."
            )

        # Infer metadata
        metadata = self._infer_metadata(
            file_path,
            metadata_overrides,
        )

        document_chunks: list[DocumentChunk] = []

        for raw_chunk in raw_chunks:
            chunk_text = raw_chunk["text"].strip()

            if not chunk_text:
                continue

            # Generate deterministic ID
            chunk_id = VectorStoreManager.generate_chunk_id(
                metadata.source,
                chunk_text,
            )

            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=metadata,
                )
            )

        logger.info(
            "Produced {} chunks from {}",
            len(document_chunks),
            file_path.name,
        )

        return document_chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files.

        This helper is not required by the workshop, but is included
        for completeness.
        """

        all_chunks: list[DocumentChunk] = []

        for file_path in file_paths:
            try:
                chunks = self.chunk_file(
                    file_path,
                    metadata_overrides=metadata_overrides,
                )
                all_chunks.extend(chunks)

            except Exception as exc:
                logger.error(
                    "Failed to chunk {}: {}",
                    file_path,
                    exc,
                )

        return all_chunks

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        PDF ingestion is not required for this workshop.
        """

        raise NotImplementedError(
            "PDF ingestion is not required for this workshop."
        )

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Markdown headers are used first to preserve document structure.
        Oversized sections are then recursively split.
        """

        # Read Markdown file
        text = file_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            logger.warning(
                "Markdown file is empty: {}",
                file_path,
            )
            return []

        # Split according to Markdown headers
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

        sections = markdown_splitter.split_text(text)

        # Recursively split sections that are too large
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        chunks: list[dict] = []

        for section in sections:
            section_text = section.page_content.strip()

            if not section_text:
                continue

            # Find the most specific header
            header = ""

            for header_name in (
                "Header 3",
                "Header 2",
                "Header 1",
            ):
                if section.metadata.get(header_name):
                    header = section.metadata[header_name]
                    break

            split_documents = recursive_splitter.split_documents(
                [section]
            )

            for document in split_documents:
                chunk_text = document.page_content.strip()

                if not chunk_text:
                    continue

                chunks.append(
                    {
                        "text": chunk_text,
                        "header": header,
                    }
                )

        logger.info(
            "Created {} Markdown chunks from {}",
            len(chunks),
            file_path.name,
        )

        return chunks

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer metadata from the filename and apply overrides.

        Recommended filename:

            <topic>_<difficulty>.md

        Examples:

            lstm_intermediate.md
            cnn_advanced.md
            ann_beginner.md
        """

        stem = file_path.stem

        # Default values
        topic = stem
        difficulty = "intermediate"

        # Parse topic_difficulty filename format
        if "_" in stem:
            possible_topic, possible_difficulty = stem.rsplit(
                "_",
                1,
            )

            if possible_topic.strip():
                topic = possible_topic.strip()

            if possible_difficulty.strip():
                difficulty = possible_difficulty.strip().lower()

        # Normalize known topic names
        known_topics = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "som": "SOM",
            "boltzmannmachine": "BoltzmannMachine",
            "gan": "GAN",
        }

        topic = known_topics.get(
            topic.lower(),
            topic,
        )

        # Build metadata
        metadata = {
            "topic": topic,
            "difficulty": difficulty,
            "type": "concept_explanation",
            "source": file_path.name,
            "related_topics": [],
            "is_bonus": topic in {
                "SOM",
                "BoltzmannMachine",
                "GAN",
            },
        }

        # Apply explicit overrides
        if overrides:
            metadata.update(overrides)

        return ChunkMetadata(
            topic=metadata["topic"],
            difficulty=metadata["difficulty"],
            type=metadata["type"],
            source=metadata["source"],
            related_topics=metadata["related_topics"],
            is_bonus=metadata["is_bonus"],
        )