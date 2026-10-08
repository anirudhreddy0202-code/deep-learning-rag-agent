"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.
        """

        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(db_path)
            )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "ChromaDB initialized: collection='{}', items={}",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )

        except Exception as exc:
            logger.exception("Failed to initialize ChromaDB")
            raise RuntimeError(
                f"Could not initialize ChromaDB: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.
        """

        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists.
        """

        if self._collection is None:
            raise RuntimeError("ChromaDB collection is not initialized.")

        result = self._collection.get(ids=[chunk_id])

        ids = result.get("ids", [])

        return chunk_id in ids

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.
        """

        result = IngestionResult()

        if not chunks:
            return result

        if self._collection is None:
            raise RuntimeError("ChromaDB collection is not initialized.")

        new_chunks: list[DocumentChunk] = []

        # Check for duplicates first
        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    logger.debug(
                        "Skipping duplicate chunk: {}",
                        chunk.chunk_id,
                    )
                    continue

                new_chunks.append(chunk)

            except Exception as exc:
                error_message = (
                    f"Duplicate check failed for "
                    f"{chunk.chunk_id}: {exc}"
                )
                result.errors.append(error_message)
                logger.error(error_message)

        # Process new chunks in batches
        batch_size = 100

        for start in range(0, len(new_chunks), batch_size):
            batch = new_chunks[start : start + batch_size]

            try:
                texts = [chunk.chunk_text for chunk in batch]

                embeddings = self._embeddings.embed_documents(texts)

                ids = [chunk.chunk_id for chunk in batch]

                documents = [chunk.chunk_text for chunk in batch]

                metadatas = [
                    chunk.metadata.to_dict()
                    for chunk in batch
                ]

                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

                result.ingested += len(batch)

                for chunk in batch:
                    source = chunk.metadata.source
                    if source not in result.document_ids:
                        result.document_ids.append(source)

            except Exception as exc:
                logger.error(
                    "Batch ingestion failed: {}",
                    exc,
                )

                # Try chunks individually so one bad chunk does not
                # prevent the remaining chunks from being ingested.
                for chunk in batch:
                    try:
                        embedding = self._embeddings.embed_documents(
                            [chunk.chunk_text]
                        )[0]

                        self._collection.upsert(
                            ids=[chunk.chunk_id],
                            embeddings=[embedding],
                            documents=[chunk.chunk_text],
                            metadatas=[chunk.metadata.to_dict()],
                        )

                        result.ingested += 1

                        source = chunk.metadata.source
                        if source not in result.document_ids:
                            result.document_ids.append(source)

                    except Exception as chunk_exc:
                        error_message = (
                            f"Failed to ingest chunk "
                            f"{chunk.chunk_id}: {chunk_exc}"
                        )
                        result.errors.append(error_message)
                        logger.error(error_message)

        logger.info(
            "Ingestion complete: ingested={}, skipped={}, errors={}",
            result.ingested,
            result.skipped,
            len(result.errors),
        )

        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.
        """

        if self._collection is None:
            raise RuntimeError("ChromaDB collection is not initialized.")

        k = k or self._settings.retrieval_k

        if self._collection.count() == 0:
            return []

        query_embedding = self._embeddings.embed_query(query_text)

        # Build ChromaDB metadata filter
        conditions = []

        if topic_filter:
            conditions.append({"topic": topic_filter})

        if difficulty_filter:
            conditions.append({"difficulty": difficulty_filter})

        if len(conditions) == 0:
            where_filter = None
        elif len(conditions) == 1:
            where_filter = conditions[0]
        else:
            where_filter = {"$and": conditions}

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, self._collection.count()),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where_filter is not None:
            query_kwargs["where"] = where_filter

        results = self._collection.query(**query_kwargs)

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            # For cosine distance:
            # similarity = 1 - distance
            score = max(0.0, min(1.0, 1.0 - float(distance)))

            if score < self._settings.similarity_threshold:
                continue

            chunk_metadata = ChunkMetadata.from_dict(metadata)

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=chunk_metadata,
                    score=score,
                )
            )

        retrieved.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return retrieved

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.
        """

        if self._collection is None:
            raise RuntimeError("ChromaDB collection is not initialized.")

        if self._collection.count() == 0:
            return []

        results = self._collection.get(
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        documents: dict[str, dict] = {}

        for metadata in metadatas:
            if not metadata:
                continue

            source = metadata.get("source", "")
            topic = metadata.get("topic", "")

            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": topic,
                    "chunk_count": 0,
                }

            documents[source]["chunk_count"] += 1

        return sorted(
            documents.values(),
            key=lambda item: item["source"].lower(),
        )

    def get_document_chunks(
        self,
        source: str,
    ) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.
        """

        if self._collection is None:
            raise RuntimeError("ChromaDB collection is not initialized.")

        results = self._collection.get(
            where={"source": source},
            include=[
                "documents",
                "metadatas",
            ],
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        chunks: list[DocumentChunk] = []

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
        ):
            if not metadata or document is None:
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=ChunkMetadata.from_dict(metadata),
                )
            )

        # ChromaDB does not have a guaranteed document position field
        # in the current metadata model, so preserve returned order.
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.
        """

        if self._collection is None:
            raise RuntimeError("ChromaDB collection is not initialized.")

        total_chunks = self._collection.count()

        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "topics": [],
                "sources": [],
                "bonus_topics_present": False,
            }

        results = self._collection.get(
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        topics = set()
        sources = set()
        bonus_topics_present = False

        for metadata in metadatas:
            if not metadata:
                continue

            topic = metadata.get("topic")
            source = metadata.get("source")

            if topic:
                topics.add(topic)

            if source:
                sources.add(source)

            is_bonus = str(
                metadata.get("is_bonus", "false")
            ).lower() == "true"

            if is_bonus:
                bonus_topics_present = True

        return {
            "total_chunks": total_chunks,
            "topics": sorted(topics),
            "sources": sorted(sources),
            "bonus_topics_present": bonus_topics_present,
        }

    # -----------------------------------------------------------------------
    # Document Deletion
    # -----------------------------------------------------------------------

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.
        """

        if self._collection is None:
            raise RuntimeError("ChromaDB collection is not initialized.")

        existing = self._collection.get(
            where={"source": source},
            include=[],
        )

        ids = existing.get("ids", [])

        if not ids:
            return 0

        self._collection.delete(ids=ids)

        logger.info(
            "Deleted {} chunks from source '{}'",
            len(ids),
            source,
        )

        return len(ids)