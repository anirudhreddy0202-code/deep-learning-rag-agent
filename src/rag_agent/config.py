"""
config.py
=========
Centralised configuration and LLM provider factory.

All settings are loaded from environment variables via pydantic-settings.
No magic numbers or hardcoded values should exist anywhere else in the
codebase — import from here instead.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    """Supported LLM backend providers."""

    GROQ = "groq"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class EmbeddingProvider(str, Enum):
    """Supported embedding backend providers."""

    LOCAL = "local"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All fields map directly to entries in .env.example.
    Validation is handled automatically by pydantic-settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider
    llm_provider: LLMProvider = LLMProvider.GROQ
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        alias="GROQ_MODEL",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_MODEL",
    )

    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        alias="LMSTUDIO_BASE_URL",
    )
    lmstudio_model: str = Field(
        default="local-model",
        alias="LMSTUDIO_MODEL",
    )

    # Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # Vector store
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        alias="CHROMA_DB_PATH",
    )
    chroma_collection_name: str = Field(
        default="deep_learning_corpus",
        alias="CHROMA_COLLECTION_NAME",
    )

    # Retrieval
    retrieval_k: int = Field(
        default=4,
        alias="RETRIEVAL_K",
    )
    similarity_threshold: float = Field(
        default=0.3,
        alias="SIMILARITY_THRESHOLD",
    )
    max_context_tokens: int = Field(
        default=3000,
        alias="MAX_CONTEXT_TOKENS",
    )

    # Application
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )
    app_title: str = Field(
        default="Deep Learning Interview Prep Agent",
        alias="APP_TITLE",
    )
    corpus_dir: str = Field(
        default="./data/corpus",
        alias="CORPUS_DIR",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Uses lru_cache so the .env file is only parsed once per process.
    Call get_settings() anywhere in the codebase to access configuration.
    """
    return Settings()


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------


class LLMFactory:
    """
    Creates and returns a configured LangChain chat model based on the
    active LLM_PROVIDER environment variable.

    Swapping providers requires only a .env change — no code changes.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(self) -> BaseChatModel:
        """
        Instantiate and return the configured chat model.

        Returns
        -------
        BaseChatModel
            A LangChain-compatible chat model ready for use in chains
            and LangGraph nodes.

        Raises
        ------
        ValueError
            If the configured provider is not supported.

        EnvironmentError
            If required credentials are missing for the chosen provider.
        """
        provider = self._settings.llm_provider

        if provider == LLMProvider.GROQ:
            return self._create_groq()

        elif provider == LLMProvider.OLLAMA:
            return self._create_ollama()

        elif provider == LLMProvider.LMSTUDIO:
            return self._create_lmstudio()

        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}"
            )

    def _create_groq(self) -> BaseChatModel:
        """
        Create a Groq-backed chat model.

        Requires GROQ_API_KEY in environment.
        """
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=self._settings.groq_model,
            api_key=self._settings.groq_api_key,
        )

    def _create_ollama(self) -> BaseChatModel:
        """
        Create a locally-hosted Ollama chat model.

        No API key required. Ollama must be running:
        `ollama serve`
        """
        raise NotImplementedError

    def _create_lmstudio(self) -> BaseChatModel:
        """
        Create an LM Studio chat model via its OpenAI-compatible
        local server.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Embedding Factory
# ---------------------------------------------------------------------------


class EmbeddingFactory:
    """
    Creates and returns a configured LangChain embedding model.

    Local embeddings via sentence-transformers require no API key and
    run entirely on CPU — appropriate for development and class use.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(self):
        """
        Instantiate and return the configured embedding model.

        Returns
        -------
        Embeddings
            A LangChain-compatible embedding model.

        Raises
        ------
        ValueError
            If the configured provider is not supported.
        """
        provider = self._settings.embedding_provider

        if provider == EmbeddingProvider.LOCAL:
            return self._create_local()

        elif provider == EmbeddingProvider.OPENAI:
            return self._create_openai()

        else:
            raise ValueError(
                f"Unsupported embedding provider: {provider}"
            )

    def _create_local(self):
        """
        Create a local sentence-transformers embedding model.

        The model is controlled by EMBEDDING_MODEL in .env.
        """
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model
        )

    def _create_openai(self):
        """
        Create an OpenAI embedding model.

        Not required for the workshop because the configured
        embedding provider is LOCAL.
        """
        raise NotImplementedError