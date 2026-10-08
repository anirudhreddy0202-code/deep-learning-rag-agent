"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

from rag_agent.agent.prompts import SYSTEM_PROMPT
from rag_agent.agent.state import AgentResponse, AgentState
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Backend Resources
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_llm():
    """Create one shared LLM instance."""
    settings = get_settings()
    return LLMFactory(settings).create()


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStoreManager:
    """Create one shared vector store manager."""
    return VectorStoreManager()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_state_value(state, key, default=None):
    """
    Read a value from either a dictionary-style LangGraph state
    or an object-style AgentState.

    This makes the nodes compatible with both representations.
    """
    if isinstance(state, dict):
        return state.get(key, default)

    return getattr(state, key, default)


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.
    """

    messages = _get_state_value(state, "messages", [])

    original_query = ""

    # Find the most recent human message.
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            original_query = str(message.content)
            break

    if not original_query:
        return {
            "original_query": "",
            "rewritten_query": "",
        }

    try:
        llm = _get_llm()

        rewrite_prompt = (
            "Rewrite the user's question into a short, keyword-rich query "
            "for semantic vector search over deep learning study notes. "
            "Keep the important technical terms. Do not answer the question. "
            "Return only the rewritten search query.\n\n"
            f"User question: {original_query}"
        )

        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a query rewriting assistant for a RAG system."
                ),
                HumanMessage(content=rewrite_prompt),
            ]
        )

        rewritten = str(response.content).strip()

        if not rewritten:
            rewritten = original_query

        return {
            "original_query": original_query,
            "rewritten_query": rewritten,
        }

    except Exception:
        # If query rewriting fails, use the original question.
        return {
            "original_query": original_query,
            "rewritten_query": original_query,
        }


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB.
    """

    rewritten_query = _get_state_value(state, "rewritten_query", "")
    original_query = _get_state_value(state, "original_query", "")

    query = rewritten_query or original_query

    if not query:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }

    try:
        store = _get_vector_store()

        topic_filter = _get_state_value(state, "topic_filter", None)
        difficulty_filter = _get_state_value(
            state,
            "difficulty_filter",
            None,
        )

        chunks = store.query(
            query_text=query,
            topic_filter=topic_filter,
            difficulty_filter=difficulty_filter,
        )

        if not chunks:
            return {
                "retrieved_chunks": [],
                "no_context_found": True,
            }

        return {
            "retrieved_chunks": chunks,
            "no_context_found": False,
        }

    except Exception:
        return {
            "retrieved_chunks": [],
            "no_context_found": True,
        }


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.
    """

    settings = get_settings()

    no_context_found = _get_state_value(
        state,
        "no_context_found",
        False,
    )

    retrieved_chunks = _get_state_value(
        state,
        "retrieved_chunks",
        [],
    )

    rewritten_query = _get_state_value(
        state,
        "rewritten_query",
        "",
    )

    original_query = _get_state_value(
        state,
        "original_query",
        "",
    )

    messages = _get_state_value(
        state,
        "messages",
        [],
    )

    # -----------------------------------------------------------------------
    # Hallucination Guard
    # -----------------------------------------------------------------------

    if no_context_found or not retrieved_chunks:
        no_context_message = (
            "I was unable to find relevant information in the corpus for "
            "your query. Please try a more specific question based on the "
            "uploaded deep learning study material."
        )

        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=rewritten_query,
        )

        return {
            "final_response": response,
            "messages": [
                AIMessage(content=no_context_message)
            ],
        }

    # -----------------------------------------------------------------------
    # Build Retrieved Context
    # -----------------------------------------------------------------------

    context_parts = []
    sources = []
    scores = []

    for chunk in retrieved_chunks:
        metadata = chunk.metadata

        source = metadata.source
        topic = metadata.topic

        citation = f"{topic} | {source}"

        context_parts.append(
            f"[SOURCE: {citation}]\n"
            f"{chunk.chunk_text}\n"
        )

        if citation not in sources:
            sources.append(citation)

        scores.append(float(chunk.score))

    context = "\n".join(context_parts)

    average_confidence = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    # -----------------------------------------------------------------------
    # Build Conversation Messages
    # -----------------------------------------------------------------------

    llm = _get_llm()

    try:
        trimmed_history = trim_messages(
            messages,
            max_tokens=settings.max_context_tokens,
            strategy="last",
            token_counter=llm,
            include_system=False,
            allow_partial=False,
        )

    except Exception:
        # Safe fallback if token counting is unavailable.
        trimmed_history = messages[-10:]

    generation_messages = [
        SystemMessage(content=SYSTEM_PROMPT),

        SystemMessage(
            content=(
                "Use ONLY the retrieved context below to answer the "
                "user's question. Do not invent information that is "
                "not supported by the context.\n\n"
                "Retrieved context:\n"
                f"{context}"
            )
        ),

        *trimmed_history,

        HumanMessage(
            content=original_query or rewritten_query
        ),
    ]

    # -----------------------------------------------------------------------
    # Generate Answer
    # -----------------------------------------------------------------------

    try:
        llm_response = llm.invoke(generation_messages)

        answer = str(llm_response.content).strip()

    except Exception as exc:
        answer = (
            "Sorry, an error occurred while generating the answer: "
            f"{exc}"
        )

    # -----------------------------------------------------------------------
    # Construct Response
    # -----------------------------------------------------------------------

    response = AgentResponse(
        answer=answer,
        sources=sources,
        confidence=average_confidence,
        no_context_found=False,
        rewritten_query=rewritten_query,
    )

    new_ai_message = AIMessage(content=answer)

    return {
        "final_response": response,
        "messages": [new_ai_message],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Decide whether the graph should generate an answer or end.

    Context found -> generation
    No context -> end
    """

    no_context_found = _get_state_value(
        state,
        "no_context_found",
        False,
    )

    retrieved_chunks = _get_state_value(
        state,
        "retrieved_chunks",
        [],
    )

    if no_context_found or not retrieved_chunks:
        return "end"

    return "generate"