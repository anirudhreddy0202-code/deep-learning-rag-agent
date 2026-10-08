"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the cached VectorStoreManager."""
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the cached DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the cached compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """Initialise Streamlit session-state values."""

    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """Render the document ingestion panel."""

    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.sidebar.write(
            f"**{len(uploaded_files)} file(s) selected**"
        )

    ingest_clicked = st.sidebar.button(
        "📥 Ingest Documents",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if ingest_clicked and uploaded_files:

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_paths: list[Path] = []

            for uploaded_file in uploaded_files:
                destination = temp_path / uploaded_file.name

                destination.write_bytes(
                    uploaded_file.getbuffer()
                )

                file_paths.append(destination)

            with st.spinner("Chunking and ingesting documents..."):
                chunks = chunker.chunk_files(file_paths)

                if not chunks:
                    st.sidebar.error(
                        "No chunks were created from the uploaded files."
                    )
                    return

                result = store.ingest(chunks)

        st.session_state["last_ingestion_result"] = result

        if result.ingested > 0:
            st.sidebar.success(
                f"✅ {result.ingested} chunks added, "
                f"{result.skipped} duplicates skipped."
            )
        elif result.skipped > 0:
            st.sidebar.warning(
                f"⚠️ No new chunks added. "
                f"{result.skipped} duplicates skipped."
            )

        if result.errors:
            for error in result.errors:
                st.sidebar.error(error)

        # Refresh document list
        st.session_state["ingested_documents"] = (
            store.list_documents()
        )

    # Refresh the document list even if the app was restarted
    try:
        documents = store.list_documents()
        st.session_state["ingested_documents"] = documents
    except Exception:
        documents = st.session_state["ingested_documents"]

    # Display documents
    if documents:
        st.sidebar.subheader("📚 Documents")

        for document in documents:
            source = document.get("source", "Unknown")
            topic = document.get("topic", "Unknown")
            chunk_count = document.get("chunk_count", 0)

            st.sidebar.write(
                f"**{source}**  \n"
                f"Topic: {topic} | Chunks: {chunk_count}"
            )

            if st.sidebar.button(
                f"🗑 Remove {source}",
                key=f"delete_{source}",
                use_container_width=True,
            ):
                try:
                    deleted = store.delete_document(source)

                    st.session_state["ingested_documents"] = (
                        store.list_documents()
                    )

                    if st.session_state["selected_document"] == source:
                        st.session_state["selected_document"] = None

                    st.sidebar.success(
                        f"Removed {deleted} chunks from {source}."
                    )

                    st.rerun()

                except Exception as exc:
                    st.sidebar.error(
                        f"Could not remove {source}: {exc}"
                    )

    else:
        st.sidebar.info(
            "Upload a Markdown (.md) file to populate the corpus."
        )


# ---------------------------------------------------------------------------
# Corpus Statistics
# ---------------------------------------------------------------------------


def render_corpus_stats(
    store: VectorStoreManager,
) -> None:
    """Render corpus health information."""

    st.sidebar.subheader("📊 Corpus Health")

    try:
        stats = store.get_collection_stats()

        st.sidebar.metric(
            "Total Chunks",
            stats.get("total_chunks", 0),
        )

        topics = stats.get("topics", [])

        if topics:
            st.sidebar.write(
                "**Topics:** " + ", ".join(topics)
            )
        else:
            st.sidebar.write("**Topics:** None")

        if stats.get("bonus_topics_present"):
            st.sidebar.success(
                "✅ Bonus topics present"
            )
        else:
            st.sidebar.warning(
                "⚠️ No bonus topics yet"
            )

    except Exception as exc:
        st.sidebar.warning(
            f"Corpus statistics unavailable: {exc}"
        )


# ---------------------------------------------------------------------------
# Document Viewer
# ---------------------------------------------------------------------------


def render_document_viewer(
    store: VectorStoreManager,
) -> None:
    """Render the document viewer."""

    st.subheader("📄 Document Viewer")

    try:
        documents = store.list_documents()
    except Exception as exc:
        st.error(
            f"Could not load documents: {exc}"
        )
        return

    if not documents:
        st.info(
            "Ingest a Markdown document using the sidebar "
            "to view its content here."
        )
        return

    sources = [
        document["source"]
        for document in documents
    ]

    current_selection = st.session_state.get(
        "selected_document"
    )

    if current_selection not in sources:
        current_selection = sources[0]

    selected_source = st.selectbox(
        "Select document",
        options=sources,
        index=sources.index(current_selection),
    )

    st.session_state["selected_document"] = selected_source

    try:
        chunks = store.get_document_chunks(
            selected_source
        )
    except Exception as exc:
        st.error(
            f"Could not load document chunks: {exc}"
        )
        return

    st.caption(
        f"{len(chunks)} chunks in {selected_source}"
    )

    with st.container(height=500):
        for index, chunk in enumerate(chunks, start=1):

            metadata = chunk.metadata

            st.markdown(
                f"**Chunk {index}**  \n"
                f"`{metadata.topic}` · "
                f"`{metadata.difficulty}` · "
                f"`{metadata.type}`"
            )

            st.write(chunk.chunk_text)

            st.divider()


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """Render the RAG chat interface."""

    st.subheader("💬 Interview Prep Chat")

    # ---------------------------------------------------------------
    # Filters
    # ---------------------------------------------------------------

    try:
        store = get_vector_store()
        stats = store.get_collection_stats()
        available_topics = stats.get("topics", [])
    except Exception:
        available_topics = []

    topics = ["All Topics"] + sorted(
        set(available_topics)
    )

    difficulties = [
        "All Difficulties",
        "beginner",
        "intermediate",
        "advanced",
    ]

    col_topic, col_diff = st.columns(2)

    with col_topic:
        selected_topic = st.selectbox(
            "Topic",
            topics,
            index=0,
        )

    with col_diff:
        selected_difficulty = st.selectbox(
            "Difficulty",
            difficulties,
            index=0,
        )

    st.session_state["topic_filter"] = (
        None
        if selected_topic == "All Topics"
        else selected_topic
    )

    st.session_state["difficulty_filter"] = (
        None
        if selected_difficulty == "All Difficulties"
        else selected_difficulty
    )

    # ---------------------------------------------------------------
    # Chat history
    # ---------------------------------------------------------------

    chat_container = st.container(height=400)

    with chat_container:
        for message in st.session_state.chat_history:

            with st.chat_message(
                message["role"]
            ):
                st.markdown(
                    message["content"]
                )

                if message.get("sources"):
                    with st.expander(
                        "📎 Sources"
                    ):
                        for source in message["sources"]:
                            st.caption(source)

                if message.get(
                    "no_context_found"
                ):
                    st.warning(
                        "⚠️ No relevant content found "
                        "in the corpus."
                    )

    # ---------------------------------------------------------------
    # Chat input
    # ---------------------------------------------------------------

    query = st.chat_input(
        "Ask about a deep learning topic..."
    )

    if not query:
        return

    # Display/store user message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Build LangGraph input
    graph_input = {
        "messages": [
            HumanMessage(content=query)
        ],
        "original_query": query,
        "topic_filter": st.session_state[
            "topic_filter"
        ],
        "difficulty_filter": st.session_state[
            "difficulty_filter"
        ],
    }

    config = {
        "configurable": {
            "thread_id": st.session_state[
                "thread_id"
            ]
        }
    }

    # Run RAG agent
    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the knowledge base..."
        ):
            try:
                result = graph.invoke(
                    graph_input,
                    config=config,
                )

                response = result.get(
                    "final_response"
                )

                if isinstance(
                    response,
                    AgentResponse,
                ):
                    answer = response.answer
                    sources = response.sources
                    no_context_found = (
                        response.no_context_found
                    )
                else:
                    answer = (
                        "The agent did not return "
                        "a valid response."
                    )
                    sources = []
                    no_context_found = True

            except Exception as exc:
                answer = (
                    f"Sorry, an error occurred "
                    f"while processing your question: {exc}"
                )
                sources = []
                no_context_found = True

        st.markdown(answer)

        if sources:
            with st.expander("📎 Sources"):
                for source in sources:
                    st.caption(source)

        if no_context_found:
            st.warning(
                "⚠️ No relevant content found in the corpus."
            )

    # Save assistant response
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "no_context_found": no_context_found,
        }
    )


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""

    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(
        f"🧠 {settings.app_title}"
    )

    st.caption(
        "RAG-powered interview preparation — "
        "built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(
        store,
        chunker,
    )

    render_corpus_stats(store)

    # Main content
    viewer_col, chat_col = st.columns(
        [1, 1],
        gap="large",
    )

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()