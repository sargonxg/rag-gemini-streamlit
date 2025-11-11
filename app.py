from __future__ import annotations

from textwrap import shorten
from typing import Any, Dict, Sequence

import streamlit as st

from rag_utils import (
    build_faiss,
    chunk_text,
    extract_text_from_pdf,
    generate_answer,
    search,
    summarise_corpus,
    summarise_document,
    upload_files_to_gemini,
)


MAX_FILES = 50


def _format_score(value: float) -> str:
    try:
        if value is None or value != value:  # NaN-safe check
            return "N/A"
        return f"{value:.3f}"
    except Exception:
        return "N/A"


def _initialise_state() -> None:
    defaults: Dict[str, object] = {
        "store": None,
        "file_refs": [],
        "documents": [],
        "corpus_overview": {},
        "messages": [],
        "uploaded_manifest": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _reset_workspace() -> None:
    for key in ("store", "file_refs", "documents", "corpus_overview", "messages"):
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["uploaded_manifest"] = []


def _render_metric(label: str, value: int | float | None) -> str:
    try:
        formatted = f"{(value or 0):,}"
    except Exception:
        formatted = str(value)
    return (
        "<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{formatted}</div>"
        "</div>"
    )


st.set_page_config(
    page_title="Gemini Diplomatic Briefing",
    page_icon="🕊️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --briefing-blue: #0f4c81;
        --briefing-sky: #c5e4ff;
        --briefing-ice: rgba(255, 255, 255, 0.85);
    }
    .stApp {
        background: linear-gradient(150deg, #dff0ff 0%, #f4fbff 55%, #ffffff 100%);
    }
    .hero-banner {
        padding: 1.9rem 2.6rem;
        border-radius: 1.6rem;
        background: rgba(15, 76, 129, 0.12);
        border: 1px solid rgba(15, 76, 129, 0.22);
        box-shadow: 0 30px 40px rgba(15, 76, 129, 0.18);
        backdrop-filter: blur(9px);
    }
    .hero-title {
        font-size: 2.55rem;
        color: var(--briefing-blue);
        margin-bottom: 0.35rem;
        font-weight: 700;
    }
    .hero-subtitle {
        font-size: 1.08rem;
        color: #134063;
        margin-bottom: 0;
    }
    .metric-card {
        background: var(--briefing-ice);
        border-radius: 1rem;
        padding: 1.05rem;
        border: 1px solid rgba(15, 76, 129, 0.14);
        box-shadow: 0 12px 28px rgba(15, 76, 129, 0.09);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--briefing-blue);
        margin-bottom: 0.2rem;
    }
    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.75rem;
        color: #2b6a9b;
        margin-bottom: 0.3rem;
    }
    .source-card {
        border-radius: 0.95rem;
        background: rgba(224, 243, 255, 0.92);
        border: 1px solid rgba(15, 76, 129, 0.12);
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .source-card h4 {
        margin: 0 0 0.25rem 0;
        color: var(--briefing-blue);
        font-size: 1rem;
        font-weight: 650;
    }
    .mission-bullet {
        font-size: 0.92rem;
        color: #114870;
    }
    .uploaded-summary {
        background: rgba(255, 255, 255, 0.75);
        border-radius: 1rem;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(15, 76, 129, 0.12);
        box-shadow: 0 10px 20px rgba(15, 76, 129, 0.08);
    }
    .stChatMessage .stMarkdown {
        color: #0f2747;
        font-size: 1rem;
    }
    .chat-instructions {
        color: #1f4870;
        font-size: 0.95rem;
    }
    .corpus-table .stDataFrame {
        border-radius: 1rem;
        border: 1px solid rgba(15, 118, 206, 0.16);
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_initialise_state()

with st.container():
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Gemini Diplomatic Briefing Room</div>
            <p class="hero-subtitle">
                Curate up to 50 dossiers, policy briefs, or field reports and interrogate them with a retrieval-augmented Gemini analyst. Crafted for diplomats, mission planners, and UN coordination teams operating in complex environments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Mission Controls")
    st.markdown(
        """
        <ul class="mission-bullet">
            <li>Authenticate with your Gemini API key.</li>
            <li>Upload up to 50 PDF dossiers to brief.</li>
            <li>Build the knowledge base for retrieval.</li>
            <li>Engage the analyst with targeted questions.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    retrieval_k = st.slider("Context passages", min_value=3, max_value=12, value=6, step=1)
    st.caption(
        "Broader context widens evidence at the cost of longer responses."
    )
    st.button(
        "♻️ Reset briefing room",
        use_container_width=True,
        on_click=_reset_workspace,
    )


api_key = st.sidebar.text_input(
    "🔑 Gemini API key",
    type="password",
    help="The key is only stored in memory for this session.",
)
if not api_key:
    st.info(
        "Provide an API key in the sidebar to initiate the diplomatic briefing workflow.",
        icon="🔐",
    )
    st.stop()


uploaded_files = st.file_uploader(
    "📚 Upload diplomatic PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Briefing books, situation reports, or agreements (maximum of 50 files).",
)


def _preview_manifest(files: Sequence[Any]) -> None:
    if not files:
        return
    manifest_rows = []
    for file in files:
        size_kb = (file.size or 0) / 1024
        manifest_rows.append(
            {
                "Document": file.name,
                "Size (KB)": f"{size_kb:,.1f}",
            }
        )
    st.session_state.uploaded_manifest = manifest_rows
    st.markdown("#### Pending dossiers")
    st.markdown("These files will be processed into the mission corpus once you build it.")
    st.dataframe(manifest_rows, use_container_width=True, hide_index=True)


if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.warning(
            f"Only the first {MAX_FILES} files will be processed to preserve responsiveness."
        )
        uploaded_files = uploaded_files[:MAX_FILES]
    _preview_manifest(uploaded_files)

    if st.button("🔍 Build Knowledge Base", use_container_width=True):
        with st.spinner("Indexing dossiers and syncing with Gemini File Search..."):
            all_chunks = []
            file_payloads = []
            document_summaries = []
            for uploaded in uploaded_files:
                uploaded.seek(0)
                file_bytes = uploaded.read()
                file_payloads.append(
                    {
                        "name": uploaded.name,
                        "bytes": file_bytes,
                        "mime_type": uploaded.type,
                    }
                )
                pages = extract_text_from_pdf(file_bytes)
                chunks = chunk_text(pages, uploaded.name)
                all_chunks.extend(chunks)
                document_summaries.append(
                    summarise_document(uploaded.name, pages, chunks)
                )

            if all_chunks:
                st.session_state.store = build_faiss(all_chunks, api_key)
                st.session_state.documents = document_summaries
                st.session_state.corpus_overview = summarise_corpus(document_summaries)
                st.session_state.messages = []
                st.session_state.uploaded_manifest = []
                try:
                    st.session_state.file_refs = upload_files_to_gemini(
                        file_payloads, api_key
                    )
                except Exception as exc:
                    st.session_state.file_refs = []
                    st.warning(
                        "Knowledge base ready locally, but Gemini File Search sync failed. "
                        "Local retrieval will continue to function.\n\n"
                        f"Details: {exc}"
                    )
                else:
                    st.success(
                        "Knowledge base indexed and Gemini File Search connected. "
                        "Ready for diplomatic briefing."
                    )
            else:
                st.warning("No readable text detected in the uploaded dossiers.")
elif st.session_state.uploaded_manifest:
    st.markdown("#### Pending dossiers")
    st.dataframe(
        st.session_state.uploaded_manifest,
        use_container_width=True,
        hide_index=True,
    )


if st.session_state.store:
    overview = st.session_state.get("corpus_overview", {})
    documents = st.session_state.get("documents", [])

    st.subheader("Corpus Intelligence Snapshot")
    metric_cols = st.columns(4)
    metrics = [
        ("Documents", overview.get("documents", 0)),
        ("Pages", overview.get("pages", 0)),
        ("Chunks", overview.get("chunks", 0)),
        ("Words", overview.get("words", 0)),
    ]
    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            card_html = _render_metric(label, value)
            st.markdown(card_html, unsafe_allow_html=True)

    if documents:
        dossier_tab, density_tab = st.tabs(["Uploaded Dossiers", "Chunk Density"])
        with dossier_tab:
            table_rows = [
                {
                    "Document": doc.get("document"),
                    "Pages": doc.get("pages"),
                    "Chunks": doc.get("chunks"),
                    "Words": doc.get("words"),
                    "Characters": doc.get("characters"),
                    "Avg chunk chars": doc.get("avg_chunk_chars"),
                }
                for doc in documents
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
        with density_tab:
            density_rows = [
                {
                    "Document": doc.get("document"),
                    "Words / Page": round(
                        (doc.get("words", 0) / max(doc.get("pages", 1), 1)), 1
                    ),
                    "Chunks / Page": round(
                        (doc.get("chunks", 0) / max(doc.get("pages", 1), 1)), 1
                    ),
                }
                for doc in documents
            ]
            st.dataframe(density_rows, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Strategic Dialogue")
    if not st.session_state.messages:
        st.markdown(
            "<p class='chat-instructions'>Begin the conversation with a briefing request, a policy comparison, or a scenario-specific question to see synthesised, cited answers.</p>",
            unsafe_allow_html=True,
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("role") == "assistant" and message.get("sources"):
                st.markdown("**Sources consulted**")
                for src in message["sources"]:
                    score_display = _format_score(src.get("score"))
                    st.markdown(
                        f"- {src['doc']} — page {src['page']} *(score {score_display})*"
                    )

    if prompt := st.chat_input("Ask about the uploaded corpus"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Retrieving intelligence and crafting response..."):
                hits = search(prompt, st.session_state.store, api_key, k=retrieval_k)
                contexts = [st.session_state.store["chunks"][h[0]] for h in hits]
                file_refs = st.session_state.get("file_refs", [])
                answer = generate_answer(
                    prompt,
                    contexts,
                    api_key,
                    file_refs=file_refs,
                )

            st.markdown(answer)

            sources_payload = []
            if contexts:
                st.markdown("**Sources consulted**")
                for chunk, hit in zip(contexts, hits):
                    meta = chunk.get("meta", {})
                    doc_name = meta.get("doc_name") or meta.get("doc_slug", "Document")
                    page = meta.get("page", "?")
                    snippet = shorten(
                        chunk.get("text", ""), width=220, placeholder="…"
                    )
                    score = float(hit[1]) if len(hit) > 1 else float("nan")
                    score_display = _format_score(score)
                    st.markdown(
                        f"<div class='source-card'><h4>{doc_name} · page {page}</h4>"
                        f"<p><em>Relevance:</em> {score_display}</p>"
                        f"<p>{snippet}</p></div>",
                        unsafe_allow_html=True,
                    )
                    sources_payload.append(
                        {
                            "doc": doc_name,
                            "page": page,
                            "score": score,
                            "snippet": chunk.get("text", ""),
                        }
                    )
            else:
                st.info("No supporting passages were retrieved for this query.")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources_payload,
            }
        )
