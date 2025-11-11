from textwrap import shorten

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

st.set_page_config(page_title="Gemini Diplomatic Briefing", page_icon="🕊️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f2ff 0%, #f5fbff 50%, #ffffff 100%);
    }
    .hero-banner {
        padding: 1.8rem 2.4rem;
        border-radius: 1.5rem;
        background: rgba(15, 118, 206, 0.15);
        border: 1px solid rgba(15, 118, 206, 0.25);
        box-shadow: 0 25px 35px rgba(79, 165, 216, 0.15);
        backdrop-filter: blur(6px);
    }
    .hero-title {
        font-size: 2.4rem;
        color: #0f4c81;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #134063;
        margin-bottom: 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 1rem;
        padding: 1.1rem;
        border: 1px solid rgba(15, 118, 206, 0.18);
        box-shadow: 0 12px 24px rgba(15, 118, 206, 0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f4c81;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.75rem;
        color: #2b6a9b;
        margin-bottom: 0.4rem;
    }
    .source-card {
        border-radius: 0.9rem;
        background: rgba(239, 250, 255, 0.85);
        border: 1px solid rgba(15, 118, 206, 0.16);
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .source-card h4 {
        margin: 0 0 0.25rem 0;
        color: #0f4c81;
        font-size: 1rem;
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


with st.container():
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Gemini Diplomatic Briefing Room</div>
            <p class="hero-subtitle">
                Upload up to 50 dossiers, policy briefs, or field reports and interrogate them with a retrieval-augmented Gemini analyst. Built for high-stakes decision-makers across missions and UN agencies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Mission Controls")
    st.write(
        """
        1. Provide your Gemini API key.
        2. Upload PDF dossiers (max 50).
        3. Process to build a searchable diplomatic corpus.
        4. Ask nuanced questions in the chat interface.
        """
    )
    retrieval_k = st.slider("Context passages", min_value=3, max_value=12, value=6, step=1)
    st.caption(
        "Higher values surface broader evidence at the cost of slightly longer response times."
    )


api_key = st.sidebar.text_input("🔑 Gemini API key", type="password", help="Stored only for this session.")
if not api_key:
    st.stop()


if "store" not in st.session_state:
    st.session_state.store = None
if "file_refs" not in st.session_state:
    st.session_state.file_refs = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "corpus_overview" not in st.session_state:
    st.session_state.corpus_overview = {}
if "messages" not in st.session_state:
    st.session_state.messages = []


uploaded_files = st.file_uploader(
    "📚 Upload diplomatic PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Batch briefing books, situation reports, or agreements (maximum of 50 files).",
)

if uploaded_files:
    if len(uploaded_files) > MAX_FILES:
        st.warning(
            f"Only the first {MAX_FILES} files will be processed to preserve responsiveness."
        )
        uploaded_files = uploaded_files[:MAX_FILES]

    if st.button("🔍 Build Knowledge Base", use_container_width=True):
        with st.spinner("Indexing dossiers and syncing with Gemini File Search..."):
            all_chunks = []
            file_payloads = []
            document_summaries = []
            for uploaded in uploaded_files:
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
                document_summaries.append(summarise_document(uploaded.name, pages, chunks))

            if all_chunks:
                st.session_state.store = build_faiss(all_chunks, api_key)
                st.session_state.documents = document_summaries
                st.session_state.corpus_overview = summarise_corpus(document_summaries)
                st.session_state.messages = []
                try:
                    st.session_state.file_refs = upload_files_to_gemini(file_payloads, api_key)
                except Exception as exc:
                    st.session_state.file_refs = []
                    st.warning(
                        "Knowledge base ready locally, but Gemini File Search sync failed. "
                        "Local retrieval will continue to function.\n\n"
                        f"Details: {exc}"
                    )
                else:
                    st.success(
                        "Knowledge base indexed and Gemini File Search connected. Ready for diplomatic briefing."
                    )
            else:
                st.warning("No readable text detected in the uploaded dossiers.")


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
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value:,}</div></div>",
                unsafe_allow_html=True,
            )

    if documents:
        st.markdown("### Uploaded Dossiers")
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

    st.divider()

    st.subheader("Strategic Dialogue")
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
                answer = generate_answer(prompt, contexts, api_key, file_refs=file_refs)

            st.markdown(answer)

            sources_payload = []
            if contexts:
                st.markdown("**Sources consulted**")
                for chunk, hit in zip(contexts, hits):
                    meta = chunk.get("meta", {})
                    doc_name = meta.get("doc_name") or meta.get("doc_slug", "Document")
                    page = meta.get("page", "?")
                    snippet = shorten(chunk.get("text", ""), width=220, placeholder="…")
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
