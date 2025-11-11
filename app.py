import streamlit as st
from rag_utils import (
    extract_text_from_pdf,
    chunk_text,
    build_faiss,
    search,
    generate_answer,
    upload_files_to_gemini,
)

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🤖", layout="wide")

st.markdown("<h2 style='color:#0F766E;'>Gemini RAG Chatbot</h2>", unsafe_allow_html=True)
st.write("Upload PDFs, ask questions, and get grounded answers with source citations.")

api_key = st.text_input("🔑 Enter your Google Gemini API key:", type="password")
if not api_key:
    st.stop()

uploaded_files = st.file_uploader("📚 Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if "store" not in st.session_state:
    st.session_state.store = None
if "file_refs" not in st.session_state:
    st.session_state.file_refs = []

if uploaded_files and st.button("🔍 Process Documents"):
    with st.spinner("Indexing documents and syncing with Gemini File Search..."):
        all_chunks = []
        file_payloads = []
        for f in uploaded_files:
            file_bytes = f.read()
            file_payloads.append({"name": f.name, "bytes": file_bytes, "mime_type": f.type})
            pages = extract_text_from_pdf(file_bytes)
            chunks = chunk_text(pages)
            all_chunks.extend(chunks)
        if all_chunks:
            st.session_state.store = build_faiss(all_chunks, api_key)
            try:
                st.session_state.file_refs = upload_files_to_gemini(file_payloads, api_key)
            except Exception as exc:
                st.session_state.file_refs = []
                st.warning(
                    "Document index created locally, but syncing with Gemini File Search failed. "
                    "You can still query the local index.\n\n"
                    f"Details: {exc}"
                )
            else:
                st.success(
                    f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} document(s) and enabled Gemini File Search."
                )
        else:
            st.warning("No readable text found in the uploaded documents.")

if st.session_state.store:
    st.divider()
    query = st.text_input("💬 Ask a question about your documents:")
    if query:
        with st.spinner("Retrieving context and generating answer..."):
            hits = search(query, st.session_state.store, api_key, k=6)
            contexts = [st.session_state.store["chunks"][h[0]] for h in hits]
            file_refs = st.session_state.get("file_refs", [])
            answer = generate_answer(query, contexts, api_key, file_refs=file_refs)
        st.markdown("### 🧠 Answer")
        st.write(answer)
        with st.expander("Show retrieved sources"):
            for idx, (c, hit) in enumerate(zip(contexts, hits), start=1):
                score = hit[1]
                st.markdown(
                    f"**{idx}. {c['id']} (p.{c['meta']['page']}) — score {score:.3f}**\n\n{c['text']}"
                )
