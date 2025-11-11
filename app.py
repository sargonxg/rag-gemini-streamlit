import streamlit as st
from rag_utils import extract_text_from_pdf, chunk_text, build_faiss, search, generate_answer

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🤖", layout="wide")

st.markdown("<h2 style='color:#0F766E;'>Gemini RAG Chatbot</h2>", unsafe_allow_html=True)
st.write("Upload PDFs, ask questions, and get grounded answers with source citations.")

api_key = st.text_input("🔑 Enter your Google Gemini API key:", type="password")
if not api_key:
    st.stop()

uploaded_files = st.file_uploader("📚 Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if "store" not in st.session_state:
    st.session_state.store = None

if uploaded_files and st.button("🔍 Process Documents"):
    all_chunks = []
    for f in uploaded_files:
        pages = extract_text_from_pdf(f.read())
        chunks = chunk_text(pages)
        all_chunks.extend(chunks)
    st.session_state.store = build_faiss(all_chunks, api_key)
    st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} document(s).")

if st.session_state.store:
    st.divider()
    query = st.text_input("💬 Ask a question about your documents:")
    if query:
        with st.spinner("Retrieving context and generating answer..."):
            hits = search(query, st.session_state.store, api_key, k=5)
            contexts = [st.session_state.store["chunks"][h[0]] for h in hits]
            answer = generate_answer(query, contexts, api_key)
        st.markdown("### 🧠 Answer")
        st.write(answer)
        with st.expander("Show retrieved sources"):
            for c in contexts:
                st.markdown(f"**{c['id']} (p.{c['meta']['page']})** — {c['text'][:300]}...")
