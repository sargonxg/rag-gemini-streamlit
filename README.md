# Gemini RAG Chatbot (Streamlit)

A professional-grade Retrieval-Augmented Generation (RAG) chatbot powered by **Google Gemini 1.5 Pro**.

## 🚀 Features
- PDF upload and text extraction
- Automatic chunking and vectorization using Gemini embeddings
- Semantic search with FAISS
- Grounded Q&A with citations

## 🧩 Requirements
- Google Gemini API key (`GOOGLE_API_KEY`)
- Python 3.10+

## ▶️ Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## ☁️ Deploy to Streamlit Cloud
1. Upload this folder to a new GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Select the repo and `app.py` as the entry point.
4. Add `GOOGLE_API_KEY` to your Streamlit secrets or enter it manually in the UI.
