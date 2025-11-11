import re
from typing import List, Dict, Tuple, Any
import numpy as np
import fitz
import google.generativeai as genai

def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)

def embed_texts(texts: List[str], model_name="text-embedding-004", task_type="retrieval_document"):
    if not isinstance(texts, list):
        texts = [texts]
    resp = genai.embed_content(model=model_name, content=texts, task_type=task_type)
    if "embeddings" in resp:
        vecs = [np.array(e["values"], dtype=np.float32) for e in resp["embeddings"]]
    else:
        vecs = [np.array(resp["embedding"]["values"], dtype=np.float32)]
    return np.vstack(vecs)

def extract_text_from_pdf(file_bytes: bytes):
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append({"page": i+1, "text": text})
    return pages

def chunk_text(pages: List[Dict[str, Any]], chunk_size=1000, overlap=200):
    chunks = []
    for p in pages:
        txt = re.sub(r"\s+", " ", p["text"]).strip()
        i = 0
        j = 0
        while i < len(txt):
            chunks.append({
                "id": f"p{p['page']}_c{j}",
                "text": txt[i:i+chunk_size],
                "meta": {"page": p["page"]}
            })
            i += chunk_size - overlap
            j += 1
    return chunks

def build_faiss(chunks: List[Dict[str, Any]], api_key: str):
    import faiss
    configure_gemini(api_key)
    texts = [c["text"] for c in chunks]
    X = embed_texts(texts)
    faiss.normalize_L2(X)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    return {"index": index, "chunks": chunks, "dim": dim}

def search(query: str, store: Dict[str, Any], api_key: str, k=6):
    import faiss
    configure_gemini(api_key)
    q = embed_texts([query], task_type="retrieval_query")
    faiss.normalize_L2(q)
    D, I = store["index"].search(q, k)
    hits = [(int(idx), float(dist)) for idx, dist in zip(I[0], D[0]) if idx != -1]
    return hits

def build_prompt(question: str, contexts: List[Dict[str, Any]]):
    ctx = "\n\n".join([f"[{c['id']} p.{c['meta']['page']}] {c['text']}" for c in contexts])
    return f"You are a meticulous analyst. Use ONLY the context.\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"

def generate_answer(question: str, contexts: List[Dict[str, Any]], api_key: str):
    configure_gemini(api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = build_prompt(question, contexts)
    resp = model.generate_content(prompt)
    return resp.text.strip() if hasattr(resp, "text") else ""
