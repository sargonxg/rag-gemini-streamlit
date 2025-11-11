import mimetypes
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import fitz
import google.generativeai as genai
import numpy as np

try:  # pragma: no cover - optional dependency surface
    from google.generativeai import types as genai_types
except Exception:  # pragma: no cover - very defensive
    genai_types = None

def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)

def _as_embedding_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("Embedding vector must be 1-dimensional.")
    return arr


def _normalise_embedding_response(resp: Any) -> List[np.ndarray]:
    """Convert the embedding response into numpy arrays.

    The google-generativeai client has changed its return types a few times. This
    helper keeps the code resilient by checking for both dict-style and attribute
    access patterns.
    """

    def _values_from(obj: Any) -> Sequence[float]:
        if obj is None:
            return ()
        if isinstance(obj, dict):
            values = obj.get("values") or obj.get("embedding", {}).get("values")
            if values is not None:
                return values
        values = getattr(obj, "values", None)
        if values is not None:
            return values
        embedding = getattr(obj, "embedding", None)
        if embedding is not None:
            return _values_from(embedding)
        raise ValueError("Unable to locate embedding values in response")

    embeddings: Iterable[Any]
    if isinstance(resp, dict):
        if "embeddings" in resp:
            embeddings = resp["embeddings"]
        elif "embedding" in resp:
            embeddings = [resp["embedding"]]
        else:
            raise ValueError("Unexpected embedding response structure.")
    else:
        embeddings = getattr(resp, "embeddings", None)
        if embeddings is None:
            embedding = getattr(resp, "embedding", None)
            if embedding is None:
                # A few client versions expose the raw values directly.
                values = getattr(resp, "values", None)
                if values is None:
                    raise ValueError("Embedding response missing expected fields.")
                embeddings = [resp]
            else:
                embeddings = [embedding]

    vectors = []
    for emb in embeddings:
        vectors.append(_as_embedding_array(_values_from(emb)))
    return vectors


def embed_texts(texts: List[str], model_name="text-embedding-004", task_type="retrieval_document"):
    if not isinstance(texts, list):
        texts = [texts]
    resp = genai.embed_content(model=model_name, content=texts, task_type=task_type)
    vectors = _normalise_embedding_response(resp)
    return np.vstack(vectors)

def extract_text_from_pdf(file_bytes: bytes):
    pages = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append({"page": i+1, "text": text})
    return pages

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-")
    return slug.lower() or "document"


def chunk_text(
    pages: List[Dict[str, Any]],
    document_name: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    doc_slug = _slugify(Path(document_name).stem if document_name else "document")
    for p in pages:
        txt = re.sub(r"\s+", " ", p["text"]).strip()
        if not txt:
            continue
        i = 0
        j = 0
        while i < len(txt):
            chunk_text_value = txt[i : i + chunk_size]
            chunks.append(
                {
                    "id": f"{doc_slug}-p{p['page']}-c{j}",
                    "text": chunk_text_value,
                    "meta": {
                        "page": p["page"],
                        "doc_name": document_name,
                        "doc_slug": doc_slug,
                    },
                }
            )
            i += max(chunk_size - overlap, 1)
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
    def format_context(chunk: Dict[str, Any]) -> str:
        meta = chunk.get("meta", {})
        doc_name = meta.get("doc_name") or meta.get("doc_slug") or "document"
        page = meta.get("page", "?")
        return f"[{doc_name} – page {page} | {chunk['id']}] {chunk['text']}"

    ctx = "\n\n".join([format_context(c) for c in contexts])
    return (
        "You are an expert research assistant. Carefully read the provided context "
        "before answering. Use numbered citations that reference the chunk id and "
        "page number from the context when supporting each claim. If the "
        "information is not present, say you do not know."
        "\n\nContext:\n"
        f"{ctx}"
        "\n\nQuestion: "
        f"{question}\nAnswer:"
    )

def _wait_for_file_activation(file_obj: Any, timeout: float = 180.0, poll_interval: float = 2.0):
    deadline = time.time() + timeout
    current = file_obj
    while True:
        state = getattr(current, "state", None)
        state_name = getattr(state, "name", None) if state is not None else None
        if isinstance(state, str):
            state_name = state
        if state_name in {None, "ACTIVE"}:
            return current
        if state_name not in {"PROCESSING", "PENDING"}:
            raise RuntimeError(f"File {getattr(current, 'name', 'unknown')} failed with state: {state_name}")
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for Gemini file processing to complete.")
        time.sleep(poll_interval)
        current = genai.get_file(current.name)


def upload_files_to_gemini(files: List[Dict[str, Any]], api_key: str) -> List[Dict[str, str]]:
    """Upload PDF (or text) payloads to Gemini File API for File Search."""

    configure_gemini(api_key)
    uploaded: List[Dict[str, str]] = []

    for file_info in files:
        name = file_info.get("name") or "document.pdf"
        data = file_info.get("bytes")
        if data is None:
            continue
        mime_type = file_info.get("mime_type") or mimetypes.guess_type(name)[0] or "application/octet-stream"
        suffix = Path(name).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = tmp.name
        try:
            uploaded_file = genai.upload_file(
                path=temp_path,
                display_name=name,
                mime_type=mime_type,
            )
            ready_file = _wait_for_file_activation(uploaded_file)
            uploaded.append(
                {
                    "name": name,
                    "uri": getattr(ready_file, "uri", getattr(ready_file, "file_uri", "")),
                    "mime_type": getattr(ready_file, "mime_type", mime_type),
                }
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
    return uploaded



def _count_words(text: str) -> int:
    tokens = re.findall(r"\b\w+\b", text)
    return len(tokens)


def summarise_document(
    document_name: str,
    pages: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total_words = sum(_count_words(p.get("text", "")) for p in pages)
    total_characters = sum(len(p.get("text", "")) for p in pages)
    doc_chunks = [c for c in chunks if c.get("meta", {}).get("doc_name") == document_name]
    avg_chunk_chars = (total_characters / len(doc_chunks)) if doc_chunks else 0.0
    return {
        "document": document_name,
        "pages": len(pages),
        "chunks": len(doc_chunks),
        "words": total_words,
        "characters": total_characters,
        "avg_chunk_chars": round(avg_chunk_chars, 1),
    }


def summarise_corpus(doc_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_pages = sum(doc.get("pages", 0) for doc in doc_summaries)
    total_chunks = sum(doc.get("chunks", 0) for doc in doc_summaries)
    total_words = sum(doc.get("words", 0) for doc in doc_summaries)
    total_documents = len(doc_summaries)
    return {
        "documents": total_documents,
        "pages": total_pages,
        "chunks": total_chunks,
        "words": total_words,
    }


def generate_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    api_key: str,
    file_refs: List[Dict[str, str]] | None = None,
    model_name: str = "gemini-1.5-pro",
):
    configure_gemini(api_key)
    tools = None
    if file_refs and genai_types is not None and hasattr(genai_types, "FileSearch"):
        tools = [genai_types.FileSearch()]
    model_kwargs = {"model_name": model_name, "system_instruction": (
        "You are a meticulous analyst answering questions about uploaded documents. "
        "Prioritise faithfulness to the provided material, cite chunk ids and page "
        "numbers, and avoid fabricating information."
    )}
    if tools:
        model_kwargs["tools"] = tools
    model = genai.GenerativeModel(**model_kwargs)
    prompt = build_prompt(question, contexts)

    parts: List[Dict[str, Any]] = []
    if file_refs:
        for ref in file_refs:
            uri = ref.get("uri")
            if not uri:
                continue
            parts.append({
                "file_data": {
                    "file_uri": uri,
                    "mime_type": ref.get("mime_type", "application/pdf"),
                }
            })
    parts.append({"text": prompt})

    response = model.generate_content({"contents": [{"role": "user", "parts": parts}]})

    if hasattr(response, "text") and response.text:
        return response.text.strip()

    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                return text.strip()

    return ""
