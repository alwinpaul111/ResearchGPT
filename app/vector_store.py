"""
Vector store wrapper around FAISS (default) via LangChain's community
integration. Persists to disk so uploaded papers survive a restart.
Swap VECTOR_BACKEND="chroma" in config.py to use ChromaDB instead -
both are wired up so you can discuss the tradeoffs in interviews:
FAISS = fastest for pure similarity search, in-memory + file persistence.
Chroma = built-in metadata filtering, easier incremental updates, its own
on-disk DB engine.
"""
import os
from typing import List

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

from app.config import FAISS_INDEX_PATH, CHROMA_PERSIST_DIR, VECTOR_BACKEND, TOP_K
from app.embeddings import get_embedding_model
from app.chunking import Chunk


def _chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    return [
        Document(
            page_content=c.text,
            metadata={
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "chunk_id": c.id,
            },
        )
        for c in chunks
    ]


def build_or_update_index(chunks: List[Chunk]):
    """Embed chunks and add them to the persistent vector store."""
    documents = _chunks_to_documents(chunks)
    embeddings = get_embedding_model()

    if VECTOR_BACKEND == "chroma":
        if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            store = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
            store.add_documents(documents)
        else:
            store = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PERSIST_DIR)
        store.persist()
        return store

    # default: FAISS
    if os.path.exists(FAISS_INDEX_PATH):
        store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        store.add_documents(documents)
    else:
        store = FAISS.from_documents(documents, embeddings)
    store.save_local(FAISS_INDEX_PATH)
    return store


def load_index():
    """Load the existing vector store from disk (or None if it doesn't exist yet)."""
    embeddings = get_embedding_model()

    if VECTOR_BACKEND == "chroma":
        if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        return None

    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None


def similarity_search(query: str, k: int = TOP_K):
    """Retrieve the top-k most relevant chunks for a query, with scores."""
    store = load_index()
    if store is None:
        return []
    return store.similarity_search_with_relevance_scores(query, k=k)


def get_first_page_chunks():
    """Directly return chunks tagged as page 1 for every indexed document.
    Bypasses similarity search entirely, since title/author blocks on page 1
    rarely match well against generic questions like 'who are the authors'.
    """
    store = load_index()
    if store is None:
        return []

    results = []
    if VECTOR_BACKEND == "chroma":
        try:
            raw = store.get(where={"page_number": 1})
            for text, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
                from langchain_core.documents import Document
                results.append((Document(page_content=text, metadata=meta), 1.0))
        except Exception:
            pass
        return results

    # FAISS: scan the in-memory docstore directly
    try:
        for doc in store.docstore._dict.values():
            if doc.metadata.get("page_number") == 1:
                results.append((doc, 1.0))
    except Exception:
        pass
    return results


def index_exists() -> bool:
    if VECTOR_BACKEND == "chroma":
        return os.path.exists(CHROMA_PERSIST_DIR) and bool(os.listdir(CHROMA_PERSIST_DIR))
    return os.path.exists(FAISS_INDEX_PATH)
