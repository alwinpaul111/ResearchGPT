"""
Central configuration for PaperLens.
All tunable parameters live here so you can adjust the pipeline
without touching the core logic.
"""
import os
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Chunking ----------
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# ---------- Embeddings ----------
# Small, fast, strong open-source embedding model (384-dim)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Vector store ----------
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss")  # "faiss" or "chroma"
FAISS_INDEX_PATH = str(VECTORSTORE_DIR / "faiss_index")
CHROMA_PERSIST_DIR = str(VECTORSTORE_DIR / "chroma_db")

# ---------- Retrieval ----------
TOP_K = 6  # number of chunks retrieved per query

# ---------- LLM ----------
# Groq gives a generous free tier and runs Llama-3 fast, good for a
# no-cost student deployment. Set GROQ_API_KEY as an environment variable.
# Alternative: swap in HuggingFace Inference API (see llm.py) using your
# alwinn HF account + HUGGINGFACEHUB_API_TOKEN.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "huggingface"
GROQ_MODEL = "llama-3.3-70b-versatile"
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1100

# ---------- Conversation memory ----------
MAX_HISTORY_TURNS = 6  # how many past Q&A turns to keep in context
