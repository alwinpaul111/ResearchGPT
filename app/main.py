"""
FastAPI backend for PaperLens.

Endpoints:
  POST /upload   - upload one or more PDFs, they get chunked + indexed
  POST /ask      - ask a question against the indexed papers
  GET  /health   - health check
  POST /reset    - clear conversation memory for a session

Run locally:
  uvicorn app.main:app --reload --port 8000
"""
import shutil
import uuid
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import UPLOAD_DIR
from app.ingest import ingest_pdfs
from app.rag_pipeline import answer_question, ConversationMemory
from app.vector_store import index_exists

app = FastAPI(title="PaperLens API", description="RAG system for research paper Q&A", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> ConversationMemory
# For a portfolio project this is fine; swap for Redis if you scale it.
SESSIONS: Dict[str, ConversationMemory] = {}


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


class AskResponse(BaseModel):
    answer: str
    citations: list


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": index_exists()}


@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    saved_paths = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")
        dest = Path(UPLOAD_DIR) / f"{uuid.uuid4().hex}_{file.filename}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(dest))

    result = ingest_pdfs(saved_paths)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])
    return result


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if not index_exists():
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    memory = SESSIONS.setdefault(payload.session_id, ConversationMemory())
    response = answer_question(payload.question, memory=memory)

    return AskResponse(
        answer=response.answer,
        citations=[c.__dict__ for c in response.citations],
    )


@app.post("/reset")
def reset_session(session_id: str = "default"):
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
