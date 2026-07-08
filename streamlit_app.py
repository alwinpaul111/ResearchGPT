"""
PaperLens - Streamlit frontend.

Two modes, controlled by RUN_MODE env var:
  - "api"    (default): talks to the FastAPI backend over HTTP (use this
              when you deploy FastAPI + Streamlit as separate services,
              e.g. FastAPI on Render + Streamlit on Streamlit Cloud)
  - "direct": imports the RAG pipeline directly in-process (simplest for
              a single-container Streamlit Cloud deployment, same pattern
              you used for the hate speech detector)
"""
import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

RUN_MODE = os.getenv("RUN_MODE", "direct")
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PaperLens", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, citations)

# ---------- Direct mode setup ----------
if RUN_MODE == "direct":
    from app.ingest import ingest_pdfs
    from app.rag_pipeline import answer_question, ConversationMemory
    from app.vector_store import index_exists
    from app.config import UPLOAD_DIR

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()


def index_ready() -> bool:
    if RUN_MODE == "direct":
        return index_exists()
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json().get("index_ready", False)
    except requests.RequestException:
        return False


def do_upload(uploaded_files):
    if RUN_MODE == "direct":
        saved_paths = []
        for f in uploaded_files:
            dest = Path(UPLOAD_DIR) / f.name
            with open(dest, "wb") as out:
                out.write(f.read())
            saved_paths.append(str(dest))
        return ingest_pdfs(saved_paths)
    else:
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
        r = requests.post(f"{API_URL}/upload", files=files, timeout=120)
        r.raise_for_status()
        return r.json()


def do_ask(question: str):
    if RUN_MODE == "direct":
        response = answer_question(question, memory=st.session_state.memory)
        return response.answer, [c.__dict__ for c in response.citations]
    else:
        r = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "session_id": st.session_state.session_id},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        return data["answer"], data["citations"]


# ---------- Sidebar: upload ----------
with st.sidebar:
    st.title("PaperLens")
    st.caption("Chat with your research papers, grounded with citations.")
    st.divider()

    st.subheader("1. Upload papers")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Extracting text, chunking, and embedding..."):
            try:
                result = do_upload(uploaded_files)
                st.session_state.chat_history = []
                if RUN_MODE == "direct":
                    st.session_state.memory.clear()
                else:
                    try:
                        requests.post(f"{API_URL}/reset", params={"session_id": st.session_state.session_id})
                    except requests.RequestException:
                        pass
                st.success(
                    f"Indexed {result['documents_processed']} document(s), "
                    f"{result['pages_processed']} pages, {result['chunks_created']} chunks. "
                    f"Previous documents and conversation were replaced."
                )
            except Exception as e:
                st.error(f"Failed to process PDFs: {e}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        if RUN_MODE == "direct":
            st.session_state.memory.clear()
        else:
            try:
                requests.post(f"{API_URL}/reset", params={"session_id": st.session_state.session_id})
            except requests.RequestException:
                pass
        st.rerun()

    st.divider()
    st.caption("Built by Alwin Paul · DistilBERT-style project pattern")

# ---------- Main chat area ----------
st.header("Ask a question about your papers")

if not index_ready():
    st.info("Upload at least one PDF to get started.")

for question, answer, citations in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
        if citations:
            with st.expander(f"{len(citations)} source(s)"):
                for c in citations:
                    st.markdown(
                        f"**{c['doc_name']}** — page {c['page_number']} "
                        f"(relevance: {c['relevance_score']})"
                    )
                    st.caption(c["snippet"])

question = st.chat_input("Ask something about the uploaded papers...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating answer..."):
            try:
                answer, citations = do_ask(question)
            except Exception as e:
                answer, citations = f"Error: {e}", []
        st.write(answer)
        if citations:
            with st.expander(f"{len(citations)} source(s)"):
                for c in citations:
                    st.markdown(
                        f"**{c['doc_name']}** — page {c['page_number']} "
                        f"(relevance: {c['relevance_score']})"
                    )
                    st.caption(c["snippet"])
    st.session_state.chat_history.append((question, answer, citations))
