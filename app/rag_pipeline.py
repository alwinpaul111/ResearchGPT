"""
Core RAG orchestration:
  1. embed the user's question
  2. retrieve top-k relevant chunks from the vector store
  3. build a grounded prompt (retrieved context + conversation memory)
  4. call the LLM
  5. return the answer + structured citations (doc name + page number)
"""
from dataclasses import dataclass, field
from typing import List, Dict

from app.vector_store import similarity_search
from app.llm import generate_answer
from app.config import MAX_HISTORY_TURNS, TOP_K


@dataclass
class Citation:
    doc_name: str
    page_number: int
    snippet: str
    relevance_score: float


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]


class ConversationMemory:
    """Simple sliding-window memory of (question, answer) turns."""

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self.turns: List[Dict[str, str]] = []

    def add(self, question: str, answer: str):
        self.turns.append({"question": question, "answer": answer})
        self.turns = self.turns[-self.max_turns:]

    def as_context(self) -> str:
        if not self.turns:
            return ""
        formatted = "\n".join(
            f"Q: {t['question']}\nA: {t['answer']}" for t in self.turns
        )
        return f"Previous conversation:\n{formatted}\n"

    def clear(self):
        self.turns = []


PROMPT_TEMPLATE = """You are ResearchGPT, an assistant that answers questions strictly using \
the provided research paper excerpts. Follow these rules:
- Only use information found in the CONTEXT below. If the answer isn't in the context, say so honestly.
- When you state a fact, refer to it naturally (e.g. "According to the paper...").
- Be precise and concise. Use bullet points for multi-part answers.
- Do not fabricate citations, numbers, or study results.

{history}
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _build_context(results) -> str:
    blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        blocks.append(
            f"[Source {i} | {doc.metadata['doc_name']} | Page {doc.metadata['page_number']}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks)


SUMMARY_TRIGGERS = (
    "what is this paper about",
    "what is the paper about",
    "summarize this paper",
    "summarize the paper",
    "give me a summary",
    "what does this paper discuss",
)


def _is_broad_summary_question(question: str) -> bool:
    q = question.lower().strip()
    return any(trigger in q for trigger in SUMMARY_TRIGGERS)


def answer_question(question: str, memory: ConversationMemory = None, k: int = TOP_K) -> RAGResponse:
    results = similarity_search(question, k=k)

    # Broad "what is this about" questions retrieve poorly with pure
    # similarity search (no chunk closely matches such a generic query).
    # For these, always pull in the earliest-page chunks too, since
    # introductions/abstracts are what actually answer this kind of question.
    if _is_broad_summary_question(question):
        intro_results = similarity_search("introduction background overview objectives", k=k)
        seen_ids = {doc.metadata.get("chunk_id") for doc, _ in results}
        for doc, score in intro_results:
            if doc.metadata.get("chunk_id") not in seen_ids:
                results.append((doc, score))
                seen_ids.add(doc.metadata.get("chunk_id"))

    if not results:
        return RAGResponse(
            answer="No documents have been indexed yet. Please upload a PDF first.",
            citations=[],
        )

    context = _build_context(results)
    history = memory.as_context() if memory else ""
    prompt = PROMPT_TEMPLATE.format(history=history, context=context, question=question)

    answer_text = generate_answer(prompt)

    citations = [
        Citation(
            doc_name=doc.metadata["doc_name"],
            page_number=doc.metadata["page_number"],
            snippet=doc.page_content[:220].strip() + ("..." if len(doc.page_content) > 220 else ""),
            relevance_score=round(float(score), 3),
        )
        for doc, score in results
    ]

    if memory is not None:
        memory.add(question, answer_text)

    return RAGResponse(answer=answer_text, citations=citations)
