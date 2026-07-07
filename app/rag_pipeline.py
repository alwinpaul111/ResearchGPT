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

from app.vector_store import similarity_search, get_first_page_chunks
from app.llm import generate_answer
from app.config import MAX_HISTORY_TURNS, TOP_K

MAX_BROAD_QUESTION_CHUNKS = 8
MAX_CHUNKS_PER_DOC_FOR_BROAD_QUESTIONS = 3


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


PROMPT_TEMPLATE = """You are PaperLens, an assistant that answers questions strictly using \
the provided research paper excerpts. Follow these rules:
- Only use information found in the CONTEXT below. If the answer isn't in the context, say so honestly.
- The CONTEXT may contain excerpts from more than one document, each labeled with its document name.
  If multiple documents appear, treat each one separately and address them individually rather than
  blending their content together. Do not confuse a document's reference list (its bibliography,
  citing other authors' work) with the content of the document itself.
- If the question refers to "paper1", "paper2", "the first paper", "the second paper", or similar,
  this means the first and second documents listed in the CONTEXT by upload order, not a literal
  search for text matching "paper1". Map these references to the actual document names shown.
- Write your answer as clean prose in your own words. Do not copy structural labels like
  "[Page 3]" or "=== Document: ... ===" into your answer — those are for your reference only.
  Do not repeat the same sentence more than once.
- Answer only the QUESTION given below. Do not invent, ask, or answer any other question,
  and do not continue the conversation past your answer to this one question.
- When you state a fact, refer to it naturally (e.g. "According to the paper...").
- Be precise and concise.
- For multi-part answers, write each point as a short complete sentence on its own line,
  starting with a plain dash "- " followed immediately by the sentence text (e.g. "- Self-attention connects all positions in constant time.").
  Never output a bullet marker with no text after it.
- Do not fabricate citations, numbers, or study results.

{history}
CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _build_context(results, max_chars_per_chunk: int = 900) -> str:
    # Group chunks by document name so the LLM sees "N excerpts from document X",
    # not a flat numbered list that can be misread as N separate documents.
    by_doc = {}
    for doc, score in results:
        name = doc.metadata["doc_name"]
        by_doc.setdefault(name, []).append(doc)

    doc_names = list(by_doc.keys())
    blocks = [f"There are {len(doc_names)} document(s) in this context: {', '.join(doc_names)}.\n"]

    for name, docs in by_doc.items():
        blocks.append(f"=== Document: {name} ({len(docs)} excerpt(s)) ===")
        for doc in docs:
            text = doc.page_content
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "..."
            blocks.append(f"[Page {doc.metadata['page_number']}]\n{text}")

    return "\n\n".join(blocks)


SUMMARY_TRIGGERS = (
    "what is this paper about",
    "what is the paper about",
    "summarize this paper",
    "summarize the paper",
    "summarize both",
    "summarize each",
    "give me a summary",
    "what does this paper discuss",
    "who are the authors",
    "who is the author",
    "who wrote this paper",
    "explain paper",
    "explain both",
    "compare the paper",
    "compare both",
    "what are these papers",
    "what are the papers",
)


def _is_broad_summary_question(question: str) -> bool:
    q = question.lower().strip()
    if any(trigger in q for trigger in SUMMARY_TRIGGERS):
        return True
    # Catches phrasing like "paper1 and paper2", "both papers", "each paper"
    if ("paper1" in q or "paper 1" in q or "both papers" in q or "each paper" in q) :
        return True
    return False


def answer_question(question: str, memory: ConversationMemory = None, k: int = TOP_K) -> RAGResponse:
    results = similarity_search(question, k=k)

    # Broad "what is this about" questions retrieve poorly with pure
    # similarity search (no chunk closely matches such a generic query).
    # For these, always pull in the earliest-page chunks too, since
    # introductions/abstracts are what actually answer this kind of question.
    if _is_broad_summary_question(question):
        seen_ids = {doc.metadata.get("chunk_id") for doc, _ in results}

        # Page 1 almost always has the title, authors, and abstract/intro —
        # pull it in directly rather than hoping similarity search finds it.
        for doc, score in get_first_page_chunks():
            if doc.metadata.get("chunk_id") not in seen_ids:
                results.append((doc, score))
                seen_ids.add(doc.metadata.get("chunk_id"))

        intro_results = similarity_search("introduction background overview objectives authors", k=k)
        for doc, score in intro_results:
            if doc.metadata.get("chunk_id") not in seen_ids:
                results.append((doc, score))
                seen_ids.add(doc.metadata.get("chunk_id"))

        # Cap chunks per document first, so with multiple documents no single
        # one dominates the context and every document stays represented.
        per_doc_counts = {}
        capped_results = []
        for doc, score in results:
            name = doc.metadata["doc_name"]
            per_doc_counts[name] = per_doc_counts.get(name, 0)
            if per_doc_counts[name] < MAX_CHUNKS_PER_DOC_FOR_BROAD_QUESTIONS:
                capped_results.append((doc, score))
                per_doc_counts[name] += 1
        results = capped_results

        # Cap total chunks sent to the LLM - merging in page-1 + intro search
        # on top of the original top-k can otherwise blow past the LLM
        # provider's tokens-per-minute rate limit on a single request, and
        # overwhelm a small model's ability to synthesize coherently.
        results = results[:MAX_BROAD_QUESTION_CHUNKS]

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
