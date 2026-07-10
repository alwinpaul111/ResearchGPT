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
import re

from app.vector_store import similarity_search, get_first_page_chunks
from app.llm import generate_answer
from app.config import MAX_HISTORY_TURNS, TOP_K

MAX_BROAD_QUESTION_CHUNKS = 10
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


def _strip_invented_followup(text: str) -> str:
    """Defensive cleanup: even with explicit prompt instructions, models can
    still append a self-invented 'Q: ... A: ...' continuation past the real
    answer. Truncate at the first such marker if one appears."""
    import re
    match = re.search(r'\n\s*Q\s*[:.]', text)
    if match:
        text = text[:match.start()].rstrip()
    return text


def _strip_repetition_loop(text: str, max_repeats: int = 1) -> str:
    """Defensive cleanup: models can get stuck restating a near-identical
    sentence over and over (a known degenerate-generation failure mode,
    especially with noisy/garbled context from math-heavy PDF extraction).
    Truncate the answer at the point a sentence starts repeating."""
    import re
    # Split on sentence-ending punctuation while keeping it simple - this
    # doesn't need to be perfect, just needs to catch obvious loops.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = {}
    output = []
    for sentence in sentences:
        # Normalize: ignore case, digits, and whitespace differences so
        # "(15) The reparameterization trick..." vs "(16) The
        # reparameterization trick..." are still recognized as the same
        # repeated content.
        normalized = re.sub(r'\d+', '', sentence.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        if not normalized:
            output.append(sentence)
            continue
        seen[normalized] = seen.get(normalized, 0) + 1
        if seen[normalized] > max_repeats:
            break
        output.append(sentence)
    return ' '.join(output).strip()


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
    "give me a summary",
    "what does this paper discuss",
    "who are the authors",
    "who is the author",
    "who wrote this paper",
    "what are these papers",
    "what are the papers",
)


def _is_broad_summary_question(question: str) -> bool:
    q = question.lower().strip()
    if any(trigger in q for trigger in SUMMARY_TRIGGERS):
        return True
    # Catch any variant of "summarize"/"summary"/"summarise" regardless of
    # what follows it ("summarize all three papers", "summarise each one",
    # "give a summary of every paper", etc.) rather than matching a fixed
    # list of exact phrases, which breaks on any new wording.
    if "summar" in q:
        return True
    # Catch "explain"/"compare" combined with any mention of multiple papers.
    if ("explain" in q or "compare" in q or "describe" in q) and (
        "paper" in q or "document" in q
    ):
        return True
    # Catch explicit multi-document references: "paper1", "both papers",
    # "each paper", "all three papers", "every paper", numeric counts, etc.
    if re.search(r'\bpaper\s*\d\b', q):
        return True
    if any(phrase in q for phrase in (
        "both papers", "each paper", "all papers", "every paper",
        "all three", "all the papers", "these papers", "these documents",
    )):
        return True
    return False


def answer_question(question: str, memory: ConversationMemory = None, k: int = TOP_K) -> RAGResponse:
    initial_results = similarity_search(question, k=k)

    # Broad "what is this about" questions retrieve poorly with pure
    # similarity search (no chunk closely matches such a generic query).
    # For these, always pull in the earliest-page chunks too, since
    # introductions/abstracts/bylines are what actually answer this kind
    # of question.
    if _is_broad_summary_question(question):
        # Build the merged pool in PRIORITY order: page-1 chunks first
        # (guarantees title/author info survives the per-doc cap below),
        # then intro-search results, then the original similarity search
        # last. If page-1 chunks went last, a question like "who are the
        # authors" - which also matches a paper's own reference list well -
        # could fill the per-doc cap with citation text before the real
        # byline chunk ever got a chance to be included.
        seen_ids = set()
        results = []

        for doc, score in get_first_page_chunks():
            if doc.metadata.get("chunk_id") not in seen_ids:
                results.append((doc, score))
                seen_ids.add(doc.metadata.get("chunk_id"))

        intro_results = similarity_search("introduction background overview objectives authors", k=k)
        for doc, score in intro_results:
            if doc.metadata.get("chunk_id") not in seen_ids:
                results.append((doc, score))
                seen_ids.add(doc.metadata.get("chunk_id"))

        for doc, score in initial_results:
            if doc.metadata.get("chunk_id") not in seen_ids:
                results.append((doc, score))
                seen_ids.add(doc.metadata.get("chunk_id"))

        # Cap chunks per document, so with multiple documents no single
        # one dominates the context and every document stays represented.
        # Because results are now in priority order, page-1 content always
        # claims its slot before lower-priority chunks compete for the cap.
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
    else:
        results = initial_results

    if not results:
        return RAGResponse(
            answer="No documents have been indexed yet. Please upload a PDF first.",
            citations=[],
        )

    context = _build_context(results)
    history = memory.as_context() if memory else ""
    prompt = PROMPT_TEMPLATE.format(history=history, context=context, question=question)

    answer_text = generate_answer(prompt)
    answer_text = _strip_invented_followup(answer_text)
    answer_text = _strip_repetition_loop(answer_text)

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
