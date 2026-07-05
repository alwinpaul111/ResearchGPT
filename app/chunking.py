"""
Chunking: splits page-level text into overlapping chunks small enough
to embed meaningfully, while keeping track of which document + page
each chunk came from (needed for citations).
"""
from dataclasses import dataclass, field
from typing import List
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.pdf_loader import PageContent
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    id: str
    text: str
    doc_name: str
    page_number: int
    chunk_index: int


def chunk_pages(pages: List[PageContent]) -> List[Chunk]:
    """Split each page's text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for page in pages:
        pieces = splitter.split_text(page.text)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=piece,
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    chunk_index=idx,
                )
            )
    return chunks
