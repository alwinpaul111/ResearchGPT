
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


@dataclass
class PageContent:
    doc_name: str
    page_number: int  # 1-indexed
    text: str


def extract_pages(pdf_path: str) -> List[PageContent]:
    """Extract text page-by-page from a single PDF file."""
    doc_name = Path(pdf_path).name
    pages: List[PageContent] = []

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:  # skip blank pages (e.g. figure-only pages)
                pages.append(PageContent(doc_name=doc_name, page_number=i + 1, text=text))

    return pages


def extract_pages_from_multiple(pdf_paths: List[str]) -> List[PageContent]:
    """Extract text from multiple PDFs, tagging each page with its source file."""
    all_pages: List[PageContent] = []
    for path in pdf_paths:
        all_pages.extend(extract_pages(path))
    return all_pages
