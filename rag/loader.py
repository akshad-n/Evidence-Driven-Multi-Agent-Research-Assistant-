import io
import re
from typing import Dict, Any, List, Union
from pypdf import PdfReader
from utils.logging import logger

def clean_extracted_text(text: str) -> str:
    """Cleans extracted text by standardizing linebreaks and whitespaces."""
    if not text:
        return ""
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace consecutive horizontal whitespace with single space
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def load_pdf(
    source: Union[str, bytes, io.BytesIO],
    filename: str = "uploaded_paper.pdf"
) -> Dict[str, Any]:
    """
    Extracts text, metadata, and per-page content from a PDF file.
    Supports file path or raw bytes/BytesIO.
    """
    logger.info(f"Extracting PDF text from: {filename}")
    try:
        if isinstance(source, bytes):
            stream = io.BytesIO(source)
        elif isinstance(source, io.BytesIO):
            stream = source
        else:
            stream = open(source, "rb")

        reader = PdfReader(stream)
        num_pages = len(reader.pages)
        
        pages_content: List[Dict[str, Any]] = []
        full_text_parts = []
        
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            cleaned = clean_extracted_text(page_text)
            if cleaned:
                pages_content.append({
                    "page_number": idx + 1,
                    "text": cleaned
                })
                full_text_parts.append(cleaned)

        raw_meta = reader.metadata or {}
        extracted_meta = {
            "title": str(raw_meta.get("/Title", "")).strip() or filename,
            "author": str(raw_meta.get("/Author", "")).strip() or "Not specified",
            "subject": str(raw_meta.get("/Subject", "")).strip(),
            "producer": str(raw_meta.get("/Producer", "")).strip(),
        }

        full_text = "\n\n".join(full_text_parts)
        text_size = len(full_text)

        logger.info(f"Successfully processed PDF '{filename}': {num_pages} pages, {text_size} characters.")
        return {
            "filename": filename,
            "num_pages": num_pages,
            "text_size_chars": text_size,
            "metadata": extracted_meta,
            "pages": pages_content,
            "full_text": full_text,
            "status": "success",
            "error": None
        }

    except Exception as e:
        logger.error(f"Failed to extract PDF '{filename}': {e}")
        return {
            "filename": filename,
            "num_pages": 0,
            "text_size_chars": 0,
            "metadata": {},
            "pages": [],
            "full_text": "",
            "status": "error",
            "error": f"Failed to read PDF document: {str(e)}"
        }
