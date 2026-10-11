import re
import os
from typing import Tuple

MAX_PDF_SIZE_MB = 25
MAX_PDF_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024

def sanitize_filename(filename: str) -> str:
    """Sanitizes file name to prevent path traversal and unsafe characters."""
    base = os.path.basename(filename)
    clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    return clean or "document.pdf"

def validate_uploaded_pdf(filename: str, content: bytes, max_size_mb: int = MAX_PDF_SIZE_MB) -> Tuple[bool, str]:
    """Validates that the uploaded file is a legitimate PDF within size limits."""
    if not filename.lower().endswith(".pdf"):
        return False, "Invalid file format. Only .pdf files are permitted."
    
    max_bytes = max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        return False, f"File size exceeds maximum allowed limit of {max_size_mb}MB."
    
    if len(content) < 4 or not content.startswith(b"%PDF"):
        return False, "File does not appear to be a valid PDF document (missing PDF magic header)."
        
    return True, "Valid PDF"

def validate_research_question(question: str) -> Tuple[bool, str]:
    """Validates that a research question is sufficiently detailed."""
    q = question.strip()
    if not q:
        return False, "Research question cannot be empty."
    if len(q) < 5:
        return False, "Please provide a more descriptive research question (at least 5 characters)."
    return True, "Valid question"
