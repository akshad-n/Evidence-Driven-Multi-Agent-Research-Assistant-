from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logging import logger

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def chunk_document(
    pages_or_text: Any,
    filename: str = "document.pdf",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Splits document text into overlapping chunks with preserved metadata.
    Accepts list of page dicts (with 'page_number' and 'text') or a single string.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks: List[Dict[str, Any]] = []

    if isinstance(pages_or_text, list):
        # List of pages
        chunk_idx = 0
        for page in pages_or_text:
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "")
            if not page_text.strip():
                continue
                
            page_splits = splitter.split_text(page_text)
            for split in page_splits:
                chunks.append({
                    "chunk_id": f"{filename}_p{page_num}_c{chunk_idx}",
                    "content": split,
                    "metadata": {
                        "source": filename,
                        "page": page_num,
                        "chunk_index": chunk_idx
                    }
                })
                chunk_idx += 1
    elif isinstance(pages_or_text, str):
        # Single string
        splits = splitter.split_text(pages_or_text)
        for idx, split in enumerate(splits):
            chunks.append({
                "chunk_id": f"{filename}_c{idx}",
                "content": split,
                "metadata": {
                    "source": filename,
                    "page": 1,
                    "chunk_index": idx
                }
            })

    logger.info(f"Chunked document '{filename}' into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks
