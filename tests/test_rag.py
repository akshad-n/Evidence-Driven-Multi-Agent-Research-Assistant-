import io
import pytest
from pypdf import PdfWriter
from rag.loader import load_pdf
from rag.chunker import chunk_document
from rag.embeddings import DeterministicDemoEmbeddings
from rag.vector_store import VectorStoreManager

def create_sample_pdf_bytes(text: str = "Generative AI improves clinical diagnosis accuracy.") -> bytes:
    """Helper to generate a minimal valid PDF in memory for testing."""
    # We can create a blank page with annotations or use PdfWriter to write a minimal valid PDF
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Put text in metadata
    writer.add_metadata({"/Title": "Test Paper", "/Author": "Dr. Researcher"})
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()

def test_pdf_loader():
    pdf_bytes = create_sample_pdf_bytes()
    result = load_pdf(pdf_bytes, filename="test.pdf")
    
    assert result["status"] == "success"
    assert result["num_pages"] == 1
    assert result["metadata"]["title"] == "Test Paper"
    assert result["metadata"]["author"] == "Dr. Researcher"

def test_pdf_loader_invalid():
    result = load_pdf(b"Not a real PDF", filename="corrupted.pdf")
    assert result["status"] == "error"
    assert result["error"] is not None

def test_chunker():
    sample_text = (
        "Generative Artificial Intelligence is transforming modern medicine. "
        "Automated documentation decreases clinical burden. "
        "However, hallucinations present a major clinical safety challenge. "
    ) * 10
    chunks = chunk_document(sample_text, filename="med_ai.pdf", chunk_size=200, chunk_overlap=50)
    
    assert len(chunks) > 1
    for c in chunks:
        assert "chunk_id" in c
        assert "content" in c
        assert c["metadata"]["source"] == "med_ai.pdf"

def test_embeddings():
    emb = DeterministicDemoEmbeddings(dimension=384)
    vec1 = emb.embed_query("Clinical diagnosis and healthcare")
    vec2 = emb.embed_query("Transformers in clinical medicine")
    
    assert len(vec1) == 384
    assert len(vec2) == 384
    assert isinstance(vec1[0], float)

def test_vector_store():
    # Use ephemeral/temp dir
    manager = VectorStoreManager(
        persist_directory="./test_chroma_db",
        collection_name="test_collection",
        demo_mode=True
    )
    test_chunks = [
        {
            "chunk_id": "c1",
            "content": "Deep learning models enhance diagnostic accuracy in radiology and pathology.",
            "metadata": {"source": "radiology.pdf", "page": 1}
        },
        {
            "chunk_id": "c2",
            "content": "Inference latency is high for large multi-modal transformer architectures.",
            "metadata": {"source": "latency.pdf", "page": 3}
        }
    ]
    added = manager.add_chunks(test_chunks, session_id="test_session")
    assert added == 2
    
    results = manager.similarity_search("radiology diagnostic accuracy", k=1, session_id="test_session")
    assert len(results) >= 1
    assert "radiology" in results[0]["content"].lower()
