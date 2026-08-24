"""Tests for the token-aware chunker."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.chunker import chunk_document, chunk_documents
from src.models import Document, FileType


def _make_doc(text: str, source: str = "test.md") -> Document:
    return Document(
        text=text,
        source_path=source,
        file_type=FileType.MARKDOWN,
    )


class TestChunkDocument:
    """Tests for chunk_document()."""

    def test_short_document_single_chunk(self):
        doc = _make_doc("Hello world, this is a short document.")
        chunks = chunk_document(doc, chunk_size=400, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == "Hello world, this is a short document."

    def test_long_document_multiple_chunks(self):
        # Create a document that should produce multiple chunks
        paragraphs = [f"Paragraph {i}. " * 50 for i in range(10)]
        text = "\n\n".join(paragraphs)
        doc = _make_doc(text)
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        """All chunks should be within reasonable bounds of chunk_size."""
        text = ("This is a test sentence. " * 200).strip()
        doc = _make_doc(text)
        chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)

        for chunk in chunks:
            # Allow some variance since we split on sentence boundaries
            assert chunk.token_count <= 80, (
                f"Chunk has {chunk.token_count} tokens, expected <= 80"
            )

    def test_overlap_exists(self):
        """Adjacent chunks should share some text (overlap)."""
        text = "\n\n".join([f"Topic {i} with details. " * 30 for i in range(5)])
        doc = _make_doc(text)
        chunks = chunk_document(doc, chunk_size=50, chunk_overlap=15)

        if len(chunks) >= 2:
            # Check that the end of chunk N overlaps with the start of chunk N+1
            words_end = set(chunks[0].text.split()[-20:])
            words_start = set(chunks[1].text.split()[:20])
            overlap = words_end & words_start
            assert len(overlap) > 0, "Expected overlap between adjacent chunks"

    def test_metadata_preserved(self):
        doc = _make_doc("Some text.", source="docs/readme.md")
        doc.metadata = {"filename": "readme.md", "extension": ".md"}
        chunks = chunk_document(doc)
        assert chunks[0].source == "docs/readme.md"
        assert chunks[0].metadata.get("filename") == "readme.md"

    def test_chunk_ids_deterministic(self):
        doc = _make_doc("Deterministic test.")
        chunks_1 = chunk_document(doc)
        chunks_2 = chunk_document(doc)
        assert chunks_1[0].id == chunks_2[0].id

    def test_chunk_ids_unique(self):
        text = "\n\n".join([f"Section {i}" * 50 for i in range(5)])
        doc = _make_doc(text)
        chunks = chunk_document(doc, chunk_size=50)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


class TestChunkDocuments:
    """Tests for chunk_documents() (multi-document)."""

    def test_multiple_documents(self):
        docs = [
            _make_doc("Document one content. " * 20, source="one.md"),
            _make_doc("Document two content. " * 20, source="two.md"),
        ]
        chunks = chunk_documents(docs, chunk_size=50)
        assert len(chunks) > 2  # Each doc should produce at least one chunk

    def test_empty_list(self):
        chunks = chunk_documents([])
        assert chunks == []
