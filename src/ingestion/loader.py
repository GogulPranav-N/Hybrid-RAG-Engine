"""Load documents from the data directory (markdown, PDF, Python files)."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from src.models import Document, FileType

logger = logging.getLogger(__name__)

# Map file extensions to our FileType enum
_EXT_MAP: dict[str, FileType] = {
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".pdf": FileType.PDF,
    ".py": FileType.PYTHON,
    ".txt": FileType.TEXT,
    ".rst": FileType.TEXT,
}

_SUPPORTED_EXTENSIONS = set(_EXT_MAP.keys())


def _load_text_file(path: Path) -> str:
    """Read a plain text / markdown / python file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    text_parts: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n\n".join(text_parts)


def load_file(path: Path) -> Document | None:
    """Load a single file into a Document, or None if unsupported."""
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        logger.debug("Skipping unsupported file: %s", path)
        return None

    file_type = _EXT_MAP[ext]

    try:
        if file_type == FileType.PDF:
            text = _load_pdf(path)
        else:
            text = _load_text_file(path)
    except Exception:
        logger.exception("Failed to load file: %s", path)
        return None

    if not text.strip():
        logger.warning("Empty file skipped: %s", path)
        return None

    return Document(
        text=text,
        source_path=str(path),
        file_type=file_type,
        metadata={"filename": path.name, "extension": ext},
    )


def load_directory(data_dir: str | Path) -> list[Document]:
    """
    Recursively load all supported files from a directory.

    Returns a list of Document objects with text and metadata.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    documents: list[Document] = []
    for path in sorted(data_path.rglob("*")):
        if path.is_file():
            doc = load_file(path)
            if doc is not None:
                documents.append(doc)

    logger.info("Loaded %d documents from %s", len(documents), data_dir)
    return documents
