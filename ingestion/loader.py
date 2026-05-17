from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int | None = None
    chunk_index: int = 0


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".html", ".htm"}
WORDS_PER_CHUNK = 400


def load_file(file_path: Path) -> list[DocumentChunk]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        return _load_docx(file_path)
    elif suffix in (".txt", ".md"):
        return _load_text(file_path)
    elif suffix == ".csv":
        return _load_csv(file_path)
    elif suffix in (".html", ".htm"):
        return _load_html(file_path)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")


def _load_pdf(path: Path) -> list[DocumentChunk]:
    import fitz  # pymupdf
    doc = fitz.open(str(path))
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            for i, chunk_text in enumerate(_split_text(text)):
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=path.name,
                    page=page_num + 1,
                    chunk_index=len(chunks) + i,
                ))
    doc.close()
    return chunks


def _load_docx(path: Path) -> list[DocumentChunk]:
    from docx import Document
    doc = Document(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [
        DocumentChunk(text=chunk, source=path.name, chunk_index=i)
        for i, chunk in enumerate(_split_text(full_text))
    ]


def _load_text(path: Path) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        DocumentChunk(text=chunk, source=path.name, chunk_index=i)
        for i, chunk in enumerate(_split_text(text))
    ]


def _load_csv(path: Path) -> list[DocumentChunk]:
    # CSVs are treated as a single structured chunk — not split
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [DocumentChunk(text=text[:8000], source=path.name, chunk_index=0)]


def _load_html(path: Path) -> list[DocumentChunk]:
    from bs4 import BeautifulSoup
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    return [
        DocumentChunk(text=chunk, source=path.name, chunk_index=i)
        for i, chunk in enumerate(_split_text(text))
    ]


def _split_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return [text] if text.strip() else []
    chunks = []
    for i in range(0, len(words), WORDS_PER_CHUNK):
        chunk = " ".join(words[i:i + WORDS_PER_CHUNK])
        if chunk.strip():
            chunks.append(chunk)
    return chunks or [text]
