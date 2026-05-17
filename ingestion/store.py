import os

import chromadb

from ingestion.loader import DocumentChunk

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def _collection_name(company_id: str) -> str:
    # ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens
    return f"co-{company_id.replace('-', '')[:60]}"


def store_chunks(company_id: str, chunks: list[DocumentChunk]) -> None:
    client = _get_client()
    collection = client.get_or_create_collection(_collection_name(company_id))
    ids = [f"{company_id}_{c.source}_{c.chunk_index}" for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [
        {"source": c.source, "page": c.page or 0, "chunk_index": c.chunk_index}
        for c in chunks
    ]
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)


def query_chunks(company_id: str, query: str, n_results: int = 5) -> list[str]:
    client = _get_client()
    try:
        collection = client.get_collection(_collection_name(company_id))
        results = collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


def delete_company_chunks(company_id: str) -> None:
    client = _get_client()
    try:
        client.delete_collection(_collection_name(company_id))
    except Exception:
        pass
