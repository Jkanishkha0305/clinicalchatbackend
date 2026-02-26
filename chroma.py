"""
ChromaDB Cloud singleton — no Flask dependency.
"""
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

_chroma_client = None
_chroma_collection = None


def get_chroma_collection():
    """Return (and lazily initialise) the ChromaDB collection singleton."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE", "clinicalchat")

    if not api_key or not tenant:
        raise RuntimeError("CHROMA_API_KEY and CHROMA_TENANT must be set")

    _chroma_client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )
    _chroma_collection = _chroma_client.get_or_create_collection(
        name="clinical_trials_embeddings"
    )
    return _chroma_collection
