# src/retriever.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
from openai.types import Completion
from openai import AsyncOpenAI

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

from openai import OpenAI
openai = OpenAI(api_key=OPENAI_API_KEY)


def get_query_embedding(text: str) -> list:
    """Get OpenAI embedding for a text query."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def retrieve_chunks(query: str, k: int = 4) -> list[dict]:
    """Semantic search from Qdrant with full payloads."""
    query_vector = get_query_embedding(query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k,
        with_payload=True,
        with_vectors=False
    )

    chunks = []
    for r in results:
        payload = r.payload or {}

        chunks.append({
            "score": r.score,
            "content": payload.get("content", ""),
            "title": payload.get("title", ""),
            "summary": payload.get("summary", ""),
            "url": payload.get("url", ""),
            "source": payload.get("source", ""),
            "chunk_id": payload.get("chunk_id", ""),
            "chunk_number": payload.get("chunk_number", -1),
        })

    return chunks
