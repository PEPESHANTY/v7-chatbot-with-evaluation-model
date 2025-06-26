# src/raw_dump_qdrant_chunk.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")


def fetch_chunk_raw(id: int = 1):
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    response = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[id],
        with_payload=True,
        with_vectors=True
    )

    if not response:
        print(f"❌ No chunk found for ID {id}")
        return

    chunk = response[0]
    payload = chunk.payload

    print(f"\n✅ Raw Chunk ID {id}")
    print("=" * 50)
    for k, v in payload.items():
        preview = v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v
        print(f"{k}: {preview}")


if __name__ == "__main__":
    fetch_chunk_raw(1)
