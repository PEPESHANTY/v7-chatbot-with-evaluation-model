# src/read_qdrant_live_chunk.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def print_nested_payload(payload, indent=0):
    """Print nested payload keys and values nicely."""
    for key, value in payload.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_nested_payload(value, indent + 1)
        else:
            preview = value if isinstance(value, str) and len(value) < 200 else str(value)[:200] + "..."
            print("  " * indent + f"{key}: {preview}")

def fetch_chunk_by_id(chunk_id: int):
    load_dotenv()

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    collection = os.getenv("QDRANT_COLLECTION_NAME")

    result = client.retrieve(
        collection_name=collection,
        ids=[chunk_id],
        with_vectors=True,
        with_payload=True
    )

    if not result:
        print(f"❌ No chunk found with ID {chunk_id}")
        return

    point = result[0]
    print("\n📦 CHUNK DUMP FROM QDRANT")
    print("=" * 60)
    print(f"🆔 ID: {point.id}")
    print(f"🧠 Vector: ({len(point.vector)} dimensions)\n")

    print("📑 Payload:")
    print_nested_payload(point.payload)


if __name__ == "__main__":
    fetch_chunk_by_id(24350)
