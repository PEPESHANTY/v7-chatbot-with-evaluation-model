from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection = os.getenv("QDRANT_COLLECTION_NAME")

# Scan from a high number down manually
START = 23700  # or whatever your guess is
END = 23600


found = []
for chunk_id in range(START, END, -1):
    result = client.retrieve(collection_name=collection, ids=[chunk_id], with_payload=True, with_vectors=False)
    if result:
        found.append(result[0])

# Print them
for pt in found:
    print(f"🧠 Chunk ID: {pt.id}")
    print(pt.payload.get("title", "❓ No title"))
    print("-" * 50)
