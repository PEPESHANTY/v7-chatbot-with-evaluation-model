from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rice_chunks")

# ✅ Create an index on the "url" field so it can be filtered
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="url",
    field_schema="keyword"  # 🚨 must be "keyword" for filtering
)

print("✅ Payload index created for field: url")
