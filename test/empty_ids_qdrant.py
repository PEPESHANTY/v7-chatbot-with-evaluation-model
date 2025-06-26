import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collection = os.getenv("QDRANT_COLLECTION_NAME")

empty_ids = []

# Loop in batches to avoid rate limits or memory issues
BATCH_SIZE = 50
for i in range(0, 23716, BATCH_SIZE):
    end = min(i + BATCH_SIZE, 23716)
    ids_batch = list(range(i, end))
    try:
        results = client.retrieve(
            collection_name=collection,
            ids=ids_batch,
            with_payload=True,
            with_vectors=False
        )

        for point in results:
            payload = point.payload or {}
            # Check for completely empty or missing important fields
            if (
                not payload or
                not payload.get("content") or
                not payload.get("title") or
                not payload.get("summary")
            ):
                empty_ids.append(point.id)

    except Exception as e:
        print(f"⚠️ Error retrieving IDs {i}–{end}: {e}")

# Output results
print("\n🧹 Empty Payload IDs Found:")
print(empty_ids)
print(f"\n🔢 Total empty IDs: {len(empty_ids)}")
