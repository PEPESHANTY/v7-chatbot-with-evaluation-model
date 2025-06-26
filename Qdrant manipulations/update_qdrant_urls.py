from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rice_chunks")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# 🔁 Your old→new link mapping
pdf_url_update_map = {
    "pdf://vietnam_and_irri,_a_partnership_in_rice_research_proceedings_of_a_conference_held_in_hanoi_vietnam":"https://drive.google.com/file/d/1_0StpYWNC7mutqP0kd9SUKNSIBXHzsxl/view?usp=drive_link",
    "pdf://vinh_long_province,_vietnam":"https://drive.google.com/file/d/1scitZP5JM4VxArpeN1-ePpVU5bKRTMqD/view?usp=drive_link",
    "pdf://rice_soil_fertility_classification_in_the_mekong_d":"https://drive.google.com/file/d/1e7egyDeUEUcLKP5qvJPt9-QjPbFau6-B/view?usp=drive_link",
    "":"",
    "":"",
    "":"",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    "": "",
    # Add more as needed...
}

# 🔍 For each old URL, find and update all matching chunks
for old_url, new_url in pdf_url_update_map.items():
    print(f"🔄 Updating chunks with URL: {old_url}")

    # Search for all chunks with this URL
    response = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={
            "must": [
                {
                    "key": "url",
                    "match": {"value": old_url}
                }
            ]
        },
        with_payload=True,
        limit=1000  # set large limit if needed
    )

    points = response[0]
    for point in points:
        client.set_payload(
            collection_name=COLLECTION_NAME,
            points=[point.id],
            payload={"url": new_url}
        )
        print(f"✅ Updated point {point.id} to: {new_url}")
