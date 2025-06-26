from openai import OpenAI
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from typing import List, Dict


load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rice_chunks")


def get_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def translate(text: str, target_lang: str) -> str:
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Translate this to {target_lang} in natural fluent style."},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message.content.strip()


def detect_lang(text: str) -> str:
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Detect if this sentence is in English or Vietnamese. Reply only 'en' or 'vie'."},
            {"role": "user", "content": text}
        ]
    )
    lang = completion.choices[0].message.content.strip().lower()
    return "vie" if lang.startswith("vie") else "en"


def retrieve_chunks_crosslingual(query: str, top_k: int = 5) -> list[dict]:
    lang = detect_lang(query)
    query_en = query if lang == "en" else translate(query, "English")
    query_vi = query if lang == "vie" else translate(query, "Vietnamese")

    embedding_en = get_embedding(query_en)
    embedding_vi = get_embedding(query_vi)

    results_en = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding_en,
        limit=top_k,
        with_payload=True
    )
    results_vi = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding_vi,
        limit=top_k,
        with_payload=True
    )

    combined = results_en + results_vi
    best_by_chunk = {}

    for r in combined:
        cid = r.payload.get("chunk_id")
        if cid not in best_by_chunk or r.score > best_by_chunk[cid].score:
            best_by_chunk[cid] = r

    sorted_chunks = sorted(best_by_chunk.values(), key=lambda x: x.score, reverse=True)[:top_k]

    return [
        {
            "score": r.score,
            "content": r.payload.get("content", ""),
            "title": r.payload.get("title", ""),
            "summary": r.payload.get("summary", ""),
            "url": r.payload.get("url", ""),
            "source": r.payload.get("source", ""),
            "chunk_id": r.payload.get("chunk_id", ""),
            "chunk_number": r.payload.get("chunk_number", -1),
        }
        for r in sorted_chunks
    ]
