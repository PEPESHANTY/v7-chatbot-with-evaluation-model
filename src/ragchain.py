# src/ragchain.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import retrieve_chunks

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")
openai = OpenAI(api_key=OPENAI_API_KEY)


def build_context_string(chunks: list[dict]) -> str:
    """Convert retrieved chunks into a structured text block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"Source {i} (score: {chunk['score']:.3f}):\n"
            f"Title: {chunk['title']}\n"
            f"Summary: {chunk['summary']}\n"
            f"Content: {chunk['content']}"
        )
    return "\n\n".join(parts)


def generate_answer(query: str, context: str) -> str:
    """Call GPT model with system prompt, context and query."""
    system_prompt = (
        "You are RiceAI Expert, a helpful assistant trained on sustainable rice farming methods in Vietnam.\n"
        "Use the following context from expert documents to answer the user's question in a clear and helpful way.\n"
        "Only rely on the retrieved chunks. If you don’t know the answer, say so."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]

    response = openai.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )

    return response.choices[0].message.content.strip()


def run_rag(query: str, k: int = 4):
    chunks = retrieve_chunks(query, k=k)

    if not chunks:
        print("❌ No relevant content found.")
        return

    context = build_context_string(chunks)
    answer = generate_answer(query, context)

    print(f"\n🤖 Answer:\n{answer}\n")

    print("📚 Sources used:\n")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. Title: {c['title'] or '[missing]'} (score: {c['score']:.3f})")
        print(f"   ├─ Chunk ID: {c['chunk_id']}")
        print(f"   ├─ URL: {c['url']}")
        print(f"   ├─ Source: {c['source']}")
        print(f"   └─ Preview: {c['content'][:200]}...\n")


if __name__ == "__main__":
    while True:
        q = input("\n❓ Ask something (or type 'exit'): ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        run_rag(q)
