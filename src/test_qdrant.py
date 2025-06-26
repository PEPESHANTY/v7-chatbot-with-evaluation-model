# src/test_qdrant.py

from retriever import retrieve_chunks

def main():
    while True:
        query = input("\n❓ Ask your question (or 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        print("\n🔎 Searching Qdrant...\n")
        chunks = retrieve_chunks(query, k=4)

        for i, chunk in enumerate(chunks, 1):
            print(f"\n🔹 Chunk {i} (Score: {chunk['score']:.3f})")
            print(f"📄 Title: {chunk['title']}")
            print(f"📌 Summary: {chunk['summary'][:100]}")
            print(f"🌐 URL: {chunk['url']}")
            print(f"📚 Source: {chunk['source']}")
            print(f"🆔 Chunk ID: {chunk['chunk_id']} | #: {chunk['chunk_number']}")
            print(f"📝 Content: {chunk['content'][:300]}...\n")

if __name__ == "__main__":
    main()
