import os

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from tools.crosslingual_retriever import retrieve_chunks_crosslingual
from tools.init_tools import tools


# from retriever import retrieve_chunks

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ✅ LangChain-compatible OpenAI model
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# ✅ Updated prompt template
prompt_template = """You are RiceAI Expert, a trusted agronomist and AI agent trained on sustainable and high-yield rice farming practices.
You have access to expert-level documentation, research papers, extension manuals, and government reports.

Your job is to answer rice farming queries in clear, professional, and practical language — backed by authoritative content.
You always look up the most relevant content using semantic and keyword similarity, and explain answers in depth when possible.

Previous conversation:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ✅ Conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

#retrieve_fn = tools[0].func

# ✅ Custom Retriever as valid BaseRetriever
class QdrantRetriever(BaseRetriever):
    k: int = 5  # ✅ Declare as Pydantic field
    threshold: float = 0.35

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        chunks = retrieve_chunks_crosslingual(query, top_k=self.k)
        #chunks = retrieve_fn(query, top_k=self.k)

        threshold: float = 0.35

        # ✅ filter based on score
        filtered = [chunk for chunk in chunks if chunk["score"] >= self.threshold]
        return [
            Document(
                page_content=chunk["content"],
                metadata={
                    "title": chunk["title"],
                    "summary": chunk["summary"],
                    "url": chunk["url"],
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                }
            )
            for chunk in filtered
        ]

# ✅ Conversational RAG Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=QdrantRetriever(k=5, threshold=0.35),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    #output_key="answer"
)

# ✅ Run loop
if __name__ == "__main__":
    print("🧠 RiceAI Expert Chat (LangChain + ConversationalRAG + Memory ✅)")
    while True:
        query = input("\n❓ Ask your rice-related question (or type 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        result = qa_chain.invoke({"question": query})
        print(f"\n🤖 Answer:\n{result['answer']}")

        # 🔍 Show retrieved chunks (sources)
        print("\n📚 Retrieved Chunks:")
        for i, doc in enumerate(result["source_documents"], 1):
            meta = doc.metadata
            print(f"\n{i}. Title: {meta.get('title', 'N/A')} (score: {meta.get('score', '?.???')})")
            print(f"   ├─ Chunk ID: {meta.get('chunk_id', 'N/A')}")
            print(f"   ├─ URL: {meta.get('url', 'N/A')}")
            print(f"   ├─ Summary: {meta.get('summary', 'N/A')[:150]}...")
            print(f"   └─ Preview: {doc.page_content[:200]}...")

