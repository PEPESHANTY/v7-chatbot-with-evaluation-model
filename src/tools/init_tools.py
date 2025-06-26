from langchain.agents import Tool
from .crosslingual_retriever import retrieve_chunks_crosslingual

tools = [
    Tool(
        name="SearchRiceDocs",
        func=retrieve_chunks_crosslingual,
        description=(
            "Searches the rice farming knowledge base (English + Vietnamese) "
            "using a cross-lingual semantic search engine. Always use this when answering rice queries."
        )
    )
]
