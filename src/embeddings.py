# src/embeddings.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBED_MODEL_NAME = "text-embedding-3-small"

def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBED_MODEL_NAME,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
