import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from ragchain_langchain import qa_chain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate


def run_ragas_evaluation_from_data(qa_pairs: list[dict]):
    if not qa_pairs or not isinstance(qa_pairs, list):
        return None, None, "❌ Invalid QA pairs provided."

    rag_data = []
    for qa in tqdm(qa_pairs, desc="Running RAG Chain"):
        query = qa["question"]
        expected = qa["ground_truth"]

        try:
            result = qa_chain.invoke({"question": query})
            predicted = result["answer"]
            context_chunks = [doc.page_content for doc in result["source_documents"]]
        except Exception as e:
            predicted = "Error: " + str(e)
            context_chunks = []

        rag_data.append({
            "question": query,
            "ground_truth": expected,
            "answer": predicted,
            "contexts": context_chunks
        })

    dataset = Dataset.from_list(rag_data)

    try:
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = results.to_pandas()
    except Exception as e:
        return None, None, f"❌ RAGAS Evaluation failed: {e}"

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return df, csv_bytes, None
