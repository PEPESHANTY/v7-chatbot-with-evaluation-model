import pandas as pd
from tqdm import tqdm
from io import BytesIO
from rag_evaluator import RAGEvaluator
from ragchain_langchain import qa_chain


def run_rag_evaluator_from_data(qa_pairs: list[dict]):
    try:
        evaluator = RAGEvaluator()
        results = []

        for idx, qa in enumerate(tqdm(qa_pairs), start=1):
            question = qa["question"]
            expected = qa["ground_truth"]

            try:
                output = qa_chain.invoke({"question": question})
                response = output.get("answer", "")
            except Exception as e:
                response = f"⚠️ Error: {e}"

            metrics = evaluator.evaluate_all(question, response, expected)
            metrics.update({
                "TT": idx,
                "user_input": question,
                "response": response,
                "reference": expected,
            })
            results.append(metrics)

        df = pd.DataFrame(results)
        csv_blob = df.to_csv(index=False).encode("utf-8")

        return df, csv_blob, None

    except Exception as e:
        return None, None, f"❌ RAG Evaluator failed: {e}"
