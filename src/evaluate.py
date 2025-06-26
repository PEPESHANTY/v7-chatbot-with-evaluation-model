import os
import json
import glob
import pandas as pd
from tqdm import tqdm

from ragchain_langchain import qa_chain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate
from datasets import Dataset


def get_latest_groundtruth(path="../evaluation_model/final_data"):
    files = glob.glob(os.path.join(path, "groundtruth_*.json"))
    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return files[-1] if files else None


def get_next_csv_index(folder: str, prefix: str) -> str:
    pattern = os.path.join(folder, f"{prefix}_*.csv")
    files = glob.glob(pattern)
    nums = []
    for f in files:
        try:
            num = int(os.path.basename(f).split("_")[-1].split(".")[0])
            nums.append(num)
        except:
            continue
    return max(nums + [0]) + 1


def run_predictions(qa_pairs):
    rag_data = []
    print(f"\n🧠 Running RAG model on groundtruth questions...")

    for qa in tqdm(qa_pairs):
        query = qa["question"]
        expected = qa["ground_truth"]

        result = qa_chain.invoke({"question": query})
        predicted = result["answer"]
        context_chunks = [doc.page_content for doc in result["source_documents"]]

        rag_data.append({
            "question": query,
            "ground_truth": expected,
            "answer": predicted,
            "contexts": context_chunks
        })

    return rag_data


def main():
    latest_file = get_latest_groundtruth()
    if not latest_file:
        print("❌ No groundtruth file found in final_data/")
        return

    print(f"📂 Using groundtruth: {latest_file}")
    with open(latest_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    rag_outputs = run_predictions(qa_pairs)
    dataset = Dataset.from_list(rag_outputs)

    # ✅ RAGAS Evaluation
    print("\n📊 Evaluating RAG system with RAGAS metrics...\n")
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    print(results.to_pandas().mean(numeric_only=True).round(3))

    # ✅ Save with numeric increment
    output_dir = "../evaluation_model/evaluation_score_ragas"
    os.makedirs(output_dir, exist_ok=True)

    next_index = get_next_csv_index(output_dir, "RAGAS_score")
    csv_path = os.path.join(output_dir, f"RAGAS_score_{next_index}.csv")

    results.to_pandas().to_csv(csv_path, index=False)
    print(f"\n✅ Evaluation scores saved to {csv_path}")


if __name__ == "__main__":
    main()


# import os
# import json
# import glob
# import pandas as pd
# from tqdm import tqdm
#
# from ragchain_langchain import qa_chain
# from ragas.metrics import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )
# from ragas import evaluate
# from datasets import Dataset
#
# # ✅ Locate latest groundtruth file
# def get_latest_groundtruth(path="../evaluation_model/final_data"):
#     files = glob.glob(os.path.join(path, "groundtruth_*.json"))
#     files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
#     return files[-1] if files else None
#
# # ✅ Run LangChain QA system on each groundtruth pair
# def run_predictions(qa_pairs):
#     rag_data = []
#     print(f"\n🧠 Running RAG model on groundtruth questions...")
#
#     for qa in tqdm(qa_pairs):
#         query = qa["question"]
#         expected = qa["ground_truth"]
#
#         result = qa_chain.invoke({"question": query})
#         predicted = result["answer"]
#
#         context_chunks = [doc.page_content for doc in result["source_documents"]]
#
#         rag_data.append({
#             "question": query,
#             "ground_truth": expected,
#             "answer": predicted,
#             "contexts": context_chunks
#         })
#
#     return rag_data
#
# # ✅ Entry point
# def main():
#     latest_file = get_latest_groundtruth()
#     if not latest_file:
#         print("❌ No groundtruth file found in final_data/")
#         return
#
#     print(f"📂 Using groundtruth: {latest_file}")
#     with open(latest_file, "r", encoding="utf-8") as f:
#         qa_pairs = json.load(f)
#
#     rag_outputs = run_predictions(qa_pairs)
#     dataset = Dataset.from_list(rag_outputs)
#
#     # ✅ RAGAS Evaluation
#     print("\n📊 Evaluating RAG system with RAGAS metrics...\n")
#     results = evaluate(
#         dataset,
#         metrics=[
#             faithfulness,
#             answer_relevancy,
#             context_precision,
#             context_recall,
#         ],
#     )
#
#     print(results.to_pandas().mean(numeric_only=True).round(3))
#
#     # Save results
#     df = results.to_pandas()
#     df.to_csv("../evaluation_model/final_data/evaluation_results.csv", index=False)
#     print("\n✅ Evaluation scores saved to evaluation_results_1.csv")
#
# if __name__ == "__main__":
#     main()
