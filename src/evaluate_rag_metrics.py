import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from rag_evaluator import RAGEvaluator

from ragchain_langchain import qa_chain  # Adjust if in a different module

# Get the latest groundtruth file
def get_latest_groundtruth(path="../evaluation_model/final_data"):
    files = glob.glob(os.path.join(path, "groundtruth_*.json"))
    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return files[-1] if files else None

# Auto-increment CSV output name
def get_next_csv_filename(folder=".", prefix="evaluation_results", ext="csv"):
    existing = glob.glob(os.path.join(folder, f"{prefix}_*.{ext}"))
    nums = [int(f.split("_")[-1].split(".")[0]) for f in existing if "_" in f]
    next_index = max(nums + [0]) + 1
    return os.path.join(folder, f"{prefix}_{next_index}.{ext}")

def evaluate_rag_pipeline():
    groundtruth_path = get_latest_groundtruth("../evaluation_model/final_data")
    if not groundtruth_path:
        print("❌ No groundtruth file found.")
        return

    with open(groundtruth_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    evaluator = RAGEvaluator()
    results = []

    print(f"📊 Evaluating {len(qa_pairs)} questions from: {groundtruth_path}")
    for idx, qa in enumerate(tqdm(qa_pairs), start=1):
        question = qa["question"]
        expected = qa["ground_truth"]

        try:
            output = qa_chain.invoke({"question": question})
            response = output.get("answer", "")
        except Exception as e:
            print(f"⚠️ Error on Q{idx}: {question[:40]}... → {e}")
            response = ""

        metrics = evaluator.evaluate_all(question, response, expected)
        metrics.update({
            "TT": idx,
            "user_input": question,
            "response": response,
            "reference": expected,
        })
        results.append(metrics)

    df = pd.DataFrame(results)
    out_path = get_next_csv_filename(folder="../evaluation_model/rag_evaluator_score", prefix="evaluator_score", ext="csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved: {out_path}")

if __name__ == "__main__":
    evaluate_rag_pipeline()
