import os
import json
import glob
import math
import re
import pdfplumber
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

QUESTIONS_PER_PDF = 5
PDF_DIR = "data_eval"
PDF_FILES = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

def extract_text_from_pdf(pdf_path):
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_chunks.append(txt)
    return "\n".join(text_chunks)

def get_next_eval_index(directory: str, prefix: str, extension: str) -> int:
    pattern = os.path.join(directory, f"{prefix}_*.{extension}")
    files = glob.glob(pattern)
    nums = []
    for f in files:
        try:
            n = int(os.path.basename(f).split("_")[1].split(".")[0])
            nums.append(n)
        except:
            continue
    return max(nums + [0]) + 1

def call_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        return ""

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        print("🧾 Raw DeepSeek Response:\n", content[:400])
        return content
    except Exception as e:
        print("❌ Failed to parse DeepSeek response:", e)
        print("📦 Raw response text:\n", response.text[:400])
        return ""

def clean_and_validate_response(raw_response, expected_count, source_filename):
    """
    Extracts a JSON array from messy markdown-like DeepSeek output.
    Validates structure, adds source, filters bad questions.
    """
    try:
        match = re.search(r"\[\s*{.*?}\s*]", raw_response, re.DOTALL)
        if not match:
            print("❌ Could not locate JSON array in response.")
            return []

        cleaned = match.group(0)
        data = json.loads(cleaned)

        valid_qas = []
        for item in data:
            if isinstance(item, dict) and "question" in item and "ground_truth" in item:
                if "serial number" in item["question"].lower():
                    continue
                item["source"] = source_filename
                valid_qas.append(item)

        return valid_qas[:expected_count]

    except Exception as e:
        print("❌ JSON parse or validation failed:", e)
        print("🔍 Raw response preview:\n", raw_response[:400])
        return []

def generate_questions(text, n_questions, source_filename):
    slices = 3
    per_slice = math.ceil(n_questions / slices)
    distribution = [per_slice] * slices
    distribution[-1] = n_questions - sum(distribution[:-1])  # Adjust last slice

    chunk_size = 4000
    total_len = len(text)

    zones = [
        ("beginning", text[:chunk_size]),
        ("middle", text[total_len // 2 - chunk_size // 2: total_len // 2 + chunk_size // 2]),
        ("end", text[-chunk_size:])
    ]

    all_qas = []

    for (label, content), count in zip(zones, distribution):
        if count <= 0:
            continue

        prompt = f"""
You are a research assistant evaluating a RAG system.

The section below may contain explanatory text, structured tables, or itemized lists.

Return only a JSON array of {count} complex, document-specific QA pairs based strictly on the **{label}** section of this document.

Each question must:
- Be grounded in the specific content of this section — not general knowledge
- Require domain-specific understanding (science, agriculture, or regulation)
- Be unanswerable without access to this portion of the document

⚠️ Avoid:
- Questions like "What is the serial number of X?" or anything based on index/list position
- Simple lookups that depend only on enumeration or obvious lists
- Generic trivia questions with no document grounding
- Legal or policy context unless it is **explicitly written** in the text

✅ Instead, prefer questions based on:
- Scientific measurements (e.g., GC content, dosage, genome size)
- Agronomic terminology, disease detection methods, pesticide categories
- Tabular insights (e.g., "Which banned chemicals contain heavy metals?")
- Named entries in lists (e.g., "Is Aldrin listed as a banned pesticide?")
- Trends or classifications (e.g., "How many herbicides are listed?")
- Data-rich statements, field observations, or banned substance characteristics

Ensure each "ground_truth" is strictly based on **quotes or paraphrased content** from this section.

Return ONLY a **single JSON array**. Do not include any markdown (like ```json), explanations, or headings.

Format:
[
  {{
    "question": "...",
    "ground_truth": "...",
    "source": "{source_filename}"
  }},
  ...
]

Document zone: {label}
Text:
{content}
        """

        print(f"🧠 Prompting DeepSeek for {label} ({count} questions)...")
        raw = call_deepseek(prompt)
        if not raw.strip():
            print(f"⚠️ Empty response from {label} zone of {source_filename}")
            continue

        qas = clean_and_validate_response(raw, count, source_filename)
        all_qas.extend(qas)

    if not all_qas:
        print(f"⚠️ No valid QAs generated for {source_filename}")
    return all_qas[:n_questions]

def main():
    all_qas = []

    for pdf in PDF_FILES:
        pdf_path = os.path.join("data_eval", pdf)
        print(f"\n📄 Processing: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        qa_pairs = generate_questions(text, QUESTIONS_PER_PDF, pdf)
        all_qas.extend(qa_pairs)

    index = get_next_eval_index("final_data", "groundtruth", "json")
    out_path = os.path.join("final_data", f"groundtruth_{index}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_qas, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Groundtruth saved: {out_path}")

if __name__ == "__main__":
    main()
