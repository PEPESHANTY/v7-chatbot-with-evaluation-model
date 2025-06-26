import json
import math
import re
import requests
import pdfplumber
from dotenv import load_dotenv
from typing import List, Tuple
import os

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


def extract_text_from_pdf(file_obj) -> str:
    text_chunks = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text_chunks.append(txt)
    return "\n".join(text_chunks)


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
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ Failed to parse response:", e)
        return ""


def clean_and_validate_response(raw_response, expected_count, source_filename):
    try:
        match = re.search(r"\[\s*{.*?}\s*]", raw_response, re.DOTALL)
        if not match:
            return []

        cleaned = match.group(0)
        data = json.loads(cleaned)

        return [
            {**item, "source": source_filename}
            for item in data
            if isinstance(item, dict)
            and "question" in item
            and "ground_truth" in item
            and "serial number" not in item["question"].lower()
        ][:expected_count]

    except Exception as e:
        print("❌ JSON validation failed:", e)
        return []


def generate_questions(text, n_questions, source_filename):
    slices = 3
    per_slice = math.ceil(n_questions / slices)
    distribution = [per_slice] * slices
    distribution[-1] = n_questions - sum(distribution[:-1])
    chunk_size = 4000
    total_len = len(text)

    zones = [
        ("beginning", text[:chunk_size]),
        ("middle", text[total_len // 2 - chunk_size // 2: total_len // 2 + chunk_size // 2]),
        ("end", text[-chunk_size:])
    ]

    all_qas = []

    for (label, content), count in zip(zones, distribution):
        if count <= 0 or not content.strip():
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
- Phrases like "according to the document" or "in the document", "as listed", or "in this table"
- Questions based on serial numbers, bullet points, or row positions

✅ Instead, prefer questions based on:
- Scientific measurements (e.g., GC content, dosage, genome size)
- Agronomic terminology, disease detection methods, pesticide categories
- Tabular insights (e.g., "Which banned chemicals contain heavy metals?")
- Named entries in lists (e.g., "Is Aldrin listed as a banned pesticide?")
- Trends or classifications (e.g., "How many herbicides are listed?")
- Data-rich statements, field observations, or banned substance characteristics

Ensure each "ground_truth" is strictly based on **quotes or paraphrased content** from this section.

Return ONLY a **single JSON array**. Do not include any markdown (like ```json), explanations, or headings.

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

    return all_qas[:n_questions]

def run_groundtruth_generation(pdf_files: List, questions_per_pdf: int = 5) -> tuple[list, str] | tuple[None, None]:
    all_qas = []

    for file in pdf_files:
        filename = file.name
        try:
            text = extract_text_from_pdf(file)
            qa_pairs = generate_questions(text, questions_per_pdf, filename)
            all_qas.extend(qa_pairs)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    if not all_qas:
        return None, None

    json_str = json.dumps(all_qas, indent=2, ensure_ascii=False)
    return all_qas, json_str
