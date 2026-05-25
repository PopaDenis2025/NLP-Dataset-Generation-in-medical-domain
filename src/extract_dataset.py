import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from crewai import Task, Crew, Process

from src.config import RAW_PAGES_JSONL, INTERMEDIATE_DIR
from src.agents import create_medical_extractor_agent


load_dotenv()

MODEL_USED = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OUTPUT_FILE = INTERMEDIATE_DIR / "extracted_records.jsonl"


def load_raw_pages(limit: int | None = None) -> List[Dict[str, Any]]:
    pages = []

    with open(RAW_PAGES_JSONL, mode="r", encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)

            if page.get("status") == "ok":
                pages.append(page)

            if limit is not None and len(pages) >= limit:
                break

    return pages


def trim_text(text: str, max_chars: int = 7000) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars]


def build_extraction_task(page: Dict[str, Any], agent) -> Task:
    source_name = page["source_name"]
    source_url = page["url"]
    source_topic = page["topic"]
    title = page.get("title") or ""
    raw_text = trim_text(page["raw_text"])

    description = f"""
You receive a trusted English medical source page.

Source name: {source_name}
Source URL: {source_url}
Source topic: {source_topic}
Page title: {title}

Your job:
Extract 3 to 8 dataset rows from this page.

Each row must contain:
- symptom
- disease
- treatment_recommendation
- source_name
- source_url
- source_topic
- confidence
- evidence
- model_used
- extraction_status
- notes

Rules:
1. Use English only.
2. Extract only information supported by the source text.
3. Do not invent diseases, symptoms, or treatment.
4. The treatment recommendation must be general and educational.
5. Do not give personal medical advice.
6. Do not recommend prescription medication as direct instruction.
7. If the source does not support a relation clearly, do not include it.
8. Confidence must be between 0.0 and 1.0.
9. extraction_status must be one of:
   - extracted
   - needs_review
   - rejected

Return ONLY valid JSON.
No markdown.
No extra text.

Required JSON format:
{{
  "records": [
    {{
      "symptom": "fever",
      "disease": "influenza",
      "treatment_recommendation": "Rest, drink fluids, and seek medical care if symptoms worsen or risk factors are present.",
      "source_name": "{source_name}",
      "source_url": "{source_url}",
      "source_topic": "{source_topic}",
      "confidence": 0.85,
      "evidence": "The page links fever with influenza symptoms and recommends rest and fluids.",
      "model_used": "{MODEL_USED}",
      "extraction_status": "extracted",
      "notes": "General recommendation only."
    }}
  ]
}}

Source text:
{raw_text}
"""

    return Task(
        description=description,
        expected_output="A valid JSON object containing a records list.",
        agent=agent,
    )


def extract_json_from_text(text: str) -> Dict[str, Any] | None:
    """
    Încearcă să citească JSON direct.
    Dacă modelul pune text extra, caută primul bloc dintre { și }.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def save_records(records: List[Dict[str, Any]]) -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, mode="w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    pages = load_raw_pages(limit=20)

    print(f"Loaded {len(pages)} raw pages.")
    print(f"Using model: {MODEL_USED}")

    extractor_agent = create_medical_extractor_agent()
    all_records = []

    for index, page in enumerate(pages, start=1):
        print("=" * 80)
        print(f"Processing page {index}/{len(pages)}")
        print(f"Source: {page['source_name']}")
        print(f"Topic: {page['topic']}")
        print(f"URL: {page['url']}")

        task = build_extraction_task(page, extractor_agent)

        crew = Crew(
            agents=[extractor_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        result_text = str(result)
        parsed = extract_json_from_text(result_text)

        if parsed is None:
            print("Could not parse JSON from model output.")
            continue

        records = parsed.get("records", [])

        for record in records:
            record["source_name"] = page["source_name"]
            record["source_url"] = page["url"]
            record["source_topic"] = page["topic"]
            record["model_used"] = MODEL_USED

        print(f"Extracted records: {len(records)}")
        all_records.extend(records)

    save_records(all_records)

    print("=" * 80)
    print("Extraction done.")
    print(f"Total records: {len(all_records)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()