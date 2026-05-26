import csv
import json
import re
from typing import Any, Dict, List, Optional

from crewai import Task, Crew, Process

from src.agents import create_medical_validator_agent
from src.config import INTERMEDIATE_DIR, FINAL_DIR


INPUT_FILE = INTERMEDIATE_DIR / "extracted_records.jsonl"
OUTPUT_JSONL = FINAL_DIR / "medical_dataset.jsonl"
OUTPUT_CSV = FINAL_DIR / "medical_dataset.csv"


REQUIRED_FIELDS = [
    "symptom",
    "disease",
    "treatment_recommendation",
    "source_name",
    "source_url",
    "source_topic",
    "confidence",
    "evidence",
    "model_used",
    "extraction_status",
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}

    for key in REQUIRED_FIELDS:
        cleaned[key] = record.get(key)

    cleaned["symptom"] = normalize_text(cleaned["symptom"]).lower()
    cleaned["disease"] = normalize_text(cleaned["disease"]).lower()
    cleaned["treatment_recommendation"] = normalize_text(
        cleaned["treatment_recommendation"]
    )
    cleaned["source_name"] = normalize_text(cleaned["source_name"])
    cleaned["source_url"] = normalize_text(cleaned["source_url"])
    cleaned["source_topic"] = normalize_text(cleaned["source_topic"]).lower()
    cleaned["evidence"] = normalize_text(cleaned["evidence"])
    cleaned["model_used"] = normalize_text(cleaned["model_used"])
    cleaned["extraction_status"] = normalize_text(
        cleaned["extraction_status"]
    ).lower()

    try:
        cleaned["confidence"] = float(cleaned["confidence"])
    except (TypeError, ValueError):
        cleaned["confidence"] = 0.0

    cleaned["validation_status"] = "unchecked"
    cleaned["validation_notes"] = ""

    return cleaned


def load_records() -> List[Dict[str, Any]]:
    records = []

    if not INPUT_FILE.exists():
        print(f"Missing file: {INPUT_FILE}")
        print("Run first: python -m src.extract_dataset")
        return records

    with open(INPUT_FILE, mode="r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                raw = json.loads(line)
                records.append(normalize_record(raw))
            except json.JSONDecodeError:
                continue

    return records


def basic_rule_validation(record: Dict[str, Any]) -> Dict[str, Any]:
    symptom = record["symptom"]
    disease = record["disease"]
    treatment = record["treatment_recommendation"]
    confidence = record["confidence"]

    problems = []

    if not symptom:
        problems.append("missing symptom")

    if not disease:
        problems.append("missing disease")

    if not treatment:
        problems.append("missing treatment recommendation")

    if confidence < 0.60:
        problems.append("low confidence")

    risky_words = [
        "dosage",
        "take 2",
        "take two",
        "prescribe",
        "must take",
        "guaranteed",
        "cure",
        "antibiotic",
    ]

    treatment_lower = treatment.lower()

    for word in risky_words:
        if word in treatment_lower:
            problems.append(f"risky treatment wording: {word}")

    if problems:
        record["validation_status"] = "needs_review"
        record["validation_notes"] = "; ".join(problems)
    else:
        record["validation_status"] = "valid"
        record["validation_notes"] = "Passed rule-based validation."

    return record


def deduplicate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_records = []

    for record in records:
        key = (
            record["symptom"].lower(),
            record["disease"].lower(),
            record["source_topic"].lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_records.append(record)

    return unique_records


def build_validation_task(record: Dict[str, Any], agent) -> Task:
    description = f"""
You receive one extracted medical dataset record.

Your job:
Validate whether this record is suitable for an educational NLP dataset.

Check:
1. The symptom is a real medical symptom.
2. The disease is a real medical condition.
3. The treatment recommendation is general and safe.
4. The treatment is not personal medical advice.
5. The record is clear enough for a dataset.
6. The record should not contain exact dosage instructions.
7. The record should not claim a guaranteed cure.

Return ONLY valid JSON.
No markdown.
No explanation.
The first character must be {{
The last character must be }}

Required JSON format:
{{
  "validation_status": "valid",
  "validation_notes": "Short reason.",
  "clean_symptom": "fever",
  "clean_disease": "influenza",
  "clean_treatment_recommendation": "Rest, drink fluids, and seek medical care if symptoms worsen."
}}

Allowed validation_status values:
- valid
- needs_review
- rejected

Record:
{json.dumps(record, ensure_ascii=False)}
"""

    return Task(
        description=description,
        expected_output="A valid JSON object with validation result.",
        agent=agent,
    )


def extract_json(text: str) -> Optional[Dict[str, Any]]:
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


def llm_validate_records(
    records: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Validare cu agentul LLM pe primele N rânduri.
    Pentru început rulăm limitat, ca să nu dureze enorm local.
    """
    validator_agent = create_medical_validator_agent()
    validated = []

    for index, record in enumerate(records, start=1):
        if index > limit:
            validated.append(record)
            continue

        print(f"LLM validating record {index}/{min(limit, len(records))}")

        task = build_validation_task(record, validator_agent)

        crew = Crew(
            agents=[validator_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            parsed = extract_json(str(result))

            if parsed is None:
                record["validation_status"] = "needs_review"
                record["validation_notes"] = "LLM validation returned invalid JSON."
                validated.append(record)
                continue

            status = parsed.get("validation_status", "needs_review")
            notes = parsed.get("validation_notes", "")

            record["validation_status"] = status
            record["validation_notes"] = normalize_text(notes)

            if parsed.get("clean_symptom"):
                record["symptom"] = normalize_text(parsed["clean_symptom"]).lower()

            if parsed.get("clean_disease"):
                record["disease"] = normalize_text(parsed["clean_disease"]).lower()

            if parsed.get("clean_treatment_recommendation"):
                record["treatment_recommendation"] = normalize_text(
                    parsed["clean_treatment_recommendation"]
                )

        except Exception as exc:
            record["validation_status"] = "needs_review"
            record["validation_notes"] = f"LLM validation failed: {exc}"

        validated.append(record)

    return validated


def save_jsonl(records: List[Dict[str, Any]]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSONL, mode="w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(records: List[Dict[str, Any]]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "symptom",
        "disease",
        "treatment_recommendation",
        "source_name",
        "source_url",
        "source_topic",
        "confidence",
        "evidence",
        "model_used",
        "extraction_status",
        "validation_status",
        "validation_notes",
    ]

    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def main() -> None:
    records = load_records()
    print(f"Loaded records: {len(records)}")

    if not records:
        return

    records = [basic_rule_validation(record) for record in records]
    print(f"After rule validation: {len(records)}")

    records = deduplicate(records)
    print(f"After deduplication: {len(records)}")

    # Pentru test local: validează doar primele 10 cu LLM.
    # După ce merge bine, crești la 30, 50 etc.
    records = llm_validate_records(records, limit=10)

    valid_count = sum(1 for r in records if r["validation_status"] == "valid")
    review_count = sum(1 for r in records if r["validation_status"] == "needs_review")
    rejected_count = sum(1 for r in records if r["validation_status"] == "rejected")

    save_jsonl(records)
    save_csv(records)

    print("=" * 80)
    print("Validation done.")
    print(f"Valid: {valid_count}")
    print(f"Needs review: {review_count}")
    print(f"Rejected: {rejected_count}")
    print(f"Saved JSONL to: {OUTPUT_JSONL}")
    print(f"Saved CSV to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()