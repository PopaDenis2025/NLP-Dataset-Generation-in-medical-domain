import json
import re
from typing import Any, Dict, List, Optional

from crewai import Crew, Process

from nlp_medical_dataset.agents.medical_agents import create_medical_validator_agent
from nlp_medical_dataset.config.settings import (
    EXTRACTED_RECORDS_JSONL,
    FINAL_DATASET_CSV,
    FINAL_DATASET_JSONL,
    LLM_VALIDATION_LIMIT,
)
from nlp_medical_dataset.tasks.medical_tasks import create_validation_task
from nlp_medical_dataset.utils.file_io import read_jsonl, write_csv, write_jsonl
from nlp_medical_dataset.utils.text_cleaning import clean_text, normalize_lower


FIELDNAMES = [
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


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    confidence = record.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "symptom": normalize_lower(record.get("symptom")),
        "disease": normalize_lower(record.get("disease")),
        "treatment_recommendation": clean_text(record.get("treatment_recommendation")),
        "source_name": clean_text(record.get("source_name")),
        "source_url": clean_text(record.get("source_url")),
        "source_topic": normalize_lower(record.get("source_topic")),
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": clean_text(record.get("evidence")),
        "model_used": clean_text(record.get("model_used")),
        "extraction_status": normalize_lower(record.get("extraction_status")) or "extracted",
        "validation_status": "unchecked",
        "validation_notes": "",
    }


def rule_validate(record: Dict[str, Any]) -> Dict[str, Any]:
    problems = []

    if not record["symptom"]:
        problems.append("missing symptom")
    if not record["disease"]:
        problems.append("missing disease")
    if len(record["treatment_recommendation"]) < 25:
        problems.append("treatment too short")
    if len(record["evidence"]) < 20:
        problems.append("evidence too short")
    if record["confidence"] < 0.6:
        problems.append("low confidence")

    risky = [
        "dosage", "dose", "take 2", "take two", "prescribe",
        "must take", "guaranteed", "cure", "antibiotic"
    ]
    treatment = record["treatment_recommendation"].lower()
    for word in risky:
        if word in treatment:
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
    unique = []

    for record in records:
        key = (
            record["symptom"].lower(),
            record["disease"].lower(),
            record["source_url"].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    return unique


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def llm_validate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agent = create_medical_validator_agent()
    output = []

    for idx, record in enumerate(records, start=1):
        if idx > LLM_VALIDATION_LIMIT:
            output.append(record)
            continue

        print(f"LLM validating row {idx}/{min(LLM_VALIDATION_LIMIT, len(records))}")
        task = create_validation_task(agent, record)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

        try:
            result = crew.kickoff()
            parsed = extract_json_object(str(result))
            if not parsed:
                record["validation_status"] = "needs_review"
                record["validation_notes"] = "LLM returned invalid JSON."
                output.append(record)
                continue

            record["validation_status"] = clean_text(parsed.get("validation_status")) or record["validation_status"]
            record["validation_notes"] = clean_text(parsed.get("validation_notes")) or record["validation_notes"]

            if parsed.get("clean_symptom"):
                record["symptom"] = normalize_lower(parsed.get("clean_symptom"))
            if parsed.get("clean_disease"):
                record["disease"] = normalize_lower(parsed.get("clean_disease"))
            if parsed.get("clean_treatment_recommendation"):
                record["treatment_recommendation"] = clean_text(parsed.get("clean_treatment_recommendation"))

        except Exception as exc:
            record["validation_status"] = "needs_review"
            record["validation_notes"] = f"LLM validation failed: {exc}"

        output.append(record)

    return output


def validate_dataset() -> List[Dict[str, Any]]:
    raw = read_jsonl(EXTRACTED_RECORDS_JSONL)
    records = [rule_validate(normalize_record(row)) for row in raw]
    records = deduplicate(records)
    records = llm_validate(records)

    write_jsonl(FINAL_DATASET_JSONL, records)
    write_csv(FINAL_DATASET_CSV, records, FIELDNAMES)
    return records
