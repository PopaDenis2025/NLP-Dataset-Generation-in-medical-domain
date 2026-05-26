import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

from crewai import Task, Crew, Process

from src.agents import create_dataset_tester_agent
from src.config import FINAL_DIR, BASE_DIR


INPUT_CSV = FINAL_DIR / "medical_dataset.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORT_FILE = REPORTS_DIR / "dataset_quality_report.md"
METRICS_FILE = REPORTS_DIR / "dataset_metrics.json"


REQUIRED_COLUMNS = [
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


def load_dataset() -> List[Dict[str, Any]]:
    if not INPUT_CSV.exists():
        print(f"Missing file: {INPUT_CSV}")
        print("Run first: python -m src.validate_dataset")
        return []

    with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def is_empty(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_rows = len(records)

    missing_by_column = {}
    for col in REQUIRED_COLUMNS:
        missing_by_column[col] = sum(1 for r in records if is_empty(r.get(col)))

    symptoms = [r.get("symptom", "").strip().lower() for r in records if r.get("symptom")]
    diseases = [r.get("disease", "").strip().lower() for r in records if r.get("disease")]
    sources = [r.get("source_name", "").strip() for r in records if r.get("source_name")]
    statuses = [r.get("validation_status", "").strip().lower() for r in records]

    confidence_values = [to_float(r.get("confidence")) for r in records]
    avg_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    duplicate_keys = []
    seen = set()

    for r in records:
        key = (
            r.get("symptom", "").strip().lower(),
            r.get("disease", "").strip().lower(),
            r.get("source_topic", "").strip().lower(),
        )

        if key in seen:
            duplicate_keys.append(key)
        else:
            seen.add(key)

    duplicate_count = len(duplicate_keys)
    duplicate_rate = duplicate_count / total_rows if total_rows else 0.0

    valid_count = sum(1 for s in statuses if s == "valid")
    review_count = sum(1 for s in statuses if s == "needs_review")
    rejected_count = sum(1 for s in statuses if s == "rejected")

    treatment_too_short = sum(
        1
        for r in records
        if len(r.get("treatment_recommendation", "").strip()) < 25
    )

    evidence_too_short = sum(
        1
        for r in records
        if len(r.get("evidence", "").strip()) < 20
    )

    risky_terms = [
        "guaranteed",
        "cure",
        "must take",
        "dosage",
        "take 2",
        "take two",
        "prescribe",
        "antibiotic",
    ]

    risky_rows = 0
    for r in records:
        treatment = r.get("treatment_recommendation", "").lower()
        if any(term in treatment for term in risky_terms):
            risky_rows += 1

    metrics = {
        "total_rows": total_rows,
        "unique_symptoms": len(set(symptoms)),
        "unique_diseases": len(set(diseases)),
        "unique_sources": len(set(sources)),
        "average_confidence": round(avg_confidence, 4),
        "duplicate_count": duplicate_count,
        "duplicate_rate": round(duplicate_rate, 4),
        "valid_count": valid_count,
        "needs_review_count": review_count,
        "rejected_count": rejected_count,
        "treatment_too_short_count": treatment_too_short,
        "evidence_too_short_count": evidence_too_short,
        "risky_treatment_count": risky_rows,
        "missing_by_column": missing_by_column,
        "top_symptoms": Counter(symptoms).most_common(10),
        "top_diseases": Counter(diseases).most_common(10),
        "source_distribution": Counter(sources).most_common(),
        "validation_distribution": Counter(statuses).most_common(),
    }

    return metrics


def build_rule_based_report(metrics: Dict[str, Any]) -> str:
    total_rows = metrics["total_rows"]

    if total_rows == 0:
        quality_score = 0
    else:
        score = 100

        if metrics["duplicate_rate"] > 0.05:
            score -= 15

        if metrics["average_confidence"] < 0.70:
            score -= 15

        if metrics["risky_treatment_count"] > 0:
            score -= 20

        if metrics["treatment_too_short_count"] > 0:
            score -= 10

        if metrics["evidence_too_short_count"] > 0:
            score -= 10

        if metrics["missing_by_column"]:
            total_missing = sum(metrics["missing_by_column"].values())
            if total_missing > 0:
                score -= 20

        quality_score = max(score, 0)

    report = f"""# Dataset Quality Report

## Overview

This report evaluates the generated medical NLP dataset.

## Main Metrics

| Metric | Value |
|---|---:|
| Total rows | {metrics["total_rows"]} |
| Unique symptoms | {metrics["unique_symptoms"]} |
| Unique diseases | {metrics["unique_diseases"]} |
| Unique sources | {metrics["unique_sources"]} |
| Average confidence | {metrics["average_confidence"]} |
| Duplicate count | {metrics["duplicate_count"]} |
| Duplicate rate | {metrics["duplicate_rate"]} |
| Valid rows | {metrics["valid_count"]} |
| Needs review rows | {metrics["needs_review_count"]} |
| Rejected rows | {metrics["rejected_count"]} |
| Risky treatment rows | {metrics["risky_treatment_count"]} |
| Quality score | {quality_score}/100 |

## Missing Values

| Column | Missing |
|---|---:|
"""

    for col, count in metrics["missing_by_column"].items():
        report += f"| {col} | {count} |\n"

    report += "\n## Top Symptoms\n\n"
    report += "| Symptom | Count |\n|---|---:|\n"

    for symptom, count in metrics["top_symptoms"]:
        report += f"| {symptom} | {count} |\n"

    report += "\n## Top Diseases\n\n"
    report += "| Disease | Count |\n|---|---:|\n"

    for disease, count in metrics["top_diseases"]:
        report += f"| {disease} | {count} |\n"

    report += "\n## Source Distribution\n\n"
    report += "| Source | Count |\n|---|---:|\n"

    for source, count in metrics["source_distribution"]:
        report += f"| {source} | {count} |\n"

    report += "\n## Rule-Based Assessment\n\n"

    if quality_score >= 80:
        report += "The dataset has good initial quality and can be used for further experiments.\n"
    elif quality_score >= 60:
        report += "The dataset is usable, but it requires manual review and more cleaning.\n"
    else:
        report += "The dataset needs major improvements before it can be used reliably.\n"

    report += "\n## Recommended Next Steps\n\n"
    report += "1. Increase the number of trusted source pages.\n"
    report += "2. Validate more records with the LLM validator agent.\n"
    report += "3. Remove or rewrite rows with risky treatment wording.\n"
    report += "4. Add more disease categories to improve coverage.\n"
    report += "5. Manually review a sample of the dataset.\n"

    return report


def build_llm_task(metrics: Dict[str, Any], agent) -> Task:
    description = f"""
You receive dataset quality metrics for a generated medical NLP dataset.

Your job:
Write a concise academic evaluation of the dataset.

Mention:
1. dataset strengths
2. dataset weaknesses
3. data quality risks
4. medical safety risks
5. recommended improvements
6. whether the dataset is acceptable as an initial academic prototype

Do not give medical advice.
Focus on dataset quality.

Return markdown only.

Metrics:
{json.dumps(metrics, indent=2)}
"""

    return Task(
        description=description,
        expected_output="A markdown dataset quality assessment.",
        agent=agent,
    )


def generate_llm_assessment(metrics: Dict[str, Any]) -> str:
    agent = create_dataset_tester_agent()
    task = build_llm_task(metrics, agent)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    try:
        result = crew.kickoff()
        return str(result)
    except Exception as exc:
        return f"LLM assessment failed: {exc}"


def save_outputs(metrics: Dict[str, Any], report: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_FILE, mode="w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, mode="w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    records = load_dataset()
    print(f"Loaded final dataset rows: {len(records)}")

    if not records:
        return

    metrics = compute_metrics(records)
    base_report = build_rule_based_report(metrics)

    print("Running Dataset Tester Agent...")
    llm_report = generate_llm_assessment(metrics)

    final_report = base_report
    final_report += "\n\n---\n\n"
    final_report += "# Agent-Based Assessment\n\n"
    final_report += llm_report
    final_report += "\n"

    save_outputs(metrics, final_report)

    print("=" * 80)
    print("Dataset testing done.")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()