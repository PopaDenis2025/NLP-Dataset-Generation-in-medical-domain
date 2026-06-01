import json
from collections import Counter
from typing import Any, Dict, List

from crewai import Crew, Process

from nlp_medical_dataset.agents.medical_agents import create_dataset_tester_agent
from nlp_medical_dataset.config.settings import FINAL_DATASET_CSV, METRICS_JSON, QUALITY_REPORT_MD
from nlp_medical_dataset.tasks.medical_tasks import create_report_task
from nlp_medical_dataset.utils.file_io import write_jsonl
import csv


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


def load_csv_rows() -> List[Dict[str, Any]]:
    if not FINAL_DATASET_CSV.exists():
        return []
    with FINAL_DATASET_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    symptoms = [r.get("symptom", "").lower() for r in rows if r.get("symptom")]
    diseases = [r.get("disease", "").lower() for r in rows if r.get("disease")]
    sources = [r.get("source_name", "") for r in rows if r.get("source_name")]
    statuses = [r.get("validation_status", "").lower() for r in rows]
    confidences = [to_float(r.get("confidence")) for r in rows]

    missing = {
        col: sum(1 for r in rows if not str(r.get(col, "")).strip())
        for col in REQUIRED_COLUMNS
    }

    seen = set()
    duplicates = 0
    for r in rows:
        key = (r.get("symptom", "").lower(), r.get("disease", "").lower(), r.get("source_url", "").lower())
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    risky_terms = ["guaranteed", "must take", "dosage", "dose", "prescribe", "cure"]
    risky = sum(
        1 for r in rows
        if any(term in r.get("treatment_recommendation", "").lower() for term in risky_terms)
    )

    return {
        "total_rows": total,
        "unique_symptoms": len(set(symptoms)),
        "unique_diseases": len(set(diseases)),
        "unique_sources": len(set(sources)),
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
        "duplicate_count": duplicates,
        "duplicate_rate": round(duplicates / total, 4) if total else 0,
        "valid_count": sum(1 for s in statuses if s == "valid"),
        "needs_review_count": sum(1 for s in statuses if s == "needs_review"),
        "rejected_count": sum(1 for s in statuses if s == "rejected"),
        "risky_treatment_count": risky,
        "missing_by_column": missing,
        "top_symptoms": Counter(symptoms).most_common(10),
        "top_diseases": Counter(diseases).most_common(10),
        "source_distribution": Counter(sources).most_common(),
    }


def rule_report(metrics: Dict[str, Any]) -> str:
    score = 100
    if metrics["duplicate_rate"] > 0.05:
        score -= 15
    if metrics["average_confidence"] < 0.70:
        score -= 15
    if metrics["risky_treatment_count"] > 0:
        score -= 20
    if sum(metrics["missing_by_column"].values()) > 0:
        score -= 20
    score = max(score, 0)

    lines = [
        "# Dataset Quality Report",
        "",
        "## Main Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total rows | {metrics['total_rows']} |",
        f"| Unique symptoms | {metrics['unique_symptoms']} |",
        f"| Unique diseases | {metrics['unique_diseases']} |",
        f"| Unique sources | {metrics['unique_sources']} |",
        f"| Average confidence | {metrics['average_confidence']} |",
        f"| Duplicate rate | {metrics['duplicate_rate']} |",
        f"| Valid rows | {metrics['valid_count']} |",
        f"| Needs review rows | {metrics['needs_review_count']} |",
        f"| Risky treatment rows | {metrics['risky_treatment_count']} |",
        f"| Quality score | {score}/100 |",
        "",
        "## Missing Values",
        "",
        "| Column | Missing |",
        "|---|---:|",
    ]

    for col, value in metrics["missing_by_column"].items():
        lines.append(f"| {col} | {value} |")

    lines.extend([
        "",
        "## Recommendation",
        "",
        "Increase the number of trusted medical pages, review risky rows manually, and check disease class balance before model training.",
    ])
    return "\n".join(lines)


def agent_report(metrics: Dict[str, Any]) -> str:
    agent = create_dataset_tester_agent()
    task = create_report_task(agent, metrics)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    try:
        return str(crew.kickoff())
    except Exception as exc:
        return f"Agent report failed: {exc}"


def generate_reports() -> Dict[str, Any]:
    rows = load_csv_rows()
    metrics = compute_metrics(rows)

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    report = rule_report(metrics)
    report += "\n\n---\n\n# Agent-Based Assessment\n\n"
    report += agent_report(metrics)
    report += "\n"

    QUALITY_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_MD.write_text(report, encoding="utf-8")
    return metrics
