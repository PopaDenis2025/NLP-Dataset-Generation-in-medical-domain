import json
from typing import Any, Dict
from crewai import Task
from nlp_medical_dataset.config.settings import (
    OLLAMA_MODEL,
    RECORDS_PER_PAGE_MIN,
    RECORDS_PER_PAGE_MAX,
)


def create_extraction_task(agent, page: Dict[str, Any]) -> Task:
    text = str(page.get("raw_text", ""))[:12000]
    description = f"""
Extract between {RECORDS_PER_PAGE_MIN} and {RECORDS_PER_PAGE_MAX} useful medical dataset rows from this page.

Rules:
- Return ONLY valid JSON.
- The first character must be {{ and the last character must be }}.
- Use English.
- Do not invent facts not supported by the text.
- Treatment recommendation must be general educational guidance.
- No exact medication dose.
- No guaranteed cure.
- If the page is not useful, return {{"records": []}}.

JSON format:
{{
  "records": [
    {{
      "symptom": "fever",
      "disease": "influenza",
      "treatment_recommendation": "Rest, drink fluids, and seek medical care if symptoms worsen.",
      "source_name": "{page.get("source_name", "")}",
      "source_url": "{page.get("url", "")}",
      "source_topic": "{page.get("topic", "")}",
      "confidence": 0.85,
      "evidence": "Short quote or paraphrase from the source text.",
      "model_used": "{OLLAMA_MODEL}",
      "extraction_status": "extracted"
    }}
  ]
}}

Source title: {page.get("title", "")}
Source text:
{text}
"""
    return Task(
        description=description,
        expected_output="Strict JSON object with a records list.",
        agent=agent,
    )


def create_validation_task(agent, record: Dict[str, Any]) -> Task:
    description = f"""
Validate this medical dataset row.

Rules:
- Return ONLY valid JSON.
- No markdown.
- Status values: valid, needs_review, rejected.
- Reject rows with exact dose, guaranteed cure, unclear disease, unclear symptom, or unsafe advice.

JSON format:
{{
  "validation_status": "valid",
  "validation_notes": "Short reason.",
  "clean_symptom": "fever",
  "clean_disease": "influenza",
  "clean_treatment_recommendation": "Rest, drink fluids, and seek medical care if symptoms worsen."
}}

Record:
{json.dumps(record, ensure_ascii=False)}
"""
    return Task(
        description=description,
        expected_output="Strict JSON object with validation result.",
        agent=agent,
    )


def create_report_task(agent, metrics: Dict[str, Any]) -> Task:
    description = f"""
Write a concise technical dataset quality report in markdown.

Include:
1. strengths
2. weaknesses
3. data quality risks
4. medical safety risks
5. improvement plan

Metrics:
{json.dumps(metrics, indent=2, ensure_ascii=False)}
"""
    return Task(
        description=description,
        expected_output="Markdown report.",
        agent=agent,
    )
