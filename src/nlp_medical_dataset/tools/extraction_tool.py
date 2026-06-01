import json
from typing import Any, Dict, List, Optional

from crewai import Crew, Process

from nlp_medical_dataset.agents.medical_agents import create_medical_extractor_agent
from nlp_medical_dataset.config.settings import (
    EXTRACTED_RECORDS_JSONL,
    RAW_PAGES_JSONL,
    TARGET_RECORDS,
)
from nlp_medical_dataset.tasks.medical_tasks import create_extraction_task
from nlp_medical_dataset.utils.file_io import read_jsonl, write_jsonl


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


def extract_records_from_pages() -> List[Dict[str, Any]]:
    pages = [p for p in read_jsonl(RAW_PAGES_JSONL) if p.get("status") == "ok"]
    agent = create_medical_extractor_agent()
    all_records: List[Dict[str, Any]] = []

    for idx, page in enumerate(pages, start=1):
        if len(all_records) >= TARGET_RECORDS:
            break

        print(f"Extracting page {idx}/{len(pages)}: {page.get('url')}")
        task = create_extraction_task(agent, page)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

        try:
            result = crew.kickoff()
            parsed = extract_json_object(str(result))
            records = parsed.get("records", []) if parsed else []
        except Exception as exc:
            print(f"Extraction failed for {page.get('url')}: {exc}")
            records = []

        for record in records:
            record["source_name"] = record.get("source_name") or page.get("source_name")
            record["source_url"] = record.get("source_url") or page.get("url")
            record["source_topic"] = record.get("source_topic") or page.get("topic")
            all_records.append(record)

        print(f"Current records: {len(all_records)}")

    all_records = all_records[:TARGET_RECORDS]
    write_jsonl(EXTRACTED_RECORDS_JSONL, all_records)
    return all_records
