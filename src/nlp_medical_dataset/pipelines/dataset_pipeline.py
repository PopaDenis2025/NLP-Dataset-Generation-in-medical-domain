from nlp_medical_dataset.config.settings import (
    EXTRACTED_RECORDS_JSONL,
    FINAL_DATASET_CSV,
    METRICS_JSON,
    QUALITY_REPORT_MD,
    RAW_PAGES_JSONL,
    ensure_project_dirs,
)
from nlp_medical_dataset.tools.extraction_tool import extract_records_from_pages
from nlp_medical_dataset.tools.reporting_tool import generate_reports
from nlp_medical_dataset.tools.scraper_tool import scrape_sources
from nlp_medical_dataset.tools.validation_tool import validate_dataset


def run_pipeline(skip_scrape: bool = False) -> None:
    ensure_project_dirs()

    print("=" * 80)
    print("STEP 1/4 - Scraper")
    print("=" * 80)
    if skip_scrape:
        print(f"Skipped. Existing file used: {RAW_PAGES_JSONL}")
    else:
        pages = scrape_sources()
        print(f"Saved raw pages: {len(pages)} -> {RAW_PAGES_JSONL}")

    print("=" * 80)
    print("STEP 2/4 - Agent 1 extraction")
    print("=" * 80)
    extracted = extract_records_from_pages()
    print(f"Saved extracted rows: {len(extracted)} -> {EXTRACTED_RECORDS_JSONL}")

    print("=" * 80)
    print("STEP 3/4 - Agent 2 validation")
    print("=" * 80)
    final_rows = validate_dataset()
    print(f"Saved final rows: {len(final_rows)} -> {FINAL_DATASET_CSV}")

    print("=" * 80)
    print("STEP 4/4 - Agent 3 reporting")
    print("=" * 80)
    metrics = generate_reports()
    print(f"Saved metrics: {METRICS_JSON}")
    print(f"Saved report: {QUALITY_REPORT_MD}")
    print(f"Total final rows: {metrics.get('total_rows', 0)}")
