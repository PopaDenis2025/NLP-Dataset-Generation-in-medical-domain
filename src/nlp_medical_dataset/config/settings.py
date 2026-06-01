from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PACKAGE_DIR.parents[0]
PROJECT_ROOT = SRC_DIR.parents[0]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

SOURCES_CSV = RAW_DIR / "sources.csv"
RAW_PAGES_JSONL = RAW_DIR / "pages_raw.jsonl"
EXTRACTED_RECORDS_JSONL = INTERMEDIATE_DIR / "extracted_records.jsonl"
FINAL_DATASET_JSONL = FINAL_DIR / "medical_dataset.jsonl"
FINAL_DATASET_CSV = FINAL_DIR / "medical_dataset.csv"
METRICS_JSON = REPORTS_DIR / "dataset_metrics.json"
QUALITY_REPORT_MD = REPORTS_DIR / "dataset_quality_report.md"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/mistral")
TARGET_RECORDS = int(os.getenv("TARGET_RECORDS", "1000"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "300"))
RECORDS_PER_PAGE_MIN = int(os.getenv("RECORDS_PER_PAGE_MIN", "3"))
RECORDS_PER_PAGE_MAX = int(os.getenv("RECORDS_PER_PAGE_MAX", "8"))
LLM_VALIDATION_LIMIT = int(os.getenv("LLM_VALIDATION_LIMIT", "25"))

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
HEADERS = {
    "User-Agent": os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
}

def ensure_project_dirs() -> None:
    for path in [RAW_DIR, INTERMEDIATE_DIR, FINAL_DIR, REPORTS_DIR, NOTEBOOKS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
