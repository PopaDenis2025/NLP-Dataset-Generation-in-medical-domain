from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"

SOURCES_CSV = RAW_DIR / "sources.csv"
RAW_PAGES_JSONL = RAW_DIR / "pages_raw.jsonl"

REQUEST_TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; NLPMedicalDatasetProject/1.0; academic project)"
    )
}