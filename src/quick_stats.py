import json
from src.config import BASE_DIR

METRICS_FILE = BASE_DIR / "reports" / "dataset_metrics.json"


def main() -> None:
    if not METRICS_FILE.exists():
        print("Metrics file not found.")
        print("Run first: python -m src.test_dataset")
        return

    with open(METRICS_FILE, mode="r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("=" * 80)
    print("DATASET QUICK STATS")
    print("=" * 80)
    print(f"Total rows: {metrics['total_rows']}")
    print(f"Unique symptoms: {metrics['unique_symptoms']}")
    print(f"Unique diseases: {metrics['unique_diseases']}")
    print(f"Unique sources: {metrics['unique_sources']}")
    print(f"Average confidence: {metrics['average_confidence']}")
    print(f"Duplicate rate: {metrics['duplicate_rate']}")
    print(f"Valid rows: {metrics['valid_count']}")
    print(f"Needs review: {metrics['needs_review_count']}")
    print(f"Rejected: {metrics['rejected_count']}")
    print(f"Risky treatment rows: {metrics['risky_treatment_count']}")


if __name__ == "__main__":
    main()