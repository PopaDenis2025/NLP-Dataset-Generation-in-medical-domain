import json
from src.config import INTERMEDIATE_DIR

EXTRACTED_FILE = INTERMEDIATE_DIR / "extracted_records.jsonl"


def main() -> None:
    if not EXTRACTED_FILE.exists():
        print(f"File not found: {EXTRACTED_FILE}")
        print("Run first: python -m src.extract_dataset")
        return

    count = 0

    with open(EXTRACTED_FILE, mode="r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            count += 1

            print("=" * 80)
            print(f"Record: {count}")
            print(f"Symptom: {record.get('symptom')}")
            print(f"Disease: {record.get('disease')}")
            print(f"Treatment: {record.get('treatment_recommendation')}")
            print(f"Confidence: {record.get('confidence')}")
            print(f"Status: {record.get('extraction_status')}")
            print(f"Source: {record.get('source_name')}")
            print(f"URL: {record.get('source_url')}")
            print(f"Evidence: {record.get('evidence')}")

            if count == 5:
                break

    print("=" * 80)
    print(f"Displayed {min(count, 5)} records.")


if __name__ == "__main__":
    main()