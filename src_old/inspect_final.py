import csv
from src.config import FINAL_DIR

FINAL_CSV = FINAL_DIR / "medical_dataset.csv"


def main() -> None:
    if not FINAL_CSV.exists():
        print(f"File not found: {FINAL_CSV}")
        print("Run first: python -m src.validate_dataset")
        return

    with open(FINAL_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        count = 0
        for row in reader:
            count += 1
            print("=" * 80)
            print(f"Record: {count}")
            print(f"Symptom: {row.get('symptom')}")
            print(f"Disease: {row.get('disease')}")
            print(f"Treatment: {row.get('treatment_recommendation')}")
            print(f"Confidence: {row.get('confidence')}")
            print(f"Validation: {row.get('validation_status')}")
            print(f"Notes: {row.get('validation_notes')}")

            if count == 5:
                break

        print("=" * 80)
        print(f"Displayed: {min(count, 5)} records")


if __name__ == "__main__":
    main()