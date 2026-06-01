import json
from src.config import RAW_PAGES_JSONL


def main() -> None:
    with open(RAW_PAGES_JSONL, mode="r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            record = json.loads(line)

            print("=" * 80)
            print(f"Record: {index}")
            print(f"Source: {record['source_name']}")
            print(f"Topic: {record['topic']}")
            print(f"URL: {record['url']}")
            print(f"Status: {record['status']}")
            print(f"Title: {record['title']}")
            print(f"Text preview: {record['raw_text'][:500]}")

            if index == 3:
                break


if __name__ == "__main__":
    main()