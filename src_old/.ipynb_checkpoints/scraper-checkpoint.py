import csv
import json
import re
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import SOURCES_CSV, RAW_PAGES_JSONL, HEADERS, REQUEST_TIMEOUT
from src.schemas import RawMedicalPage


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_title(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        return clean_text(soup.title.string)

    h1 = soup.find("h1")
    if h1:
        return clean_text(h1.get_text(" ", strip=True))

    return None


def extract_main_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    return clean_text(text)


def load_sources() -> List[Dict[str, str]]:
    sources = []

    with open(SOURCES_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            sources.append(
                {
                    "source_name": row["source_name"].strip(),
                    "url": row["url"].strip(),
                    "topic": row["topic"].strip(),
                }
            )

    return sources


def fetch_page(source: Dict[str, str]) -> RawMedicalPage:
    try:
        response = requests.get(
            source["url"],
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        title = extract_title(soup)
        raw_text = extract_main_text(soup)

        if len(raw_text) < 300:
            return RawMedicalPage(
                source_name=source["source_name"],
                url=source["url"],
                topic=source["topic"],
                title=title,
                raw_text=raw_text,
                status="too_short",
                error="Extracted text is too short.",
            )

        return RawMedicalPage(
            source_name=source["source_name"],
            url=source["url"],
            topic=source["topic"],
            title=title,
            raw_text=raw_text,
            status="ok",
            error=None,
        )

    except Exception as exc:
        return RawMedicalPage(
            source_name=source["source_name"],
            url=source["url"],
            topic=source["topic"],
            title=None,
            raw_text="",
            status="failed",
            error=str(exc),
        )


def save_jsonl(records: List[RawMedicalPage]) -> None:
    with open(RAW_PAGES_JSONL, mode="w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def main() -> None:
    sources = load_sources()

    print(f"Loaded {len(sources)} sources from {SOURCES_CSV}")

    records = []

    for source in tqdm(sources, desc="Fetching medical pages"):
        page = fetch_page(source)
        records.append(page)

    save_jsonl(records)

    ok_count = sum(1 for r in records if r.status == "ok")
    failed_count = sum(1 for r in records if r.status == "failed")
    short_count = sum(1 for r in records if r.status == "too_short")

    print("Done.")
    print(f"Saved to: {RAW_PAGES_JSONL}")
    print(f"OK: {ok_count}")
    print(f"Failed: {failed_count}")
    print(f"Too short: {short_count}")


if __name__ == "__main__":
    main()