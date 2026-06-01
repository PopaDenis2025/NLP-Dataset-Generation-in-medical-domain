import csv
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from nlp_medical_dataset.config.settings import (
    HEADERS,
    MAX_PAGES,
    REQUEST_TIMEOUT,
    RAW_PAGES_JSONL,
    SOURCES_CSV,
)
from nlp_medical_dataset.schemas.medical_record import RawMedicalPage
from nlp_medical_dataset.utils.file_io import write_jsonl
from nlp_medical_dataset.utils.text_cleaning import clean_text


def load_sources() -> List[Dict[str, str]]:
    if not SOURCES_CSV.exists():
        raise FileNotFoundError(f"Missing file: {SOURCES_CSV}")

    sources = []
    with SOURCES_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = clean_text(row.get("url"))
            if not url:
                continue
            sources.append({
                "source_name": clean_text(row.get("source_name")) or "unknown",
                "url": url,
                "topic": clean_text(row.get("topic")) or "general",
            })

    unique = []
    seen = set()
    for item in sources:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        unique.append(item)

    return unique[:MAX_PAGES]


def extract_title(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        return clean_text(soup.title.string)
    h1 = soup.find("h1")
    return clean_text(h1.get_text(" ", strip=True)) if h1 else None


def extract_main_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    return clean_text(main.get_text(" ", strip=True))


def fetch_page(source: Dict[str, str], retries: int = 2) -> RawMedicalPage:
    last_error = None

    for attempt in range(retries + 1):
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
            last_error = str(exc)
            time.sleep(1 + attempt)

    return RawMedicalPage(
        source_name=source["source_name"],
        url=source["url"],
        topic=source["topic"],
        title=None,
        raw_text="",
        status="failed",
        error=last_error,
    )


def scrape_sources() -> List[Dict[str, str]]:
    sources = load_sources()
    pages = []

    for source in tqdm(sources, desc="Scraping medical pages"):
        pages.append(fetch_page(source).model_dump())

    write_jsonl(RAW_PAGES_JSONL, pages)
    return pages
