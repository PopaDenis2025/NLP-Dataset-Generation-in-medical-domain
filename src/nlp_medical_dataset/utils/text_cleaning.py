import re
from typing import Any


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_lower(value: Any) -> str:
    return clean_text(value).lower()


def shorten(text: str, max_chars: int = 12000) -> str:
    text = clean_text(text)
    return text[:max_chars]
