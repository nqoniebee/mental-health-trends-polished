"""Shared text-cleaning helpers used both offline (scripts/prepare_data.py)
and at runtime (Streamlit pages) so cleaning stays consistent."""

import re

import pandas as pd

_URL_RE = re.compile(r"http\S+|www\.\S+")
_REDDIT_MENTION_RE = re.compile(r"/?u/\w+|/?r/\w+")
_AT_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip URLs, usernames, and excess whitespace. Defense-in-depth
    anonymization even though source datasets are already pseudonymous."""
    if not isinstance(text, str):
        return ""
    text = _URL_RE.sub(" ", text)
    text = _REDDIT_MENTION_RE.sub(" ", text)
    text = _AT_MENTION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def filter_english(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Drop rows whose text isn't detected as English. Best-effort; short
    strings are kept as-is since language detection is unreliable on them."""
    from langdetect import LangDetectException, detect, DetectorFactory
    DetectorFactory.seed = 42

    def _is_english(text: str) -> bool:
        if len(text) < 20:
            return True
        try:
            return detect(text) == "en"
        except LangDetectException:
            return False

    mask = df[text_col].map(_is_english)
    return df[mask].reset_index(drop=True)


def deduplicate(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    return df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
