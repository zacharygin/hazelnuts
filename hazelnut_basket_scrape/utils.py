"""Shared utilities for hazelnut price scrapers."""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT      = Path(__file__).parent
RAW_DIR   = ROOT / 'data' / 'raw'
PROC_DIR  = ROOT / 'data' / 'processed'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'tr-TR,tr;q=0.9,en;q=0.8',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

SCHEMA = [
    'source', 'date', 'exchange', 'product', 'variety', 'grade',
    'min_price_tlkg', 'max_price_tlkg', 'avg_price_tlkg', 'last_price_tlkg',
    'volume_kg', 'transaction_count', 'transaction_value_tl', 'url',
]


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def tr_float(val: str) -> float | None:
    """Convert Turkish-formatted number string to float."""
    if not val or val.strip() in ('', '-', '—'):
        return None
    cleaned = re.sub(r'[^\d,\.]', '', val.strip())
    # Turkish: 1.234,56 → 1234.56
    if ',' in cleaned and '.' in cleaned:
        cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '.')
    try:
        return float(cleaned)
    except ValueError:
        return None


def empty_row() -> dict:
    return {col: None for col in SCHEMA}


def save_raw(data: str | bytes, path: Path, mode: str = 'w') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding='utf-8' if isinstance(data, str) else None) as f:
        f.write(data)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / name
    df.to_csv(path, index=False, encoding='utf-8-sig')
    logging.info('Saved %d rows → %s', len(df), path)
    return path


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        datefmt='%H:%M:%S',
    )


def polite_sleep(seconds: float = 1.0) -> None:
    time.sleep(seconds)
