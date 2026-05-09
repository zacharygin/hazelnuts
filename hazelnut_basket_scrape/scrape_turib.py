"""
TURIB (Türkiye Ürün İhtisas Borsası) Scraper
Source: turib.com.tr — ELÜS hazelnut derivatives + daily bulletins

NOTE: TURIB live/historical ELÜS price data is behind a paid data-vendor subscription
(see turib.com.tr/veri-dagitim-sirketleri for authorised distributors).
This scraper does two things:
  1. Downloads any freely available PDF bulletins from the bulletin archive page.
  2. Attempts to parse price tables from those PDFs using pdfplumber.

For full historical ELÜS time series, contact a TURIB data distributor.
"""

import logging
import re
from io import BytesIO
from pathlib import Path

import bs4
import pandas as pd
import pdfplumber
import requests

from utils import (
    RAW_DIR, empty_row, get_session, polite_sleep,
    save_processed, save_raw, setup_logging, tr_float,
)

URL_BULTEN = 'https://www.turib.com.tr/gunluk-bulten/'
SOURCE     = 'turib'
RAW_SUBDIR = RAW_DIR / 'turib'


def fetch_bulletin_links(session) -> list[tuple[str, str]]:
    """Return list of (title, pdf_url) from the bulletin archive page."""
    resp = session.get(URL_BULTEN, timeout=15)
    resp.raise_for_status()
    save_raw(resp.text, RAW_SUBDIR / 'bulletin_index.html')

    soup  = bs4.BeautifulSoup(resp.text, 'lxml')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.lower().endswith('.pdf'):
            title = a.get_text(strip=True) or href.split('/')[-1]
            links.append((title, href))
    return links


def parse_bulletin_pdf(pdf_bytes: bytes, source_url: str) -> list[dict]:
    """Extract price rows from a TURIB daily bulletin PDF."""
    records = []
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if not row or len(row) < 4:
                            continue
                        # Heuristic: rows containing 'Fındık' or 'FINDIK'
                        row_text = ' '.join(str(c) for c in row if c)
                        if not re.search(r'f[iı]nd[iı]k', row_text, re.IGNORECASE):
                            continue
                        rec = empty_row()
                        rec.update({
                            'source':  SOURCE,
                            'product': row[0] if row[0] else None,
                            'url':     source_url,
                        })
                        # Try to extract numeric price columns
                        nums = [tr_float(str(c)) for c in row if c and re.search(r'\d', str(c))]
                        if len(nums) >= 1: rec['last_price_tlkg'] = nums[-1]
                        if len(nums) >= 2: rec['avg_price_tlkg']  = nums[-2]
                        records.append(rec)
    except Exception as e:
        logging.warning('PDF parse error: %s', e)
    return records


def run() -> pd.DataFrame:
    session = get_session()
    logging.info('TURIB — fetching bulletin index: %s', URL_BULTEN)

    pdf_links = fetch_bulletin_links(session)
    logging.info('TURIB: found %d PDF bulletin links', len(pdf_links))

    if not pdf_links:
        logging.warning('TURIB: no free PDFs found. '
                        'Historical ELÜS data requires a paid data-vendor subscription.')
        return pd.DataFrame(columns=pd.Index(['source', 'date', 'product',
                                              'avg_price_tlkg', 'last_price_tlkg', 'url']))

    all_records: list[dict] = []
    for title, url in pdf_links:
        logging.info('  Downloading: %s', title[:60])
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            fname = re.sub(r'[^\w\-.]', '_', url.split('/')[-1])
            save_raw(r.content, RAW_SUBDIR / fname, mode='wb')
            records = parse_bulletin_pdf(r.content, url)
            logging.info('    → %d price rows parsed', len(records))
            all_records.extend(records)
        except Exception as e:
            logging.warning('  Failed %s: %s', url[:60], e)
        polite_sleep(1.0)

    if not all_records:
        logging.warning('TURIB: no price data extracted from PDFs')
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    save_processed(df, 'turib_hazelnut.csv')
    return df


if __name__ == '__main__':
    setup_logging()
    run()
