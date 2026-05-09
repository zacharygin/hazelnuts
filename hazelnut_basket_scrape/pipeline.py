"""
Hazelnut price scraper pipeline — main runner.

Usage:
    python pipeline.py                  # full run (TOBB 2005-present + TB.org + TURIB)
    python pipeline.py --tobb-only      # TOBB only
    python pipeline.py --year-start 2015
    python pipeline.py --no-turib       # skip TURIB (slow, mostly paywalled)

Outputs:
    data/processed/tobb_hazelnut.csv    — daily exchange prices, multiple regions, 2005-present
    data/processed/tb_org_hazelnut.csv  — free-market variety prices, 2019-present
    data/processed/turib_hazelnut.csv   — TURIB PDF bulletins (sparse)
    data/processed/hazelnut_combined.csv — merged, deduplicated master
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

import scrape_tobb
import scrape_tb_org
import scrape_turib
from utils import PROC_DIR, SCHEMA, setup_logging, save_processed


def merge_all(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all source frames, deduplicate, sort."""
    if not frames:
        return pd.DataFrame(columns=pd.Index(SCHEMA))

    combined = pd.concat(frames, ignore_index=True)

    # Ensure all schema columns present
    for col in SCHEMA:
        if col not in combined.columns:
            combined[col] = None

    combined = combined[SCHEMA]
    combined['date'] = pd.to_datetime(combined['date'], format='mixed', dayfirst=True)
    combined = combined.dropna(subset=['date'])
    combined = combined.drop_duplicates(subset=['source', 'date', 'exchange', 'product', 'variety'])
    combined = combined.sort_values(['source', 'exchange', 'variety', 'date']).reset_index(drop=True)
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tobb-only',   action='store_true')
    ap.add_argument('--no-turib',    action='store_true')
    ap.add_argument('--year-start',  type=int, default=2005)
    ap.add_argument('--verbose',     action='store_true')
    args = ap.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger(__name__)

    frames: list[pd.DataFrame] = []

    def _run(label, fn):
        try:
            df = fn()
            if not df.empty:
                frames.append(df)
                log.info('%s: %d rows', label, len(df))
            return df
        except Exception as e:
            log.error('%s failed: %s', label, e)
            return pd.DataFrame()

    # --- TOBB ---
    log.info('=== TOBB ===')
    _run('TOBB', lambda: scrape_tobb.run(year_start=args.year_start))

    if not args.tobb_only:
        # --- TB.org ---
        log.info('=== TB.org ===')
        _run('TB.org', scrape_tb_org.run)

        # --- TURIB ---
        if not args.no_turib:
            log.info('=== TURIB ===')
            _run('TURIB', scrape_turib.run)

    # --- Merge ---
    log.info('=== Merging ===')
    combined = merge_all(frames)
    save_processed(combined, 'hazelnut_combined.csv')

    log.info('Done. Combined: %d rows from %d sources',
             len(combined), combined['source'].nunique() if not combined.empty else 0)

    if not combined.empty:
        print('\n--- Summary ---')
        summary = (
            combined.groupby('source')
            .agg(rows=('date', 'count'),
                 earliest=('date', 'min'),
                 latest=('date', 'max'))
        )
        print(summary.to_string())


if __name__ == '__main__':
    main()
