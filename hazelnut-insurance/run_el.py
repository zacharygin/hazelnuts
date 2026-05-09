"""
Run the full expected loss calculation and print results.

Usage:
    python run_el.py

Skips any trigger whose data hasn't been downloaded yet and reports
what's missing at the end.
"""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.pricing.expected_loss import compute_expected_loss, print_el_table

result = compute_expected_loss(
    half_life=15.0,
    include_frost=True,
    include_hail=True,
    include_drought=True,
    include_lira=True,
)

print_el_table(result)

# Report which triggers are still missing data
df = result["per_trigger"]
present = set(df[df["trigger"] != "TOTAL"]["trigger"].tolist())
all_triggers = {"Production", "Frost", "Drought", "Hail", "Lira Depreciation",
                "Efb Outbreak", "Export Disruption", "Logistics Disruption"}
missing = all_triggers - present
if missing:
    print(f"NOT YET PRICED (data missing): {', '.join(sorted(missing))}")
    print("  Frost/Hail: run  python -m src.data.era5_downloader --type all --start 1990 --end 2024")
    print("  Drought:    run  python -m src.data.spei_downloader")
