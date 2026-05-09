# Data Sources

All raw data lives under `data/raw/`. This document explains each source: what it measures, how it was obtained, how the derived metric is computed, and where the raw files sit.

---

## 1. ERA5 Frost Degree-Hours

**Files**
- `data/raw/era5/era5_temp_YYYY.nc` — hourly 2m temperature per year (Feb–May)
- `data/raw/era5_frost_monthly.csv` — derived annual frost DH by phenological phase

**What it measures**
Production-weighted accumulated degree-hours below a crop-damage threshold during the full spring frost risk window (February–May). Higher DH = more cold stress on hazelnut flowers, catkins, and developing nuts.

**Source**
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) — ERA5 reanalysis, dataset `reanalysis-era5-single-levels`, variable `2m_temperature`. Downloaded via `cdsapi`. Requires free CDS account and `~/.cdsapirc`.

**Spatial domain**
Bounding box: lat 40–42°N, lon 29–41°E.  
Point extraction at nearest ERA5 grid cell (0.25°) to each province centroid.

| Province | Lat | Lon | Production weight |
|---|---|---|---|
| Ordu | 40.98 | 37.88 | 31.1% |
| Giresun | 40.91 | 38.39 | 16.3% |
| Samsun | 41.29 | 36.33 | 14.1% |
| Sakarya | 40.74 | 30.40 | 12.8% |
| Düzce | 40.84 | 31.16 | 10.7% |
| Trabzon | 41.00 | 39.72 | 6.1% |

Weights from TÜİK provincial hazelnut production data. "Other" provinces (8.9%) excluded — no representative point available; weights normalised to sum to 1.0 internally.

**Computation**
Five-phase frost degree-hours, production-weighted:

```
feb_dh     = Σ_p  w_p × Σ_hour∈Feb1-28   max(0, -5.0°C − T_hour)
emarch_dh  = Σ_p  w_p × Σ_hour∈Mar1-14   max(0, -5.0°C − T_hour)
lmarch_dh  = Σ_p  w_p × Σ_hour∈Mar15-31  max(0, -3.0°C − T_hour)
april_dh   = Σ_p  w_p × Σ_hour∈Apr1-30   max(0, -2.3°C − T_hour)
may_dh     = Σ_p  w_p × Σ_hour∈May1-31   max(0, -2.3°C − T_hour)
frost_dh   = feb_dh + emarch_dh + lmarch_dh + april_dh + may_dh
```

**Thresholds — literature basis**

| Phase | Threshold | Basis |
|---|---|---|
| Feb 1–Mar 14 | -5.0°C | Catkin elongation / female flower emergence; multiple sources confirm -5°C as critical for open catkins (Taghavi et al., ISHS; Turkish climate studies) |
| Mar 15–31 | -3.0°C | Post-pollination, fertilization beginning; -3°C is partially supported; the -2.3°C no-damage finding cited below was April-specific |
| Apr 1–30 | -2.3°C | Nut set, pre-fertilization ovule; BIO Web of Conferences 2021 (Russian vars): -2.3°C caused no damage — damage onset below this value |
| May 1–31 | -2.3°C | Early nut development / fertilization period; no direct literature — applying same threshold as April (same stage) |

Note: fertilization in *Corylus avellana* occurs 4–5 months after pollination (late May/June), so April/May damage acts on the pre-fertilization ovule and surrounding leaf/shoot tissue rather than the fertilized nut directly.

**Coverage**
- Hourly files: 1940–2024 (download in progress via `scripts/fetch_era5_frost_historical.py --start 1940 --end 2024 --force`)
- `feb_dh` and `may_dh` are zero for years where only March–April was previously downloaded; will be populated as files are replaced.

**Code**
- `hazelnut-insurance/src/triggers/frost.py` — `compute_dh(year)`
- `scripts/fetch_era5_frost_historical.py` — full historical download and CSV rebuild

---

## 2. ERA5 Monthly Precipitation

**Files**
- `data/raw/era5_precip_monthly.csv` — spatially averaged monthly precip (mm)
- `data/raw/era5_monthly/data_stream-moda_stepType-avgad.nc` — raw monthly totals

**What it measures**
Monthly mean total precipitation averaged over the hazelnut belt, used for:
- **Pollination trigger:** Feb–Mar excess rainfall disrupts wind pollination of catkins.
- **Drought characterization:** raw JJA precipitation as a drought indicator (preferred over SPEI — see Section 3).

**Source**
CDS ERA5 monthly averaged data on single levels, variable `total_precipitation` (`tp`). Downloaded at 0.25° resolution.

**Coverage**: 1950–2024 (monthly)

---

## 3. Drought — Raw JJA Precipitation vs SPEI

**Files**
- `data/raw/era5_precip_monthly.csv` — monthly precipitation (JJA months already present)
- `data/raw/spei/spei03_era5.csv` — SPEI-03 August (legacy; poor signal)

**Why SPEI underperforms here**

SPEI is a *standardized* index: it converts the absolute precipitation-ET balance into a z-score relative to the long-run climatological distribution at that location. This removes two things that are actually relevant:

1. **Absolute moisture level.** The Black Sea hazelnut belt receives 800–1,200 mm/year. A "moderate drought" at SPEI -1.0 here means the region still has ~600–700 mm — more than most hazelnut-growing regions worldwide. SPEI cannot distinguish "dry for the Black Sea but still adequate" from "genuinely damaging." Raw mm preserves this information.

2. **Trend.** Standardization is done against a historical baseline. If the region is drying over time (as in many Mediterranean-adjacent areas), recent years will still centre around SPEI ≈ 0 even if absolute rainfall has fallen 15%. This matters for climate-change-adjusted pricing.

The practical result: our SPEI-03 August regressor had R² ≈ 0.004 and p ≈ 0.80 against buyer price excess — statistically zero signal. Raw June + July + August precipitation (in mm, summed or as separate month columns) retains the absolute scale and should be tested instead.

**Preferred approach going forward**
Use raw ERA5 monthly `tp` (mm) for Jun, Jul, Aug individually — already downloaded in `era5_precip_monthly.csv`. Sum to `jja_precip_mm`. Expected signal direction: low JJA precip → mild yield drag at extreme deficits; but the relationship is likely non-linear with a high threshold given hazelnut drought tolerance.

**SPEI-03 reference** (retained for completeness)
Vicente-Serrano, Beguería & López-Moreno (2010), "A Multiscalar Drought Index Sensitive to Global Warming," *Journal of Climate* 23(7), 1696–1718.

**Coverage**: 1950–2024 (monthly precip); SPEI 1950–2024

---

## 4. FAOSTAT Turkey Hazelnut Production

**Files**
- `data/raw/faostat/turkey_hazelnut_production.csv`

**What it measures**
Official annual Turkey hazelnut production in metric tons.

**Source**
[FAOSTAT](https://www.fao.org/faostat/), Production Quantities of Agricultural Products, Item 225 (Hazelnuts, with shell). Retrieved via the `faostat` Python package or bulk CSV download. FAO aggregates data from national reporting agencies (TÜİK for Turkey).

**Coverage**: 1961–2024

**Notes**
FAOSTAT figures can lag 1–2 years for the most recent season. For 2023–2025, supplement with TÜİK or USDA GAIN estimates (see sections 5 and 6 below). The 2025 TÜİK announcement (~520,000 MT) is hardcoded in `trigger_params.yaml` production overrides.

---

## 5. TÜİK Hazelnut Balance Sheet

**Files**
- `data/raw/tuik_hazelnut_balance.csv` — cleaned panel
- `data/raw/tuik_nuts_balance_2024.xls` — raw official release

**What it measures**
Detailed annual supply and utilization balance for Turkish hazelnuts: production, imports, exports (total and to EU), domestic consumption, seed use, losses, and stock changes.

**Source**
[Turkish Statistical Institute (TÜİK)](https://www.tuik.gov.tr/) — Nuts and Oil Seeds Balance Sheet, published annually. Covers marketing years (e.g., 2000/01).

**Coverage**: 2000–2024

**Key columns**
`production_mt`, `imports_mt`, `exports_mt`, `domestic_use_mt`, `processing_mt`, `losses_mt`, `stock_change_mt`, `per_capita_kg`, `self_sufficiency_pct`

---

## 6. USDA GAIN Hazelnut Production Estimates

**Files**
- `data/raw/usda_gain_hazelnut_estimates.csv`

**What it measures**
USDA Foreign Agricultural Service (FAS) agricultural attaché estimates of Turkey hazelnut production, reported at the marketing year level alongside the TÜİK official figure.

**Source**
[USDA FAS GAIN Reports](https://apps.fas.usda.gov/newgainapi/), Turkey-specific hazelnut reports published annually. Estimates are based on field surveys and often diverge from TÜİK official data during contested years.

**Coverage**: 2009–present

**Use**
Cross-check against FAOSTAT/TÜİK, and as an early production signal when TÜİK data is not yet available.

---

## 7. TRY/USD Exchange Rate

**Files**
- `data/raw/fx/tryusd_annual.csv`

**What it measures**
Annual closing TRY/USD exchange rate and year-on-year depreciation rate (e.g., -0.44 means TRY lost 44% of its USD value during the year).

**Source**
[Yahoo Finance](https://finance.yahoo.com/) — ticker `TRYUSD=X`, retrieved via `yfinance` Python package. Annual close = last trading day of each calendar year.

**Coverage**: 2006–2024

**Hazelnut relevance**
Turkish hazelnuts are priced and exported in USD/EUR, but producers incur costs in TRY. Sharp TRY depreciation compresses real farm margins (input costs rise in local terms) even if export revenues hold. This is partially captured in the equity basket via the Turkcell/Ülker positions, which are Turkish-listed assets.

---

## 8. Giresun Commodity Exchange Spot Prices

**Files**
- `data/raw/giresun_spot_prices_monthly.csv` — monthly VWAP, 2000–2023
- `data/raw/giresun_spot_prices_cropyear.csv` — crop-year averages
- `data/raw/giresun_spot_prices_annual.csv` — calendar-year averages

**What it measures**
Volume-weighted average price (VWAP) of physical hazelnut transactions on the Giresun Commodity Exchange (Giresun Ticaret Borsası), the primary price discovery venue for Turkish in-shell hazelnuts. Prices in both TRY/kg and USD/kg, for in-shell and shelled product.

**Source**
Giresun Ticaret Borsası — monthly transaction records. No public API; data collected manually or via web scraping (`scripts/scrape_giresun.py`). Exchange records begin commercially in the 1990s; reliable monthly data from 2000.

**Coverage**: 2000–2023

**Note on crop-year alignment**
Hazelnut harvest runs July–August; the "crop year" for pricing is approximately September of harvest year through August of the following year. The `crop_year` column in these files identifies the harvest year. The August opening bid (first day of new crop) is excluded from crop-year mean calculation as it reflects new-crop price discovery, not the realized mean.

---

## 9. Equity Basket Proxy

**Files**
- `data/raw/basket/equity_basket_monthly.csv` — monthly closing prices
- `data/raw/basket/equity_basket_prices.csv` — annual September prices (aligned to crop year)
- `data/raw/basket/basket_weights.csv` — ridge regression weights

**What it measures**
Monthly closing prices for 10 publicly traded proxies that capture demand-side exposure to hazelnuts and soft commodities:

| Ticker | Company | Currency | Role |
|---|---|---|---|
| TUR | iShares MSCI Turkey ETF | USD | Turkey sovereign / macro |
| ULKER.IS | Ülker Bisküvi | TRY | Turkey confectionery buyer |
| MDLZ | Mondelēz International | USD | Global hazelnut-chocolate buyer |
| NESTLE | Nestlé SA | USD | Global hazelnut-chocolate buyer |
| BARN.SW | Barry Callebaut | CHF | Largest cocoa/choc ingredients processor |
| HSY | Hershey Company | USD | US confectionery buyer |
| BG | Bunge Global SA | USD | Agricultural commodity handler |
| CC=F | Cocoa futures front-month | USD | Substitute crop / demand signal |
| DBA | Invesco DB Agriculture ETF | USD | Broad agricultural price signal |
| TRYUSD=X | Turkish Lira / US Dollar | — | FX rate for TRY-priced assets |

**Source**
Yahoo Finance via `yfinance` Python package (see `notebooks/hazelnut_basket.ipynb`).

**Coverage**: Varies by ticker (Nestlé from ~2000; Bunge, Ülker from ~2004; TUR from ~2008).

**Derived metrics**
`basket_weights.csv` contains ridge regression coefficients from a two-stage model:
- **Model A (priceable):** basket assets only → weights used for hedging/pricing
- **Model B (validation):** basket + control variables (almond futures, Turkey 10Y yield, NDVI) → explained variance

---

## 10. Control / Supplementary Variables

These are used in Model B regression validation but are not priceable basket components.

### Almond Futures (ZA=F)
Yahoo Finance, USD/ton. Acts as a control for global tree-nut supply conditions — if almonds and hazelnuts move together, the basket explains more variance. Coverage: 2007–present.

### Turkey 10-Year Government Bond Yield
[FRED](https://fred.stlouisfed.org/) series `IRLTLT01TRM156N` — Turkey 10Y yield (%). Retrieved via `pandas_datareader`. Captures sovereign risk premium in Turkish assets; correlated with TRY depreciation periods. Coverage: 2005–present.

### NDVI — Giresun/Samsun Hazelnut Belt
**File**: `data/raw/basket/ndvi_giresun.csv` *(downloading)*

**Source**: [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/) REST API, product MOD13A3.061 (MODIS Terra Vegetation Indices, 1 km, monthly). Requires free NASA Earthdata account.

**What it measures**
April–May mean NDVI (Normalized Difference Vegetation Index) averaged over a 0.25° point grid spanning the Samsun–Ordu–Giresun–Trabzon corridor (lon 35.0–40.5°E, lat 40.1–41.5°N, ~138 grid points). April–May NDVI captures pre-bloom canopy health as a leading indicator of the coming harvest.

**Computation**
```
Grid points: 0.25° spacing across bbox → ~138 points
Raw NDVI:    integer ×10,000 (e.g., 6500 = NDVI 0.65)
Fill/bad:    values < -2000 raw (< -0.2 after scaling) masked to NaN
Annual mean: spatial mean over all valid points, filtered to months 4-5
```

**Coverage**: 2000–2025 (when AppEEARS task `d8646219-20d3-4e4e-a4fa-6e0e364b48a9` completes)

**Fetch script**: `scripts/fetch_ndvi_giresun.py`

---

## 11. TMO / Basket Prices

**File**: `data/raw/basket/basket_dataset.csv`

**What it measures**
Annual TMO (Toprak Mahsulleri Ofisi — Turkish Grain Board) announced hazelnut purchase price in EUR/kg, alongside comparable almond, walnut, and hazelnut EUR reference prices.

**Source**
TMO publishes the government support price each harvest year; collected from TMO annual reports and ISHS (International Society for Horticultural Science) publications.

**Coverage**: 1985–present (sparse for 1985–1999)

**Note**
The TMO price acts as a price floor in bad crop years (the government buys at support price to stabilize farmer income). This creates a non-linear downside hedge for producers: in severe frost/drought years, the TMO floor limits spot price collapse.

---

## 12. Named Perils (Binary Events)

These are not stored as CSVs; historical instances are hardcoded in `hazelnut-insurance/src/data/named_events.py`.

| Peril | Definition | Probability source | Literature |
|---|---|---|---|
| EFB Outbreak | Eastern Filbert Blight (*Anisogramma anomala*) enters Turkey | Jeffreys prior on 0 events / 90 yr; 0.5× Turkey distance discount → **0.3%/yr** | EFSA PLH Panel (2018), *EFSA Journal* 16(2):5184 |
| Export Disruption | Government restriction on hazelnut exports >10% of Aug–Dec volume | 1–2 qualifying events in 35 yr = **3–6%/yr**; conservative floor 3% | USDA FAS GAIN; TMO intervention history; ISHS (2001) |
| Logistics Disruption | Turkish Straits closed to commercial shipping >30 days | 0 events / 90 yr under Montreux Convention; P(Turkey at war) × P(closure\|war) × P(>30d\|closed) ≈ **0.15%/yr** | Montreux Convention (1936) Arts 2, 10, 19; Lieber Institute (2022) |

---

## 13. Unquantifiable / Deprecated Triggers

These triggers were investigated but cannot be priced reliably with available data. They are retained here for documentation purposes.

### Hail (ERA5 Convective Precipitation)

**Status: Deprecated as a priceable trigger.**

**Files (not used in pricing)**
- `data/raw/era5/era5_hail_YYYY.nc` — hourly convective precipitation per year
- `data/raw/era5_hail_monthly.csv` — derived annual max 6-hr CP per province-month

**Why it fails**
ERA5 reports `convective_precipitation` (variable `cp`) — total convective rainfall, not hail. Problems:
1. ERA5 at 0.25° (~31 km) cannot resolve individual hail cells, which are typically 1–10 km in diameter.
2. ERA5 `cp` captures all convective rain regardless of hail occurrence. Heavy convective rain with no hail scores identically to an actual hail event.
3. No ERA5 native hail variable exists. Proper hail data would require TSMS (Turkish State Meteorological Service) station records or EUMETSAT MSG SEVIRI satellite products.
4. Empirical calibration in `notebooks/hazelnut_pricing.ipynb` showed hail_cp_max ≥ 70mm had R² = 0.045, p = 0.37 — no statistically significant relationship to buyer price excess.

**ESWD reviewed and rejected**: The European Severe Weather Database (eswd.eu) was reviewed as a potential hail data source for the Turkish hazelnut belt. Turkey coverage consists of sparse, ad-hoc point observations with no programmatic API and no systematic temporal coverage. The record is insufficient to construct a usable annual time series for the six production provinces. No priceable trigger can be constructed from ESWD.

**Reference**: Polat et al. (2016), "Severe Hail Climatology of Turkey," *Monthly Weather Review* 144(1): 1,489 severe hail cases on 1,107 hail days over 1925–2014. Hail accounts for >60% of all weather-related insured agricultural losses in Turkey 2007–2013. The peril is real; the data is not.

**Code**: `hazelnut-insurance/src/triggers/hail.py` (retained but excluded from aggregation)

---

## Data Flow Summary

```
CDS (ERA5 hourly t2m)          → era5_temp_YYYY.nc    → era5_frost_monthly.csv
CDS (ERA5 monthly tp)          → era5_monthly/*.nc    → era5_precip_monthly.csv
                                                         (JJA sum → jja_precip_mm; Feb–Mar → pollination)
CDS (ERA5 hourly cp)           → era5_hail_YYYY.nc    → era5_hail_monthly.csv  [DEPRECATED]
NASA AppEEARS (MODIS NDVI)     → ndvi_giresun.csv
FAOSTAT                        → faostat/turkey_hazelnut_production.csv
TÜİK                           → tuik_hazelnut_balance.csv
USDA FAS GAIN                  → usda_gain_hazelnut_estimates.csv
Yahoo Finance (yfinance)       → basket/equity_basket_monthly.csv
                               → fx/tryusd_annual.csv
Giresun Ticaret Borsası        → giresun_spot_prices_*.csv
TMO / ISHS                     → basket/basket_dataset.csv
```

---

## Re-running / Refreshing Data

| Source | Command |
|---|---|
| ERA5 frost 1990–2024 | `python -m src.data.era5_downloader --start 1990 --end 2024 --type frost` (from `hazelnut-insurance/`) |
| ERA5 frost 1950–1989 | `python scripts/fetch_era5_frost_historical.py --start 1950 --end 1989` |
| ERA5 hail 1990–2024 | `python -m src.data.era5_downloader --start 1990 --end 2024 --type hail` |
| NDVI (AppEEARS) | `python scripts/fetch_ndvi_giresun.py` (resumes from cached task ID) |
| Equity basket prices | Run `notebooks/hazelnut_basket.ipynb` |
| Frost CSV only (no download) | `python scripts/fetch_era5_frost_historical.py --rebuild-csv-only` |
