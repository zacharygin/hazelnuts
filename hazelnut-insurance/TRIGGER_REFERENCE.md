# Turkish Hazelnut Parametric Insurance — Trigger Reference

**Contract region:** Black Sea hazelnut belt, Turkey (Ordu, Giresun, Trabzon, Samsun, Düzce)
**Production weighting:** Province-level output shares from FAOSTAT / TUIK
**Data source:** ERA5 reanalysis (ECMWF), FAOSTAT QCL, Yahoo Finance TRYUSD=X
**Settlement currency:** USD (or EUR) as % of notional

---

## Expected Loss Methodology

### Core Formula

For every trigger, the actuarially fair expected loss (EL) is:

$$
\text{EL} = \int_{-\infty}^{\infty} \text{payout}(x) \cdot f(x;\,\hat{\theta}) \; dx
$$

where $x$ is the trigger metric for a given year, $\text{payout}(x)$ is the piecewise-linear payout function defined by the trigger's bands, and $f(x;\,\hat{\theta})$ is a probability density fitted to the historical record of $x$.

This decomposes as:

$$
\text{EL} = \underbrace{P(\text{trigger fires})}_{\text{frequency}} \times \underbrace{\mathbb{E}[\text{payout} \mid \text{trigger fires}]}_{\text{severity}}
$$

EL is expressed as a fraction of notional (e.g. EL = 0.05 means the contract is expected to pay out 5% of face value per year on average).

### Recency Weighting

Not all historical years are equally informative. Climate change and structural shifts in the Turkish economy mean recent years better represent current risk. Each observation year $t$ is assigned an exponentially decaying weight:

$$
w(t) = \exp\!\left(-\frac{\ln 2}{\tau} \cdot (T - t)\right)
$$

where $T$ is the current year and $\tau = 15$ years is the half-life (a year 15 years ago carries half the weight of the current year). Weights are normalised to sum to 1.

The effective sample size (Kish approximation) measures how much information the weighted sample contains:

$$
n_{\text{eff}} = \frac{1}{\sum_t w_t^2}
$$

A half-life of 15 years over a 35-year record gives $n_{\text{eff}} \approx 21$ — roughly equivalent to an unweighted 21-year sample.

### Distribution Fitting

Two distribution families are used depending on the trigger metric's characteristics:

**Kernel Density Estimate (KDE)** — used for continuous, two-sided metrics (lira depreciation, pollination precip):

$$
\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} w_i \cdot K\!\left(\frac{x - x_i}{h}\right)
$$

where $K$ is a Gaussian kernel and bandwidth $h$ is set by Scott's rule: $h = 1.06\,\hat{\sigma}\,n_{\text{eff}}^{-1/5}$.

**Zero-Inflated Gamma** — used for non-negative metrics with a mass at zero (frost DH, hail CP), where most years have no event:

$$
f(x) = p_0 \cdot \delta(x=0) + (1-p_0) \cdot \text{Gamma}(x;\,\alpha,\,\beta)
$$

$p_0$ is the empirical fraction of zero years; $\alpha$ and $\beta$ are fitted via weighted method of moments on the positive observations.

### Numerical Integration

The integral $\int \text{payout}(x) \cdot f(x) \, dx$ is computed on a fine grid of 2,000 points spanning the metric's support:

$$
\text{EL} \approx \sum_{k=1}^{2000} \text{payout}(x_k) \cdot \hat{f}(x_k) \cdot \Delta x
$$

The payout function is piecewise linear within each band and zero outside the trigger range, so the integrand is nonzero only where the trigger is active.

### Named Perils (Binary Events)

For events with no continuous metric, EL reduces to the classical actuarial formula:

$$
\text{EL} = P(\text{event occurs in year}) \times \text{payout fraction}
$$

Probabilities are estimated from historical event counts using a Jeffreys prior on the Poisson rate: $\hat{\lambda} = (n_{\text{events}} + 0.5) / T_{\text{years}}$.

### Premium Loading

The indicated gross premium is:

$$
\text{Premium} = \text{EL} \times (1 + \theta) + c
$$

where $\theta \approx 0.35$–$0.40$ is the risk load (compensates the insurer for variance and capital cost) and $c \approx 1$–$2\%$ of notional is a fixed capital charge. At $\theta = 0.40$, the gross premium is approximately $1.4 \times \text{EL}$.

---

## 1. Spring Frost

**Peril:** Late spring frost kills open catkins and young leaves during the critical bloom window, destroying the season's flowers before fertilisation.

**Settlement:** May (following the March–April risk window)

**Metric:** Cumulative degree-hours (DH) below the frost threshold at each province, production-weighted across the basket.

$$
DH = \sum_{t \in \text{window}} \max(0,\ T_{\text{threshold}} - T_t)
$$

| Window | Threshold | Rationale |
|---|---|---|
| March 15 – March 31 | −3.0 °C | Buds partially closed; more cold-tolerant |
| April 1 – April 30 | −1.5 °C | Open catkins and young leaves; highly sensitive |

**Data source:** ERA5 hourly 2m temperature (`2m_temperature`), 6-hourly resolution

**Payout schedule:**

| DH (degree-hours) | Payout |
|---|---|
| 0 – 25 | 0% |
| 25 – 50 | 25% – 50% |
| 50 – 100 | 50% – 90% |
| 100+ | 90% – 100% |

*Linear interpolation within each band.*

**Expected Loss Calculation:**

Distribution: Zero-Inflated Gamma (most years have DH = 0; positive DH years follow a Gamma tail).

$$
\text{EL}_{\text{frost}} = \int_0^{\infty} \text{payout}_{\text{frost}}(x) \cdot \left[ p_0\,\delta(x) + (1-p_0)\,\text{Gamma}(x;\alpha,\beta) \right] dx
$$

Since $\text{payout}(0) = 0$, the zero-mass term drops out:

$$
\text{EL}_{\text{frost}} = (1 - p_0) \int_{25}^{\infty} \text{payout}_{\text{frost}}(x) \cdot \text{Gamma}(x;\alpha,\beta) \, dx
$$

$p_0$ = fraction of years with DH < 25 (no payout); $\alpha, \beta$ fitted to positive DH observations with recency weights.

**Notes:**
- Catastrophic tail trigger — only event with near-100% payout potential
- ERA5 likely understates actual DH by 30–50% due to cold-air drainage in valley orchards (31 km grid cannot resolve micro-topography)
- Key documented Turkish frost years: 2007, 2009, 2011

---

## 2. Pollination Failure

**Peril:** Sustained rainfall during the wind-pollination window washes pollen from male catkins before airborne transfer to female flowers, reducing fertilisation and nut set.

**Reference:** Mehlenbacher (1991), *Genetic Resources of Temperate Fruit and Nut Crops*, ISHS Acta Horticulturae 290: 791–836.

**Settlement:** April (bloom window fully observed)

**Metric:** Production-weighted total accumulated precipitation (mm) across hazelnut provinces during February and March.

$$
P_{\text{bloom}} = \sum_{m \in \{\text{Feb, Mar}\}} P_{m,\text{weighted}}
$$

**Data source:** ERA5 monthly means (`total_precipitation`), converted from mean daily rate (m/day) to monthly total (mm/month).

**Payout schedule:**

| Feb + Mar Precip (mm) | Payout |
|---|---|
| 0 – 250 | 0% |
| 250 – 270 | 0% – 8% |
| 270 – 300 | 8% – 15% |
| 300+ | 15% (cap) |

*Linear interpolation within each band.*

**Deductible rationale:** Black Sea coast median Feb+Mar precip is ~210 mm (1950–2024); 250 mm represents the 90th percentile. The deductible absorbs normal wet-season rainfall and only triggers on genuinely exceptional bloom-period rain.

**Cap rationale:** Pollination failure is spatially heterogeneous — provinces experiencing heavy rain may be partially offset by drier neighbouring provinces. Basket-level impact is partial.

**Expected Loss Calculation:**

Distribution: KDE fitted to 75 years of Feb+Mar precip (1950–2024) with recency weights. Trigger fires when precip *exceeds* the deductible (upper-tail event), so:

$$
\text{EL}_{\text{poll}} = \int_{250}^{\infty} \text{payout}_{\text{poll}}(x) \cdot \hat{f}_{\text{KDE}}(x) \, dx
$$

$$
P(\text{fire}) = \int_{250}^{\infty} \hat{f}_{\text{KDE}}(x) \, dx = 1 - \hat{F}_{\text{KDE}}(250)
$$

**Current estimate (1950–2024):** EL ≈ **1.4%** | P(fire) ≈ **9%** | E[payout | fire] ≈ **8%**

---

## 3. Summer Hail

**Peril:** Severe convective hailstorms during kernel development mechanically damage nuts, split husks, and cause bruising that leads to mould and grade-down losses.

**Reference:** Polat et al. (2016), "Severe Hail Climatology of Turkey," *Monthly Weather Review* 144(1). Hail accounts for >60% of weather-related insured agricultural losses in Turkey (2007–2013).

**Settlement:** September (after harvest completion)

**Metric:** Maximum 6-hour accumulated convective precipitation (mm) at any hazelnut province during June – August.

**Data source:** ERA5 hourly convective precipitation (`convective_precipitation`), aggregated to 6-hour windows.

**Payout schedule:**

| Max 6h Convective Precip (mm) | Payout |
|---|---|
| 0 – 8 | 0% |
| 8 – 15 | 5% – 15% |
| 15 – 25 | 15% – 25% |
| 25+ | 25% (cap) |

*Linear interpolation within each band.*

**Expected Loss Calculation:**

Distribution: Zero-Inflated Gamma (most years have no qualifying convective event at any province; severe storms follow a Gamma tail on the positive values).

$$
\text{EL}_{\text{hail}} = (1 - p_0) \int_{8}^{\infty} \text{payout}_{\text{hail}}(x) \cdot \text{Gamma}(x;\alpha,\beta) \, dx
$$

$p_0$ = fraction of years where max 6h CP across all provinces remains below 8 mm.

**Deductible rationale:** 8 mm/6h separates light convective rain from hail-producing thunderstorms.

**Cap rationale:** A single storm rarely covers the full basket. Even a severe hailstorm hitting one province leaves others intact, limiting basket-level loss.

---

## 4. Lira Depreciation

**Peril:** Rapid TRY/USD depreciation inflates the cost of USD-priced farm inputs (fertiliser, fuel, machinery), compressing grower margins and reducing orchard reinvestment, leading to lower yields in subsequent seasons.

**Settlement:** December (full calendar year observed)

**Metric:** Annual TRY/USD depreciation rate.

$$
d_t = \frac{S_t - S_{t-1}}{S_{t-1}}, \quad S_t = \text{TRY per USD at year-end}
$$

*Negative values indicate TRY weakening (e.g. −0.45 = TRY lost 45% vs USD).*

**Data source:** Yahoo Finance `TRYUSD=X` annual close prices (2005–present).

**Payout schedule:**

| Annual Depreciation | Payout |
|---|---|
| > −20% | 0% |
| −20% to −40% | 0% – 10% |
| −40% to −60% | 10% – 20% |
| < −60% | 20% (cap) |

*Linear interpolation within each band.*

**Deductible rationale:** TRY has depreciated 15–20%/year on average since 2010. The −20% deductible absorbs structural trend depreciation; the trigger activates only on acute stress episodes.

**Cap rationale:** Above −60% depreciation, hazelnut export prices tend to rally in lira terms (hazelnuts are a USD-priced commodity), partially offsetting further input cost inflation. Farmers on forward export contracts remain exposed.

**Expected Loss Calculation:**

Distribution: KDE fitted to 20 years of annual TRY/USD depreciation (2005–2024) with recency weights. Trigger fires when depreciation is *more negative* than −20% (lower-tail event):

$$
\text{EL}_{\text{lira}} = \int_{-1}^{-0.20} \text{payout}_{\text{lira}}(d) \cdot \hat{f}_{\text{KDE}}(d) \, dd
$$

$$
P(\text{fire}) = \hat{F}_{\text{KDE}}(-0.20) = \int_{-\infty}^{-0.20} \hat{f}_{\text{KDE}}(d) \, dd
$$

**Current estimate (2005–2024):** EL ≈ **2.4%** | P(fire) ≈ **42%** | E[payout | fire] ≈ **5.8%**

---

## 5. Named Perils

These events are priced as flat `P × payout` — no metric index; settlement requires confirmed event occurrence.

### 5a. Eastern Filbert Blight (EFB) Outbreak

**Peril:** *Anisogramma anomala* fungal infection devastates hazelnut orchards. Not currently present in Turkey or Europe; classified as EU quarantine organism.

**Reference:** EFSA PLH Panel (2018), EFSA Journal 16(2):5184.

**Settlement:** Upon confirmed EPPO/EFSA detection in Turkish hazelnut regions.

| Parameter | Value |
|---|---|
| Annual probability | 0.3% |
| Payout | 50% |
| **Expected Loss** | **0.15%** |

**Expected Loss Calculation:**

$$
\text{EL}_{\text{EFB}} = P(\text{outbreak}) \times \text{payout} = 0.003 \times 0.50 = \mathbf{0.15\%}
$$

Probability from Jeffreys prior: $\hat{\lambda} = (0 + 0.5) / 90 \approx 0.56\%$, discounted by $0.5\times$ for geographic distance → $0.3\%$/yr.

---

### 5b. Export Disruption

**Peril:** Turkish government restriction on hazelnut exports (quota, ban, or TMO intervention) that reduces export volume by >10% during August–December.

**Reference:** USDA FAS GAIN reports; FISKOBIRLIK/TMO intervention history.

**Settlement:** Upon confirmed export restriction announcement affecting the current crop year.

| Parameter | Value |
|---|---|
| Annual probability | 3.0% |
| Payout | 25% |
| **Expected Loss** | **0.75%** |

**Expected Loss Calculation:**

$$
\text{EL}_{\text{export}} = P(\text{restriction}) \times \text{payout} = 0.030 \times 0.25 = \mathbf{0.75\%}
$$

Rate: 1–2 qualifying events in 35 years = 2.9–5.7%/yr; 3% is the conservative floor.

---

### 5c. Bosphorus / Logistics Disruption

**Peril:** Commercial closure of the Turkish Straits for >30 days, preventing hazelnut export shipments.

**Reference:** Montreux Convention (1936), Arts. 2, 10, 19; Lieber Institute (2022).

**Settlement:** Upon confirmed commercial shipping closure exceeding 30 days.

| Parameter | Value |
|---|---|
| Annual probability | 0.15% |
| Payout | 15% |
| **Expected Loss** | **0.02%** |

**Expected Loss Calculation:**

$$
\text{EL}_{\text{Bosphorus}} = P(\text{closure}) \times \text{payout} = 0.0015 \times 0.15 = \mathbf{0.02\%}
$$

$$
P(\text{closure}) = P(\text{Turkey at war}) \times P(\text{closure} \mid \text{war}) \times P(\text{>30 days} \mid \text{closed}) \approx 1.5\% \times 15\% \times 50\% = 0.11\%
$$

Jeffreys prior baseline: $(0 + 0.5)/90 = 0.56\%$; war-conditional model: $0.11\%$; midpoint used: $0.15\%$/yr.

---

## Summary Table

| Trigger | Settlement | Metric | Max Payout | EL (current estimate) | Data |
|---|---|---|---|---|---|
| Spring Frost | May | Degree-hours below −2°C | 100% | ~0% now → **2–4%** when complete | ERA5 hourly temp (downloading) |
| Pollination Failure | April | Feb+Mar precip (mm) | 15% | **1.4%** | ERA5 monthly 1950–2024 ✓ |
| Summer Hail | September | Max 6h conv. precip (mm) | 25% | TBD | ERA5 hourly precip (downloading) |
| Lira Depreciation | December | Annual TRY/USD change | 20% | **2.4%** | Yahoo Finance 2005–2024 ✓ |
| EFB Outbreak | On event | Confirmed detection | 50% | 0.15% | Literature / EFSA ✓ |
| Export Disruption | On event | Confirmed restriction | 25% | 0.75% | USDA FAS / TMO ✓ |
| Bosphorus Disruption | On event | Confirmed closure | 15% | 0.02% | Montreux / news ✓ |
| **TOTAL** | | | | **~4.7–8.7%** | |

*Risk-loaded premium at 1.4× loading: approximately **6.6–12.2%** of notional once frost and hail data complete.*

---

## Payout Cap Justifications

### Spring Frost — 100%

A catastrophic late frost (April, −5°C or below) kills essentially all open flowers simultaneously across the entire basket. The Black Sea hazelnut belt is geographically compact — a single cold air mass covers all major provinces in one night. Documented events (1991, 2009) produced 40–60% national production drops; given that ~70% of Turkey's output comes from the core basket provinces, a basket-level 80–100% loss is agronomically credible. No other peril has this spatial correlation structure. This is consistent with IFC/World Bank parametric frost contracts for tree crops (coffee, citrus), which routinely use 100% caps where total flower kill is possible.

### Pollination Failure — 15%

Deliberately shallow. Wind pollination failure is spatially heterogeneous — rain hits some provinces harder than others, and trees at different elevations and aspects pollinate at different times within the February–March window. Even in the wettest bloom season on record, some orchards achieve successful fertilisation. The mechanism is probabilistic (reduced pollen transfer, not zero), and pollination failure at the basket level is a frequency risk, not a severity risk. Cross-checking against observed data: the worst documented wet-bloom years in Turkey produced nut set reductions of 15–20%, not total failures. The 15% cap matches the observed upper bound.

### Summer Hail — 25%

Capped due to the spatial concentration of hailstorms. A storm cell covers 10–50 km², not the 500+ km of coastline spanning the basket. A severe storm completely destroying Ordu (the largest province, ~30% of production) leaves Giresun, Trabzon, Samsun, and Düzce intact — a basket-level loss of roughly 30% at most. ERA5 convective precipitation is also an imperfect hail proxy, introducing basis risk that argues for conservatism. The 25% cap is consistent with Turkish private agricultural hail insurance on spatially diversified portfolios (industry benchmark: 15–30%).

### Lira Depreciation — 20%

Bounded from above by the natural export price hedge. Hazelnuts are priced internationally in USD. When TRY collapses, the lira value of export revenues rises in proportion, partially offsetting higher USD-priced input costs (fertiliser, fuel, machinery). Above approximately −60% annual depreciation, this export price rally becomes the dominant effect and growers selling into the export market can actually benefit in lira terms. The trigger is designed to cover the window where input costs spike faster than export revenues adjust — typically the first 6–12 months of a currency crisis. The 20% cap reflects the net input cost squeeze after the export price offset. Growers on fixed-price forward contracts remain exposed beyond this, which is acknowledged as residual basis risk.

### EFB Outbreak — 50%

If Eastern Filbert Blight reaches Turkish orchards it is a multi-year catastrophic event — infected trees require 3–5 years to recover or be replanted, and the fungus spreads rapidly through orchards. 50% reflects a severe but not total first-year impact on the basket (some provinces will not yet be affected at point of detection). A higher cap would require epidemiological spread modelling beyond the scope of this contract.

### Export Disruption — 25%

A seasonal export ban or quota typically lasts one crop year and affects volume, not price across all markets. Turkey's domestic consumption (~50,000 MT/yr) and non-EU export routes (Middle East, Russia) provide partial revenue offsets. The 25% figure is consistent with documented revenue impact from analogous interventions (e.g. the 2017 hazelnut export quota episode) and accounts for the fact that affected exporters can partially redirect shipments to alternative buyers at lower margins.

### Bosphorus Disruption — 15%

A 30-day commercial closure is economically disruptive but not catastrophic for hazelnut exporters. Alternative logistics routes exist — overland rail through Anatolia, Black Sea ports routed via the Adriatic, or holding inventory — at a meaningful cost premium. The 15% cap covers the incremental logistics cost and revenue delay rather than total revenue loss. A closure long enough to cause total loss (>6 months) has probability approaching zero under the Montreux framework, since it requires Turkey to be an active war belligerent.

---

## Aggregation Rules

### Payout in a Given Year

All triggers are additive, subject to an annual cap:

$$
\text{payout}_{\text{year}} = \min\!\left(\sum_{i} \text{payout}_i,\ 1.0\right)
$$

Each trigger $i$ pays its piecewise-linear amount independently; the sum is capped at 100% of notional. A year with a catastrophic frost (90% payout) and a severe hail event (20% payout) settles at 100%, not 110%.

### Trigger Independence Classification

| Relationship | Triggers | Rule | Rationale |
|---|---|---|---|
| **Independent** | Frost, Hail, Pollination | Additive | Different seasons, different atmospheric drivers |
| **Independent** | Weather perils vs. Lira | Additive | Physical weather and macroeconomic events are uncorrelated |
| **Positively correlated** | Lira + Export disruption | Additive, but sub-capped | Currency crises cluster with government commodity intervention |
| **Substitutive** | Frost + any future price trigger | Max, not sum | A frost-driven price spike measures the same underlying event |

### Political/Macro Sub-Cap

Lira depreciation, export disruption, and Bosphorus closure are all driven by the same underlying geopolitical and macroeconomic stress environment. In a compound crisis (e.g. war, sanctions, or sovereign debt event), all three could fire simultaneously. To prevent the insurer from being triple-exposed to a single macro scenario, a combined sub-cap applies:

$$
\text{payout}_{\text{macro}} = \min\!\left(\text{payout}_{\text{lira}} + \text{payout}_{\text{export}} + \text{payout}_{\text{Bosphorus}},\ 0.35\right)
$$

The 35% sub-cap reflects the maximum plausible combined input cost squeeze and revenue disruption from a macro shock, net of the natural export price offset.

### Total EL vs. Payout Cap

For EL calculation purposes, individual ELs sum directly:

$$
\text{EL}_{\text{total}} = \sum_i \text{EL}_i
$$

This is exact for independent triggers by linearity of expectation. The 100% cap reduces effective EL only when multiple large triggers co-fire — with current calibration (total EL ~5–9%), the cap correction is negligible (~0.1–0.2%). It becomes material only in extreme tail scenarios where frost fires at near-maximum severity simultaneously with another trigger.

### Structural Note: Frost Dominance

Once ERA5 frost data is complete, frost EL is expected to represent ~40–50% of total EL. The contract is structurally a **spring frost contract with supplementary covers** for hail, pollination failure, currency stress, and named perils. The supplementary triggers add frequency (they fire more often) at lower severity (capped at 15–25%), while frost provides the catastrophic tail exposure.

### Optional True-Up (November)

If TUIK production statistics are available post-harvest:

$$
\text{true-up} = \max\!\left(0,\ \text{actual\_production\_loss} - \text{weather\_payout\_already\_paid}\right)
$$

This closes the basis risk gap for years where weather triggers understate actual crop damage (e.g. disease, orchard management failures). The true-up is optional and subject to data quality review.

---

## Basis Risk

The primary unhedged exposures are:

1. **Biennial bearing**: Hazelnut trees naturally alternate heavy/light years (~25–30% of production variance), independent of any weather trigger
2. **ERA5 spatial smoothing**: 31 km grid understates frost severity in valley orchards; hail coverage is approximate
3. **TUIK data quality**: Turkish national production statistics are subject to revision and political noise
4. **Disease (non-EFB)**: Grey mould (*Botrytis*), bacterial blight — not modelled
5. **Orchard management**: Irrigation, pruning, fertilisation decisions affect yield independently of weather

---

*Generated: 2026-05-06 | Model version: ERA5-based parametric v0.1*
*Data: ERA5 © ECMWF; FAOSTAT © FAO; FX data © Yahoo Finance*
