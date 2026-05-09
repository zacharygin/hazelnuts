"""
Distributional expected loss engine.

Replaces the historical-mean estimator with fitted distributions, weighted
by recency so that recent years count more than distant history.

Why recency weighting matters
------------------------------
A 1942 frost event is less informative about 2030 frost risk than a 2018
event, because climate state, land use, and orchard composition have all
shifted. Exponential decay weighting formalises this: each year gets weight
proportional to exp(-ln2 * age / half_life). At half_life=15yr, data from
2009 gets half the weight of 2024; data from 1994 gets one quarter.

Distribution choices per trigger
----------------------------------
  Production shortfall   Weighted KDE (non-parametric) + Weighted Normal
                         Both are computed; KDE is the primary estimate.

  Drought (SPEI)         Normal by construction (SPEI is a probability
                         transform of precip → approximately N(0,1)).
                         We verify this holds empirically and fall back
                         to KDE if the fit is poor.

  Frost (degree-hours)   Zero-inflated Gamma:
                           - p0  = weighted P(no damaging frost)
                           - Positive DH | DH>0 ~ Gamma(alpha, rate)
                         This separates "frost occurrence" from
                         "frost severity given it occurs."

  Hail                   Same zero-inflated Gamma structure as frost.

All distributions return an ELResult with:
  el          expected payout as fraction of notional
  p_fires     probability the trigger crosses its threshold
  cond_el     E[payout | trigger fires]
  el_var      Var[payout] — needed for risk margin
  half_life   the recency decay parameter used

Usage
-----
  from src.pricing.distributions import fit_and_integrate, exponential_weights

  # Production shortfall example (data available now)
  series = production.metric_series()          # DataFrame[year, shortfall]
  result = fit_and_integrate(
      values   = series['shortfall'].values,
      years    = series['year'].values,
      payout_fn= production.compute_payout,
      dist_type= 'kde',
      half_life= 15,
  )
  print(result)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.stats as stats

logger = logging.getLogger(__name__)

DistType = Literal["kde", "normal", "skewnorm", "zero_inflated_gamma"]


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------

def exponential_weights(
    years: np.ndarray,
    half_life: float = 15.0,
    reference_year: int | None = None,
) -> np.ndarray:
    """
    Recency weights using exponential decay.

        w(t) = exp(-ln2 / half_life × (T - t))

    At t = T (most recent year):        w = 1.0
    At t = T - half_life:               w = 0.5
    At t = T - 2 × half_life:           w = 0.25

    Weights are normalised to sum to 1.

    Parameters
    ----------
    years       : array of calendar years
    half_life   : years for weight to halve (default 15)
    reference_year : treated as the "present" for decay; defaults to max(years)
    """
    years = np.asarray(years, dtype=float)
    T = float(reference_year or years.max())
    decay_rate = np.log(2.0) / half_life
    w = np.exp(-decay_rate * (T - years))
    return w / w.sum()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ELResult:
    el: float           # expected payout as fraction of notional
    p_fires: float      # P(trigger crosses threshold in a given year)
    cond_el: float      # E[payout | trigger fires]
    el_var: float       # Var[payout] — for risk margin calculation
    half_life: float    # decay half-life used
    dist_type: str      # which distribution was fitted
    n_obs: int          # number of observations used
    n_weighted_obs: float  # effective sample size = 1 / Σw²

    def premium_estimate(self, risk_load: float = 0.35, capital_charge: float = 0.015) -> float:
        """
        Indicative premium = EL × (1 + risk_load) + capital_charge.
        Industry standard for parametric weather contracts:
          risk_load: 30-50% multiplier on EL
          capital_charge: 1-2% of notional
        """
        return self.el * (1 + risk_load) + capital_charge

    def __repr__(self) -> str:
        return (
            f"ELResult(el={self.el*100:.2f}%, p_fires={self.p_fires*100:.1f}%, "
            f"cond_el={self.cond_el*100:.1f}%, "
            f"el_var={self.el_var*1e4:.1f}bps², "
            f"half_life={self.half_life}yr, dist={self.dist_type}, "
            f"n={self.n_obs} obs / {self.n_weighted_obs:.1f} eff)"
        )


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------

def _effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size: 1 / Σw²  (with normalised weights)."""
    w = weights / weights.sum()
    return 1.0 / (w ** 2).sum()


def fit_weighted_kde(
    values: np.ndarray,
    weights: np.ndarray,
    bw_method: str | float = "scott",
) -> stats.gaussian_kde:
    """
    Fit a weighted kernel density estimator.

    scipy.stats.gaussian_kde accepts a `weights` parameter (since scipy 1.2).
    The bandwidth is set by Scott's rule on the weighted effective sample size.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return stats.gaussian_kde(values, bw_method=bw_method, weights=w)


def fit_weighted_normal(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """
    Weighted maximum-likelihood Normal: returns (mu, sigma).
    Uses weighted mean and corrected weighted variance (reliability weights).
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mu = np.dot(w, values)
    # Bessel correction for reliability weights: 1 / (1 - Σw²)
    v2 = np.dot(w, (values - mu) ** 2)
    sigma = np.sqrt(v2 / (1.0 - (w ** 2).sum()))
    return float(mu), float(sigma)


def fit_weighted_skewnorm(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """
    Fit a weighted skewed Normal distribution.
    Uses weighted MLE via scipy.optimize.

    Returns (a, loc, scale) as in scipy.stats.skewnorm.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    def neg_wll(params):
        a, loc, scale = params
        if scale <= 0:
            return np.inf
        ll = stats.skewnorm.logpdf(values, a, loc, scale)
        return -np.dot(w, ll)

    mu0, s0 = fit_weighted_normal(values, w)
    x0 = [0.0, mu0, s0]
    bounds = [(-10, 10), (None, None), (1e-6, None)]
    res = opt.minimize(neg_wll, x0, method="L-BFGS-B", bounds=bounds)
    a, loc, scale = res.x
    return float(a), float(loc), float(scale)


def fit_zero_inflated_gamma(
    values: np.ndarray,
    weights: np.ndarray,
    zero_threshold: float = 1e-6,
) -> tuple[float, float, float]:
    """
    Fit a Zero-Inflated Gamma to non-negative metrics (degree-hours, convective precip).

    Model:
        P(X = 0) = p0
        X | X > 0  ~  Gamma(alpha, rate=1/scale)

    Returns (p0, alpha, scale) where scale = 1/rate.

    p0 is estimated as the weighted fraction of zero (or near-zero) observations.
    Gamma parameters are estimated from positive observations via weighted MOM.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    zero_mask = values <= zero_threshold
    p0 = float(w[zero_mask].sum())

    pos_vals = values[~zero_mask]
    pos_w = w[~zero_mask]
    if pos_w.sum() == 0:
        logger.warning("No positive values for Gamma fit; returning p0=1")
        return 1.0, 1.0, 1.0

    pos_w = pos_w / pos_w.sum()

    # Method of weighted moments
    mu = np.dot(pos_w, pos_vals)
    var = np.dot(pos_w, (pos_vals - mu) ** 2) / (1.0 - (pos_w ** 2).sum())
    var = max(var, 1e-9)

    alpha = mu ** 2 / var
    scale = var / mu  # scale = 1/rate
    return float(p0), float(alpha), float(scale)


# ---------------------------------------------------------------------------
# Expected payout integration
# ---------------------------------------------------------------------------

def _integrate_payout(
    pdf_fn: Callable[[np.ndarray], np.ndarray],
    payout_fn: Callable[[float], float],
    lower: float = -1.5,
    upper: float = 1.0,
    n_points: int = 2000,
) -> tuple[float, float]:
    """
    Numerically integrate E[payout] and E[payout²] using a grid.

    Returns (E[payout], E[payout²]).
    Uses a fine grid rather than scipy.integrate.quad to handle the
    piecewise-linear payout function efficiently.
    """
    x = np.linspace(lower, upper, n_points)
    dx = x[1] - x[0]
    p = np.array([payout_fn(xi) for xi in x])
    dens = pdf_fn(x)
    dens = np.maximum(dens, 0.0)  # KDE can produce tiny negatives at boundary

    el = float(np.sum(p * dens) * dx)
    el2 = float(np.sum(p ** 2 * dens) * dx)
    return el, el2


def expected_payout_kde(
    kde: stats.gaussian_kde,
    payout_fn: Callable[[float], float],
    threshold: float,
    lower: float = -1.5,
    upper: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Compute EL, p_fires, cond_EL, el_var from a KDE.

    threshold: the metric value at which payout first becomes > 0
               (used to compute p_fires).
    """
    el, el2 = _integrate_payout(kde.evaluate, payout_fn, lower, upper)
    el_var = max(el2 - el ** 2, 0.0)
    p_fires = float(kde.integrate_box_1d(-np.inf, threshold))
    cond_el = el / p_fires if p_fires > 1e-10 else 0.0
    return el, p_fires, cond_el, el_var


def expected_payout_normal(
    mu: float,
    sigma: float,
    payout_fn: Callable[[float], float],
    threshold: float,
    lower: float = -1.5,
    upper: float = 1.0,
) -> tuple[float, float, float, float]:
    dist = stats.norm(mu, sigma)
    el, el2 = _integrate_payout(dist.pdf, payout_fn, lower, upper)
    el_var = max(el2 - el ** 2, 0.0)
    p_fires = float(dist.cdf(threshold))
    cond_el = el / p_fires if p_fires > 1e-10 else 0.0
    return el, p_fires, cond_el, el_var


def expected_payout_zero_inflated_gamma(
    p0: float,
    alpha: float,
    scale: float,
    payout_fn: Callable[[float], float],
    threshold: float = 0.0,
    upper: float = 500.0,
) -> tuple[float, float, float, float]:
    """
    E[payout] for a zero-inflated Gamma metric (frost DH, hail CP).

    The zero mass contributes 0 to payout (below any threshold).
    The Gamma component is integrated from threshold to upper.
    """
    dist = stats.gamma(a=alpha, scale=scale)
    pos_pdf = lambda x: (1.0 - p0) * dist.pdf(x)
    el, el2 = _integrate_payout(pos_pdf, payout_fn, threshold, upper)
    el_var = max(el2 - el ** 2, 0.0)
    p_fires = float((1.0 - p0) * (1.0 - dist.cdf(threshold)))
    cond_el = el / p_fires if p_fires > 1e-10 else 0.0
    return el, p_fires, cond_el, el_var


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def fit_and_integrate(
    values: np.ndarray,
    years: np.ndarray,
    payout_fn: Callable[[float], float],
    trigger_threshold: float,
    dist_type: DistType = "kde",
    half_life: float = 15.0,
    reference_year: int | None = None,
    integration_lower: float = -1.5,
    integration_upper: float = 1.0,
    threshold_direction: str = "below",
) -> ELResult:
    """
    Fit a weighted distribution and integrate the payout function over it.

    Parameters
    ----------
    values              : 1-D array of historical trigger metric values
    years               : 1-D array of corresponding calendar years
    payout_fn           : metric_value → payout fraction (0–1)
    trigger_threshold   : metric value at which payout first exceeds 0
                          (used to compute p_fires)
    dist_type           : 'kde' | 'normal' | 'skewnorm' | 'zero_inflated_gamma'
    half_life           : recency decay half-life in years (default 15)
    reference_year      : year treated as "present" (default max(years))
    integration_lower   : lower limit for numerical integration
    integration_upper   : upper limit for numerical integration
    threshold_direction : 'below' (trigger fires when metric < threshold, e.g. frost, lira)
                          'above' (trigger fires when metric > threshold, e.g. pollination, hail)

    Returns
    -------
    ELResult
    """
    values = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=float)

    assert len(values) == len(years), "values and years must have same length"
    mask = ~np.isnan(values)
    values, years = values[mask], years[mask]

    weights = exponential_weights(years, half_life=half_life, reference_year=reference_year)
    eff_n = _effective_n(weights)

    if dist_type == "kde":
        kde = fit_weighted_kde(values, weights)
        el, p_fires, cond_el, el_var = expected_payout_kde(
            kde, payout_fn, trigger_threshold,
            lower=integration_lower, upper=integration_upper,
        )
        if threshold_direction == "above":
            p_fires = 1.0 - p_fires
    elif dist_type == "normal":
        mu, sigma = fit_weighted_normal(values, weights)
        el, p_fires, cond_el, el_var = expected_payout_normal(
            mu, sigma, payout_fn, trigger_threshold,
            lower=integration_lower, upper=integration_upper,
        )
        if threshold_direction == "above":
            p_fires = 1.0 - p_fires
    elif dist_type == "skewnorm":
        a, loc, scale = fit_weighted_skewnorm(values, weights)
        dist = stats.skewnorm(a, loc, scale)
        el, el2 = _integrate_payout(dist.pdf, payout_fn, integration_lower, integration_upper)
        el_var = max(el2 - el ** 2, 0.0)
        p_fires = float(dist.cdf(trigger_threshold))
        if threshold_direction == "above":
            p_fires = 1.0 - p_fires
        cond_el = el / p_fires if p_fires > 1e-10 else 0.0
    elif dist_type == "zero_inflated_gamma":
        p0, alpha, scale = fit_zero_inflated_gamma(values, weights)
        el, p_fires, cond_el, el_var = expected_payout_zero_inflated_gamma(
            p0, alpha, scale, payout_fn,
            threshold=trigger_threshold,
            upper=integration_upper,
        )
        if threshold_direction == "above":
            p_fires = 1.0 - p_fires
    else:
        raise ValueError(f"Unknown dist_type: {dist_type!r}")

    # Recompute cond_el after direction correction
    if threshold_direction == "above":
        cond_el = el / p_fires if p_fires > 1e-10 else 0.0

    return ELResult(
        el=el,
        p_fires=p_fires,
        cond_el=cond_el,
        el_var=el_var,
        half_life=half_life,
        dist_type=dist_type,
        n_obs=len(values),
        n_weighted_obs=eff_n,
    )


def sensitivity_table(
    values: np.ndarray,
    years: np.ndarray,
    payout_fn: Callable[[float], float],
    trigger_threshold: float,
    dist_type: DistType = "kde",
    half_lives: list[float] | None = None,
    reference_year: int | None = None,
    integration_lower: float = -1.5,
    integration_upper: float = 1.0,
) -> list[ELResult]:
    """
    Compute EL under a range of half-life assumptions.
    Useful for communicating sensitivity to the recency-weighting choice.
    """
    if half_lives is None:
        half_lives = [5.0, 10.0, 15.0, 20.0, 30.0, float("inf")]

    results = []
    for hl in half_lives:
        if hl == float("inf"):
            # Equal weights = no recency decay
            w = np.ones(len(years)) / len(years)
        else:
            w = None  # fit_and_integrate will compute

        if hl == float("inf"):
            # Manually fit with equal weights
            kde = fit_weighted_kde(values, np.ones(len(values)))
            el, p_fires, cond_el, el_var = expected_payout_kde(
                kde, payout_fn, trigger_threshold,
                lower=integration_lower, upper=integration_upper,
            )
            r = ELResult(
                el=el, p_fires=p_fires, cond_el=cond_el, el_var=el_var,
                half_life=float("inf"), dist_type=dist_type,
                n_obs=len(values), n_weighted_obs=len(values),
            )
        else:
            r = fit_and_integrate(
                values, years, payout_fn, trigger_threshold,
                dist_type=dist_type, half_life=hl,
                reference_year=reference_year,
                integration_lower=integration_lower,
                integration_upper=integration_upper,
            )
        results.append(r)
    return results
