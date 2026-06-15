"""
utils/forecasting.py — Senior Population Forecasting
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Capture the real import error so it surfaces in Streamlit instead of
# being swallowed and replaced with a misleading "not installed" message.
_PROPHET_OK = False
_PROPHET_IMPORT_ERROR = None
try:
    from prophet import Prophet
    _PROPHET_OK = True
except Exception as e:
    _PROPHET_IMPORT_ERROR = e


FORECAST_YEARS    = 10
CONFIDENCE_WIDTH  = 0.90
CHANGEPOINT_PRIOR = 0.3
MIN_HISTORY_PTS   = 5


def _build_series(pop, level, pa_filter, sz_filter):
    s = pop[pop["AgeNum"] >= 65].copy()
    if level == "national":
        series = s.groupby("Time")["Pop"].sum().reset_index()
    elif level == "pa":
        series = (
            s[s["PA"].str.upper() == pa_filter.strip().upper()]
            .groupby("Time")["Pop"].sum().reset_index()
        )
    elif level == "subzone":
        mask = (
            (s["PA"].str.upper() == pa_filter.strip().upper()) &
            (s["SZ"].str.upper() == sz_filter.strip().upper())
        )
        series = s[mask].groupby("Time")["Pop"].sum().reset_index()
    else:
        raise ValueError(f"Unknown level: {level!r}")
    series.columns = ["year", "seniors"]
    return series[series["seniors"] > 0].sort_values("year").reset_index(drop=True)


def _fit_prophet(series, horizon):
    df = pd.DataFrame({
        "ds": pd.to_datetime(series["year"].astype(str) + "-01-01"),
        "y":  series["seniors"].astype(float),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
            changepoint_prior_scale=CHANGEPOINT_PRIOR,
            interval_width=CONFIDENCE_WIDTH,
            growth="linear",
        )
        m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="YS")
    fc = m.predict(future)
    last_hist = series["year"].max()
    out = fc[fc["ds"].dt.year > last_hist][["ds","yhat","yhat_lower","yhat_upper"]].copy()
    for col in ["yhat","yhat_lower","yhat_upper"]:
        out[col] = out[col].clip(lower=0).round(0).astype(int)
    out["year"] = out["ds"].dt.year
    return out[["year","yhat","yhat_lower","yhat_upper"]].reset_index(drop=True)


def _check_prophet():
    """Raise with the real error message if Prophet failed to import."""
    if not _PROPHET_OK:
        raise ImportError(
            f"Prophet import failed: {_PROPHET_IMPORT_ERROR}\n"
            "Try running in your venv:\n"
            "  pip install prophet\n"
            "If already installed, the issue is likely a missing C++ dependency on Windows.\n"
            "Fix: pip install pystan==2.19.1.1  OR  conda install -c conda-forge prophet"
        )


def forecast_senior_population(
    pop: pd.DataFrame,
    level: str = "national",
    pa_filter: Optional[str] = None,
    sz_filter: Optional[str] = None,
    horizon: int = FORECAST_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _check_prophet()
    _empty = pd.DataFrame(columns=["year","yhat","yhat_lower","yhat_upper"])
    hist = _build_series(pop, level, pa_filter, sz_filter)
    if len(hist) < MIN_HISTORY_PTS:
        return hist, _empty
    return hist, _fit_prophet(hist, horizon)


def forecast_top_pa_seniors(
    pop: pd.DataFrame,
    top_n: int = 10,
    horizon: int = FORECAST_YEARS,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    _check_prophet()
    top_pas = (
        pop[(pop["Time"] == 2025) & (pop["AgeNum"] >= 65)]
        .groupby("PA")["Pop"].sum()
        .nlargest(top_n).index.tolist()
    )
    results = {}
    for pa in top_pas:
        try:
            h, f = forecast_senior_population(pop, level="pa", pa_filter=pa, horizon=horizon)
            if not f.empty:
                results[pa] = (h, f)
        except Exception:
            pass
    return results


def forecast_aging_index(
    pop: pd.DataFrame,
    pa_filter: Optional[str] = None,
    horizon: int = FORECAST_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _empty = pd.DataFrame(columns=["year","ai_yhat","ai_lower","ai_upper"])
    level = "pa" if pa_filter else "national"
    hist_s, fc_s = forecast_senior_population(pop, level=level, pa_filter=pa_filter, horizon=horizon)
    if fc_s.empty:
        return pd.DataFrame(columns=["year","aging_index"]), _empty

    total = pop.copy()
    if pa_filter:
        total = total[total["PA"].str.upper() == pa_filter.strip().upper()]
    tot = (
        total.groupby("Time")["Pop"].sum().reset_index()
        .rename(columns={"Time":"year","Pop":"total"})
        .query("total > 0").sort_values("year")
    )
    from numpy.polynomial.polynomial import polyfit, polyval
    yrs = tot["year"].values
    coef = polyfit(yrs - yrs[0], tot["total"].values, deg=1)
    fc_years = fc_s["year"].values
    total_proj = np.array([max(1, polyval(y - yrs[0], coef)) for y in fc_years])

    hist_ai = hist_s.merge(tot, on="year")
    hist_ai["aging_index"] = (hist_ai["seniors"] / hist_ai["total"]).round(4)

    fc_ai = pd.DataFrame({
        "year":     fc_years,
        "ai_yhat":  (fc_s["yhat"].values  / total_proj).clip(0,1).round(4),
        "ai_lower": (fc_s["yhat_lower"].values / total_proj).clip(0,1).round(4),
        "ai_upper": (fc_s["yhat_upper"].values / total_proj).clip(0,1).round(4),
    })
    return hist_ai[["year","aging_index"]], fc_ai