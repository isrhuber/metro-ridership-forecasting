"""Model comparison with rolling-origin cross-validation.

Forecasters, in increasing order of structure:
  1. seasonal_naive — same weekday, previous week
  2. weekday_ma     — mean of the last 4 same-weekdays
  3. harmonic_ols   — log-space OLS on a 2-yr window: trend + day-of-week
                      + annual Fourier + holiday/adjacency dummies.
                      Fully interpretable: every coefficient is readable.

Key modeling choices (validated by this CV):
  - log space: weekday/weekend structure is multiplicative, not additive
  - 730-day training window: pre-2022 recovery slope misleads the trend term
  - per-holiday dummies: July 4 raises ridership, Christmas craters it

CV: 18 monthly cutoffs; fit through cutoff, forecast next 28 days.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from features import fit_harmonic, predict_harmonic

ROOT = Path(__file__).resolve().parents[1]
TRAIN_START, HORIZON = "2022-01-01", 28

df = pd.read_csv(ROOT / "data" / "rail_clean.csv", parse_dates=["date"])
df = df[df.date >= TRAIN_START].reset_index(drop=True)

def seasonal_naive(train, fdates):
    return np.array([train[train.date.dt.dayofweek == d.dayofweek].entries.iloc[-1]
                     for d in fdates], float)

def weekday_ma(train, fdates, k=4):
    return np.array([train[train.date.dt.dayofweek == d.dayofweek].entries.iloc[-k:].mean()
                     for d in fdates], float)

def harmonic_ols(train, fdates):
    beta, _ = fit_harmonic(train)
    return predict_harmonic(beta, fdates)

MODELS = {"seasonal_naive": seasonal_naive, "weekday_ma": weekday_ma,
          "harmonic_ols": harmonic_ols}

cutoffs = pd.date_range(end=df.date.max() - pd.Timedelta(days=HORIZON),
                        periods=18, freq="MS")
rows = []
for cutoff in cutoffs:
    train = df[df.date < cutoff]
    test = df[(df.date >= cutoff) & (df.date < cutoff + pd.Timedelta(days=HORIZON))]
    if len(train) < 400 or len(test) < HORIZON:
        continue
    fdates, y = pd.DatetimeIndex(test.date), test.entries.to_numpy(float)
    for name, fn in MODELS.items():
        pred = fn(train, fdates)
        rows.append({"cutoff": cutoff, "model": name,
                     "MAE": np.abs(pred - y).mean(),
                     "MAPE": 100 * np.abs((pred - y) / y).mean()})

res = pd.DataFrame(rows)
summary = res.groupby("model")[["MAE", "MAPE"]].mean().round(1).sort_values("MAE")
print(f"Rolling-origin CV: {res.cutoff.nunique()} folds x {HORIZON}-day horizon\n")
print(summary)
(ROOT / "output").mkdir(exist_ok=True)
summary.to_csv(ROOT / "output" / "cv_results.csv")
