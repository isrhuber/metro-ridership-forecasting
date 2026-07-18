"""Final 90-day forecast with empirical prediction intervals.

Intervals come from the distribution of in-sample log-residuals rather
than a normality assumption (daily transit data has fat tails: events,
weather, closures).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from features import fit_harmonic, predict_harmonic

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 90

df = pd.read_csv(ROOT / "data" / "rail_clean.csv", parse_dates=["date"])
df = df[df.date >= "2022-01-01"].reset_index(drop=True)

beta, names = fit_harmonic(df)
fitted = predict_harmonic(beta, df.date)
resid = np.log(df.entries.to_numpy(float)) - np.log(fitted)
lo, hi = np.quantile(resid[-730:], [0.05, 0.95])

future = pd.date_range(df.date.max() + pd.Timedelta(days=1), periods=HORIZON)
point = predict_harmonic(beta, future)
fc = pd.DataFrame({"date": future, "forecast": point.round(0),
                   "lo90": (point * np.exp(lo)).round(0),
                   "hi90": (point * np.exp(hi)).round(0)})
fc.to_csv(ROOT / "output" / "forecast_90d.csv", index=False)

# readable coefficients (log space -> % effects)
coef = pd.Series(beta, index=names)
effects = (100 * (np.exp(coef) - 1)).round(1)
effects.to_csv(ROOT / "output" / "coefficients_pct.csv")
print("Selected effects (% vs baseline):")
print(effects[["trend_per_year", "dow_5", "dow_6", "hol_july4",
               "hol_christmas", "christmas_week", "black_friday"]].to_string())

fig, ax = plt.subplots(figsize=(12, 5))
hist = df[df.date >= df.date.max() - pd.Timedelta(days=365)]
ax.plot(hist.date, hist.entries, lw=0.6, color="#999", label="observed")
ax.plot(hist.date, predict_harmonic(beta, hist.date), lw=1.2, color="#2266aa", label="model fit")
ax.plot(fc.date, fc.forecast, lw=1.4, color="#cc5500", label="90-day forecast")
ax.fill_between(fc.date, fc.lo90, fc.hi90, alpha=0.25, color="#cc5500", label="90% interval")
ax.legend(); ax.set_title("Metrorail daily ridership — 90-day forecast (harmonic OLS)")
plt.tight_layout(); plt.savefig(ROOT / "output" / "02_forecast.png", dpi=110)
print("\nsaved output/02_forecast.png and forecast_90d.csv")
