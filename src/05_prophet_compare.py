"""OPTIONAL: benchmark the harmonic OLS against Prophet.

Runs locally (pip install -r requirements.txt), not in the build sandbox.
Uses the same rolling-origin CV as 03_models.py so numbers are comparable.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from prophet import Prophet

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 28

df = pd.read_csv(ROOT / "data" / "rail_clean.csv", parse_dates=["date"])
df = df[df.date >= "2022-01-01"].reset_index(drop=True)

cutoffs = pd.date_range(end=df.date.max() - pd.Timedelta(days=HORIZON),
                        periods=18, freq="MS")
maes, mapes = [], []
for cutoff in cutoffs:
    train = df[df.date < cutoff]
    test = df[(df.date >= cutoff) & (df.date < cutoff + pd.Timedelta(days=HORIZON))]
    if len(train) < 400 or len(test) < HORIZON:
        continue
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False)
    m.add_country_holidays(country_name="US")
    tr = train.rename(columns={"date": "ds", "entries": "y"})
    tr["y"] = np.log(tr["y"])
    m.fit(tr)
    fc = m.predict(pd.DataFrame({"ds": test.date}))
    pred, y = np.exp(fc.yhat.to_numpy()), test.entries.to_numpy(float)
    maes.append(np.abs(pred - y).mean())
    mapes.append(100 * np.abs((pred - y) / y).mean())

print(f"Prophet (log space):  MAE {np.mean(maes):.0f}  MAPE {np.mean(mapes):.1f}")
print("Compare with output/cv_results.csv from 03_models.py")
