"""EDA for Metrorail daily ridership: trend, seasonality, holidays."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "rail_clean.csv", parse_dates=["date"])
df["dow"] = df.date.dt.day_name()
df["year"] = df.date.dt.year
df["month"] = df.date.dt.month

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle("WMATA Metrorail Daily Ridership — EDA", fontsize=14, y=0.98)

# 1. full history with 28-day rolling mean
ax = axes[0, 0]
ax.plot(df.date, df.entries, lw=0.3, alpha=0.5, color="#888")
ax.plot(df.date, df.entries.rolling(28, center=True).mean(), lw=1.8, color="#2266aa")
ax.axvspan(pd.Timestamp("2020-03-13"), pd.Timestamp("2021-06-01"), alpha=0.12, color="red")
ax.set_title("Daily entries, 2019–2026 (COVID era shaded)")

# 2. weekly pattern, recent two years
ax = axes[0, 1]
recent = df[df.date >= df.date.max() - pd.Timedelta(days=730)]
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
data = [recent.loc[recent.dow == d, "entries"] for d in order]
ax.boxplot(data, labels=[d[:3] for d in order])
ax.set_title("Day-of-week pattern (last 2 yrs)"); ax.set_xlabel("")

# 3. annual seasonality: monthly means by year (post-recovery)
ax = axes[1, 0]
for yr in range(2022, 2027):
    m = df[df.year == yr].groupby("month").entries.mean()
    ax.plot(m.index, m.values, marker="o", ms=3, label=str(yr))
ax.legend(fontsize=8); ax.set_title("Monthly mean by year (post-recovery)")
ax.set_xticks(range(1, 13))

# 4. recovery ratio vs 2019 baseline
ax = axes[1, 1]
base = df[df.year == 2019].groupby(df[df.year == 2019].date.dt.dayofyear).entries.mean()
yearly = df.groupby("year").entries.mean()
ax.bar(yearly.index.astype(str), 100 * yearly / yearly[2019], color="#2266aa")
ax.axhline(100, ls="--", color="#888", lw=1)
ax.set_title("Avg daily ridership as % of 2019")

plt.suptitle("WMATA Metrorail Daily Ridership — EDA")
plt.tight_layout()
plt.savefig(OUT / "01_eda_overview.png", dpi=110)
print("saved", OUT / "01_eda_overview.png")

# quick stats for the write-up
print("\n2019 avg: %d" % df[df.year==2019].entries.mean())
print("2020 min day: %d" % df[df.year==2020].entries.min())
print("2025 avg: %d | 2026 avg: %d" % (df[df.year==2025].entries.mean(), df[df.year==2026].entries.mean()))
print("recovery vs 2019: %.0f%%" % (100*df[df.year==2026].entries.mean()/df[df.year==2019].entries.mean()))
