"""Clean the WMATA Ridership Data Portal export.

The portal export contains both Metro Rail and Metro Bus daily rows.
This script filters to rail, enforces a complete daily calendar, and
flags/repairs impossible values (e.g. negative entries).
Output: data/rail_clean.csv (date, entries)
"""
from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "metrorail_daily.csv"
OUT = RAW.parent / "rail_clean.csv"

def main() -> None:
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df.columns = ["date", "mode", "entries"]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y %H:%M")

    rail = (
        df.loc[df["mode"] == "Metro Rail", ["date", "entries"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # enforce complete daily calendar
    full = pd.DataFrame(
        {"date": pd.date_range(rail.date.min(), rail.date.max(), freq="D")}
    )
    rail = full.merge(rail, on="date", how="left")

    # impossible values: negative or zero riders on a system that never
    # fully closes -> treat as recording errors, interpolate
    bad = (rail.entries <= 0) | rail.entries.isna()
    print(f"rows: {len(rail)}, flagged bad/missing: {bad.sum()}")
    if bad.any():
        print(rail.loc[bad, "date"].dt.date.tolist())
        rail.loc[bad, "entries"] = pd.NA
        rail["entries"] = (
            rail["entries"].astype("Float64").interpolate(limit_direction="both")
        )

    rail["entries"] = rail["entries"].round().astype("int64")
    rail.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {rail.date.min().date()} -> {rail.date.max().date()}")

if __name__ == "__main__":
    main()
