"""Shared feature engineering: calendar features + US federal holidays."""
import numpy as np
import pandas as pd

def federal_holidays(years):
    """Major US federal holidays, computed with pandas offsets."""
    rows = []
    for y in years:
        rows += [
            ("newyear",      pd.Timestamp(y, 1, 1)),
            ("mlk",          pd.Timestamp(y, 1, 1) + pd.offsets.WeekOfMonth(week=2, weekday=0)),
            ("presidents",   pd.Timestamp(y, 2, 1) + pd.offsets.WeekOfMonth(week=2, weekday=0)),
            ("memorial",     pd.Timestamp(y, 5, 31) - pd.offsets.Week(weekday=0) if pd.Timestamp(y,5,31).weekday()!=0 else pd.Timestamp(y,5,31)),
            ("july4",        pd.Timestamp(y, 7, 4)),
            ("labor",        pd.Timestamp(y, 9, 1) + pd.offsets.WeekOfMonth(week=0, weekday=0) if pd.Timestamp(y,9,1).weekday()!=0 else pd.Timestamp(y,9,1)),
            ("thanksgiving", pd.Timestamp(y, 11, 1) + pd.offsets.WeekOfMonth(week=3, weekday=3)),
            ("christmas",    pd.Timestamp(y, 12, 25)),
        ]
    return pd.DataFrame(rows, columns=["holiday", "date"])

def design_matrix(dates: pd.Series, k_annual: int = 4):
    """Trend + day-of-week + annual Fourier + holiday & adjacency dummies.

    Holiday effects get individual coefficients because they differ wildly:
    July 4 fireworks *raise* ridership; Thanksgiving/Christmas crater it.
    Adjacency terms capture travel behavior around, not on, the holiday.
    """
    d = pd.DataFrame({"date": pd.to_datetime(dates)})
    t = (d.date - pd.Timestamp("2022-01-01")).dt.days.to_numpy(float)
    X = [np.ones_like(t), t / 365.25]
    names = ["intercept", "trend_per_year"]
    for dow in range(1, 7):                      # Monday = reference
        X.append((d.date.dt.dayofweek == dow).to_numpy(float))
        names.append(f"dow_{dow}")
    doy = d.date.dt.dayofyear.to_numpy(float)
    for k in range(1, k_annual + 1):
        X.append(np.sin(2 * np.pi * k * doy / 365.25)); names.append(f"sin{k}")
        X.append(np.cos(2 * np.pi * k * doy / 365.25)); names.append(f"cos{k}")
    hol = federal_holidays(range(d.date.dt.year.min() - 1, d.date.dt.year.max() + 2))
    for name, grp in hol.groupby("holiday"):
        X.append(d.date.isin(grp.date).to_numpy(float)); names.append(f"hol_{name}")
    tg = hol[hol.holiday == "thanksgiving"].date
    X.append(d.date.isin(tg + pd.Timedelta(days=1)).to_numpy(float)); names.append("black_friday")
    xmas = hol[hol.holiday == "christmas"].date
    week = pd.DatetimeIndex([dd + pd.Timedelta(days=o) for dd in xmas for o in range(-2, 7) if o != 0])
    X.append(d.date.isin(week).to_numpy(float)); names.append("christmas_week")
    j4 = hol[hol.holiday == "july4"].date
    X.append(d.date.isin(list(j4 - pd.Timedelta(days=1)) + list(j4 + pd.Timedelta(days=1))).to_numpy(float))
    names.append("july4_adjacent")
    return np.column_stack(X), names

def fit_harmonic(train: pd.DataFrame, window_days: int = 730):
    """Fit log-space harmonic OLS on the trailing window. Returns beta."""
    train = train[train.date >= train.date.max() - pd.Timedelta(days=window_days)]
    X, names = design_matrix(train.date)
    beta, *_ = np.linalg.lstsq(X, np.log(train.entries.to_numpy(float)), rcond=None)
    return beta, names

def predict_harmonic(beta, dates):
    X, _ = design_matrix(pd.Series(dates))
    return np.exp(X @ beta)
