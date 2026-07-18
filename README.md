# Forecasting DC Metro Ridership

Time-series forecasting of WMATA Metrorail daily ridership in Python —
seasonality decomposition, holiday effects, and model comparison with
rolling-origin cross-validation, framed as a transit budget-planning problem.

**Author:** Ian S. Huber · [Portfolio](https://isrhuber.github.io/portfolio/)

## Data

Daily systemwide ridership (2019–present) from WMATA's
[Ridership Data Portal](https://www.wmata.com/initiatives/ridership-portal/).
To reproduce: open the Metrorail daily dashboard, download button → Data →
"download all rows as a text file", save as `data/metrorail_daily.csv`.
Raw data is not committed.

## Structure

    src/01_clean.py      load portal export, filter to rail, fix anomalies
    src/02_explore.py    EDA: trend, weekly/annual seasonality, COVID, holidays
    src/03_models.py     baseline / SARIMAX / Prophet + rolling-origin CV
    src/04_forecast.py   final model, 90-day forecast with intervals
    output/              figures and forecast tables

## Setup

    pip install -r requirements.txt
    python src/01_clean.py && python src/02_explore.py
    python src/03_models.py && python src/04_forecast.py
