"""Reproducible hourly PJM West load analysis and 30-day forecast.

Run with the bundled project Python:
    python forecast_energy.py

Outputs are written beside this script in outputs/.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".model_deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "PJMW_MW_Hourly.xlsx"
OUT = ROOT / "outputs"
HOLDOUT_DAYS = 365
FORECAST_DAYS = 30


def features(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Calendar, smooth annual seasonality, and trend features."""
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    elapsed = (index - origin).total_seconds().to_numpy() / (365.2425 * 86400)
    cols = [np.ones(len(index)), elapsed]
    for period, harmonics, values in ((24, 3, hour), (7, 2, dow), (365.2425, 3, doy - 1)):
        for k in range(1, harmonics + 1):
            angle = 2 * np.pi * k * values / period
            cols.extend((np.sin(angle), np.cos(angle)))
    cols.extend(((dow >= 5).astype(float), ((hour >= 16) & (hour <= 20)).astype(float)))
    return np.column_stack(cols)


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    err = predicted - actual
    return {
        "n": int(len(actual)),
        "MAE_MW": float(np.mean(np.abs(err))),
        "RMSE_MW": float(np.sqrt(np.mean(err**2))),
        "MAPE_pct": float(np.mean(np.abs(err) / np.maximum(np.abs(actual), 1)) * 100),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = pd.read_excel(SOURCE, usecols=["Datetime", "PJMW_MW"], parse_dates=["Datetime"])
    raw = raw.dropna(subset=["Datetime", "PJMW_MW"]).sort_values("Datetime")
    duplicate_rows = int(raw.duplicated("Datetime", keep=False).sum())
    # Repeated timestamps are averaged; absent hours remain absent for evaluation.
    hourly = raw.groupby("Datetime")["PJMW_MW"].mean().sort_index().asfreq("h")
    observed = hourly.dropna()
    origin = observed.index.min()
    cutoff = observed.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train = observed.loc[observed.index < cutoff]
    test = observed.loc[observed.index >= cutoff]

    x_train, x_test = features(train.index, origin), features(test.index, origin)
    y_train = train.to_numpy()
    model_specs = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=160, min_samples_leaf=2,
            max_features=0.8, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=350, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.85, objective="reg:squarederror",
            n_jobs=-1, random_state=42, verbosity=0),
    }
    fitted = {}
    predictions = {}
    for name, estimator in model_specs.items():
        estimator.fit(x_train, y_train)
        fitted[name] = estimator
        predictions[name] = estimator.predict(x_test)

    # Keep SARIMA estimation bounded to the most recent two training years.
    # This preserves hourly seasonality while making fitting practical.
    sarima_train = hourly.loc[
        (hourly.index < cutoff)
        & (hourly.index >= train.index.max() - pd.Timedelta(days=730))
    ].interpolate(method="time")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sarima = SARIMAX(sarima_train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24),
            trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(
                disp=False, maxiter=60)
    fitted["SARIMA"] = sarima
    sarima_holdout_index = pd.date_range(cutoff, test.index.max(), freq="h")
    sarima_holdout = sarima.get_forecast(steps=len(sarima_holdout_index)).predicted_mean
    predictions["SARIMA"] = pd.Series(np.asarray(sarima_holdout), index=sarima_holdout_index).reindex(test.index).to_numpy()
    prior_week = hourly.reindex(test.index - pd.Timedelta(days=7)).to_numpy()
    valid_baseline = np.isfinite(prior_week)

    holdout = {"cutoff": cutoff.isoformat(), "observed_test_hours": int(len(test)),
        "models": {name: scores(test.to_numpy(), pred) for name, pred in predictions.items()},
        "weekly_seasonal_naive": scores(test.to_numpy()[valid_baseline], prior_week[valid_baseline])}
    comparison = [{"Model": name, **metric} for name, metric in holdout["models"].items()]
    comparison.append({"Model": "Weekly seasonal naive", **holdout["weekly_seasonal_naive"]})
    pd.DataFrame(comparison).sort_values("MAE_MW").to_csv(OUT / "model_comparison.csv", index=False)
    future_index = pd.date_range(hourly.index.max() + pd.Timedelta(hours=1), periods=FORECAST_DAYS * 24, freq="h")
    future_features = features(future_index, origin)
    forecast_frame = pd.DataFrame({"Datetime": future_index})
    for name in model_specs:
        forecast_frame[name.replace(" ", "_") + "_MW"] = fitted[name].predict(future_features)
    forecast_frame["SARIMA_MW"] = np.asarray(fitted["SARIMA"].get_forecast(
        steps=len(future_index)).predicted_mean)
    forecast_frame.to_csv(OUT / "forecast_30_days.csv", index=False)

    profile = observed.rename("PJMW_MW").to_frame()
    profile["hour"] = profile.index.hour
    profile["day_of_week"] = profile.index.day_name()
    profile["month"] = profile.index.month
    (profile.groupby("hour")["PJMW_MW"].mean().rename("mean_MW").to_csv(OUT / "hourly_profile.csv"))
    (profile.groupby("day_of_week")["PJMW_MW"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).rename("mean_MW").to_csv(OUT / "weekday_profile.csv"))
    (profile.groupby("month")["PJMW_MW"].mean().rename("mean_MW").to_csv(OUT / "monthly_profile.csv"))

    summary = {
        "source_file": SOURCE.name,
        "series": "PJM West hourly load (MW)",
        "source_rows": int(len(raw)),
        "duplicate_timestamp_rows_averaged": duplicate_rows,
        "unique_hourly_timestamps": int(len(hourly)),
        "missing_hourly_timestamps": int(hourly.isna().sum()),
        "observed_start": observed.index.min().isoformat(),
        "observed_end": observed.index.max().isoformat(),
        "observed_hours": int(len(observed)),
        "train_observed_hours": int(len(train)),
        "test_start": cutoff.isoformat(),
        "test_end": test.index.max().isoformat(),
        "load_min_MW": float(observed.min()),
        "load_mean_MW": float(observed.mean()),
        "load_max_MW": float(observed.max()),
        "holdout_metrics": holdout,
        "model_description": {
            "Linear Regression": "Calendar and Fourier features with linear trend.",
            "Random Forest": "160-tree ensemble fit to the same calendar features.",
            "XGBoost": "Gradient-boosted trees fit to the same calendar features.",
            "SARIMA": "SARIMA(1,0,1)(1,0,1,24) fitted to the latest two training years.",
        },
        "limitations": [
            "The data ends on 2018-08-03, so the 30-day output is a historical demonstration, not a current operational forecast.",
            "The supplied series has no holiday calendar, weather, or market/system covariates.",
            "The regression does not model abrupt events or changing seasonal load relationships.",
        ],
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
