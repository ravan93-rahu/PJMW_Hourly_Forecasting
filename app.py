from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.tsa.statespace.sarimax import SARIMAX


HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "model_bundle_compact.joblib"


@st.cache_resource
def load_models():
    if not BUNDLE.exists():
        raise FileNotFoundError(
            "model_bundle_compact.joblib not found. Run the final deployment cell "
            "in PJMW_Hourly_Forecasting.ipynb first."
        )
    bundle = joblib.load(BUNDLE)
    sarima_data = bundle["sarima"]
    sarima_index = pd.date_range(sarima_data["start"], periods=len(sarima_data["endog"]), freq="h")
    sarima_endog = pd.Series(sarima_data["endog"], index=sarima_index, name="PJMW_MW")
    sarima_model = SARIMAX(
        sarima_endog,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).filter(sarima_data["params"])
    bundle["models"]["SARIMA"] = sarima_model
    return bundle


try:
    bundle = load_models()
except Exception as error:
    st.error(f"Could not load compact model bundle: {error}")
    st.stop()

models = bundle["models"]
origin = bundle["origin"]
last_timestamp = pd.Timestamp(bundle["last_timestamp"])


def make_features(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    index = pd.DatetimeIndex(index)
    hour, dow, doy = index.hour.to_numpy(), index.dayofweek.to_numpy(), index.dayofyear.to_numpy()
    elapsed = (index - origin).total_seconds().to_numpy() / (365.2425 * 86400)
    columns = [np.ones(len(index)), elapsed]
    for period, harmonics, values in ((24, 3, hour), (7, 2, dow), (365.2425, 3, doy - 1)):
        for k in range(1, harmonics + 1):
            angle = 2 * np.pi * k * values / period
            columns.extend([np.sin(angle), np.cos(angle)])
    columns.extend([(dow >= 5).astype(float), ((hour >= 16) & (hour <= 20)).astype(float)])
    holidays = USFederalHolidayCalendar().holidays(
        start=index.min().normalize(), end=index.max().normalize()).normalize()
    columns.append(index.normalize().isin(holidays).astype(float))
    return np.column_stack(columns)


st.set_page_config(page_title="PJM West Load Forecast", layout="wide")
st.title("PJM West Hourly Load Forecast")
st.warning(
    f"Historical demonstration. Source data ends {last_timestamp:%Y-%m-%d %H:%M}; "
    "this is not a current operational forecast."
)
model_name = st.selectbox("Forecast model", ["Random Forest", "XGBoost", "Linear Regression", "SARIMA"])
days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)
index = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=days * 24, freq="h")
if model_name == "SARIMA":
    values = np.asarray(models[model_name].get_forecast(steps=len(index)).predicted_mean)
else:
    values = models[model_name].predict(make_features(index, origin))
result = pd.DataFrame({"Datetime": index, "Forecast_MW": values})
st.line_chart(result.set_index("Datetime"))
st.dataframe(result, use_container_width=True)
st.download_button("Download forecast CSV", result.to_csv(index=False), "pjm_forecast.csv", "text/csv")
