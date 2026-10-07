from pathlib import Path
import os
import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "model_bundle.joblib"
FILE_ID = "148OQ0KopGjiABR1ifwZA4r1J2NsvB59U"

@st.cache_resource
def download_and_load_model():
    if not BUNDLE.exists():
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        with st.spinner("Model 2GB file Google Drive varun download hot ahe, krupaya thamba..."):
            gdown.download(url, str(BUNDLE), quiet=False, candidate_limit=5)
    
    return joblib.load(BUNDLE)

try:
    bundle = download_and_load_model()
except Exception as e:
    st.error(f"Model load hotana samasya ali: {e}")
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
