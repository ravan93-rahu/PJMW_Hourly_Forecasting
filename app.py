from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pandas.tseries.holiday import USFederalHolidayCalendar

# 1. Page Configuration (Wide Layout)
st.set_page_config(
    page_title="PJM West Random Forest Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "model_bundle_random_forest.joblib"

@st.cache_resource
def load_model_bundle():
    if not BUNDLE.exists():
        raise FileNotFoundError(
            "Random Forest bundle is missing. Run the deployment cell in "
            "PJMW_Hourly_Forecasting.ipynb first."
        )
    return joblib.load(BUNDLE)

try:
    bundle = load_model_bundle()
except Exception as error:
    st.error(f"Could not load Random Forest bundle: {error}")
    st.stop()

model = bundle["model"]
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

# 2. Sidebar Controls
st.sidebar.title("⚡ Control Panel")
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info("Model: **Random Forest**")
st.sidebar.write(f"**Source Data Ends:**\n{last_timestamp:%Y-%m-%d %H:%M}")

st.sidebar.markdown("---")
days = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)

st.sidebar.markdown("---")
st.sidebar.text("Data Science Portfolio App")

# 3. Main Dashboard Header
st.title("⚡ PJM West Hourly Load Forecast Dashboard")
st.warning(
    f"Historical demonstration. Source data ends {last_timestamp:%Y-%m-%d %H:%M}; "
    "this is not a current operational forecast."
)

# Generate Predictions
index = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=days * 24, freq="h")
values = model.predict(make_features(index, origin))
result = pd.DataFrame({"Datetime": index, "Forecast_MW": values})

# 4. Top Metrics Cards (KPIs)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📊 Average Forecast Load", value=f"{result['Forecast_MW'].mean():,.2f} MW")
with col2:
    st.metric(label="📈 Peak Load (Max)", value=f"{result['Forecast_MW'].max():,.2f} MW")
with col3:
    st.metric(label="📉 Minimum Load", value=f"{result['Forecast_MW'].min():,.2f} MW")

st.markdown("---")

# 5. Interactive Plotly Chart (Replacing simple line_chart)
st.subheader("📈 Hourly Power Consumption Forecast Trend")
fig = px.line(
    result, 
    x='Datetime', 
    y='Forecast_MW',
    labels={'Datetime': 'Date & Time', 'Forecast_MW': 'Forecasted Load (MW)'},
    title=f"Electricity Demand Forecast for Next {days} Days"
)
fig.update_traces(line_color='#00CC96', line_width=2)
fig.update_layout(
    xaxis_title="Timeline",
    yaxis_title="Load in MW",
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# 6. Data Preview and Download Section
st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📋 Forecast Data Preview")
    st.dataframe(result.head(10), use_container_width=True)

with col_right:
    st.subheader("📥 Export Results")
    st.markdown("Download the complete forecast results as a CSV file.")
    st.download_button(
        label="Download forecast CSV",
        data=result.to_csv(index=False).encode('utf-8'),
        file_name="pjm_random_forest_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )
