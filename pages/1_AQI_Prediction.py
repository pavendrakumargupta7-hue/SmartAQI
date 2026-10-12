
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.predict import predict_aqi, POLLUTANT_COLUMNS


DATA_FILE = ROOT / "data" / "processed" / "aqi_model_data.csv"


st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 AQI Prediction for Indian Cities")
st.write(
    "Estimate Air Quality Index (AQI) using pollutant measurements "
    "and the trained Random Forest model."
)

st.info(
    "This dashboard estimates AQI from entered pollutant values. "
    "It does not yet forecast future AQI."
)


# Load historical dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE, parse_dates=["Date"])


try:
    historical_data = load_data()
except Exception as exc:
    st.error(f"Could not load processed dataset: {exc}")
    st.stop()


cities = sorted(historical_data["City"].dropna().unique().tolist())

st.header("1. Select City and Date")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("Choose a city", cities)

with col2:
    prediction_date = st.date_input(
        "Select date",
        value=pd.Timestamp("2020-07-01").date(),
        min_value=pd.Timestamp("2015-01-01").date(),
        max_value=pd.Timestamp("2035-12-31").date(),
    )


st.header("2. Enter Pollutant Measurements")

st.caption(
    "Enter pollutant measurements in the same units as the training dataset. "
    "These fields accept numeric values; use 0 only when the measured value is actually zero."
)

defaults = {
    "PM2.5": 60.0,
    "PM10": 100.0,
    "NO": 10.0,
    "NO2": 20.0,
    "NOx": 30.0,
    "NH3": 10.0,
    "CO": 1.0,
    "SO2": 10.0,
    "O3": 30.0,
    "Benzene": 2.0,
    "Toluene": 5.0,
}

pollutants = {}

left, right = st.columns(2)

for index, name in enumerate(POLLUTANT_COLUMNS):
    target_column = left if index % 2 == 0 else right

    with target_column:
        pollutants[name] = st.number_input(
            name,
            min_value=0.0,
            value=defaults[name],
            step=0.1,
            format="%.2f",
            key=f"pollutant_{name}",
        )


if st.button("Predict AQI", type="primary", use_container_width=True):
    try:
        predicted_aqi = predict_aqi(
            city=city,
            date=prediction_date.isoformat(),
            pollutants=pollutants,
        )

        st.header("3. AQI Prediction")

        metric_col, detail_col = st.columns([1, 2])

        with metric_col:
            st.metric("Estimated AQI", f"{predicted_aqi:.2f}")

        with detail_col:
            st.write(f"**City:** {city}")
            st.write(f"**Selected date:** {prediction_date}")
            st.write("**Model:** Random Forest Regressor")

        if predicted_aqi <= 50:
            st.success("Indicative category: Good")
        elif predicted_aqi <= 100:
            st.info("Indicative category: Satisfactory")
        elif predicted_aqi <= 200:
            st.warning("Indicative category: Moderate")
        elif predicted_aqi <= 300:
            st.warning("Indicative category: Poor")
        elif predicted_aqi <= 400:
            st.error("Indicative category: Very Poor")
        else:
            st.error("Indicative category: Severe")

        st.caption(
            "The category is indicative and uses the project's AQI scale. "
            "This is a model estimate, not an official real-time AQI reading."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


st.divider()
st.header("4. Historical AQI Trend")

city_history = historical_data[
    (historical_data["City"] == city)
    & historical_data["AQI"].notna()
].copy()

if not city_history.empty:
    city_history = city_history.sort_values("Date")

    st.write("Minimum historical AQI:", city_history["AQI"].min())
    st.write("Maximum historical AQI:", city_history["AQI"].max())
    st.write(
        "Records with AQI above 500:",
        int((city_history["AQI"] > 500).sum())
    )

    st.line_chart(
        city_history.set_index("Date")["AQI"],
        x_label="Date",
        y_label="Historical AQI",
    )

    st.caption(
        f"Historical records available for {city}: {len(city_history):,}"
    )
else:
    st.warning("No historical AQI records found for this city.")
