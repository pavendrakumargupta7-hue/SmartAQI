
import streamlit as st
import pandas as pd
from pathlib import Path

from src.forecasting.aqi_forecast import forecast_next_days


# ==========================================
# PAGE TITLE
# ==========================================
st.title("🌤️ Future AQI Forecast")

st.write(
    "Select an Indian city to estimate its AQI "
    "for the next 1 to 30 days using historical data."
)


# ==========================================
# DATASET PATH
# ==========================================
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "aqi_model_data.csv"


# ==========================================
# LOAD CITY LIST
# ==========================================
@st.cache_data
def load_cities():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    if "City" not in df.columns:
        raise ValueError("The dataset does not contain a City column.")

    return sorted(
        df["City"].dropna().astype(str).unique().tolist()
    )


# ==========================================
# CITY AND FORECAST SETTINGS
# ==========================================
try:
    cities = load_cities()

    if not cities:
        st.error("No cities were found in the dataset.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        selected_city = st.selectbox(
            "Select City",
            cities,
            index=cities.index("Ahmedabad")
            if "Ahmedabad" in cities else 0
        )

    with col2:
        forecast_days = st.slider(
            "Forecast Period (Days)",
            min_value=1,
            max_value=30,
            value=10
        )

    st.info(
        "Forecast values are calculated from the historical "
        "data available in your processed dataset."
    )

    # ==========================================
    # GENERATE FORECAST AUTOMATICALLY
    # ==========================================
    with st.spinner("Generating AQI forecast..."):
        forecast_df = forecast_next_days(
            selected_city,
            days=forecast_days
        )

    if forecast_df is None or forecast_df.empty:
        st.warning(
            f"No forecast was generated for {selected_city}. "
            "Check the forecasting module and historical data."
        )
        st.stop()

    forecast_df = forecast_df.copy()

    required_columns = ["Date", "Predicted_AQI"]
    missing_columns = [
        column for column in required_columns
        if column not in forecast_df.columns
    ]

    if missing_columns:
        st.error(
            "Forecast output is missing these columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    forecast_df["Date"] = pd.to_datetime(
        forecast_df["Date"], errors="coerce"
    )
    forecast_df["Predicted_AQI"] = pd.to_numeric(
        forecast_df["Predicted_AQI"], errors="coerce"
    )

    forecast_df = forecast_df.dropna(
        subset=["Date", "Predicted_AQI"]
    ).sort_values("Date")

    if forecast_df.empty:
        st.warning("The forecasting module returned no valid values.")
        st.stop()

    # ==========================================
    # SUMMARY METRICS
    # ==========================================
    st.subheader(f"AQI Forecast for {selected_city}")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Forecast Days",
        len(forecast_df)
    )

    metric2.metric(
        "Average Predicted AQI",
        f"{forecast_df['Predicted_AQI'].mean():.1f}"
    )

    metric3.metric(
        "Maximum Predicted AQI",
        f"{forecast_df['Predicted_AQI'].max():.1f}"
    )

    # ==========================================
    # FORECAST CHART
    # ==========================================
    st.subheader("📈 Predicted AQI Trend")

    chart_df = forecast_df.set_index("Date")[["Predicted_AQI"]]
    st.line_chart(chart_df)

    # ==========================================
    # FORECAST TABLE
    # ==========================================
    st.subheader("📋 Daily Forecast Values")

    display_df = forecast_df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    display_df["Predicted_AQI"] = display_df[
        "Predicted_AQI"
    ].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ==========================================
    # DOWNLOAD CSV
    # ==========================================
    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast as CSV",
        data=csv_data,
        file_name=f"{selected_city}_aqi_forecast.csv",
        mime="text/csv"
    )

    # ==========================================
    # RESEARCH LIMITATION
    # ==========================================
    st.warning(
        "Research limitation: These are experimental estimates, "
        "not official AQI forecasts. The historical dataset ends "
        "on 2020-07-01, so check the forecast dates shown above. "
        "The estimates are not real-time AQI readings."
    )

except FileNotFoundError as e:
    st.error(str(e))

except Exception as e:
    st.error(f"Unable to generate the AQI forecast: {e}")
