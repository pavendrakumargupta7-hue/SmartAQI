
from pathlib import Path

import pandas as pd
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AQI City Analytics",
    page_icon="🌍",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "processed" / "aqi_model_data.csv"
)


# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_data(file_path, modified_time):
    df = pd.read_csv(file_path, parse_dates=["Date"])

    df["AQI"] = pd.to_numeric(
        df["AQI"], errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "AQI", "City"]
    )

    # Keep only the project's validated AQI range.
    df = df[df["AQI"].between(0, 500)].copy()

    return df


# =====================================================
# AQI CATEGORY
# =====================================================

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    return "Severe"


# =====================================================
# MAIN DASHBOARD
# =====================================================

st.title("🌍 AQI Analytics Dashboard")

st.write(
    "Explore historical air quality, compare Indian cities, "
    "and analyze AQI categories using the available dataset."
)

if not DATA_FILE.exists():
    st.error(
        f"Dataset not found: {DATA_FILE}\n\n"
        "Please run src/data/prepare_dataset.py first."
    )
    st.stop()

df = load_data(
    str(DATA_FILE),
    DATA_FILE.stat().st_mtime
)

if df.empty:
    st.warning("No valid AQI records were found.")
    st.stop()


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.header("Dashboard Filters")

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

start_date = st.sidebar.date_input(
    "Start date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

if start_date > end_date:
    st.error("Start date must be before or equal to end date.")
    st.stop()

filtered_df = df[
    (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No records available for this date range.")
    st.stop()


# =====================================================
# CITY SELECTOR
# =====================================================

cities = sorted(filtered_df["City"].dropna().unique())

if not cities:
    st.warning("No cities available for this date range.")
    st.stop()

selected_city = st.sidebar.selectbox(
    "Select city for trend analysis",
    cities
)

selected_city_df = filtered_df[
    filtered_df["City"] == selected_city
].copy()


# =====================================================
# OVERVIEW METRICS
# =====================================================

st.subheader("📊 Air Quality Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Selected City",
    selected_city
)

col2.metric(
    "Historical Records",
    f"{len(selected_city_df):,}"
)

col3.metric(
    "Average AQI",
    f"{selected_city_df['AQI'].mean():.1f}"
)

col4.metric(
    "Maximum AQI",
    f"{selected_city_df['AQI'].max():.1f}"
)

st.caption(
    f"Date range: {start_date} to {end_date}"
)



# =====================================================
# AQI CATEGORY DISTRIBUTION
# =====================================================

st.subheader(
    f"🧭 AQI Category Distribution: {selected_city}"
)

category_order = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe"
]

selected_city_df["AQI Category"] = (
    selected_city_df["AQI"].apply(get_aqi_category)
)

category_counts = (
    selected_city_df["AQI Category"]
    .value_counts()
    .reindex(category_order, fill_value=0)
)

left, right = st.columns(2)

with left:
    st.write("**Records by AQI Category**")
    st.bar_chart(category_counts)

with right:
    category_table = category_counts.rename(
        "Number of Records"
    ).to_frame()

    category_table["Percentage (%)"] = (
        category_table["Number of Records"]
        / len(selected_city_df) * 100
    ).round(2)

    st.dataframe(
        category_table,
        use_container_width=True
    )


# =====================================================
# CITY-WISE COMPARISON
# =====================================================

st.subheader("🏙️ City-wise AQI Comparison")

city_comparison = (
    filtered_df.groupby("City")["AQI"]
    .agg(
        Average_AQI="mean",
        Maximum_AQI="max",
        Minimum_AQI="min",
        Records="count"
    )
    .reset_index()
)

city_comparison = city_comparison.sort_values(
    "Average_AQI",
    ascending=False
)

city_comparison["Average_AQI"] = (
    city_comparison["Average_AQI"].round(2)
)

city_comparison["Maximum_AQI"] = (
    city_comparison["Maximum_AQI"].round(2)
)

city_comparison["Minimum_AQI"] = (
    city_comparison["Minimum_AQI"].round(2)
)

left, right = st.columns(2)

with left:
    st.write("**Highest Average AQI Cities**")

    st.bar_chart(
        city_comparison.head(10).set_index("City")[
            ["Average_AQI"]
        ]
    )

with right:
    st.write("**Lowest Average AQI Cities**")

    st.bar_chart(
        city_comparison.tail(10)
        .sort_values("Average_AQI")
        .set_index("City")[["Average_AQI"]]
    )

st.write("**Complete City Comparison Table**")

st.dataframe(
    city_comparison.rename(
        columns={
            "Average_AQI": "Average AQI",
            "Maximum_AQI": "Maximum AQI",
            "Minimum_AQI": "Minimum AQI"
        }
    ),
    use_container_width=True,
    hide_index=True
)



# =====================================================
# SELECTED CITY HISTORICAL TREND
# =====================================================

st.subheader(
    f"📈 Historical AQI Trend: {selected_city}"
)

city_history = selected_city_df.copy()

daily_trend = (
    city_history.groupby("Date")["AQI"]
    .mean()
    .sort_index()
)

if not daily_trend.empty:

    st.line_chart(
        daily_trend,
        y_label="AQI",
        x_label="Date"
    )

    trend_col1, trend_col2, trend_col3 = st.columns(3)

    trend_col1.metric(
        "City Average AQI",
        f"{city_history['AQI'].mean():.2f}"
    )

    trend_col2.metric(
        "City Maximum AQI",
        f"{city_history['AQI'].max():.2f}"
    )

    trend_col3.metric(
        "City Records",
        f"{len(city_history):,}"
    )

else:
    st.warning(
        f"No historical AQI records found for {selected_city}."
    )


# =====================================================
# DATA PREVIEW
# =====================================================

with st.expander("View filtered historical records"):
    st.dataframe(
        filtered_df.sort_values(
            "Date", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Data source: processed historical AQI dataset. "
    "This page analyzes historical data; it does not "
    "forecast future AQI."
)


# =====================================================
# DOWNLOAD ANALYTICS REPORT
# =====================================================

st.subheader("📥 Download Analytics Report")

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.write("**Download filtered historical records**")

    historical_csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Download Historical Data (CSV)",
        data=historical_csv,
        file_name="aqi_historical_data.csv",
        mime="text/csv",
        key="download_historical_aqi"
    )

with download_col2:
    st.write("**Download city-wise comparison**")

    comparison_csv = city_comparison.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Download City Comparison (CSV)",
        data=comparison_csv,
        file_name="aqi_city_comparison.csv",
        mime="text/csv",
        key="download_city_comparison"
    )
