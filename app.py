import streamlit as st

st.set_page_config(
    page_title="SmartAQI | Indian Cities",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

prediction_page = st.Page(
    "pages/1_AQI_Prediction.py",
    title="AQI Prediction",
    icon="🌡️",
    default=True,
)

analytics_page = st.Page(
    "pages/2_City_Analytics.py",
    title="City Analytics",
    icon="📊",
)

forecast_page = st.Page(
    "pages/3_AQI_Forecast.py",
    title="AQI Forecast",
    icon="📈",
)

navigation = st.navigation(
    {
        "SmartAQI Dashboard": [
            prediction_page,
            analytics_page,
            forecast_page,
        ]
    }
)

navigation.run()
