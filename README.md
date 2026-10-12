# SmartAQI - AQI Prediction for Indian Cities

## Project Overview
SmartAQI is a machine learning project that estimates Air Quality Index (AQI) for Indian cities using historical air-quality data. It also provides city-wise analysis and future AQI forecasting.

## Features
- AQI estimation using a trained Machine Learning model
- Future AQI forecasting for 1 to 30 days
- City-wise AQI trend analysis
- Interactive Streamlit dashboard
- Historical air-quality data processing

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib / Plotly (where used)
- Joblib

## Dataset
The project uses historical air-quality data for Indian cities. Refer to the original dataset source and its license before redistributing the data.

## How to Run

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Start the dashboard:

   streamlit run app.py

## Machine Learning
The project uses a trained Random Forest model for AQI estimation. Historical data is also used for future AQI forecasting.

## Author
Pavendra Kumar Gupta
