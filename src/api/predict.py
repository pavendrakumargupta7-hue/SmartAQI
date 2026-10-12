
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    ROOT
    / "src"
    / "models"
    / "artifacts"
    / "aqi_random_forest.joblib"
)


POLLUTANT_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
]


def load_model():
    """Load the saved AQI model."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


def predict_aqi(city, date, pollutants):
    """
    Estimate AQI for a city, date, and pollutant measurements.

    pollutants must contain the pollutant names listed above.
    Missing pollutant values are handled by the saved pipeline.
    """
    if not isinstance(pollutants, dict):
        raise TypeError("pollutants must be a dictionary.")

    missing = [
        name for name in POLLUTANT_COLUMNS
        if name not in pollutants
    ]

    if missing:
        raise ValueError(
            f"Missing pollutant fields: {', '.join(missing)}"
        )

    prediction_date = pd.to_datetime(date, errors="raise")

    row = {
        name: pollutants[name]
        for name in POLLUTANT_COLUMNS
    }

    row["Year"] = prediction_date.year
    row["Month"] = prediction_date.month
    row["DayOfYear"] = prediction_date.dayofyear
    row["City"] = city

    input_data = pd.DataFrame([row])

    model = load_model()
    predicted_aqi = float(model.predict(input_data)[0])

    return round(max(0.0, predicted_aqi), 2)


if __name__ == "__main__":
    example_pollutants = {
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

    estimated_aqi = predict_aqi(
        city="Delhi",
        date="2020-07-01",
        pollutants=example_pollutants,
    )

    print("City: Delhi")
    print("Date: 2020-07-01")
    print("Estimated AQI:", estimated_aqi)
