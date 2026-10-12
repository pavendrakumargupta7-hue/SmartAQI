
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# =====================================================
# PATHS AND SETTINGS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "aqi_model_data.csv"

FORECAST_DAYS = 7
MIN_HISTORY_DAYS = 30
RANDOM_STATE = 42


# =====================================================
# LOAD AND PREPARE HISTORICAL AQI
# =====================================================

def load_city_history(city):
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])

    required_columns = {"Date", "City", "AQI"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    df = df.dropna(subset=["Date", "City", "AQI"])

    # Apply the project's AQI validation range.
    df = df[df["AQI"].between(0, 500)]

    city_df = df[df["City"].str.casefold() == city.casefold()]

    # Use daily mean if a city has multiple records per date.
    daily = (
        city_df.groupby("Date", as_index=False)["AQI"]
        .mean()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(daily) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Not enough historical records for {city}. "
            f"At least {MIN_HISTORY_DAYS} daily records are required."
        )

    # Create a continuous daily timeline.
    daily = daily.set_index("Date").asfreq("D")
    daily["AQI"] = daily["AQI"].interpolate(
        method="time",
        limit_area="inside"
    )

    daily = daily.dropna(subset=["AQI"]).reset_index()

    if len(daily) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Not enough continuous AQI history for {city}."
        )

    return daily


# =====================================================
# CREATE TIME-SERIES FEATURES
# =====================================================

def create_features(daily):
    result = daily.copy()

    # Past AQI values.
    for lag in [1, 2, 3, 7, 14, 30]:
        result[f"lag_{lag}"] = result["AQI"].shift(lag)

    # Rolling averages use only earlier observations.
    result["rolling_mean_7"] = (
        result["AQI"].shift(1).rolling(7).mean()
    )

    result["rolling_mean_30"] = (
        result["AQI"].shift(1).rolling(30).mean()
    )

    result["day_of_week"] = result["Date"].dt.dayofweek
    result["month"] = result["Date"].dt.month
    result["day_of_year"] = result["Date"].dt.dayofyear

    return result


FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "day_of_week",
    "month",
    "day_of_year",
]


# =====================================================
# BUILD FORECASTING MODEL
# =====================================================

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=2
            )
        )
    ])


# =====================================================
# EVALUATE ON THE LATEST HISTORICAL PERIOD
# =====================================================

def evaluate_forecast(city):
    daily = load_city_history(city)
    featured = create_features(daily)

    supervised = featured.dropna(
        subset=FEATURE_COLUMNS + ["AQI"]
    ).reset_index(drop=True)

    if len(supervised) < 60:
        raise ValueError(
            "Not enough usable history for a reliable time-based split."
        )

    split_index = int(len(supervised) * 0.80)

    train = supervised.iloc[:split_index]
    test = supervised.iloc[split_index:]

    model = build_model()

    model.fit(
        train[FEATURE_COLUMNS],
        train["AQI"]
    )

    predictions = model.predict(test[FEATURE_COLUMNS])
    predictions = np.clip(predictions, 0, 500)

    mae = mean_absolute_error(test["AQI"], predictions)
    rmse = np.sqrt(mean_squared_error(test["AQI"], predictions))

    results = pd.DataFrame({
        "Date": test["Date"].values,
        "Actual_AQI": test["AQI"].values,
        "Predicted_AQI": predictions
    })

    return {
        "city": city,
        "train_records": len(train),
        "test_records": len(test),
        "test_start": test["Date"].min(),
        "test_end": test["Date"].max(),
        "mae": mae,
        "rmse": rmse,
        "results": results
    }


# =====================================================
# FORECAST THE NEXT 7 DAYS
# =====================================================

def forecast_next_days(city, days=FORECAST_DAYS):
    if not isinstance(days, int) or not 1 <= days <= 30:
        raise ValueError("Forecast days must be an integer from 1 to 30.")

    daily = load_city_history(city)

    featured = create_features(daily)
    supervised = featured.dropna(
        subset=FEATURE_COLUMNS + ["AQI"]
    )

    model = build_model()
    model.fit(
        supervised[FEATURE_COLUMNS],
        supervised["AQI"]
    )

    # Keep a separate history for recursive forecasting.
    history = daily.set_index("Date")["AQI"].copy()
    last_date = history.index.max()

    forecast_rows = []

    for step in range(1, days + 1):
        forecast_date = last_date + pd.Timedelta(days=step)

        row = {
            "Date": forecast_date,
            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "lag_7": history.iloc[-7],
            "lag_14": history.iloc[-14],
            "lag_30": history.iloc[-30],
            "rolling_mean_7": history.iloc[-7:].mean(),
            "rolling_mean_30": history.iloc[-30:].mean(),
            "day_of_week": forecast_date.dayofweek,
            "month": forecast_date.month,
            "day_of_year": forecast_date.dayofyear,
        }

        input_df = pd.DataFrame([row])[FEATURE_COLUMNS]

        predicted_aqi = float(
            np.clip(model.predict(input_df)[0], 0, 500)
        )

        forecast_rows.append({
            "Date": forecast_date,
            "Predicted_AQI": round(predicted_aqi, 2)
        })

        # Feed each prediction into the next day's lag features.
        history.loc[forecast_date] = predicted_aqi

    return pd.DataFrame(forecast_rows)


# =====================================================
# RUN THIS FILE DIRECTLY FOR A QUICK TEST
# =====================================================

if __name__ == "__main__":
    city_name = "Ahmedabad"

    print("=" * 55)
    print("AQI TIME-SERIES FORECASTING")
    print("=" * 55)

    try:
        evaluation = evaluate_forecast(city_name)

        print(f"\nCity: {evaluation['city']}")
        print(f"Training records: {evaluation['train_records']}")
        print(f"Testing records: {evaluation['test_records']}")
        print(f"Test period: {evaluation['test_start']} to "
              f"{evaluation['test_end']}")
        print(f"MAE: {evaluation['mae']:.2f}")
        print(f"RMSE: {evaluation['rmse']:.2f}")

        print("\nNext 7 days (experimental estimates):")
        print(forecast_next_days(city_name).to_string(index=False))

    except (FileNotFoundError, ValueError) as error:
        print(f"\nForecasting error: {error}")
