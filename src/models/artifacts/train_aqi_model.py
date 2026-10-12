
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "aqi_model_data.csv"
MODEL_DIR = ROOT / "src" / "models" / "artifacts"
MODEL_FILE = MODEL_DIR / "aqi_random_forest.joblib"


def main():
    print("=" * 55)
    print("AQI RANDOM FOREST MODEL TRAINING")
    print("=" * 55)

    # 1. Load processed dataset
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.dropna(subset=["AQI"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 2. Create date-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # 3. Select features and target
    pollutant_columns = [
        "PM2.5", "PM10", "NO", "NO2", "NOx",
        "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"
    ]

    numeric_features = pollutant_columns + [
        "Year", "Month", "DayOfYear"
    ]
    categorical_features = ["City"]

    X = df[numeric_features + categorical_features]
    y = df["AQI"]

    # 4. Chronological train-test split
    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Training date range:",
          df["Date"].iloc[0].date(), "to",
          df["Date"].iloc[split_index - 1].date())
    print("Testing date range:",
          df["Date"].iloc[split_index].date(), "to",
          df["Date"].iloc[-1].date())

    # 5. Preprocessing pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # 6. Create model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 7. Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # 8. Evaluate on unseen later dates
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nMODEL EVALUATION")
    print("-" * 40)
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    # 9. Save trained pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("\nModel saved to:")
    print(MODEL_FILE)

    # 10. Show sample predictions
    results = pd.DataFrame({
        "City": df["City"].iloc[split_index:].values,
        "Actual_AQI": y_test.values,
        "Predicted_AQI": np.round(predictions, 2)
    })

    print("\nSAMPLE PREDICTIONS")
    print(results.head(10).to_string(index=False))
    print("\nModel training completed!")


if __name__ == "__main__":
    main()
