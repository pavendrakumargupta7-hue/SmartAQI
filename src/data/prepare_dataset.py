
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "data" / "raw" / "city_day.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "aqi_model_data.csv"


def main():
    print("Loading raw AQI dataset...")

    df = pd.read_csv(RAW_FILE)

    initial_rows = len(df)

    # Keep only records with a known AQI target.
    df = df.dropna(subset=["AQI"]).copy()
    df = df[df["AQI"].between(0, 500)].copy()

    # AQI_Bucket is derived from AQI, so exclude it
    # to prevent target leakage during AQI prediction.
    df = df.drop(columns=["AQI_Bucket"])

    # Xylene has more than 61% missing values.
    # Exclude it from this initial model dataset.
    df = df.drop(columns=["Xylene"])

    # Convert Date to a consistent datetime format.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPREPROCESSING SUMMARY")
    print("-" * 40)
    print("Original records:", initial_rows)
    print("Processed records:", len(df))
    print("Removed records:", initial_rows - len(df))
    print("Remaining columns:", len(df.columns))
    print("Output file:", OUTPUT_FILE)

    print("\nMissing values remaining:")
    print(df.isnull().sum().to_string())

    print("\nProcessed dataset saved successfully!")


if __name__ == "__main__":
    main()
