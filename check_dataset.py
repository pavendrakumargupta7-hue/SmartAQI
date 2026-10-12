
from pathlib import Path
import pandas as pd

DATA_FILE = Path("data/raw/city_day.csv")


def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        print("Please check the file location.")
        return

    df = pd.read_csv(DATA_FILE)

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- First Five Rows ---")
    print(df.head())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Date Range ---")
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        print("Start:", dates.min())
        print("End:", dates.max())

    print("\n--- Available Cities ---")
    if "City" in df.columns:
        print("Total cities:", df["City"].nunique())
        print(df["City"].dropna().unique()[:20])


if __name__ == "__main__":
    main()
