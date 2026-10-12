
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "city_day.csv"


def inspect_dataset():
    print("=" * 50)
    print("AQI DATASET INSPECTION")
    print("=" * 50)

    if not DATASET_PATH.exists():
        print("Dataset not found:", DATASET_PATH)
        return

    df = pd.read_csv(DATASET_PATH)

    print("\n1. Dataset Shape")
    print("Total records:", df.shape[0])
    print("Total columns:", df.shape[1])

    print("\n2. Column Names")
    print(df.columns.tolist())

    print("\n3. First Five Records")
    print(df.head().to_string(index=False))

    if "City" in df.columns:
        print("\n4. Total Cities:", df["City"].nunique())
        print(sorted(df["City"].dropna().unique()))

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], errors="coerce")
        print("\n5. Date Range")
        print("Start date:", dates.min())
        print("End date:", dates.max())

    print("\n6. Missing Values")
    print(df.isnull().sum().to_string())

    for column in ["AQI", "AQI_Bucket"]:
        if column in df.columns:
            print(f"\n{column} Summary")
            print(df[column].value_counts(dropna=False).head(10))

    print("\nDataset inspection completed!")


if __name__ == "__main__":
    inspect_dataset()
