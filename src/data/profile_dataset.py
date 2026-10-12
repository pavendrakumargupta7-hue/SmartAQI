
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "raw" / "city_day.csv"


def main():
    df = pd.read_csv(DATASET)

    print("=" * 55)
    print("AQI DATA QUALITY REPORT")
    print("=" * 55)

    print("\n1. Total Records:", len(df))
    print("2. Duplicate Rows:", df.duplicated().sum())

    print("\n3. Missing Values Percentage")
    missing_percent = (
        df.isnull().mean().mul(100).round(2).sort_values(ascending=False)
    )
    print(missing_percent.to_string())

    print("\n4. AQI Target Analysis")
    print("AQI available:", df["AQI"].notna().sum())
    print("AQI missing:", df["AQI"].isna().sum())

    print("\n5. Records by City")
    print(df.groupby("City").size().sort_values(ascending=False).to_string())

    print("\n6. AQI Records by City")
    print(
        df.groupby("City")["AQI"]
        .count()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nData quality analysis completed!")


if __name__ == "__main__":
    main()
