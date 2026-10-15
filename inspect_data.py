
import pandas as pd
from pathlib import Path

# Load dataset
file_path = Path("Dataset.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset.csv project folder mein nahi mili."
    )

df = pd.read_csv(file_path)

print("=" * 50)
print("NETFLIX DATASET INSPECTION")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nContent type distribution:")
print(df["type"].value_counts())

print("\nRelease year range:")
print(df["release_year"].min(), "-", df["release_year"].max())

print("\nRating distribution:")
print(df["rating"].value_counts().head(10))

print("\nFirst five records:")
print(df.head())

print("\nInspection completed successfully!")
