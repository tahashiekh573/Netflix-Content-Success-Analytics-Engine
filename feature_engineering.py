
import pandas as pd
import numpy as np
from pathlib import Path

# Load dataset
df = pd.read_csv("Dataset.csv")

# Standardize text columns
text_columns = [
    "type", "title", "director", "country",
    "rating", "duration", "listed_in"
]

for col in text_columns:
    df[col] = df[col].astype("string").str.strip()

# Replace placeholder values with missing values
placeholders = ["", "Not Given", "Unknown", "N/A"]

for col in ["director", "country", "rating", "duration"]:
    df[col] = df[col].replace(placeholders, pd.NA)

# Convert date_added to datetime
df["date_added"] = pd.to_datetime(
    df["date_added"], errors="coerce"
)

# Extract date-related features
df["date_added_year"] = df["date_added"].dt.year
df["date_added_month"] = df["date_added"].dt.month

# Extract duration features
df["duration_value"] = pd.to_numeric(
    df["duration"].str.extract(r"(\d+)")[0],
    errors="coerce"
)

# Movies: duration_value represents minutes
df["movie_duration_minutes"] = np.where(
    df["type"].eq("Movie"),
    df["duration_value"],
    np.nan
)

# TV Shows: duration_value represents seasons
df["tv_seasons"] = np.where(
    df["type"].eq("TV Show"),
    df["duration_value"],
    np.nan
)

# Count listed genres/categories
df["genre_count"] = (
    df["listed_in"]
    .fillna("")
    .apply(
        lambda value: len([
            item for item in value.split(",")
            if item.strip()
        ])
    )
)

# Count listed countries
df["country_count"] = (
    df["country"]
    .fillna("")
    .apply(
        lambda value: len([
            item for item in value.split(",")
            if item.strip()
        ])
    )
)

# Create metadata availability features
df["has_director"] = df["director"].notna().astype(int)
df["has_country"] = df["country"].notna().astype(int)
df["has_rating"] = df["rating"].notna().astype(int)

# Save processed dataset
output_path = Path("processed_netflix.csv")
df.to_csv(output_path, index=False)

print("=" * 50)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 50)

print("\nProcessed dataset shape:", df.shape)

print("\nNew features:")
print([
    "date_added_year",
    "date_added_month",
    "duration_value",
    "movie_duration_minutes",
    "tv_seasons",
    "genre_count",
    "country_count",
    "has_director",
    "has_country",
    "has_rating"
])

print("\nMissing values after cleaning:")
print(df[[
    "date_added", "rating", "country", "director",
    "duration_value"
]].isnull().sum())

print("\nSample engineered features:")
print(df[[
    "title", "type", "release_year",
    "movie_duration_minutes", "tv_seasons",
    "genre_count", "country_count"
]].head(10))

print(f"\nSaved file: {output_path.resolve()}")
