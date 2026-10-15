
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv("processed_netflix.csv")

# Ensure date fields are correctly interpreted
df["date_added"] = pd.to_datetime(
    df["date_added"], errors="coerce"
)

df["date_added_year"] = df["date_added"].dt.year

# 1. Content type distribution
type_counts = df["type"].value_counts().rename_axis(
    "content_type"
).reset_index(name="title_count")

type_counts["percentage"] = (
    type_counts["title_count"] / len(df) * 100
).round(2)

type_counts.to_csv(
    OUTPUT_DIR / "content_type_distribution.csv",
    index=False
)

# 2. Titles added per year
yearly = (
    df.dropna(subset=["date_added_year"])
    .groupby("date_added_year")
    .size()
    .reset_index(name="titles_added")
    .sort_values("date_added_year")
)

yearly.to_csv(
    OUTPUT_DIR / "yearly_additions.csv",
    index=False
)

# 3. Content type by release year
release_type = pd.crosstab(
    df["release_year"], df["type"]
).reset_index()

release_type.to_csv(
    OUTPUT_DIR / "release_year_by_type.csv",
    index=False
)

# 4. Top countries (split multi-country entries)
country_df = df.dropna(subset=["country"]).copy()
country_df["country"] = country_df["country"].str.split(",")
country_df = country_df.explode("country")
country_df["country"] = country_df["country"].str.strip()
country_df = country_df[country_df["country"] != ""]

top_countries = (
    country_df["country"]
    .value_counts()
    .head(10)
    .rename_axis("country")
    .reset_index(name="title_count")
)

top_countries.to_csv(
    OUTPUT_DIR / "top_countries.csv",
    index=False
)

# 5. Top genre/category labels
genre_df = df.dropna(subset=["listed_in"]).copy()
genre_df["listed_in"] = genre_df["listed_in"].str.split(",")
genre_df = genre_df.explode("listed_in")
genre_df["listed_in"] = genre_df["listed_in"].str.strip()

top_genres = (
    genre_df["listed_in"]
    .value_counts()
    .head(10)
    .rename_axis("genre")
    .reset_index(name="title_count")
)

top_genres.to_csv(
    OUTPUT_DIR / "top_genres.csv",
    index=False
)

# 6. Rating distribution
rating_counts = (
    df["rating"]
    .dropna()
    .value_counts()
    .rename_axis("rating")
    .reset_index(name="title_count")
)

rating_counts.to_csv(
    OUTPUT_DIR / "rating_distribution.csv",
    index=False
)

# Chart 1: Content type distribution
plt.figure(figsize=(8, 5))
plt.bar(
    type_counts["content_type"],
    type_counts["title_count"]
)
plt.title("Netflix Content Type Distribution")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "content_type_distribution.png", dpi=150)
plt.close()

# Chart 2: Yearly additions
plt.figure(figsize=(10, 5))
plt.plot(
    yearly["date_added_year"],
    yearly["titles_added"],
    marker="o"
)
plt.title("Netflix Titles Added by Year")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "yearly_additions.png", dpi=150)
plt.close()

# Chart 3: Top countries
plt.figure(figsize=(10, 6))
plt.barh(
    top_countries["country"][::-1],
    top_countries["title_count"][::-1]
)
plt.title("Top 10 Countries by Title Count")
plt.xlabel("Number of Titles")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_countries.png", dpi=150)
plt.close()

# Chart 4: Top genres
plt.figure(figsize=(10, 6))
plt.barh(
    top_genres["genre"][::-1],
    top_genres["title_count"][::-1]
)
plt.title("Top 10 Netflix Categories")
plt.xlabel("Number of Titles")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_genres.png", dpi=150)
plt.close()

# Generate automated text insights
movie_count = int((df["type"] == "Movie").sum())
tv_count = int((df["type"] == "TV Show").sum())

movie_pct = movie_count / len(df) * 100
tv_pct = tv_count / len(df) * 100

peak_year = None
peak_count = None

if not yearly.empty:
    peak_row = yearly.loc[yearly["titles_added"].idxmax()]
    peak_year = int(peak_row["date_added_year"])
    peak_count = int(peak_row["titles_added"])

with open(OUTPUT_DIR / "business_insights.txt", "w", encoding="utf-8") as f:
    f.write("NETFLIX CONTENT ANALYTICS: AUTOMATED INSIGHTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total titles analyzed: {len(df):,}\n")
    f.write(f"Movies: {movie_count:,} ({movie_pct:.2f}%)\n")
    f.write(f"TV Shows: {tv_count:,} ({tv_pct:.2f}%)\n\n")

    if peak_year is not None:
        f.write(
            f"Highest recorded additions year: {peak_year} "
            f"({peak_count:,} titles)\n"
        )

    if not top_countries.empty:
        f.write(
            f"Most represented country in split country entries: "
            f"{top_countries.iloc[0]['country']} "
            f"({int(top_countries.iloc[0]['title_count']):,} entries)\n"
        )

    if not top_genres.empty:
        f.write(
            f"Most common listed category: "
            f"{top_genres.iloc[0]['genre']} "
            f"({int(top_genres.iloc[0]['title_count']):,} entries)\n"
        )

    f.write(
        "\nInterpretation note: These results describe the supplied "
        "dataset's metadata, not viewership, revenue, or causal success.\n"
    )

print("Business analytics completed successfully.")
print(f"Reports saved in: {OUTPUT_DIR.resolve()}")
print("\nContent type distribution:")
print(type_counts.to_string(index=False))
print("\nTop countries:")
print(top_countries.head().to_string(index=False))
print("\nTop categories:")
print(top_genres.head().to_string(index=False))
print("\nAutomated insights saved to reports/business_insights.txt")
