
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# NETFLIX CONTENT SUCCESS ANALYTICS ENGINE
# Interactive dashboard using Streamlit + Plotly
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_FILE = BASE_DIR / "processed_netflix.csv"
MODEL_FILE = BASE_DIR / "model_comparison.csv"
INSIGHTS_FILE = REPORTS_DIR / "business_insights.txt"

st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- Styling --------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 15px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- Data Loading --------------------

@st.cache_data
def load_csv(file_path, file_mtime):
    """Load a CSV file and cache it until the file changes."""
    return pd.read_csv(file_path)


def read_csv_safely(file_path):
    """Read a CSV file safely, returning None if unavailable."""
    if not file_path.exists():
        return None

    try:
        return load_csv(str(file_path), file_path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Could not read {file_path.name}: {exc}")
        return None


@st.cache_data
def load_raw_dataset(file_path, file_mtime):
    return pd.read_csv(file_path)


def get_dataset():
    """Prefer the processed dataset; fall back to Dataset.csv."""
    if PROCESSED_FILE.exists():
        try:
            return load_raw_dataset(
                str(PROCESSED_FILE),
                PROCESSED_FILE.stat().st_mtime,
            ), "processed_netflix.csv"
        except Exception as exc:
            st.error(f"Processed dataset load nahi ho saka: {exc}")

    raw_file = BASE_DIR / "Dataset.csv"

    if raw_file.exists():
        try:
            return load_raw_dataset(
                str(raw_file),
                raw_file.stat().st_mtime,
            ), "Dataset.csv"
        except Exception as exc:
            st.error(f"Dataset.csv load nahi ho saka: {exc}")

    return None, None


def clean_text(series):
    """Convert values to strings and remove common missing-value labels."""
    result = series.fillna("Unknown").astype(str).str.strip()
    result = result.replace(
        {
            "": "Unknown",
            "nan": "Unknown",
            "None": "Unknown",
            "NaN": "Unknown",
        }
    )
    return result


def find_column(data, candidates):
    """Find the first matching column, ignoring capitalization."""
    column_map = {column.lower(): column for column in data.columns}

    for candidate in candidates:
        if candidate.lower() in column_map:
            return column_map[candidate.lower()]

    return None


def split_metadata_counts(data, column_name, limit=10):
    """
    Split comma-separated country/category metadata.
    Counts metadata entries, not necessarily unique titles.
    """
    if column_name not in data.columns:
        return pd.DataFrame(columns=["Name", "Count"])

    values = (
        data[column_name]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )

    values = values[
        (values != "")
        & (values.str.lower() != "nan")
        & (values.str.lower() != "unknown")
    ]

    result = values.value_counts().head(limit).reset_index()
    result.columns = ["Name", "Count"]
    return result


def create_yearly_additions(data):
    """Build yearly content additions from date_added or engineered columns."""
    if "date_added_year" in data.columns:
        years = pd.to_numeric(data["date_added_year"], errors="coerce")
    elif "date_added" in data.columns:
        dates = pd.to_datetime(data["date_added"], errors="coerce")
        years = dates.dt.year
    else:
        return pd.DataFrame(columns=["Year", "Titles Added"])

    result = years.dropna().astype(int).value_counts().sort_index()
    result = result.reset_index()
    result.columns = ["Year", "Titles Added"]
    return result


def get_duration_column(data):
    return find_column(
        data,
        [
            "duration_value",
            "duration_numeric",
            "duration",
            "movie_duration_minutes",
        ],
    )


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# -------------------- Load Dataset --------------------

df, dataset_name = get_dataset()

if df is None:
    st.title("🎬 Netflix Content Success Analytics Engine")
    st.error(
        "Dataset file nahi mili. app.py ko project folder mein save karo "
        "jahan Dataset.csv ya processed_netflix.csv mojood ho."
    )
    st.code(
        "E:\\ML-Task\\Netflix Content Success Analytics Engine\\"
        "Dataset.csv"
    )
    st.stop()

if df.empty:
    st.error("Dataset empty hai. CSV file check karo.")
    st.stop()

# Standardize commonly used fields without changing the original data.
type_col = find_column(df, ["type"])
title_col = find_column(df, ["title"])
year_col = find_column(df, ["release_year"])
rating_col = find_column(df, ["rating"])
country_col = find_column(df, ["country"])
genre_col = find_column(df, ["listed_in"])
director_col = find_column(df, ["director"])

if type_col:
    df[type_col] = clean_text(df[type_col])

if rating_col:
    df[rating_col] = clean_text(df[rating_col])

if country_col:
    df[country_col] = clean_text(df[country_col])

if genre_col:
    df[genre_col] = clean_text(df[genre_col])

if title_col:
    df[title_col] = clean_text(df[title_col])

if year_col:
    df[year_col] = safe_numeric(df[year_col])


# -------------------- Sidebar --------------------

st.sidebar.title("🎬 Netflix Analytics")
st.sidebar.caption("Content analytics and ML model evaluation")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "Overview",
        "Content Analysis",
        "Country & Genre Analysis",
        "Yearly Trends",
        "ML Model Comparison",
        "Business Insights",
        "Data Explorer",
    ],
)

st.sidebar.divider()
st.sidebar.write(f"**Dataset:** {dataset_name}")
st.sidebar.write(f"**Rows:** {len(df):,}")
st.sidebar.write(f"**Columns:** {len(df.columns):,}")

if st.sidebar.button("🔄 Refresh dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# -------------------- Header --------------------

st.markdown(
    '<div class="main-title">🎬 Netflix Content Success Analytics Engine</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Explore content distribution, genres, countries, yearly trends "
    "and machine-learning model performance."
    "</div>",
    unsafe_allow_html=True,
)


# -------------------- Common Metrics --------------------

total_titles = len(df)

if type_col:
    type_counts = df[type_col].value_counts()
    movies = int(type_counts.get("Movie", 0))
    tv_shows = int(type_counts.get("TV Show", 0))
else:
    type_counts = pd.Series(dtype=int)
    movies = 0
    tv_shows = 0

movie_percentage = (movies / total_titles * 100) if total_titles else 0
tv_percentage = (tv_shows / total_titles * 100) if total_titles else 0

if year_col and df[year_col].notna().any():
    min_year = int(df[year_col].min())
    max_year = int(df[year_col].max())
else:
    min_year = None
    max_year = None


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":
    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Titles", f"{total_titles:,}")
    col2.metric("Movies", f"{movies:,}", f"{movie_percentage:.2f}%")
    col3.metric("TV Shows", f"{tv_shows:,}", f"{tv_percentage:.2f}%")
    col4.metric(
        "Release Year Range",
        f"{min_year}–{max_year}" if min_year is not None else "N/A",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Content Type Distribution")

        if type_col:
            type_chart = (
                df[type_col]
                .value_counts()
                .rename_axis("Content Type")
                .reset_index(name="Titles")
            )

            fig = px.pie(
                type_chart,
                names="Content Type",
                values="Titles",
                hole=0.48,
                title="Movies vs TV Shows",
            )
            fig.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dataset mein 'type' column nahi hai.")

    with right:
        st.markdown("#### Rating Distribution")

        if rating_col:
            rating_data = (
                df[rating_col]
                .value_counts()
                .head(15)
                .rename_axis("Rating")
                .reset_index(name="Titles")
            )

            fig = px.bar(
                rating_data,
                x="Rating",
                y="Titles",
                title="Titles by Rating",
                text="Titles",
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dataset mein rating column nahi hai.")

    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.caption(
        "Note: Dashboard content catalogue ko analyze karta hai. "
        "Ye actual audience views, revenue ya popularity ka direct measure nahi hai."
    )


# ============================================================
# PAGE 2: CONTENT ANALYSIS
# ============================================================

elif page == "Content Analysis":
    st.subheader("🎞️ Content Analysis")

    if type_col:
        left, right = st.columns(2)

        with left:
            st.markdown("#### Movies vs TV Shows")
            counts = (
                df[type_col]
                .value_counts()
                .rename_axis("Type")
                .reset_index(name="Count")
            )

            fig = px.bar(
                counts,
                x="Type",
                y="Count",
                color="Type",
                text="Count",
                title="Number of Titles by Type",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### Type Percentage")
            percentage_data = counts.copy()
            percentage_data["Percentage"] = (
                percentage_data["Count"] / total_titles * 100
            )

            fig = px.bar(
                percentage_data,
                x="Type",
                y="Percentage",
                text=percentage_data["Percentage"].map(
                    lambda value: f"{value:.2f}%"
                ),
                title="Share of Total Catalogue",
            )
            fig.update_layout(yaxis_title="Percentage (%)")
            st.plotly_chart(fig, use_container_width=True)

    if rating_col:
        st.markdown("#### Ratings by Content Type")

        if type_col:
            rating_type_data = (
                df.groupby([rating_col, type_col], dropna=False)
                .size()
                .reset_index(name="Titles")
            )

            fig = px.bar(
                rating_type_data,
                x=rating_col,
                y="Titles",
                color=type_col,
                barmode="group",
                title="Rating Breakdown",
            )
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

    if year_col:
        year_data = df.dropna(subset=[year_col]).copy()
        year_data[year_col] = year_data[year_col].astype(int)

        if not year_data.empty:
            st.markdown("#### Release Year Distribution")

            yearly_release = (
                year_data.groupby(year_col)
                .size()
                .reset_index(name="Titles")
                .sort_values(year_col)
            )

            fig = px.line(
                yearly_release,
                x=year_col,
                y="Titles",
                markers=True,
                title="Titles by Release Year",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                yearly_release.sort_values("Titles", ascending=False).head(15),
                use_container_width=True,
            )
    else:
        st.info("Release year information available nahi hai.")


# ============================================================
# PAGE 3: COUNTRY & GENRE ANALYSIS
# ============================================================

elif page == "Country & Genre Analysis":
    st.subheader("🌍 Country & Genre Analysis")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Top Countries")

        if country_col:
            country_data = split_metadata_counts(df, country_col, limit=15)

            if not country_data.empty:
                fig = px.bar(
                    country_data.sort_values("Count"),
                    x="Count",
                    y="Name",
                    orientation="h",
                    text="Count",
                    title="Top Countries by Metadata Mentions",
                )
                fig.update_layout(yaxis_title="Country")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(country_data, use_container_width=True)
            else:
                st.info("Country information available nahi hai.")
        else:
            st.info("Dataset mein country column nahi hai.")

    with right:
        st.markdown("#### Top Genres / Categories")

        if genre_col:
            genre_data = split_metadata_counts(df, genre_col, limit=15)

            if not genre_data.empty:
                fig = px.bar(
                    genre_data.sort_values("Count"),
                    x="Count",
                    y="Name",
                    orientation="h",
                    text="Count",
                    title="Top Listed Categories",
                )
                fig.update_layout(yaxis_title="Category")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(genre_data, use_container_width=True)
            else:
                st.info("Genre information available nahi hai.")
        else:
            st.info("Dataset mein listed_in column nahi hai.")

    st.caption(
        "Country aur genre counts comma-separated metadata entries ko split "
        "karke nikale jaate hain. Ek title multiple countries ya genres mein "
        "count ho sakta hai."
    )


# ============================================================
# PAGE 4: YEARLY TRENDS
# ============================================================

elif page == "Yearly Trends":
    st.subheader("📅 Netflix Content Addition Trends")

    yearly_data = create_yearly_additions(df)

    if not yearly_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Years with Addition Dates",
                yearly_data["Year"].nunique(),
            )

        with col2:
            peak_row = yearly_data.loc[
                yearly_data["Titles Added"].idxmax()
            ]
            st.metric(
                "Peak Addition Year",
                int(peak_row["Year"]),
                f"{int(peak_row['Titles Added']):,} titles",
            )

        fig = px.line(
            yearly_data,
            x="Year",
            y="Titles Added",
            markers=True,
            title="Titles Added by Year",
        )
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Titles Added",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            yearly_data,
            x="Year",
            y="Titles Added",
            title="Yearly Additions",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Yearly Data")
        st.dataframe(
            yearly_data.sort_values("Year", ascending=False),
            use_container_width=True,
        )
    else:
        st.warning(
            "Yearly addition data available nahi hai. "
            "date_added ya date_added_year column required hai."
        )

    st.caption(
        "Ye chart catalogue mein recorded addition dates dikhata hai; "
        "ye viewing activity ya user engagement ka chart nahi hai."
    )


# ============================================================
# PAGE 5: ML MODEL COMPARISON
# ============================================================

elif page == "ML Model Comparison":
    st.subheader("🤖 Machine Learning Model Comparison")

    model_df = read_csv_safely(MODEL_FILE)

    if model_df is None:
        st.warning(
            "model_comparison.csv nahi mili. Pehle train_models.py run karo."
        )
        st.code("python train_models.py")
    elif model_df.empty:
        st.warning("Model comparison file empty hai.")
    else:
        st.markdown("#### Model Results")
        st.dataframe(model_df, use_container_width=True)

        lower_map = {column.lower(): column for column in model_df.columns}

        accuracy_col = next(
            (
                lower_map[name]
                for name in ["accuracy", "test_accuracy", "accuracy_score"]
                if name in lower_map
            ),
            None,
        )

        f1_col = next(
            (
                lower_map[name]
                for name in ["weighted_f1", "f1_weighted", "f1_score", "f1"]
                if name in lower_map
            ),
            None,
        )

        model_name_col = next(
            (
                lower_map[name]
                for name in ["model", "model_name", "name", "algorithm"]
                if name in lower_map
            ),
            None,
        )

        if model_name_col is None:
            model_name_col = model_df.columns[0]

        chart_df = model_df.copy()

        for metric_col in [accuracy_col, f1_col]:
            if metric_col:
                chart_df[metric_col] = pd.to_numeric(
                    chart_df[metric_col], errors="coerce"
                )

        metric_options = [
            column
            for column in [accuracy_col, f1_col]
            if column is not None
        ]

        if metric_options:
            selected_metric = st.selectbox(
                "Select evaluation metric",
                metric_options,
            )

            chart_df = chart_df.dropna(subset=[selected_metric])

            if not chart_df.empty:
                fig = px.bar(
                    chart_df,
                    x=model_name_col,
                    y=selected_metric,
                    color=model_name_col,
                    text=chart_df[selected_metric].map(
                        lambda value: f"{value:.4f}"
                    ),
                    title=f"Model Comparison — {selected_metric}",
                )
                fig.update_layout(
                    xaxis_title="Model",
                    yaxis_title=selected_metric,
                )
                st.plotly_chart(fig, use_container_width=True)

                best_row = chart_df.loc[
                    chart_df[selected_metric].idxmax()
                ]

                st.success(
                    f"Best model by {selected_metric}: "
                    f"{best_row[model_name_col]} "
                    f"({best_row[selected_metric]:.4f})"
                )
        else:
            st.info(
                "Accuracy ya weighted F1 column identify nahi ho saka. "
                "CSV ke column names check karo."
            )

        st.warning(
            "**Important limitation:** Current model `type` "
            "(Movie vs TV Show) predict karta hai; actual content success, "
            "views, revenue ya popularity predict nahi karta. "
            "Duration aur other metadata ki wajah se high scores aa sakte hain. "
            "Isliye in scores ko business success ka proof na samjhein."
        )


# ============================================================
# PAGE 6: BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":
    st.subheader("💡 Business Insights")

    if INSIGHTS_FILE.exists():
        try:
            with open(INSIGHTS_FILE, "r", encoding="utf-8") as file:
                insights_text = file.read()

            st.markdown(insights_text)
        except Exception as exc:
            st.error(f"Insights report read nahi ho saki: {exc}")
    else:
        st.warning(
            "reports/business_insights.txt nahi mili. "
            "Pehle generate_insights.py run karo."
        )
        st.code("python generate_insights.py")

    st.divider()
    st.markdown("#### Automatically Calculated Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.metric("Total Catalogue Titles", f"{total_titles:,}")

        if type_col and not type_counts.empty:
            most_common_type = type_counts.index[0]
            st.write(f"**Most common content type:** {most_common_type}")

    with summary_col2:
        if rating_col:
            known_ratings = df.loc[
                df[rating_col] != "Unknown", rating_col
            ]

            if not known_ratings.empty:
                st.write(
                    "**Most common rating:** "
                    f"{known_ratings.value_counts().idxmax()}"
                )

        if year_col and df[year_col].notna().any():
            st.write(
                f"**Earliest release year:** {int(df[year_col].min())}"
            )
            st.write(
                f"**Latest release year:** {int(df[year_col].max())}"
            )

    st.caption(
        "Catalogue-level findings descriptive analytics hain. "
        "Ye audience satisfaction, profit ya future success establish nahi karte."
    )


# ============================================================
# PAGE 7: DATA EXPLORER
# ============================================================

elif page == "Data Explorer":
    st.subheader("🔎 Explore Dataset")

    st.write(
        "Search aur filters use karke dataset ke records inspect karo."
    )

    filtered_df = df.copy()

    # Type filter
    if type_col:
        available_types = sorted(
            filtered_df[type_col].dropna().unique().tolist()
        )

        selected_types = st.multiselect(
            "Content Type",
            options=available_types,
            default=available_types,
        )

        filtered_df = filtered_df[
            filtered_df[type_col].isin(selected_types)
        ]

    # Rating filter
    if rating_col:
        available_ratings = sorted(
            filtered_df[rating_col].dropna().unique().tolist()
        )

        selected_ratings = st.multiselect(
            "Rating",
            options=available_ratings,
            default=available_ratings,
        )

        filtered_df = filtered_df[
            filtered_df[rating_col].isin(selected_ratings)
        ]

    # Release year filter
    if year_col and filtered_df[year_col].notna().any():
        year_values = filtered_df[year_col].dropna()

        if not year_values.empty:
            min_filter_year = int(year_values.min())
            max_filter_year = int(year_values.max())

            if min_filter_year < max_filter_year:
                selected_years = st.slider(
                    "Release Year Range",
                    min_value=min_filter_year,
                    max_value=max_filter_year,
                    value=(min_filter_year, max_filter_year),
                )

                filtered_df = filtered_df[
                    filtered_df[year_col].between(
                        selected_years[0],
                        selected_years[1],
                    )
                ]

    # Title search
    if title_col:
        search_text = st.text_input(
            "Search by title",
            placeholder="e.g. Stranger Things",
        )

        if search_text.strip():
            filtered_df = filtered_df[
                filtered_df[title_col]
                .str.contains(search_text.strip(), case=False, na=False)
            ]

    # Country text search
    if country_col:
        country_search = st.text_input(
            "Filter by country",
            placeholder="e.g. India",
        )

        if country_search.strip():
            filtered_df = filtered_df[
                filtered_df[country_col].str.contains(
                    country_search.strip(),
                    case=False,
                    na=False,
                )
            ]

    st.divider()

    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Filtered Rows", f"{len(filtered_df):,}")
    result_col2.metric(
        "Columns",
        f"{len(filtered_df.columns):,}",
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=480,
    )

    # Download filtered results
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_data,
        file_name="netflix_filtered_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("#### Missing Values")
    missing_data = (
        filtered_df.isna()
        .sum()
        .reset_index()
    )
    missing_data.columns = ["Column", "Missing Values"]
    missing_data = missing_data.sort_values(
        "Missing Values",
        ascending=False,
    )

    st.dataframe(missing_data, use_container_width=True)


# -------------------- Footer --------------------

st.divider()
st.caption(
    "Netflix Content Success Analytics Engine | "
    "Streamlit Dashboard | Descriptive analytics and model evaluation"
)
