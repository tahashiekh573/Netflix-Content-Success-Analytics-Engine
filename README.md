# Netflix Content Success Analytics Engine

## 1. Project Overview

The Netflix Content Success Analytics Engine is a data analytics and machine learning project designed to analyze Netflix's content catalogue. It explores content distribution, release years, ratings, countries, genres, and machine learning model performance through an interactive Streamlit dashboard.

The project uses Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Plotly, and Streamlit for data processing, analysis, visualization, and dashboard development.

## 2. Objectives

- Analyze the distribution of Movies and TV Shows.
- Explore Netflix content by release year and rating.
- Identify frequently listed countries and content categories.
- Analyze yearly content additions.
- Compare machine learning classification models.
- Present results through an interactive dashboard.
- Generate charts, CSV reports, and business insights.

## 3. Dataset Information

**Dataset file:** `Dataset.csv`

The dataset contains 8,790 records and 10 original columns.

| Column | Description |
|---|---|
| `show_id` | Unique content identifier |
| `type` | Movie or TV Show |
| `title` | Content title |
| `director` | Director name |
| `country` | Country information |
| `date_added` | Date added to Netflix |
| `release_year` | Original release year |
| `rating` | Content rating |
| `duration` | Movie duration or TV Show seasons |
| `listed_in` | Genres or content categories |

## 4. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Plotly

## 5. Project Structure

```text
Netflix Content Success Analytics Engine/
│
├── Dataset.csv
├── inspect_data.py
├── feature_engineering.py
├── train_models.py
├── generate_insights.py
├── plot_model_comparison.py
├── app.py
├── processed_netflix.csv
├── model_comparison.csv
├── best_model.joblib
├── requirements.txt
├── README.md
│
├── screenshots/
│
└── reports/
    ├── business_insights.txt
    ├── content_type_distribution.csv
    ├── yearly_additions.csv
    ├── release_year_by_type.csv
    ├── top_countries.csv
    ├── top_genres.csv
    ├── rating_distribution.csv
    ├── content_type_distribution.png
    ├── yearly_additions.png
    ├── top_countries.png
    ├── top_genres.png
    └── model_comparison.png
```

## 6. Installation

### Step 1: Open the project directory

Open PowerShell and run:

```powershell
cd "E:\ML-Task\Netflix Content Success Analytics Engine"
```

### Step 2: Install the required libraries

```powershell
python -m pip install -r requirements.txt
```

If the requirements file is empty, add these dependencies first:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
plotly
```

### Step 3: Run the data inspection script

```powershell
python inspect_data.py
```

This script inspects the dataset structure and basic data quality.

## 7. Data Processing and Model Training

Run the following scripts from the project directory, in order.

### Feature Engineering

```powershell
python feature_engineering.py
```

This creates `processed_netflix.csv` with additional features such as date-added year and month, numeric duration information, genre and country counts, and metadata availability indicators.

### Model Training

```powershell
python train_models.py
```

This trains and compares classification models and generates:

- `model_comparison.csv`
- `best_model.joblib`

### Generate Business Insights

```powershell
python generate_insights.py
```

This generates summary tables, charts, and the `business_insights.txt` report inside the `reports` folder.

### Plot Model Comparison

```powershell
python plot_model_comparison.py
```

This generates the model comparison visualization in the `reports` folder.

## 8. Machine Learning Models

The project compares three classification algorithms:

1. Logistic Regression
2. Random Forest
3. Extra Trees

The models are evaluated using accuracy and weighted F1 score.

### Recorded Results

| Model | Accuracy | Weighted F1 Score |
|---|---:|---:|
| Logistic Regression | 99.49% | 99.49% |
| Random Forest | 99.89% | 99.89% |
| Extra Trees | 99.54% | 99.54% |

Random Forest achieved the highest recorded scores in this experiment.

**Important:** The current machine learning task predicts content type (Movie or TV Show). It does not predict actual Netflix success, views, revenue, audience engagement, or popularity. The high classification scores should be interpreted in the context of the selected features and evaluation setup.

## 9. Key Dataset Findings

The analysis produced the following results:

- **Total titles:** 8,790
- **Movies:** 6,126 (69.69%)
- **TV Shows:** 2,664 (30.31%)
- **Release year range:** 1925–2021

Frequently occurring country entries include the United States, India, the United Kingdom, Pakistan, and Canada.

Frequently occurring categories include International Movies, Dramas, Comedies, International TV Shows, and Documentaries.

Country and genre counts are calculated from multi-value metadata. A single title may contribute to more than one country or category count.

## 10. Interactive Dashboard

Launch the Streamlit dashboard with:

```powershell
python -m streamlit run app.py
```

Open the following address in your browser:

**http://localhost:8501**

The dashboard contains these sections:

- **Overview:** Catalogue metrics and summary charts.
- **Content Analysis:** Content types, ratings, and release-year distribution.
- **Country & Genre Analysis:** Top country entries and categories.
- **Yearly Trends:** Content additions by year.
- **ML Model Comparison:** Model metrics and comparison charts.
- **Business Insights:** Generated report and summary findings.
- **Data Explorer:** Search, filters, data preview, and CSV download.

Keep the PowerShell window open while using the dashboard.

## 11. Output Files

| File | Purpose |
|---|---|
| `processed_netflix.csv` | Processed dataset with engineered features |
| `model_comparison.csv` | Classification model evaluation results |
| `best_model.joblib` | Saved best-performing model |
| `reports/business_insights.txt` | Text-based business insights |
| `reports/content_type_distribution.png` | Content type chart |
| `reports/yearly_additions.png` | Yearly additions chart |
| `reports/top_countries.png` | Country analysis chart |
| `reports/top_genres.png` | Genre analysis chart |
| `reports/model_comparison.png` | Model comparison chart |

## 12. Limitations

- The dataset describes a content catalogue and does not directly contain actual viewing figures or revenue.
- Missing director and country values may occur in the original dataset.
- Country and genre counts can include multiple entries for one title.
- Classification performance depends on the selected features and evaluation procedure.
- Predicting actual content success would require a suitable target variable and relevant data, such as viewing figures, engagement, or revenue.

## 13. Conclusion

The Netflix Content Success Analytics Engine combines data inspection, feature engineering, descriptive analytics, machine learning classification, and interactive visualization in one project. It provides an accessible way to explore Netflix catalogue patterns and compare classification models.

The Streamlit dashboard makes the analysis easier to navigate and allows users to inspect and filter the dataset interactively.

## 14. Author

**Project:** Netflix Content Success Analytics Engine  
**Task:** Task 6  
**Language:** Python  
**Dashboard Framework:** Streamlit
