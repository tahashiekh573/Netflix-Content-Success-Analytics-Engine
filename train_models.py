
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import joblib

# Load engineered dataset
df = pd.read_csv("processed_netflix.csv")

# Target: content type
target = "type"

# Features available for prediction
numeric_features = [
    "release_year",
    "date_added_year",
    "date_added_month",
    "duration_value",
    "genre_count",
    "country_count",
    "has_director",
    "has_country",
    "has_rating"
]

categorical_features = [
    "rating"
]

features = numeric_features + categorical_features

# Keep required columns and remove rows without target
df = df.dropna(subset=[target]).copy()

X = df[features].copy()
y = df[target].copy()

# Replace infinite values, if any
X = X.replace([np.inf, -np.inf], np.nan)

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Stratified split preserves Movie/TV Show proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

results = []
best_f1 = -1
best_name = None
best_pipeline = None

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1
    })

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_pipeline = pipeline

# Save comparison table
results_df = pd.DataFrame(results).sort_values(
    "weighted_f1", ascending=False
)
results_df.to_csv("model_comparison.csv", index=False)

# Save best complete pipeline
joblib.dump(best_pipeline, "best_model.joblib")

print("\nModel comparison:")
print(results_df.to_string(index=False))
print(f"\nBest model by weighted F1: {best_name}")
print("Saved: model_comparison.csv")
print("Saved: best_model.joblib")
