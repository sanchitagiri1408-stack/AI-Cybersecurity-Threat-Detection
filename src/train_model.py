
from pathlib import Path
import json
import warnings

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

MODEL_FILE = MODEL_DIR / "logistic_regression_model.joblib"
METRICS_FILE = OUTPUT_DIR / "evaluation_metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"

TARGET = "label"

def main():
    print("=" * 60)
    print("AI CYBERSECURITY THREAT DETECTION")
    print("=" * 60)

    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Dataset files not found. Check that both CSV files are "
            "inside the data/raw folder."
        )

    print("\nLoading datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")

    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise ValueError("The 'label' target column must exist in both CSV files.")

    # Remove rows without a target label.
    train_df = train_df.dropna(subset=[TARGET]).copy()
    test_df = test_df.dropna(subset=[TARGET]).copy()

    # Remove identifiers and target-related descriptive columns.
    # 'attack_cat' describes the attack category and is not a model input.
    excluded_columns = ["id", "label", "attack_cat"]

    feature_columns = [
        column for column in train_df.columns
        if column not in excluded_columns
        and column in test_df.columns
    ]

    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET].astype(int)

    # Detect categorical and numeric input columns.
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_columns = [
        column for column in feature_columns
        if column not in categorical_columns
    ]

    print(f"\nNumeric features:     {len(numeric_columns)}")
    print(f"Categorical features: {len(categorical_columns)}")

    # Prepare numeric values: fill missing values and scale them.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Prepare categorical values: fill missing values and encode categories.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    # Logistic Regression works in the current Python environment.
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )),
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)
    print("Training completed successfully.")

    print("\nEvaluating on the separate testing dataset...")
    predictions = model.predict(X_test)

    metrics = {
        "model": "Logistic Regression",
        "target": TARGET,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "feature_count": int(len(feature_columns)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
            output_dict=True,
        ),
    }

    joblib.dump(model, MODEL_FILE)

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    results = test_df[["label"]].copy()
    results["predicted_label"] = predictions
    results["actual_class"] = results["label"].map(
        {0: "Normal", 1: "Attack"}
    )
    results["predicted_class"] = results["predicted_label"].map(
        {0: "Normal", 1: "Attack"}
    )
    results.to_csv(PREDICTIONS_FILE, index=False)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nSaved files:")
    print(f"Model:      {MODEL_FILE}")
    print(f"Metrics:    {METRICS_FILE}")
    print(f"Predictions:{PREDICTIONS_FILE}")
    print("\nTraining and evaluation finished.")

if __name__ == "__main__":
    main()
