import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

from src.config import (
    TRAIN_FILE, TEST_FILE, MODEL_FILE, OUTPUT_DIR, METRICS_FILE,
    PREDICTIONS_FILE, RANDOM_STATE
)
from src.data import load_csv, get_labels, make_features

def train_and_evaluate():
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Dataset files not found. Place UNSW_NB15_training-set.csv and "
            "UNSW_NB15_testing-set.csv in data/raw/. See README.md."
        )

    train_df = load_csv(TRAIN_FILE)
    test_df = load_csv(TEST_FILE)
    y_train_raw, train_label_col = get_labels(train_df)
    y_test_raw, test_label_col = get_labels(test_df)

    # Only rows with known labels are usable.
    train_mask = y_train_raw.notna()
    test_mask = y_test_raw.notna()
    train_df, y_train = train_df.loc[train_mask].copy(), y_train_raw.loc[train_mask]
    test_df, y_test = test_df.loc[test_mask].copy(), y_test_raw.loc[test_mask]

    X_train = make_features(train_df, train_label_col)
    X_test = make_features(test_df, test_label_col)

    # Align test features to training features. Missing test columns become NaN
    # and are imputed; extra test columns are ignored.
    X_test = X_test.reindex(columns=X_train.columns)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2
        ))
    ])
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Numeric features: {len(X_train.columns)}")
    print("Training Random Forest...")
    pipeline.fit(X_train, y_train.astype(str))
    predictions = pipeline.predict(X_test)

    metrics = {
        "dataset": "UNSW-NB15 (official train/test split)",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(len(X_train.columns)),
        "features": list(X_train.columns),
        "accuracy": float(accuracy_score(y_test.astype(str), predictions)),
        "precision_attack": float(precision_score(
            y_test.astype(str), predictions, pos_label="ATTACK", zero_division=0
        )),
        "recall_attack": float(recall_score(
            y_test.astype(str), predictions, pos_label="ATTACK", zero_division=0
        )),
        "f1_attack": float(f1_score(
            y_test.astype(str), predictions, pos_label="ATTACK", zero_division=0
        )),
        "classification_report": classification_report(
            y_test.astype(str), predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": ["BENIGN", "ATTACK"],
        "confusion_matrix": confusion_matrix(
            y_test.astype(str), predictions, labels=["BENIGN", "ATTACK"]
        ).tolist()
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": list(X_train.columns)}, MODEL_FILE)

    results = test_df.copy()
    results["actual_label"] = y_test.astype(str).values
    results["predicted_label"] = predictions
    results["is_suspicious"] = results["predicted_label"].eq("ATTACK")
    results["risk_level"] = results["predicted_label"].map({
        "BENIGN": "LOW", "ATTACK": "HIGH"
    })
    results.to_csv(PREDICTIONS_FILE, index=False)

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation results (held-out test set)")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Attack precision:  {metrics['precision_attack']:.4f}")
    print(f"Attack recall:     {metrics['recall_attack']:.4f}")
    print(f"Attack F1 score:   {metrics['f1_attack']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test.astype(str), predictions, zero_division=0))
    print(f"Saved model:       {MODEL_FILE}")
    print(f"Saved metrics:     {METRICS_FILE}")
    print(f"Saved predictions: {PREDICTIONS_FILE}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
