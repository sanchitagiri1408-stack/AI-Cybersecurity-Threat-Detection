import joblib
import pandas as pd
from src.config import MODEL_FILE, OUTPUT_DIR, ALERTS_FILE
from src.data import load_csv

def predict_file(input_path, output_path=None):
    """Predict labels for an unlabeled CSV using the saved trained pipeline."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run: python main.py train")
    bundle = joblib.load(MODEL_FILE)
    model = bundle["pipeline"]
    features = bundle["features"]

    df = load_csv(input_path)
    # Ignore label columns in inference inputs; use only training feature columns.
    X = df.reindex(columns=features)
    predictions = model.predict(X)
    out = df.copy()
    out["predicted_label"] = predictions
    out["is_suspicious"] = out["predicted_label"].eq("ATTACK")
    out["risk_level"] = out["predicted_label"].map({
        "BENIGN": "LOW", "ATTACK": "HIGH"
    })
    out["alert_reason"] = out["predicted_label"].map({
        "BENIGN": "No attack pattern predicted by the model",
        "ATTACK": "Model classified this record as attack-like; analyst review required"
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = output_path or OUTPUT_DIR / "detection_results.csv"
    out.to_csv(destination, index=False)
    out[out["is_suspicious"]].to_csv(ALERTS_FILE, index=False)
    print(f"Records processed: {len(out):,}")
    print(f"Suspicious predictions: {int(out['is_suspicious'].sum()):,}")
    print(f"Results saved to: {destination}")
    print(f"Alerts saved to: {ALERTS_FILE}")
    print("These are model predictions, not confirmed incidents.")
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to a CSV with compatible numeric features")
    parser.add_argument("--output", help="Optional output CSV path")
    args = parser.parse_args()
    predict_file(args.input_csv, args.output)
