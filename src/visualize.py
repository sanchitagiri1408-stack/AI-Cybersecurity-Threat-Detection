import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import OUTPUT_DIR, METRICS_FILE, PREDICTIONS_FILE

def create_graphs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not METRICS_FILE.exists() or not PREDICTIONS_FILE.exists():
        raise FileNotFoundError("Run training first: python main.py train")
    with open(METRICS_FILE, encoding="utf-8") as f:
        metrics = json.load(f)

    sns.set_theme()
    cm = metrics["confusion_matrix"]
    labels = metrics["confusion_matrix_labels"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Held-out Test Set")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    df = pd.read_csv(PREDICTIONS_FILE)
    counts = df["predicted_label"].value_counts()
    plt.figure(figsize=(7, 5))
    counts.reindex(["BENIGN", "ATTACK"], fill_value=0).plot(kind="bar")
    plt.title("Predicted Traffic Categories")
    plt.xlabel("Predicted category")
    plt.ylabel("Number of records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_counts.png", dpi=160)
    plt.close()

    # Plot a small sample to keep the chart readable on large datasets.
    sample = df.head(500).copy()
    sample["record_number"] = range(1, len(sample) + 1)
    sample["suspicious_numeric"] = sample["predicted_label"].eq("ATTACK").astype(int)
    plt.figure(figsize=(10, 4))
    plt.scatter(sample["record_number"], sample["suspicious_numeric"], alpha=0.6)
    plt.yticks([0, 1], ["BENIGN prediction", "ATTACK prediction"])
    plt.title("Model Predictions for First 500 Test Records")
    plt.xlabel("Record number in test CSV")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_timeline_sample.png", dpi=160)
    plt.close()

    print("Graphs saved in outputs/: confusion_matrix.png, prediction_counts.png, prediction_timeline_sample.png")

if __name__ == "__main__":
    create_graphs()
