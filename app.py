from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# Project paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "logistic_regression_model.joblib"
METRICS_PATH = ROOT / "outputs" / "evaluation_metrics.json"
PREDICTIONS_PATH = ROOT / "outputs" / "test_predictions.csv"

st.set_page_config(
    page_title="CyberShield | Threat Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ CyberShield")
st.subheader("AI-Powered Cybersecurity Threat Detection")
st.caption(
    "Machine-learning demonstration using the UNSW-NB15 network "
    "intrusion detection dataset."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train_model.py` first.")
    st.stop()

if not METRICS_PATH.exists() or not PREDICTIONS_PATH.exists():
    st.error("Evaluation files are missing. Please train the model again.")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_metrics(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_data
def load_predictions(path):
    return pd.read_csv(path)

model = load_model(str(MODEL_PATH))
metrics = load_metrics(str(METRICS_PATH))
predictions = load_predictions(str(PREDICTIONS_PATH))

# Sidebar
st.sidebar.header("Dashboard Menu")
page = st.sidebar.radio(
    "Choose a section",
    ["Overview", "Prediction Results", "Model Performance"],
)

# Summary metrics
total = len(predictions)
attacks_detected = int((predictions["predicted_label"] == 1).sum())
normal_predicted = int((predictions["predicted_label"] == 0).sum())
false_positives = int(
    (
        (predictions["label"] == 0)
        & (predictions["predicted_label"] == 1)
    ).sum()
)

if page == "Overview":
    st.header("Security Overview")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Test Records", f"{total:,}")
    col2.metric("Predicted Attacks", f"{attacks_detected:,}")
    col3.metric("Predicted Normal", f"{normal_predicted:,}")
    col4.metric("False Alarms", f"{false_positives:,}")

    st.divider()
    st.subheader("Prediction Distribution")

    chart_data = pd.DataFrame({
        "Classification": ["Attack", "Normal"],
        "Records": [attacks_detected, normal_predicted],
    }).set_index("Classification")

    st.bar_chart(chart_data)

    st.info(
        "This dashboard analyzes saved predictions from the test dataset. "
        "It is a research/demo tool, not a live network monitoring system."
    )

elif page == "Prediction Results":
    st.header("Network Traffic Prediction Results")
    st.write(
        "Browse the model's predictions on the held-out test dataset."
    )

    filter_choice = st.selectbox(
        "Filter by predicted class",
        ["All", "Attack", "Normal", "Incorrect predictions"],
    )

    filtered = predictions.copy()

    if filter_choice == "Attack":
        filtered = filtered[filtered["predicted_label"] == 1]
    elif filter_choice == "Normal":
        filtered = filtered[filtered["predicted_label"] == 0]
    elif filter_choice == "Incorrect predictions":
        filtered = filtered[
            filtered["label"] != filtered["predicted_label"]
        ]

    st.write(f"Showing {len(filtered):,} records")
    st.dataframe(filtered.head(1000), use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered predictions",
        data=csv_bytes,
        file_name="filtered_predictions.csv",
        mime="text/csv",
    )

elif page == "Model Performance":
    st.header("Model Performance")
    st.caption("Evaluation on the separate UNSW-NB15 testing dataset.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Attack Precision", f"{metrics['precision']:.2%}")
    col3.metric("Attack Recall", f"{metrics['recall']:.2%}")
    col4.metric("Attack F1-score", f"{metrics['f1_score']:.2%}")

    st.subheader("Confusion Matrix")
    cm = metrics["confusion_matrix"]

    matrix_df = pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Attack"],
        columns=["Predicted Normal", "Predicted Attack"],
    )

    st.dataframe(matrix_df, use_container_width=True)

    st.subheader("Classification Report")
    report = metrics.get("classification_report", {})
    class_rows = []

    for class_name in ["Normal", "Attack"]:
        values = report.get(class_name, {})
        if values:
            class_rows.append({
                "Class": class_name,
                "Precision": values.get("precision", 0),
                "Recall": values.get("recall", 0),
                "F1-score": values.get("f1-score", 0),
                "Support": values.get("support", 0),
            })

    if class_rows:
        st.dataframe(
            pd.DataFrame(class_rows).set_index("Class"),
            use_container_width=True,
        )

st.divider()
st.caption(
    "CyberShield | Educational cybersecurity project | "
    "UNSW-NB15 dataset"
)