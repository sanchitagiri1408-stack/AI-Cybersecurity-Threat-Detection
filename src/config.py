from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"
MODEL_FILE = MODEL_DIR / "threat_detector.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"
ALERTS_FILE = OUTPUT_DIR / "alerts.csv"

# Set to "label" for official UNSW-NB15 train/test files.
# For another dataset, change this to its documented label column.
LABEL_COLUMN = "label"
RANDOM_STATE = 42
