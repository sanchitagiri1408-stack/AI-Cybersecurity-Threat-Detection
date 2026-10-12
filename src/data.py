import numpy as np
import pandas as pd
from src.config import LABEL_COLUMN

DROP_COLUMNS = {
    "label", "attack_cat", "attack_category", "class", "target",
    "id", "srcip", "dstip", "stime", "ltime", "timestamp"
}

def load_csv(path):
    """Read a CSV and normalize column names/invalid numeric values."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    # Remove accidental duplicate column names (some flow datasets contain them).
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def get_labels(df, label_column=LABEL_COLUMN):
    """Convert documented 0/1 labels or benign/attack text labels to binary labels."""
    lookup = {c.lower(): c for c in df.columns}
    actual = lookup.get(label_column.lower())
    if actual is None:
        candidates = ["label", "class", "target", "attack_cat", "class_label"]
        actual = next((lookup[c] for c in candidates if c in lookup), None)
    if actual is None:
        raise ValueError(
            f"Could not find a label column. Available columns: {list(df.columns)}. "
            "Set LABEL_COLUMN in src/config.py to the correct column."
        )

    raw = df[actual]
    if pd.api.types.is_numeric_dtype(raw):
        values = pd.to_numeric(raw, errors="coerce")
        unique = set(values.dropna().unique().tolist())
        if unique.issubset({0, 1}):
            y = values.astype("Int64")
            return y.map({0: "BENIGN", 1: "ATTACK"}), actual
        # Numeric multiclass labels: 0 is assumed benign only if documented that way.
        raise ValueError(
            f"Numeric label column {actual!r} has values {sorted(unique)[:20]}. "
            "Map labels explicitly using the dataset documentation."
        )

    text = raw.astype("string").str.strip().str.lower()
    benign_names = {"benign", "normal", "normal traffic", "0"}
    y = text.map(lambda value: "BENIGN" if value in benign_names else (
        "ATTACK" if pd.notna(value) and value != "" else pd.NA
    ))
    return y, actual

def make_features(df, label_column):
    """Select numeric features only; remove target/identifier columns to reduce leakage."""
    excluded = {c.lower() for c in DROP_COLUMNS}
    excluded.add(label_column.lower())
    candidates = [
        c for c in df.columns
        if c.lower() not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not candidates:
        raise ValueError(
            "No numeric features found. Check the dataset format and label column."
        )
    X = df[candidates].apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Remove columns entirely missing in this split.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("All selected feature columns are empty.")
    return X
