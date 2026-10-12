"""Create a safe, offline demonstration file by sampling dataset records.

This does NOT generate real network traffic and does NOT guarantee that modified
records will be classified as attacks. It is a pipeline smoke test only.
"""
import argparse
import pandas as pd
from src.config import TEST_FILE, OUTPUT_DIR
from src.data import load_csv, get_labels

def create_demo(output_path=None, rows=100):
    if not TEST_FILE.exists():
        raise FileNotFoundError("Place the UNSW-NB15 test CSV in data/raw/ first.")
    df = load_csv(TEST_FILE)
    labels, _ = get_labels(df)
    # Sample across existing labels for an offline demo.
    working = df.loc[labels.notna()].copy()
    demo = working.sample(n=min(rows, len(working)), random_state=42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = output_path or OUTPUT_DIR / "simulation_input.csv"
    demo.to_csv(destination, index=False)
    print(f"Created {len(demo)} offline sample records: {destination}")
    print("This is sampled historical data, not live traffic or a newly generated attack.")
    print("Use python main.py detect outputs/simulation_input.csv to run the detector.")
    return destination

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    create_demo(args.output, args.rows)
