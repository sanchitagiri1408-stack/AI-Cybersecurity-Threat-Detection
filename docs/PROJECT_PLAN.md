# Project implementation and proof plan

## Phase 1 — Setup
Create the project directory, virtual environment, install requirements, and verify imports.
Proof: terminal screenshot of Python version and successful `pip install -r requirements.txt`.
Commit: `chore: add project scaffold and dependencies`

## Phase 2 — Dataset
Download the official UNSW-NB15 training and testing CSVs and place them in `data/raw/`.
Proof: file names, row/column counts, and a small redacted dataset preview.
Commit: `docs: document UNSW-NB15 dataset setup`

## Phase 3 — Cleaning and features
Run training and verify labels, numeric feature count, missing values, and feature alignment.
Proof: terminal output showing rows and feature count.
Commit: `feat: add dataset validation and feature preparation`

## Phase 4 — Model
Train Random Forest and save the model.
Proof: training log and generated model filename (do not need to publish the binary model).
Commit: `feat: train random forest intrusion detector`

## Phase 5 — Evaluation
Review precision, recall, F1, classification report, and confusion matrix.
Proof: metrics JSON and confusion matrix graph.
Commit: `feat: report held-out detection metrics`

## Phase 6 — Visualization
Generate prediction-count and sample prediction plots.
Proof: PNG files opened at readable resolution.
Commit: `feat: add threat detection visualizations`

## Phase 7 — Detection workflow
Generate an offline sample and run the saved model against it.
Proof: input CSV, detection results, and alert count. State clearly that this is an offline dataset replay.
Commit: `feat: add offline detection and alert export`

## Phase 8 — Documentation and GitHub
Complete README, dataset attribution, limitations, screenshots, and reproducible commands.
Proof: repository page and successful clean clone/run if possible.
Commit: `docs: finalize README and project evidence`

## Suggested 7-day schedule
Day 1: setup and scaffold.
Day 2: download and inspect dataset.
Day 3: cleaning and feature pipeline.
Day 4: train model.
Day 5: evaluation and confusion matrix.
Day 6: detection workflow and visualization.
Day 7: documentation, screenshots, GitHub upload, and clean rerun.

## Evidence checklist
- Dataset preview and shape
- Training log
- Precision/recall/F1 output
- Confusion matrix
- Prediction counts chart
- Detection results CSV preview
- Alert CSV preview
- GitHub repository page
- Short screen recording of the end-to-end workflow
