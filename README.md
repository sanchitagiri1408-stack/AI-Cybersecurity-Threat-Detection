# AI-Powered Cybersecurity Threat Detection System

A defensive machine-learning prototype that classifies labeled network-flow records as **BENIGN** or **ATTACK**, evaluates detection performance, generates reviewable alerts, and creates visual reports.

> **Scope:** This is an educational research prototype using offline public datasets. It does not monitor a live network, block traffic, confirm incidents, or replace a production IDS/SIEM/SOC workflow.

## Problem statement

Security teams receive large volumes of network telemetry. This project demonstrates how supervised machine learning can help prioritize records that resemble known attack patterns while measuring false positives and missed attacks.

## Features

- CSV ingestion and basic data validation
- Numeric feature selection and missing-value imputation
- Random Forest binary classification
- Accuracy, attack precision, attack recall, F1, classification report
- Confusion matrix and prediction visualizations
- CSV prediction output and suspicious-record alert export
- Offline sample-based pipeline demonstration
- Reproducible command-line workflow

## Architecture

```text
Public labeled network-flow CSV
             |
             v
      Data validation
             |
             v
  Numeric feature selection
             |
             v
 Median imputation + Random Forest
             |
             v
 Held-out test evaluation
             |
             v
 Predictions --> CSV alerts --> Graphs / reports
```

## Dataset

Recommended dataset: [UNSW-NB15 official page](https://research.unsw.edu.au/projects/unsw-nb15-dataset). It contains labeled network traffic and attack categories. Follow its academic citation requirements and dataset terms.

Download the official train and test CSV files:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Place both under `data/raw/`. The expected binary target column is `label` (0 = normal, 1 = attack in the official split). Verify this against the dataset documentation before use. Raw datasets are excluded from Git by `.gitignore`.

## Requirements

- Python 3.11 or 3.12 recommended
- Git
- Internet access only to obtain the dataset and Python packages
- No GPU required

## Installation — Windows PowerShell

```powershell
cd path\to\AI-Cybersecurity-Threat-Detection
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py` is unavailable, use an installed Python executable, for example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the project

1. Put both dataset CSVs in `data/raw/`.
2. Train and evaluate:

   ```powershell
   python main.py train
   ```

3. Create graphs:

   ```powershell
   python main.py visualize
   ```

4. Create an offline sample file:

   ```powershell
   python main.py simulate --rows 100
   ```

5. Run predictions on that file:

   ```powershell
   python main.py detect outputs/simulation_input.csv
   ```

## Outputs

Files are generated in `outputs/`:
- `metrics.json` — test metrics and confusion matrix counts
- `test_predictions.csv` — actual and predicted test labels
- `alerts.csv` — records predicted as attacks
- `detection_results.csv` — predictions for an input file
- `confusion_matrix.png`
- `prediction_counts.png`
- `prediction_timeline_sample.png`
- `simulation_input.csv`

The model is saved to `models/threat_detector.joblib`.

## Evaluation

Report accuracy, attack precision, attack recall, and attack F1. In security work, recall helps measure how many labeled attacks were detected, while precision helps measure how many attack alerts were correct. Also inspect false positives and false negatives; accuracy alone can be misleading with imbalanced data.

Do not copy expected scores from another project. Run this code and report the values produced on your own dataset and environment. Dataset artifacts, duplicate flows, class imbalance, and distribution shifts can make benchmark results look better or worse than deployment performance.

## Simulation limitations

The `simulate` command samples records from the held-out dataset. It is an **offline pipeline demonstration**, not live network capture and not a synthetic attack generator. Its predictions are not guaranteed to be attacks. To demonstrate controlled synthetic patterns, add a separately documented generator only after reviewing feature semantics and validating it against held-out data.

## Repository structure

```text
AI-Cybersecurity-Threat-Detection/
├── data/raw/          # Downloaded data; not committed
├── src/               # Data, training, detection, visualization
├── models/             # Generated model; not committed by default
├── outputs/            # Generated metrics, alerts, graphs
├── docs/               # Additional design notes/screenshots
├── images/             # Curated README screenshots
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Screenshots

After running the project, copy selected graphs to `images/` and embed them here:

```markdown
![Confusion matrix](images/confusion_matrix.png)
![Prediction counts](images/prediction_counts.png)
![Detection results](images/detection_results.png)
```

Only include screenshots and results you actually generated. Do not present sample or synthetic results as real-world detections.

## Security and responsible use

- Use the system for defensive learning and authorized analysis only.
- This application does not capture packets or interact with a live network.
- Treat alerts as leads for investigation, not confirmed incidents.
- Do not upload private traffic, credentials, or sensitive logs to a public repository.
- Pickle/joblib model files should only be loaded from trusted sources.

## Future improvements

- Compare Logistic Regression, Random Forest, and Isolation Forest
- Add multiclass attack-family prediction
- Add Streamlit dashboard and FastAPI endpoint
- Add threshold tuning and precision-recall curves
- Add data-drift monitoring and model versioning
- Add tests, CI, Docker, and an analyst feedback loop

## Learning outcomes

Python data processing, supervised ML, evaluation metrics, network intrusion detection concepts, defensive alerting, reproducible experiments, visualization, and GitHub project documentation.

## Citation

Cite the UNSW-NB15 dataset and its associated research publications as directed by the official dataset page:
https://research.unsw.edu.au/projects/unsw-nb15-dataset
