# Architecture notes

1. Data source: documented public labeled flow CSVs.
2. Validation: normalize headers, drop duplicate headers, convert infinite values to missing.
3. Labels: map documented binary labels to BENIGN and ATTACK.
4. Features: select numeric traffic features and exclude labels and common identifiers.
5. Pipeline: median imputation followed by Random Forest classifier.
6. Evaluation: use the dataset's official held-out test CSV; report class-sensitive metrics.
7. Detection: reuse the same saved pipeline and exact training feature columns.
8. Alerting: export rows predicted as ATTACK for analyst review.
9. Visualization: confusion matrix, predicted class counts, and sample record plot.

Limitations: this first version is offline, binary, and dataset-specific. It does not capture packets, block traffic, or establish that a predicted record is a confirmed incident.
