# Model Card: Smart Laboratory Report Generator Quality Classifier

## Model details

- **Model:** scikit-learn `RandomForestClassifier`
- **MLflow run:** `random-forest-200-depth-10`
- **Registry:** `smart-lab-quality-classifier`, version 1 (READY)
- **Hyperparameters:** `n_estimators=200`, `max_depth=10`, `random_state=42`,
  `n_jobs=-1`
- **Training data:** `data/processed/train.csv`

The model and run selection are recorded in
[`models/best_run.json`](models/best_run.json), and the candidate settings are
defined in [`params.yaml`](params.yaml).

## Intended use

The classifier supports a lab-style report by adding a predicted wine-quality
label based on the supplied measurements. It is a demonstration of the
project's sklearn report backend, not a replacement for laboratory judgement,
validated measurement procedures, or expert interpretation.

## Dataset

- **Source:** UCI Wine Quality red-wine dataset
- **Rows before deduplication:** 1,599
- **Rows after deduplication:** 1,359
- **Label:** `good = 1` when `quality >= 6`; otherwise `good = 0`
- **Inputs:** the 11 physicochemical feature columns; the original `quality`
  column is used only to derive the label.

The configured threshold is in [`params.yaml`](params.yaml). The prepared
training and test feature columns are in `data/processed/train.csv` and
`data/processed/test.csv`.

## Training procedure

The data is split into training and test partitions using a stratified 80/20
split and random state 42. `StandardScaler` is fit on the complete training
partition and applied to both partitions. The training partition is evaluated
with shuffled, stratified 5-fold cross-validation (random state 42), and the
selected candidate is the one with the highest mean CV F1. The selected
classifier is then fit on the scaled training partition.

## Evaluation results

Held-out test metrics are from [`metrics/metrics.json`](metrics/metrics.json):

| Metric | Test result |
|---|---:|
| Accuracy | 0.7573529411764706 |
| Precision | 0.7910447761194029 |
| Recall | 0.7361111111111112 |
| F1 | 0.762589928057554 |
| ROC-AUC | 0.8369683159722222 |

## Limitations

- Wine Quality is a proxy for laboratory data; it is not a dataset of
  laboratory reports or a validation of real laboratory measurements.
- The dataset is small.
- Mapping the ordinal quality score to a binary label hides quality
  gradations.
- The scaler is fit on the full training set before cross-validation, so CV
  scores may be slightly optimistic.
- The configured DVC remote is local, so another machine needs an accessible
  shared remote to retrieve DVC-managed artifacts.

## Reproduction

From the repository root, install the pinned dependencies and reproduce the
DVC pipeline:

```bash
python -m pip install -r requirements.txt
dvc pull
dvc repro
```

The DVC remote is local; configure an accessible shared remote before running
`dvc pull` on another machine. To generate and validate the example report,
run:

```bash
python scripts/demo_report.py
```
