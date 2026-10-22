# Hydraulic condition-monitoring pipeline

This pipeline is separate from the retained legacy Wine Quality
implementation. Its explicit setup script downloads and verifies the official
UCI archive; model imports and tests do not download data. Pipeline commands
read the verified local extraction and create run-specific outputs without
changing Wine data or artifacts. Hydraulic comparisons use the dedicated
MLflow experiment documented in [hydraulic_mlflow.md](./hydraulic_mlflow.md).

## Dataset source and scientific attribution

- Dataset: UCI Machine Learning Repository, [Condition Monitoring of
  Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
- DOI: `10.24432/C5CW21`
- Local data directory:
  `data/inspection/uci-hydraulic-condition-monitoring/extracted/`
- Official archive SHA-256 checked on load:
  `24128aad2ee45eea7e6b63ebbd9992cdf25d0483a2cebefbfc13bc69079af1f2`
- Download: [official UCI ZIP archive](https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip)
- Local setup command: `python scripts/setup_hydraulic_dataset.py`
- License displayed on the official UCI page: **CC BY 4.0**. Retain
  attribution and identify modifications when sharing adapted material.
- Citation requested in the archive: Nikolai Helwig, Eliseo Pignanelli, and
  Andreas Schütze, “Condition Monitoring of a Complex Hydraulic System Using
  Multivariate Statistics,” IEEE I2MTC 2015, DOI
  [10.1109/I2MTC.2015.7151267](https://doi.org/10.1109/I2MTC.2015.7151267).

The code validates the source archive digest, audited extracted file byte
sizes, exact matrix dimensions, finite numeric values, all profile target
value distributions, and positional row-count alignment. It does not alter
raw files. The dataset contains 2,205 cycle rows; each cycle is documented as
60 seconds. There are no explicit timestamps, run IDs, or rig IDs in the
matrices. `cycle_id` is therefore a **1-based source row index**, not a
timestamp or independently verified physical cycle identifier.

## Sensor channels, units, and sampling

All channels are summarized at their native sampling rate; there is no
resampling or interpolation.

| Channel(s) | Quantity | Unit | Rate | Samples/cycle |
|---|---|---|---:|---:|
| PS1–PS6 | Pressure | bar | 100 Hz | 6,000 |
| EPS1 | Motor power | W | 100 Hz | 6,000 |
| FS1–FS2 | Volume flow | l/min | 10 Hz | 600 |
| TS1–TS4 | Temperature | C (Celsius) | 1 Hz | 60 |
| VS1 | Vibration | mm/s | 1 Hz | 60 |
| CE | Virtual cooling efficiency | % | 1 Hz | 60 |
| CP | Virtual cooling power | kW | 1 Hz | 60 |
| SE | Efficiency factor | % | 1 Hz | 60 |

The five `profile.txt` columns are, in order: cooler condition (%), valve
condition (%), pump leakage grade, accumulator pressure (bar), and stable
flag. They are cycle-level ground truth only. Pump leakage grades are kept
unchanged in `profile_targets.csv`; the binary target is derived separately:

- `0`: original grade `0`, no pump leakage.
- `1`: original grade `1` (weak) or `2` (severe), leakage present.

The observed original-grade counts are 1,221 / 492 / 492; binary counts are
1,221 no-leakage and 984 leakage-present cycles. The loader rejects a
distribution that differs from the audited source rather than silently
remapping labels.

## Feature engineering and inputs

For each of the 17 channels, the pipeline calculates eight independent
cycle-local summaries: arithmetic mean, population standard deviation
(`ddof=0`), minimum, maximum, root-mean-square, 25th percentile, median, and
75th percentile. This creates 136 derived columns and keeps each feature's
channel unit. Values must be finite; malformed rows, shape mismatches, missing
files, changed audited file sizes, and changed archive hashes fail explicitly.
No imputation is performed.

The initial classifier uses the 14 physical channels only (112 features).
Virtual channels CE, CP, and SE are summarized and retained in the feature
table for audit/report context, but excluded from model inputs to reduce
target-proxy risk. The five profile targets and `cycle_id` are never model
features. The source pump grade is retained in a separate targets file for
traceability.

## Split, leakage controls, and evaluation

The baseline uses a fixed forward-in-time 70/15/15 partition. At each
partition boundary it removes 60 cycles on both sides (a one-hour gap on
each side, given the documented 60-second cycles). For 2,205 cycles, the
retained 1-based inclusive cycle ranges are:

- train: 1–1,483;
- validation: 1,604–1,814;
- test: 1,935–2,205;
- 240 cycles are purged around the two boundaries.

The corresponding binary class counts (0, 1) are train `(827, 656)`,
validation `(122, 89)`, and test `(107, 164)`. Original grade counts are
recorded in the training split manifest. The one-hour purge is a buffer, not
a claim of statistical independence: pump-label lag-1 correlation observed
in the raw cycle order is approximately 0.978, and FS1 cycle-mean lag-1
correlation is approximately 0.9999. There are no independent run/site
identifiers, so the test estimates later-cycle behavior within this one rig
dataset, not cross-run or cross-rig generalization.

The first model is regularized, class-weighted logistic regression in a
scikit-learn `Pipeline` containing `StandardScaler`. The scaler and estimator
are fitted only on the training cycles. A most-frequent `DummyClassifier`
is fitted on the same training partition as a baseline. Candidate selection
uses validation balanced accuracy, then F1 and ROC-AUC. The fixed test set is
not used to select the model or threshold. For logistic regression, candidate
thresholds are compared on validation data only using balanced accuracy,
then F1; ties favor a threshold nearest 0.5. The Dummy baseline uses 0.5.
Exact feature-vector duplicates across retained partitions are rejected.

For each split, metrics include sample/class counts, confusion matrix,
accuracy, precision, recall, F1, balanced accuracy, and ROC-AUC when both
classes and a valid probability estimate are available. Training metrics are
in-sample and must not be interpreted as generalization performance.

### Observed run `hydraulic-20261021-b`

The command-completed run selected logistic regression. Its validation-selected
threshold was approximately `1.9954e-50`; validation ROC-AUC was `0.5305` and
balanced accuracy was `0.7459` (validation support: 122 class-0, 89 class-1).
On the later-cycle test partition (107 class-0, 164 class-1), the measured
confusion matrix was `[[107, 0], [0, 164]]`, with accuracy, precision, recall,
F1, balanced accuracy, and ROC-AUC each equal to `1.0`. The training-fitted
most-frequent baseline predicted class 0 throughout the test partition:
accuracy `0.3948`, balanced accuracy `0.5`, and F1/recall `0`.

The perfect test result is surprising given near-chance validation ROC-AUC
and the extreme validation threshold; it is recorded as observed, not treated
as proof of generalization. The strong serial dependence, condition scheduling,
single-rig scope, and threshold selection on one small validation block make
independent experimental replication essential before drawing scientific or
operational conclusions.

A subsequent audit independently reproduced the stored metrics and verified
the saved cycle membership and training-fitted scaler. It found no direct
target/cycle/split feature leakage, but confirmed strong operating-condition
shift and that a fixed 0.5 threshold predicts every validation and test cycle
as negative. The original validation-selected threshold remains recorded,
not endorsed as transferable or stable. Validation also contains long
single-class contiguous segments. See
[Hydraulic evaluation audit](./hydraulic_evaluation_audit.md) for the split
composition, score ranges, threshold sensitivity, and limits on interpretation.

A controlled comparison of Dummy, Logistic Regression, Random Forest, and
HistGradientBoosting is documented in
[hydraulic_model_comparison.md](./hydraulic_model_comparison.md). It uses
additional training-only forward blocks for tuning and reports a provisional
best-ranked candidate without claiming independent-run generalization.

## Environment and commands

Use Python 3.10 and install the pinned dependencies from `requirements.txt`.
The dataset setup script uses only the Python standard library. Run it once
from the repository root to download the archive, verify its SHA-256 and ZIP
contents, and create the extraction in the expected directory:

```powershell
python scripts/setup_hydraulic_dataset.py
```

The checksum is pinned from the verified local official UCI archive; UCI does
not publish a checksum. ZIP members are validated against audited names and
sizes, paths are checked before extraction, and archive CRCs are tested.
Existing extracted files are verified and never overwritten. The official
dataset is CC BY 4.0; cite UCI and DOI `10.24432/C5CW21` when reusing it. This
download is independent of DVC; no shared DVC remote is configured for the
legacy Wine workflow.

Run from the repository root. Choose a new, unique run ID for each training
run; commands refuse to overwrite an existing run directory.

```powershell
python -m unittest discover -s model\tests -v
python -m unittest discover -s tests -v

python -m src.hydraulic.train --run-id hydraulic-20261021-a
python -m src.hydraulic.evaluate --run-id hydraulic-20261021-a
python -m src.hydraulic.report --run-id hydraulic-20261021-a --cycle-id 2000
```

For a saved comparison, generate a report with
`python -m src.hydraulic.report --comparison-id hydraulic-compare-20261021-2104d --candidate random_forest --cycle-id 2006`.
The report is validated and written beneath
`reports/hydraulic/comparisons/<comparison-id>/`.

The default data path points to the local audited extraction. Override it
with `--data-dir` only when using the same verified extraction. To create an
explicit retrospective report that discloses its profile ground truth, pass
`--include-ground-truth`; this target is still never supplied to the model.

## Outputs and report interpretation

Each run has separate directories under:

- `data/processed/hydraulic/runs/<run-id>/`: cycle features, profile targets,
  and split membership, with targets saved separately from features.
- `models/hydraulic/runs/<run-id>/`: the fitted pipeline, model metadata, and
  split manifest.
- `metrics/hydraulic/runs/<run-id>/`: per-split metrics and Dummy baseline.
- `reports/hydraulic/runs/<run-id>/`: generated, schema-validated cycle
  report JSON files.

The report identifies the UCI dataset and citation, the 1-based row-index
cycle identifier, all channel summaries and units, feature definitions,
validation method, test metrics, provenance hash, and limitations. It
discloses ground truth only when explicitly requested. The model probability
is reported as an **uncalibrated model probability**, not uncertainty.
Predictions are model outputs, not direct laboratory measurements, certified
diagnoses, or maintenance instructions.

Hydraulic comparison runs are tracked in MLflow; the independent run-specific
training/evaluation files remain local reproducibility artifacts. This
pipeline does not add or alter DVC stages.
