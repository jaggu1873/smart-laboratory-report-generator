# Hydraulic pump-leakage model comparison

## Result and scope

Comparison ID: `hydraulic-compare-20261021-2104d`

Earlier implementation trial IDs from this same task
(`hydraulic-compare-20261021-2104`,
`hydraulic-compare-20261021-2104b`, and
`hydraulic-compare-20261021-2104c`) were preserved rather than overwritten.
They are superseded: the first did not include the Dummy in blocked-fold
selection, the second had an incorrectly tuned Dummy threshold diagnostic,
and the third preceded the saved in-sample metric fields. Use only the final
`...-2104d` outputs for submission.

This run compared a majority `DummyClassifier`, Logistic Regression, Random
Forest, and `HistGradientBoostingClassifier` on the same 112 physical-sensor
cycle-summary features and the same fixed cycle partitions. All artifacts are
isolated from the historical `hydraulic-20261021-a` and `hydraulic-20261021-b`
runs. The raw UCI source was not modified.

**Random Forest is the top candidate for further investigation under the
predeclared validation ranking, not a validated or reliable model.** The
chronological validation is condition-shifted, scheduled cycles are not
independent runs, and no run/rig identifiers are available. The unusually
strong validation/test ranking results cannot establish independent-run
generalization. No candidate is claimed to be production-ready or
scientifically validated.

Positive class: `1` means pump leakage present, mapped from original pump
grades 1 or 2. Label `0` is original grade 0.

## Split and procedures

The fixed chronological partition remains unchanged:

| Split | Source cycle IDs | N | Class 0 | Class 1 |
|---|---:|---:|---:|---:|
| Train | 1–1,483 | 1,483 | 827 | 656 |
| Validation | 1,604–1,814 | 211 | 122 | 89 |
| Test | 1,935–2,205 | 271 | 107 | 164 |

The existing 60-cycle boundary purges (240 cycles total) remain in place.
Four additional expanding forward blocks, all inside the nominal training
partition, were used for per-family hyperparameter selection. Each fold's
training data ends 60 cycles before its validation block:

| Fold | Fold training cycle IDs | Fold validation cycle IDs | Validation class 0 / 1 |
|---|---:|---:|---:|
| 1 | 1–340 | 401–700 | 111 / 189 |
| 2 | 1–640 | 701–950 | 242 / 8 |
| 3 | 1–890 | 951–1,200 | 94 / 156 |
| 4 | 1–1,140 | 1,201–1,483 | 119 / 164 |

These are temporal blocks, **not independent experimental runs**. Hyperparameter
selection used only these blocks within training. Final candidate ranking
used the mean of blocked-CV balanced accuracy and fixed-validation balanced
accuracy at threshold 0.5; ties favor lower blocked-fold standard deviation,
then validation ROC-AUC. Validation-tuned thresholds are reported as
diagnostics and do not affect model-family ranking. Test results were
calculated only after the candidates, parameters, and validation thresholds
were frozen. The CV values are tuning/selection evidence, not unbiased
generalization estimates.

StandardScaler is fitted within each Logistic Regression Pipeline on that
fold's training rows and then on the full designated training partition for
the final model. Random Forest and HistGradientBoosting do not require
scaling; each is still encapsulated in a scikit-learn Pipeline. All use seed
42. No profile target, virtual channel, cycle ID, row order, or split label is
an input feature.

Grid:

- Logistic Regression: `C = 0.01, 0.1, 1, 10`; class-weight balanced.
- Random Forest: 200 trees, square-root feature sampling, balanced-subsample
  weights; depth `8` or unlimited and minimum leaf size `5` or `15`.
- HistGradientBoosting: balanced class weights, no internal random early
  stopping; 100 or 200 iterations, learning rate `0.05` or `0.1`, 15 leaf
  nodes.
- Dummy baseline: most-frequent class, no tuning, threshold 0.5.

For probabilistic candidates, the validation diagnostic threshold maximizes
validation balanced accuracy, then F1, with ties favoring the threshold
closest to 0.5. Probabilities are uncalibrated scores. The 0.5 results below
are the primary threshold comparison; tuned-threshold results are separate.

## Selection evidence

| Candidate | Selected parameters | Blocked-CV mean BA (SD; minimum) | Validation BA at 0.5 | Validation ROC-AUC | Ranking score |
|---|---|---:|---:|---:|---:|
| Dummy baseline | none | 0.5000 (0.0000; 0.5000) | 0.5000 | 0.5000 | 0.5000 |
| Logistic Regression | `C=0.01` | 0.7244 (0.2256; 0.4979) | 0.5000 | 0.4961 | 0.6122 |
| Random Forest | unlimited depth, `min_samples_leaf=5` | 0.9120 (0.0894; 0.8013) | 0.8202 | 1.0000 | **0.8661** |
| HistGradientBoosting | 100 iterations, learning rate 0.05, 15 leaves | 0.9185 (0.1078; 0.7372) | 0.7809 | 1.0000 | 0.8497 |

Random Forest ranks first by the declared combined score, with a more
consistent minimum/SD across the four blocks than HistGradientBoosting. Its
blocked CV and validation still share the same scheduled single-rig source;
this ranking only identifies the most promising candidate for independent
follow-up.

## Metrics for every candidate

Each cell below lists `accuracy / balanced accuracy / precision / recall /
F1 / ROC-AUC`, followed by the confusion matrix `[[TN, FP], [FN, TP]]`.
Validation and test class support is shown in the split table above.

### Fixed threshold 0.5 (primary comparison)

| Candidate | Validation metrics; confusion matrix | Test metrics; confusion matrix |
|---|---|---|
| Dummy baseline | `0.5782 / 0.5000 / 0 / 0 / 0 / 0.5000`; `[[122,0],[89,0]]` | `0.3948 / 0.5000 / 0 / 0 / 0 / 0.5000`; `[[107,0],[164,0]]` |
| Logistic Regression | `0.5782 / 0.5000 / 0 / 0 / 0 / 0.4961`; `[[122,0],[89,0]]` | `0.3948 / 0.5000 / 0 / 0 / 0 / 1.0000`; `[[107,0],[164,0]]` |
| Random Forest | `0.8483 / 0.8202 / 1.0000 / 0.6404 / 0.7808 / 1.0000`; `[[122,0],[32,57]]` | `0.8635 / 0.8872 / 1.0000 / 0.7744 / 0.8729 / 1.0000`; `[[107,0],[37,127]]` |
| HistGradientBoosting | `0.8152 / 0.7809 / 1.0000 / 0.5618 / 0.7194 / 1.0000`; `[[122,0],[39,50]]` | `0.7159 / 0.7652 / 1.0000 / 0.5305 / 0.6932 / 1.0000`; `[[107,0],[77,87]]` |

### Training-partition metrics (in-sample)

These are fitted-row diagnostics, not generalization metrics. Each cell has
the same metric order and confusion-matrix format as above.

| Candidate | At threshold 0.5 | At validation-tuned threshold |
|---|---|---|
| Dummy baseline | `0.5577 / 0.5000 / 0 / 0 / 0 / 0.5000`; `[[827,0],[656,0]]` | Same, threshold 0.5 fixed |
| Logistic Regression | `0.9933 / 0.9930 / 0.9939 / 0.9909 / 0.9924 / 0.9977`; `[[823,4],[6,650]]` | `0.4423 / 0.5000 / 0.4423 / 1.0000 / 0.6134 / 0.9977`; `[[0,827],[0,656]]` |
| Random Forest | `0.9993 / 0.9994 / 0.9985 / 1.0000 / 0.9992 / 1.0000`; `[[826,1],[0,656]]` | `0.9960 / 0.9964 / 0.9909 / 1.0000 / 0.9954 / 1.0000`; `[[821,6],[0,656]]` |
| HistGradientBoosting | `1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000`; `[[827,0],[0,656]]` | `0.9858 / 0.9873 / 0.9690 / 1.0000 / 0.9842 / 1.0000`; `[[806,21],[0,656]]` |

### Validation-tuned threshold diagnostics

| Candidate | Validation threshold | Validation metrics; confusion matrix | Same threshold on test; confusion matrix |
|---|---:|---|---|
| Dummy baseline | 0.5 (fixed) | `0.5782 / 0.5000 / 0 / 0 / 0 / 0.5000`; `[[122,0],[89,0]]` | `0.3948 / 0.5000 / 0 / 0 / 0 / 0.5000`; `[[107,0],[164,0]]` |
| Logistic Regression | `2.243348337792029e-19` | `0.6967 / 0.7377 / 0.5817 / 1.0000 / 0.7355 / 0.4961`; `[[58,64],[0,89]]` | `1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000`; `[[107,0],[0,164]]` |
| Random Forest | `0.13628977735799333` | `1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000`; `[[122,0],[0,89]]` | `1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000`; `[[107,0],[0,164]]` |
| HistGradientBoosting | `0.006492485888361937` | `1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000`; `[[122,0],[0,89]]` | `1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000 / 1.0000`; `[[107,0],[0,164]]` |

The perfect tuned-threshold validation and test scores are not independent
confirmation: these thresholds were optimized on one small validation block,
and both held-out segments come from the same scheduled experiment. Their
values are shown transparently rather than treated as reliable operating
points. Logistic Regression's test ROC-AUC is 1.0 even at threshold 0.5
because ROC-AUC evaluates ranking, while that cutoff classifies every test
cycle as negative.

## Regime and feature analysis

The validation and test intervals are not representative draws from training:

| Condition | Train | Validation | Test |
|---|---|---|---|
| Cooler condition | 3%: 732; 20%: 732; 100%: 19 | 100%: 211 | 100%: 271 |
| Accumulator pressure | 90: 485; 100: 266; 115: 266; 130: 466 | 90: 61; 115: 17; 130: 133 | 90: 142; 100: 129 |

Train pump grade counts by cooler condition show grades 1/2 only in the 3%
and 20% cooler stages; the 19 training cycles at 100% are all grade 0.
Validation and test contain all pump grades at 100% cooler condition. Thus,
the fitted models have very limited training support for leakage-present
cycles at that cooler setting. Validation's first 71 cycles are all negative,
the next 70 all positive, and only the last 70 mix labels.

The largest standardized mean shifts relative to training are in PS4
summaries: validation PS4 minimum `+351.6` training SD and test `+465.9`
training SD; PS4 p25 shifts are `+135.2` and `+176.0` SD. These huge values
occur because the channel is almost constant in training and differs in later
cooler/accumulator regimes. The selected Random Forest's largest impurity
importance is FS1 flow summaries (`FS1__p75`, median, mean, RMS, and p25),
followed by EPS1 and PS3 features. Flow response is physically relevant to
pump leakage, but importance does not establish causal or regime-independent
leakage detection. The scheduled experiment entangles pump grade, operating
condition, and row chronology; this dataset lacks identifiers needed to
separate those effects.

The model classifies summaries from the **completed labeled cycle**. This is
retrospective classification, not a forecast or early-warning model. A row
index is not a timestamp, and the data do not identify independent runs or
rigs.

## Report integration

Comparison-linked reports record candidate name/version, parameters, seed,
dataset DOI/source, cycle row index, positive-class definition, validation
method, selected threshold, held-out evaluation metrics, and generalization
limitations. Candidate probability values are omitted because no calibration
or meaningful confidence interpretation is established. Reports describe
retrospective classification rather than early warning.

Example report for cycle 2004:

`reports/hydraulic/comparisons/hydraulic-compare-20261021-2104d/cycle_2004_random_forest_report.json`

## Reproduction

Run the controlled comparison with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m src.hydraulic.comparison `
  --comparison-id hydraulic-compare-reproduction-01 `
  --random-state 42
```

The command refuses to reuse an existing comparison ID. It writes models to
`models/hydraulic/comparisons/<comparison-id>/`, the complete metrics and
per-cycle test predictions to `metrics/hydraulic/comparisons/<comparison-id>/`,
and reserves a corresponding reports directory.

Generate a report for the selected candidate:

```powershell
.\.venv\Scripts\python.exe -m src.hydraulic.report `
  --comparison-id hydraulic-compare-reproduction-01 `
  --candidate random_forest `
  --cycle-id 2004
```

The report command refuses to overwrite a report with the same cycle and
candidate filename. It records the cycle's split membership, the selected
validation threshold, split metrics, source provenance, and limitations
including the condition shift, sequential dependence, and lack of independent
run/rig identifiers. Pass `--include-ground-truth` only for a completed cycle
when an explicitly retrospective label is appropriate; the target is never a
model input. Per-cycle probabilities are omitted because they are uncalibrated
model scores. No packages were installed or dependency versions changed for
this comparison.

## MLflow tracking

The comparison command logs one run per candidate to the
`hydraulic-condition-monitoring` experiment after saving the comparison
artifacts. Each run stores validation metrics separately from locked test
metrics, alongside the model artifact, parameters, split distributions,
dataset provenance, and limitations. Existing saved comparisons can be logged
without retraining:

```powershell
.\.venv\Scripts\python.exe -m src.hydraulic.mlflow_tracking `
  --comparison-id hydraulic-compare-20261021-2104d
```

See the [hydraulic MLflow guide](./hydraulic_mlflow.md) for tracking URI
configuration, run deduplication, inspection, and fresh-clone local tracking.
