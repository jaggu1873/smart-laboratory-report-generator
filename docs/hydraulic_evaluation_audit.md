# Hydraulic model evaluation audit

## Scope and disposition

This audit inspected both saved runs, their model metadata, split manifests,
saved features and profile labels, stored metrics, and the hydraulic training,
splitting, evaluation, and report code. The UCI archive and raw sensor files
were not changed. The existing model runs were not overwritten or deleted.

No target/profile/cycle/split field was found among the 112 model features.
The feature names are the eight cycle-local summaries for the 14 physical
sensor channels. The five profile targets, three virtual channels, cycle ID,
row order, and split membership are excluded from the estimator input.
Training code fits the scaler and estimator on training rows only; the saved
scaler means also match the training-feature means. The test partition is
used only after model and threshold selection in the original training flow.

Saved split membership was independently checked against the deterministic
split and 1-based cycle IDs. It correctly identifies train cycles 1–1,483,
validation cycles 1,604–1,814, test cycles 1,935–2,205, and 240 purged cycles.
Independent metric recomputation from the saved source labels and fitted
models exactly matches both stored metric files. The test scores in run B are
therefore reproducible under its recorded split and threshold; that does not
establish independent experimental validity or generalization.

## Findings on run `hydraulic-20261021-b`

Positive class is label 1: pump leakage present, mapped from original pump
grades 1 or 2. The original validation-selected threshold is
`1.995378783118002e-50`; it was not replaced during this audit.

At that threshold, validation confusion is `[[60, 62], [0, 89]]`, balanced
accuracy `0.7459`, F1 `0.7417`, and ROC-AUC `0.5305`. At a fixed 0.5 cutoff,
validation predicts all 211 cycles as negative (`[[122, 0], [89, 0]]`).
Test confusion at the recorded threshold is `[[107, 0], [0, 164]]`; accuracy,
balanced accuracy, precision, recall, F1, and ROC-AUC are each `1.0000`.
At a fixed 0.5 cutoff, test predicts every cycle as negative
(`[[107, 0], [164, 0]]`), despite the same test score ranking having ROC-AUC
1.0.

The small threshold is a symptom of severe score-scale shift, not a useful
calibrated probability boundary. Validation positive-class probabilities
range from about `2.20e-53` to `1.05e-2`; test probabilities range from about
`1.28e-53` to `2.75e-46`. The validation model decision function ranges from
`-121.25` to `-4.55`; on test it ranges from `-121.79` to `-104.91`. At the
recorded threshold, validation predicts 151/211 positive (all 89 true
positives and 62 false positives); the test ordering separates the labels
perfectly. At 0.5, all validation and test predictions are negative. The
model's probabilities are not calibrated, and displaying a tiny positive
probability with four decimal places previously rounded it to `0.0000`; report
formatting now preserves significant digits.

Balanced accuracy is a reasonable threshold objective for imbalanced classes,
but selecting its best value (then F1) among all empirical validation
thresholds on this one short, non-independent chronological block is not a
stable threshold-estimation procedure. The validation threshold is retained
only as a historical, validation-tuned decision rule. It must not be treated
as a pre-specified or transferable operating point.

## Chronology and distribution shift

Validation is not compositionally uniform: its first 71 cycles are all class
0, the next 70 are all positive, and the only portion containing both classes
is its final 70 cycles (51 class 0, 19 class 1). That final segment alone has
ROC-AUC 1.0 and balanced accuracy 1.0 at the stored cutoff; this is a
post-hoc diagnostic, not a new selection set.

The chronological partitions also cover different operating-condition
blocks:

| Profile condition | Train | Validation | Test |
|---|---|---|---|
| Cooler condition | 732 at 3%, 732 at 20%, 19 at 100% | 211 at 100% | 271 at 100% |
| Pump grades 0 / 1 / 2 | 827 / 328 / 328 | 122 / 41 / 48 | 107 / 82 / 82 |
| Accumulator pressure | 90: 485; 100: 266; 115: 266; 130: 466 | 90: 61; 115: 17; 130: 133 | 90: 142; 100: 129 |
| Stable flag 1 | 523/1,483 (35.3%) | 85/211 (40.3%) | 22/271 (8.1%) |

Pump grades repeat in long, scheduled blocks through the file (37 contiguous
grade runs), rather than independent randomized cycles. Training is mostly
from earlier cooler stages; both validation and test are exclusively from the
100% cooler stage. The test has only 90% and 100% accumulator-pressure
conditions, while validation has 90%, 115%, and 130%. For example, several
sensor-feature means shift by tens to hundreds of the training standard
deviations because those features are nearly constant during training. These
shifts explain why fitted logistic scores extrapolate to extremely negative
values in the later blocks and why one threshold behaves so differently
across the partitions.

All source matrices have the expected equal row count and the UCI
documentation's positional row-as-cycle convention is used. No timestamps,
run identifiers, or rig identifiers are present, so physical cross-file
cycle alignment and independence cannot be independently proven from IDs.
The one-hour purge reduces boundary adjacency but cannot turn these scheduled
cycles into independent experimental runs. The chronological test is not a
reliable estimate of generalization to new runs, rigs, or operating schedules.

Features use complete-cycle sensor summaries and targets describe the
condition during those cycles. This is retrospective, same-cycle
classification—not evidence of pre-cycle forecasting or early warning.
There is no direct feature leakage found, but the dataset alone does not
establish a prospectively usable prediction protocol.

## Run A and saved-result verification

Run `hydraulic-20261021-a` selected the majority DummyClassifier. Its saved
metrics likewise match independent recomputation; it predicts only class 0
and has test accuracy `0.3948`, balanced accuracy `0.5000`, and F1 `0`.
Its historical `metrics.json` omitted the `decision_threshold` field; the
evaluator verifies the 0.5 baseline/metadata thresholds rather than treating
that older schema omission as a metric mismatch.
Run B's persisted class labels, split membership, model feature order, scaler
statistics, and all split metrics were independently checked. The original
runs and reports remain unchanged.

The evaluator now fails if persisted split membership, split-manifest cycle
ranges, fitted training scaler means, or saved metrics disagree with
recomputation. It can also emit per-cycle predictions and probabilities to a
new collision-safe audit ID; see the audit output under
`metrics/hydraulic/audits/`.

## Submission recommendation

Do not present the perfect later-cycle test score as evidence of robust
generalization, and do not characterize the model as production-ready or
scientifically validated. Report the test score only with its exact locked
cycle range, observed class/condition schedule, selected threshold, threshold
sensitivity, and single-rig/no-run-ID limitations. Treat probability values
as uncalibrated scores. A defensible future assessment needs independent
experimental run IDs or an external dataset and a predeclared evaluation and
threshold policy; the current data cannot supply that evidence.

## Reproduction

From the repository root, the following command rechecks run B and writes
independently reproduced metrics plus per-cycle predictions to a new audit
directory. The audit ID must be new; existing output is never overwritten:

```powershell
.\.venv\Scripts\python.exe -m src.hydraulic.evaluate `
  --run-id hydraulic-20261021-b `
  --audit-id hydraulic-evaluation-audit-20261021-1334
```
