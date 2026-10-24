# FraudGuard — Temporal Fraud Detection

**Imbalanced classification with chronological validation and an explicit review budget.**

FraudGuard compares Logistic Regression and Random Forest for transaction
classification. The portfolio benchmark uses the ULB credit-card schema and
separates model selection, threshold selection and final evaluation.

## Benchmark protocol

- Split **distinct timestamps** into 60% training / 20% validation / 20% test.
  Transactions with the same timestamp stay together; future rows never fit scaling.
- Train on V1–V28 and Amount. Time is used only to define the split.
- Choose the model by **validation average precision**; choose its threshold by
  validation F1. Freeze both and evaluate the selected model once on test.
- Report average precision (stepwise PR area), ROC-AUC, precision/recall at the
  frozen threshold, actual threshold workload, and precision/recall among the
  highest-scoring **1%** of a batch.
- Preserve full class prevalence: no test downsampling, oversampling or SMOTE.

The top-1% metric uses an exact review capacity and stable label-independent tie
breaking. A frozen threshold may produce a different workload as the distribution
changes; the report shows both quantities.

## Reproduce the ULB experiment

The minimal benchmark environment is separate from the optional Streamlit app.
Use Python 3.12 for the pinned numerical environment:

```bash
python3.12 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
pip install -r requirements-benchmark.txt
python -m scripts.download_creditcard
python -m scripts.benchmark --data data/raw/creditcard.csv --output reports/creditcard.json
python -m pytest -o addopts='' -q
```

Data source: [official TensorFlow tutorial and mirror](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data),
with attribution to [Worldline / ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Raw transactions are excluded from Git. The downloader records the source URL,
file size and SHA-256; consult original data terms separately from the code license.

The benchmark expects the original `Time,V1..V28,Amount,Class` schema and rejects
missing, infinite or nonbinary inputs. It does not write a model over the app's
artifact: its numerical features differ from the interactive form.

## Measured result

Completed on the original **284,807 transactions / 492 fraud cases**. Logistic
Regression was selected by validation average precision. On the chronological
held-out test (60,140 transactions / 77 frauds):

| Metric | Selected Logistic Regression |
| --- | ---: |
| Average precision | 0.7510 |
| ROC-AUC | 0.9825 |
| Precision at validation-selected threshold | 0.9455 |
| Recall at validation-selected threshold | 0.6753 |
| Recall within a top-1% review budget | 0.8701 |
| Precision within that review budget | 0.1113 |

The top-1% rule reviewed 602 transactions and found 67 fraud cases. The frozen
validation threshold reviewed approximately 0.0915% of test transactions. These
are different workloads. The class-weighted model's scores are not calibrated
fraud probabilities.

[Full measured report, split ranges and provenance](reports/creditcard.json) ·
[Source manifest](data/creditcard_source.json). Values are rounded only here;
the JSON retains full precision. No confidence interval or superiority claim is
made from this single historical split.

## Evidence and limits

The benchmark emits a versioned JSON report with split times, class counts,
validation comparisons, final metrics, EDA summary, data/source hashes and library
versions. Reported results belong only to the exact file and protocol recorded
there. Dataset-specific results are linked below when a complete run is available.

Two days of historical transactions are a small time window. Card/entity IDs are
unavailable for group holdout. The anonymous PCA features were created upstream;
their fitting provenance cannot be audited here. Neither AP nor a review-budget
metric is a deployment or financial impact estimate. Rank-based cohort scoring
and calibrated fraud probabilities are different concepts.

## Existing application

A separate generic-transaction workflow remains available:

```bash
python -m pip install -e '.[dev]'
python -m scripts.train --data transactions.csv --model both
python -m streamlit run app/app.py
```

Its CSV contains `amount`, `transaction_type`, `device_type`, `transaction_time`
and binary `isFraud`. It uses the original stratified split and fixed app threshold
of 0.5. Its numbers should not be confused with the chronological ULB benchmark.
Additional training features must also be supplied at inference. Missing-value
imputation is not implemented for that application.

## Repository

- `scripts/benchmark.py`: chronological experiment and review-budget metrics.
- `fraudguard/`: preprocessing, models, scoring and data-split helpers.
- `notebooks/`: executable EDA and benchmark walkthroughs, without embedded data.
- `app/`: optional Streamlit demo.
- `tests/`: threshold, schema, split and model regression checks.

[MIT code license](LICENSE). Dataset rights remain with their respective publishers.
