# FraudGuard

**Classical machine learning for imbalanced transaction classification.**

[Source](https://github.com/ssivitskii/FraudGuard) · [Issues](https://github.com/ssivitskii/FraudGuard/issues) · [Contributing](CONTRIBUTING.md)

## What is implemented

- Logistic Regression and Random Forest with class weighting.
- A scikit-learn pipeline with scaling and one-hot encoding.
- Hour and day-of-week features extracted from transaction timestamps.
- Stratified train / validation / test splits, model selection by validation F1 and test-set evaluation.
- Saved pipelines, command-line predictions and a Streamlit interface.

**Stack:** Python · pandas · scikit-learn · joblib · Streamlit

The implemented classifiers are in [`fraudguard/models.py`](fraudguard/models.py). XGBoost and a FastAPI service are not implemented.

## Run locally

Use Python 3.11:

```bash
git clone https://github.com/ssivitskii/FraudGuard.git
cd FraudGuard
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

### Prepare data

Provide your own `data/raw/transactions.csv`. For compatibility with the supplied CLI and Streamlit form, use this schema:

| Column | Expected value |
| --- | --- |
| `amount` | Numeric transaction amount |
| `transaction_type` | Transaction category |
| `device_type` | Device category |
| `transaction_time` | Valid timestamp |
| `isFraud` | Binary target: `0` or `1` |

Use complete records and enough examples of both classes for stratification. The pipeline does not impute missing values. Extra training columns become features and must also be supplied during inference; the existing form only supplies the four fields above.

Data and trained models are not included. PaySim and other public transaction datasets need explicit schema adaptation; they are not drop-in inputs for this form.

### Train and predict

```bash
python -m scripts.train --data transactions.csv --model both

python -m scripts.predict --amount 1250 --transaction_type transfer --device_type mobile --transaction_time "2026-01-15 14:30:00" --threshold 0.5 --json

python -m streamlit run app/app.py
```

Training selects between `logreg` and `forest` and writes `models/fraud_model.joblib` when a candidate improves the initial validation F1 of zero. Ensure the artifact exists before starting inference. The Streamlit form uses a fixed threshold of `0.5`; the prediction CLI exposes `--threshold`.

## Evaluation

```bash
python -m pytest
```

Training prints precision, recall, F1, ROC-AUC and a confusion matrix. Report dataset provenance, split settings, class prevalence and threshold alongside results. This README does not claim benchmark scores: no versioned evaluation report is provided here.

The default split is approximately 55% training, 25% validation and 20% test, with seed 42. For time-dependent transaction data, also evaluate on a chronological holdout before interpreting random-split results as deployment performance.

## Repository map

`fraudguard/` contains data, features, models and evaluation; `scripts/` contains training and prediction entry points; `app/` contains the interface; `notebooks/` contains exploratory work; `tests/` contains automated tests.

A Dockerfile is included, but the local Python workflow above is the documented starting point; the container build has not been validated in this documentation update.

## License

[MIT](LICENSE)
