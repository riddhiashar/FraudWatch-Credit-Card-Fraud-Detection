# FraudWatch: Real-Time Credit Card Fraud Detection

Transaction stream -> feature engineering -> 5 ML models + ensemble -> fraud probability -> SQLite -> live dashboard.

| Model | Role |
|---|---|
| Logistic Regression | interpretable baseline |
| Random Forest | non-linear baseline |
| XGBoost | main supervised model (imbalance handled via scale_pos_weight) |
| Isolation Forest | unsupervised, trained on legitimate transactions only |
| Autoencoder (sklearn MLP) | unsupervised, reconstruction error = anomaly score |
| Ensemble | weighted blend of supervised + unsupervised scores |

Evaluation: PR-AUC, ROC-AUC, Precision@100, recall on a **time-ordered** split (train 60% / val 15% / test 25%).
The decision threshold is chosen on validation, never on test.

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get data (pick one)
**Real (use this for your resume numbers):** download `creditcard.csv` from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it at `data/creditcard.csv`.

**Quick demo:** `python make_demo_data.py` creates a synthetic file with the same columns (Time, V1-V28, Amount, Class).

## 2. Train
```bash
python train.py
```
Prints the model comparison and writes `models/` and `data/test_stream.csv`.
(Real data has ~285k rows; Random Forest and Isolation Forest take a few minutes.)

## 3. Run the dashboard
```bash
uvicorn app:app --port 8000
```
Open http://localhost:8000

Env vars: `STREAM_DELAY=0.5` (seconds between transactions), `FRAUD_BOOST=0.1`.
PowerShell: `$env:STREAM_DELAY="0.5"; uvicorn app:app --port 8000`

## API
- `GET /api/transactions`, `GET /api/stats`, `GET /api/models`, `POST /api/stream/toggle`
- `POST /api/predict` with `{"Amount": 87500, "V": {"V1": -4, "V3": -4}}` scores one transaction
- Interactive docs: http://localhost:8000/docs

## Honest notes (put these in your writeup)
- The stream is a **replay of the held-out test set** with a delay. Kafka is not used; it is a documented next step.
- `FRAUD_BOOST` replays known-fraud rows more often than their real ~0.17% rate, only so the demo shows alerts. Model metrics are unaffected.
- Kaggle features V1-V28 are anonymised PCA components, so velocity/geo features (distance from previous transaction, etc.) can't be built from this data.
- Amounts show a rupee sign in the UI; the original dataset is in euros.
- Metrics from the synthetic demo data are not meaningful; report numbers from the real dataset.

## Next steps
Kafka producer/consumer, Redis state for velocity features, PostgreSQL, SHAP explanations per alert, cost-based thresholding.
