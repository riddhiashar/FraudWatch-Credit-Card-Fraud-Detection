import numpy as np, pandas as pd
V = [f"V{i}" for i in range(1, 29)]
FEATURES = V + ["log_amount", "hour"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[V].copy()
    out["log_amount"] = np.log1p(df["Amount"])
    out["hour"] = (df["Time"] // 3600) % 24
    return out[FEATURES]

def _norm(s, lo, hi):
    return np.clip((s - lo) / (hi - lo + 1e-9), 0, 1)

def score_all(bundle, X: pd.DataFrame) -> dict:
    """Returns each model's fraud score in [0,1] plus the weighted ensemble."""
    s = {}
    s["Logistic Regression"] = bundle["lr"].predict_proba(X)[:, 1]
    s["Random Forest"] = bundle["rf"].predict_proba(X)[:, 1]
    s["XGBoost"] = bundle["xgb"].predict_proba(X)[:, 1]
    s["Isolation Forest"] = _norm(-bundle["iso"].score_samples(X), *bundle["iso_range"])
    Z = bundle["ae_scaler"].transform(X)
    err = ((bundle["ae"].predict(Z) - Z) ** 2).mean(axis=1)
    s["Autoencoder"] = _norm(err, *bundle["ae_range"])
    w = bundle["weights"]
    s["Ensemble"] = sum(w[k] * s[k] for k in w)
    return s
