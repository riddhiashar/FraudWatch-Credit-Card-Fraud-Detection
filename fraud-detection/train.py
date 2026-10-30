import json, joblib, numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from common import make_features, score_all

df = pd.read_csv("data/creditcard.csv").sort_values("Time").reset_index(drop=True)
n = len(df); a, b = int(n * .60), int(n * .75)          # time-ordered split: train | val | test
tr, va, te = df[:a], df[a:b], df[b:]
Xtr, Xva, Xte = map(make_features, (tr, va, te))
ytr, yva, yte = tr.Class.values, va.Class.values, te.Class.values
print(f"train={len(tr)} val={len(va)} test={len(te)} | fraud in train={ytr.sum()} test={yte.sum()}")

pos_w = (ytr == 0).sum() / max(ytr.sum(), 1)
bundle = {}
bundle["lr"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xtr, ytr)
bundle["rf"] = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Xtr, ytr)
bundle["xgb"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=.08, subsample=.8,
                              colsample_bytree=.8, scale_pos_weight=pos_w ** .5, eval_metric="aucpr", n_jobs=-1).fit(Xtr, ytr)

legit = Xtr[ytr == 0]                                    # unsupervised models learn "normal" only
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0, n_jobs=-1).fit(legit)
bundle["iso"] = iso
sc = -iso.score_samples(legit); bundle["iso_range"] = (float(np.median(sc)), float(np.quantile(sc, .999)))
ae_scaler = StandardScaler().fit(legit); Zl = ae_scaler.transform(legit)
ae = MLPRegressor(hidden_layer_sizes=(20, 8, 20), max_iter=60, random_state=0).fit(Zl, Zl)   # bottleneck autoencoder
bundle["ae_scaler"], bundle["ae"] = ae_scaler, ae
err = ((ae.predict(Zl) - Zl) ** 2).mean(axis=1); bundle["ae_range"] = (float(np.median(err)), float(np.quantile(err, .999)))
bundle["weights"] = {"XGBoost": .45, "Random Forest": .20, "Isolation Forest": .15, "Autoencoder": .20}

# threshold picked on validation (never on test)
sv = score_all(bundle, Xva)["Ensemble"]
p, r, t = precision_recall_curve(yva, sv); f1 = 2 * p * r / (p + r + 1e-9)
bundle["threshold"] = float(t[np.argmax(f1[:-1])])

st = score_all(bundle, Xte); metrics = {"models": {}, "threshold": bundle["threshold"],
      "test_rows": len(te), "test_fraud": int(yte.sum())}
for name, s in st.items():
    k = min(100, len(s)); top = np.argsort(-s)[:k]
    pred = s >= (bundle["threshold"] if name == "Ensemble" else .5)
    tp = int((pred & (yte == 1)).sum())
    metrics["models"][name] = {"pr_auc": round(average_precision_score(yte, s), 4),
        "roc_auc": round(roc_auc_score(yte, s), 4), "precision_at_100": round(float(yte[top].mean()), 3),
        "recall": round(tp / max(yte.sum(), 1), 3), "precision": round(tp / max(pred.sum(), 1), 3)}
    print(f"{name:20s}", metrics["models"][name])

joblib.dump(bundle, "models/bundle.joblib"); te.to_csv("data/test_stream.csv", index=False)
json.dump(metrics, open("models/metrics.json", "w"), indent=2)
print("Saved models/bundle.joblib, models/metrics.json, data/test_stream.csv")
