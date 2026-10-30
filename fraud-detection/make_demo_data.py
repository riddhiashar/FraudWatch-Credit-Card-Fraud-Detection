"""Generates a Kaggle-style creditcard.csv (Time, V1..V28, Amount, Class) for testing.
Skip this if you downloaded the real dataset into data/creditcard.csv."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n, fraud_rate = 60000, 0.004
y = (rng.random(n) < fraud_rate).astype(int)
X = rng.normal(size=(n, 28))
f = y == 1
shift = np.zeros(28); shift[[0,2,3,9,11,13,16]] = [-3,-3.5,2.5,-3,-3.5,-4,-3]
X[f] = X[f] * 1.25 + shift * 0.45
amount = np.where(f, rng.lognormal(4.1, 1.4, n), rng.lognormal(3.7, 1.3, n)).round(2)
t = np.sort(rng.uniform(0, 172800, n))
df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
df.insert(0, "Time", t); df["Amount"] = amount; df["Class"] = y
df.to_csv("data/creditcard.csv", index=False)
print(f"Wrote data/creditcard.csv  rows={n}  fraud={y.sum()}")
