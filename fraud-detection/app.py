import json, os, random, sqlite3, threading, time, joblib, pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from common import make_features, score_all

SPEED = float(os.getenv("STREAM_DELAY", "1.2"))     # seconds between transactions
FRAUD_BOOST = float(os.getenv("FRAUD_BOOST", "0.08"))  # demo only: chance to replay a known-fraud row
bundle = joblib.load("models/bundle.joblib")
metrics = json.load(open("models/metrics.json"))
stream = pd.read_csv("data/test_stream.csv")
fraud_idx = stream.index[stream.Class == 1].tolist(); legit_idx = stream.index[stream.Class == 0].tolist()
lock = threading.Lock(); state = {"running": True}
db = sqlite3.connect("fraud.db", check_same_thread=False)
db.execute("DROP TABLE IF EXISTS transactions")
db.execute("""CREATE TABLE transactions(id INTEGER PRIMARY KEY, ts REAL, amount REAL, hour INT,
              prob REAL, status TEXT, actual INT, latency_ms REAL, scores TEXT)""")

def status_of(p): return "HIGH RISK" if p >= .8 else "REVIEW" if p >= max(.5, bundle["threshold"]) else "APPROVED"

def process(row: pd.DataFrame):
    t0 = time.perf_counter(); s = score_all(bundle, make_features(row))
    return {k: float(v[0]) for k, v in s.items()}, (time.perf_counter() - t0) * 1000

def replay():
    n = 0
    while True:
        if not state["running"]: time.sleep(.3); continue
        i = random.choice(fraud_idx) if fraud_idx and random.random() < FRAUD_BOOST else random.choice(legit_idx)
        row = stream.iloc[[i]]; sc, ms = process(row); p = sc["Ensemble"]; n += 1
        with lock:
            db.execute("INSERT INTO transactions VALUES(?,?,?,?,?,?,?,?,?)", (None, time.time(),
                float(row.Amount.iloc[0]), int(row.Time.iloc[0] // 3600 % 24), p, status_of(p),
                int(row.Class.iloc[0]), ms, json.dumps(sc))); db.commit()
        time.sleep(SPEED)

@asynccontextmanager
async def lifespan(_):
    threading.Thread(target=replay, daemon=True).start(); yield

app = FastAPI(title="Fraud Detection", lifespan=lifespan)

def rows(q, a=()):
    with lock: cur = db.execute(q, a); cols = [c[0] for c in cur.description]; return [dict(zip(cols, r)) for r in cur.fetchall()]

@app.get("/api/transactions")
def transactions(limit: int = 30):
    out = rows("SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,))
    for r in out: r["scores"] = json.loads(r["scores"])
    return out

@app.get("/api/stats")
def stats():
    r = rows("""SELECT COUNT(*) n, COALESCE(SUM(status!='APPROVED'),0) flagged,
        COALESCE(SUM(status!='APPROVED' AND actual=1),0) caught, COALESCE(SUM(actual),0) fraud,
        COALESCE(SUM(status!='APPROVED' AND actual=0),0) false_pos,
        COALESCE(SUM(CASE WHEN status!='APPROVED' THEN amount END),0) amount_flagged,
        COALESCE(AVG(latency_ms),0) latency FROM transactions""")[0]
    return {**r, "running": state["running"]}

@app.get("/api/models")
def models(): return metrics

@app.post("/api/stream/toggle")
def toggle(): state["running"] = not state["running"]; return {"running": state["running"]}

class Txn(BaseModel):
    Time: float = 0; Amount: float; V: dict[str, float] = {}

@app.post("/api/predict")
def predict(t: Txn):
    d = {f"V{i}": t.V.get(f"V{i}", 0.0) for i in range(1, 29)}; d.update(Time=t.Time, Amount=t.Amount)
    sc, ms = process(pd.DataFrame([d])); return {"fraud_probability": sc["Ensemble"], "status": status_of(sc["Ensemble"]),
        "model_scores": sc, "latency_ms": ms}

@app.get("/")
def index(): return FileResponse("static/index.html")
