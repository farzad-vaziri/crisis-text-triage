"""Helpers to load evaluation sets and save predictions without any text."""
import numpy as np, pandas as pd, json, time
from common import RAW, OUT, CLASSES

PRED = OUT / "preds"
SETS = ["val", "eval", "covid", "test"]


def load_set(name):
    if name == "safety":
        d = pd.read_csv(OUT / "safety_tests.csv")
        return pd.DataFrame({"id": [f"safety_{i:02d}" for i in range(len(d))],
                             "subreddit": np.where(d.expected_high_risk == 1, "suicidewatch", "other"),
                             "text": d.text})
    return pd.read_parquet(RAW / f"{name}.parquet")


def save_preds(model, name, df, probs):
    PRED.mkdir(exist_ok=True)
    out = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
    out.insert(0, "label", df["subreddit"].values)
    out.insert(0, "id", df["id"].values)
    out.round(5).to_csv(PRED / f"{model}_{name}.csv", index=False)


def save_meta(model, **kw):
    PRED.mkdir(exist_ok=True)
    (PRED / f"{model}_meta.json").write_text(json.dumps(kw, indent=2))
