"""Model 1: TF-IDF + logistic regression (the simple baseline)."""
import time, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from common import CLASSES, SEED
from predict_io import load_set, save_preds, save_meta

tr, val = load_set("train"), load_set("val")
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=100000)
Xtr = vec.fit_transform(tr.text); Xval = vec.transform(val.text)
grid = {}
for C in [0.5, 1, 2, 4, 8]:
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED).fit(Xtr, tr.subreddit)
    grid[C] = f1_score(val.subreddit, m.predict(Xval), average="macro")
bestC = max(grid, key=grid.get)
print("C grid", grid, "best", bestC)
t0 = time.time()
m = LogisticRegression(C=bestC, class_weight="balanced", max_iter=2000, random_state=SEED).fit(Xtr, tr.subreddit)
train_secs = time.time() - t0 
assert list(m.classes_) == CLASSES
for name in ["val", "eval", "covid", "test", "safety"]:
    d = load_set(name)
    t0 = time.time(); P = m.predict_proba(vec.transform(d.text)); secs = time.time() - t0
    if name == "eval": per1000 = secs / len(d) * 1000
    save_preds("tfidf", name, d, P)
# top words per class (interpretability)
fn = np.array(vec.get_feature_names_out())
top = [{"class": c, "rank": r + 1, "term": fn[i], "weight": round(float(m.coef_[k, i]), 3)}
       for k, c in enumerate(CLASSES) for r, i in enumerate(np.argsort(-m.coef_[k])[:15])]
pd.DataFrame(top).to_csv("../data/tfidf_top_terms.csv", index=False)
save_meta("tfidf", name="TF-IDF + logistic regression", train_posts=len(tr), train_secs=round(train_secs, 1),
          infer_secs_per_1000=round(per1000, 3), C_grid={str(k): round(v, 4) for k, v in grid.items()}, C=bestC,
          hardware="2 vCPU cloud workspace, no GPU", params="1-2 grams, min_df 3, sublinear tf, 100k features, balanced classes")
print("tfidf done", round(per1000, 3), "s/1000")
