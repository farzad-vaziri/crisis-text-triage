"""Evaluate all models and write the small, text-free tables the app reads."""
import json, re, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from common import CLASSES, HIGH_RISK, OUT, SEED
from predict_io import PRED, load_set
from metrics_lib import recall_alerts, threshold_for_recall, softmax_rows, contextual_calibrate, flesch_reading_ease

GBP_PER_HOUR = 0.08   # ASSUMPTION: indicative price of a 2-vCPU cloud machine (Nov 2026), used only to compare models
SW = CLASSES.index(HIGH_RISK)
rng = np.random.default_rng(SEED)

# ---------- collect predictions: model -> set -> DataFrame(id, label, p_*) ----------
models, meta = {}, {}
for key in ["tfidf", "minilm_small", "minilm_large"]:
    f = PRED / f"{key}_meta.json"
    if not f.exists(): continue
    meta[key] = json.loads(f.read_text())
    models[key] = {s: pd.read_csv(PRED / f"{key}_{s}.csv") for s in ["val", "eval", "covid", "safety", "test"] if (PRED / f"{key}_{s}.csv").exists()}
for key in ["llm_qwen05"]:
    f = PRED / f"{key}_meta.json"
    if not f.exists(): continue
    raw = {s: pd.read_csv(PRED / f"{key}_raw_{s}.csv") for s in ["val", "eval", "covid", "safety"]}
    lp = lambda d: d[[f"lp_{c}" for c in CLASSES]].values
    cf = pd.read_csv(PRED / f"{key}_contentfree.csv")[CLASSES].mean().values
    ref_val = lp(raw["val"]).mean(0)
    for suffix, fn, label in [("", lambda L: softmax_rows(L), "raw"),
                              ("_cc", lambda L: contextual_calibrate(L, cf), "content-free calibration"),
                              ("_bc", lambda L: contextual_calibrate(L, ref_val), "calibrated on 250 validation posts")]:
        k = key + suffix
        m = json.loads(f.read_text()); m["name"] = m["name"] + ("" if not suffix else f" + {label}"); meta[k] = m
        models[k] = {}
        for s, d in raw.items():
            P = fn(lp(d)); out = pd.DataFrame(P, columns=[f"p_{c}" for c in CLASSES])
            out.insert(0, "label", d.label.values); out.insert(0, "id", d.id.values); models[k][s] = out

def P(d): return d[[f"p_{c}" for c in CLASSES]].values

# ---------- Platt calibration of the high-risk score, fitted on validation ----------
platt = {}
for k, sets in models.items():
    v = sets["val"]; x = np.log(np.clip(P(v)[:, SW], 1e-6, 1)); y = (v.label == HIGH_RISK).values
    platt[k] = LogisticRegression(C=1e6).fit(x[:, None], y)
def p_high(k, d, calibrated=True):
    s = P(d)[:, SW]
    return platt[k].predict_proba(np.log(np.clip(s, 1e-6, 1))[:, None])[:, 1] if calibrated else s

# validation threshold for 90% recall (used for safety tests, fairness and drift)
val_thr = {k: threshold_for_recall((s["val"].label == HIGH_RISK).values, p_high(k, s["val"]), 0.90) for k, s in models.items()}

rows, conf, curves, rel, band = [], [], [], [], []
for k, sets in models.items():
    for s in ["eval", "covid", "test"]:
        if s not in sets: continue
        d = sets[s]; y = d.label.values; pred = np.array(CLASSES)[P(d).argmax(1)]
        yh = y == HIGH_RISK; ph = p_high(k, d); praw = p_high(k, d, False)
        r95 = threshold_for_recall(yh, ph, 0.95); _, apt95, _ = recall_alerts(yh, ph, r95)
        rec_v, apt_v, flag_v = recall_alerts(yh, ph, val_thr[k])
        rows.append(dict(model=k, set=s, n=len(d), accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                         high_risk_recall_top_label=((pred == HIGH_RISK) & yh).sum() / yh.sum(),
                         auc_high_risk=roc_auc_score(yh, praw), alerts_per_true_at_95=apt95,
                         recall_at_val_threshold=rec_v, alerts_per_true_at_val_threshold=apt_v,
                         brier_raw=brier_score_loss(yh, praw), brier_calibrated=brier_score_loss(yh, ph),
                         share_top_label=pd.Series(pred).value_counts(normalize=True).iloc[0]))
        if s == "test": continue
        for t in CLASSES:
            for p_ in CLASSES:
                conf.append(dict(model=k, set=s, true=t, pred=p_, share=((y == t) & (pred == p_)).sum() / (y == t).sum()))
        for thr in np.unique(np.quantile(ph, np.linspace(0, 1, 201))):
            rc, apt, fl = recall_alerts(yh, ph, thr)
            curves.append(dict(model=k, set=s, threshold=thr, recall=rc, flagged_share=fl,
                               miss_share=(yh & (ph < thr)).mean(), false_alarm_share=(~yh & (ph >= thr)).mean()))
        for kind, sc in [("raw", praw), ("calibrated", ph)]:
            bins = np.clip((sc * 10).astype(int), 0, 9)
            for b in range(10):
                m_ = bins == b
                if m_.sum(): rel.append(dict(model=k, set=s, kind=kind, bin=b, mean_pred=sc[m_].mean(), frac_pos=yh[m_].mean(), n=int(m_.sum())))
        top = P(d).max(1); correct = pred == y
        for t in np.round(np.arange(0.20, 0.96, 0.05), 2):
            auto = top >= t
            band.append(dict(model=k, set=s, confidence_cutoff=t, share_to_human=1 - auto.mean(),
                             error_rate_auto=(~correct[auto]).mean() if auto.any() else np.nan,
                             missed_high_risk_auto=((yh & auto & (pred != HIGH_RISK)).sum()) / yh.sum()))
M = pd.DataFrame(rows)
for k in M.model.unique():
    mt = meta[k]; M.loc[M.model == k, "model_name"] = mt["name"]
    M.loc[M.model == k, "train_posts"] = mt["train_posts"]; M.loc[M.model == k, "train_secs"] = mt["train_secs"]
    M.loc[M.model == k, "infer_secs_per_1000"] = mt["infer_secs_per_1000"]
M["cost_gbp_per_1000"] = M.infer_secs_per_1000 / 3600 * GBP_PER_HOUR
M.round(4).to_csv(OUT / "metrics.csv", index=False)
pd.DataFrame(conf).round(4).to_csv(OUT / "confusion.csv", index=False)
pd.DataFrame(curves).round(5).to_csv(OUT / "risk_curves.csv", index=False)
pd.DataFrame(rel).round(4).to_csv(OUT / "reliability.csv", index=False)
pd.DataFrame(band).round(4).to_csv(OUT / "human_band.csv", index=False)

# ---------- safety tests ----------
st = pd.read_csv(OUT / "safety_tests.csv"); srows = []
for k, sets in models.items():
    d = sets["safety"]; ph = p_high(k, d); flag = ph >= val_thr[k]
    for i, r in st.iterrows():
        srows.append(dict(model=k, test_type=r.test_type, expected_high_risk=r.expected_high_risk, text=r.text,
                          p_high_risk=round(float(ph[i]), 3), flagged=bool(flag[i]), passed=bool(flag[i] == bool(r.expected_high_risk))))
pd.DataFrame(srows).to_csv(OUT / "safety_results.csv", index=False)

# ---------- fairness proxy by writing style (aggregates only) ----------
SLANG = set("lol lmao idk tbh ngl rn fr smh omg ur u r ya yall gonna wanna gotta im dont cant wont thats theres bc cuz cos tho thru wtf af imo irl bf gf pls plz".split())
def style(texts):
    words = texts.str.split()
    return pd.DataFrame({"length": words.str.len(),
                         "reading_ease": [flesch_reading_ease(t) for t in texts],
                         "slang_rate": [sum(w.lower().strip(".,!?") in SLANG for w in ws) / max(1, len(ws)) for ws in words]})
frows = []
for s in ["test", "eval"]:
    base = load_set(s); feats = style(base.text); feats["id"] = base.id.values
    feats["length_group"] = pd.qcut(feats.length, 4, labels=["shortest 25%", "short-mid", "mid-long", "longest 25%"])
    feats["reading_group"] = pd.cut(feats.reading_ease, [-1e9, 50, 70, 1e9], labels=["harder to read (<50)", "standard (50-70)", "easy to read (>70)"])
    feats["slang_group"] = pd.cut(feats.slang_rate, [-1, 0, 0.02, 1], labels=["no text-speak", "some (≤2%)", "frequent (>2%)"])
    for k, sets in models.items():
        if s not in sets: continue
        d = sets[s].merge(feats, on="id"); yh = (d.label == HIGH_RISK).values; flag = p_high(k, d) >= val_thr[k]
        overall = (flag & yh).sum() / yh.sum()
        for var in ["length_group", "reading_group", "slang_group"]:
            for g, gd in d.groupby(var, observed=True):
                idx = gd.index.values; yy = yh[idx]; ff = flag[idx]
                if yy.sum() < 5: continue
                hits = ff[yy]; boots = [rng.choice(hits, len(hits)).mean() for _ in range(1000)]
                frows.append(dict(model=k, set=s, variable=var, group=str(g), n_high_risk=int(yy.sum()), recall=hits.mean(),
                                  ci_lo=np.quantile(boots, 0.025), ci_hi=np.quantile(boots, 0.975), overall_recall=overall))
F = pd.DataFrame(frows); F["flag"] = F.ci_hi < F.overall_recall
F.round(4).to_csv(OUT / "fairness.csv", index=False)
pd.DataFrame([dict(model=k, val_threshold_90_recall=round(float(v), 4)) for k, v in val_thr.items()]).to_csv(OUT / "thresholds.csv", index=False)
json.dump({k: {kk: vv for kk, vv in m.items()} for k, m in meta.items()}, open(OUT / "model_meta.json", "w"), indent=2)
print(M[M.set == "eval"][["model", "accuracy", "macro_f1", "alerts_per_true_at_95", "auc_high_risk", "share_top_label"]].round(3).to_string())
