"""Crisis text triage: simple vs complex models. One-page Streamlit app reading precomputed, text-free results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
REPO = "https://github.com/farzad-vaziri/crisis-text-triage"
CLASSES = ["anxiety", "depression", "lonely", "relationships", "suicidewatch"]
CLASS_LABEL = {"anxiety": "Anxiety", "depression": "Depression", "lonely": "Loneliness",
               "relationships": "Relationships", "suicidewatch": "High risk (r/SuicideWatch)"}

# Fixed colour per model family (validated categorical slots 1-3); variants share the hue, told apart by dash/opacity.
FAMILY_COLOUR = {"tfidf": "#2a78d6", "minilm": "#1baf7a", "llm": "#eb6834"}
SHORT = {"tfidf": "TF-IDF + logistic regression", "minilm_small": "Fine-tuned MiniLM (4k posts)",
         "minilm_large": "Fine-tuned MiniLM (10k posts)", "llm_qwen05": "Zero-shot LLM, raw",
         "llm_qwen05_cc": "Zero-shot LLM, content-free calibration", "llm_qwen05_bc": "Zero-shot LLM, calibrated on validation"}
TAG = {"tfidf": "TF-IDF", "minilm_small": "MiniLM 4k", "minilm_large": "MiniLM 10k", "llm_qwen05": "LLM raw",
       "llm_qwen05_cc": "LLM + content-free calib.", "llm_qwen05_bc": "LLM + validation calib."}
POS = {"tfidf": "top right", "minilm_small": "bottom center", "minilm_large": "top center", "llm_qwen05": "top left",
       "llm_qwen05_cc": "bottom left", "llm_qwen05_bc": "top left"}
SET_LABEL = {"eval": "2019 held-out posts", "covid": "COVID-period posts (drift)"}


def family(k): return k.split("_")[0]
def colour(k): return FAMILY_COLOUR[family(k)]
def dash(k): return "dot" if k in ("minilm_small", "llm_qwen05") else ("dash" if k == "llm_qwen05_cc" else "solid")


st.set_page_config(page_title="Crisis text triage", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
h1 {font-weight: 650; letter-spacing: -0.01em;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px; height: 100%;}
.kpi .v {font-size: 1.7rem; font-weight: 650; line-height: 1.2;}
.kpi .l {font-size: .85rem; opacity: .75;}
.take {border-left: 3px solid #2a78d6; padding: 6px 12px; margin: 4px 0 18px; opacity: .9;}
.card {border: 1px solid rgba(128,128,128,.3); border-radius: 14px; padding: 18px 20px;}
.small {font-size: .85rem; opacity: .8;}
</style>""", unsafe_allow_html=True)

LAYOUT = dict(template="plotly_white", font=dict(family="Inter, Segoe UI, sans-serif", size=13),
              margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


@st.cache_data
def load():
    d = {n: pd.read_csv(DATA / f"{n}.csv") for n in
         ["metrics", "confusion", "risk_curves", "reliability", "human_band", "safety_results",
          "fairness", "privacy_counts", "split_sizes", "thresholds"]}
    d["meta"] = json.loads((DATA / "model_meta.json").read_text())
    return d


D = load()
M = D["metrics"]
available = [k for k in SHORT if k in set(M.model)]


def take(text):
    st.markdown(f"<div class='take'>{text}</div>", unsafe_allow_html=True)


# ---------------- header ----------------
st.title("Crisis text triage: simple vs complex models")
st.markdown("Three ways to sort messages to a mental-health support service by topic and risk — a simple "
            "TF-IDF model, a fine-tuned transformer and a zero-shot LLM — tested on the same real posts for accuracy, "
            "missed high-risk messages, cost and safety. Then: which one I would actually deploy, and why.")
st.caption("Portfolio demo · real public-domain posts (Reddit Mental Health Dataset, 2019–2020) · **no post text is shown anywhere** · "
           "“High risk” = posted in r/SuicideWatch, a stand-in label, not a clinical rating.")

with st.sidebar:
    st.header("Filters")
    eval_set = st.radio("Evaluation set", list(SET_LABEL), format_func=SET_LABEL.get,
                        help="Models were trained on 2019 posts only. COVID-period posts test drift.")
    default_models = [k for k in ["tfidf", "minilm_large", "minilm_small", "llm_qwen05", "llm_qwen05_bc"] if k in available]
    if "minilm_large" in default_models and "minilm_small" in default_models: default_models.remove("minilm_small")
    chosen = st.multiselect("Models", available, default=default_models, format_func=SHORT.get)
    st.divider()
    st.markdown("**Alert settings** (used by the threshold panel)")
    cost_miss = st.slider("Cost of missing a high-risk message (reviewer-minutes equivalent)", 5, 600, 60, 5)
    cost_fa = st.slider("Cost of a false alarm (reviewer-minutes)", 1, 30, 5)
    volume = st.slider("Messages per day", 100, 10000, 1000, 100)
if not chosen:
    st.info("Pick at least one model in the sidebar.")
    st.stop()

Ms = M[(M.set == eval_set) & M.model.isin(chosen)].copy()
curves = D["risk_curves"][(D["risk_curves"].set == eval_set) & D["risk_curves"].model.isin(chosen)]


def operating_point(model):
    """Cost-minimising threshold on this model's calibrated high-risk score."""
    c = curves[curves.model == model].copy()
    c["cost"] = cost_miss * c.miss_share + cost_fa * c.false_alarm_share
    return c.loc[c.cost.idxmin()]


ops = {k: operating_point(k) for k in chosen}
best_f1 = Ms.loc[Ms.macro_f1.idxmax()]
cheapest_cost = {k: ops[k].cost for k in chosen}
rec_model = min(cheapest_cost, key=cheapest_cost.get)
op = ops[rec_model]

# ---------------- KPI strip ----------------
cols = st.columns(5)
kpis = [(f"{best_f1.macro_f1:.2f}", f"Best macro F1 · {SHORT[best_f1.model]}"),
        (f"{op.recall:.0%}", f"High-risk recall at the cost-optimal threshold · {SHORT[rec_model]}"),
        (f"{op.flagged_share / max(op.flagged_share - op.false_alarm_share, 1e-9):.1f}",
         "Alerts per true high-risk message"),
        (f"{op.miss_share * 1000:.0f}", "Missed high-risk per 1,000 messages"),
        (f"{Ms.set_index('model').loc[rec_model, 'infer_secs_per_1000'] / 3600 * 0.08 * 100:.2g}p", "Compute cost per 1,000 messages (indicative)")]
for c, (v, l) in zip(cols, kpis):
    c.markdown(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>", unsafe_allow_html=True)
st.caption("The KPI model is the one with the lowest expected cost at your sidebar settings. In the evaluation sets 20% of posts are high-risk.")

# ---------------- Panel 1: scoreboard ----------------
st.subheader("1 · Scoreboard: accuracy against cost and complexity")
fig = go.Figure()
for _, r in Ms.iterrows():
    fig.add_trace(go.Scatter(x=[r.infer_secs_per_1000], y=[r.macro_f1], mode="markers+text", name=SHORT[r.model],
                             text=[TAG[r.model]], textposition=POS.get(r.model, "top center"),
                             marker=dict(size=14 + 30 * np.sqrt(r.train_secs / max(1, Ms.train_secs.max())), color=colour(r.model),
                                         opacity=0.55 if dash(r.model) != "solid" else 0.9, line=dict(width=2, color="white")),
                             hovertemplate=f"<b>{SHORT[r.model]}</b><br>Macro F1 %{{y:.3f}}<br>%{{x:.2f}} s per 1,000 messages"
                                           f"<br>Training: {r.train_secs:.0f} s on {int(r.train_posts):,} posts<extra></extra>"))
fig.update_layout(**LAYOUT, height=380, showlegend=False)
fig.update_xaxes(type="log", title="Seconds per 1,000 messages (log scale)")
fig.update_yaxes(title="Macro F1 (5 topics)", range=[0, 1])
st.plotly_chart(fig, width="stretch")
tbl = Ms[["model", "accuracy", "macro_f1", "auc_high_risk", "alerts_per_true_at_95", "train_posts", "train_secs", "infer_secs_per_1000"]].copy()
tbl["model"] = tbl.model.map(SHORT)
tbl.columns = ["Model", "Accuracy", "Macro F1", "High-risk AUC", "Alerts per true case at 95% recall", "Training posts",
               "Training time (s)", "Seconds per 1,000 messages"]
st.dataframe(tbl.round(3), hide_index=True, width="stretch")
simple = Ms[Ms.model == "tfidf"]
if len(simple) and best_f1.model != "tfidf":
    s = simple.iloc[0]
    take(f"The simplest model is {100 * (best_f1.macro_f1 - s.macro_f1):.1f} points of macro F1 behind the best "
         f"({SHORT[best_f1.model]}) and about {best_f1.infer_secs_per_1000 / max(s.infer_secs_per_1000, 1e-6):,.0f}× faster to run.")
elif len(simple):
    s = simple.iloc[0]
    others = Ms[Ms.model != "tfidf"]
    msg = f"The simplest model is also the most accurate here (macro F1 {s.macro_f1:.2f})"
    if len(others):
        o = others.loc[others.macro_f1.idxmax()]
        msg += f", ahead of {SHORT[o.model]} ({o.macro_f1:.2f}), and about {o.infer_secs_per_1000 / max(s.infer_secs_per_1000, 1e-6):,.0f}× cheaper to run."
    take(msg)

full = M[(M.set == "test") & M.model.isin(["tfidf", "minilm_large"])].set_index("model")
if len(full) == 2:
    a, b = full.loc["tfidf"], full.loc["minilm_large"]
    st.markdown(f"<span class='small'><b>Check on the full 2019 test set ({int(a.n):,} posts):</b> TF-IDF accuracy {a.accuracy:.3f} vs "
                f"fine-tuned MiniLM (10k posts) {b.accuracy:.3f}; at 95% high-risk recall, {a.alerts_per_true_at_95:.2f} vs "
                f"{b.alerts_per_true_at_95:.2f} alerts per true case. On 400 posts a 2-point gap is within noise; on 6,700 it disappears.</span>",
                unsafe_allow_html=True)

# ---------------- Panel 2: confusion ----------------
st.subheader("2 · Where the models get confused")
C = D["confusion"][(D["confusion"].set == eval_set)]
show = [k for k in chosen if k in set(C.model)][:3]
cc = st.columns(len(show)) if show else []
short_lab = {"anxiety": "Anx", "depression": "Dep", "lonely": "Lone", "relationships": "Rel", "suicidewatch": "High risk"}
for col, k in zip(cc, show):
    mat = C[C.model == k].pivot(index="true", columns="pred", values="share").reindex(index=CLASSES, columns=CLASSES).fillna(0)
    f = go.Figure(go.Heatmap(z=mat.values, x=[short_lab[c] for c in CLASSES], y=[short_lab[c] for c in CLASSES],
                             colorscale=[[0, "#f4f7fb"], [1, "#1c5aa6"]], zmin=0, zmax=1, showscale=False,
                             text=(mat.values * 100).round(0).astype(int), texttemplate="%{text}",
                             hovertemplate="True %{y} → predicted %{x}: %{z:.0%}<extra></extra>"))
    f.update_layout(**{**LAYOUT, "margin": dict(l=10, r=10, t=36, b=10)}, height=320, title=dict(text=SHORT[k], font=dict(size=13)))
    f.update_yaxes(autorange="reversed", title="true")
    f.update_xaxes(title="predicted")
    col.plotly_chart(f, width="stretch")
if show:
    trained = [k for k in show if not k.startswith("llm")] or show
    off = C[(C.model.isin(trained)) & (C.true != C.pred)].groupby(["true", "pred"]).share.mean().sort_values(ascending=False)
    (t, p), v = off.index[0], off.iloc[0]
    take(f"Biggest confusion for the {'trained ' if trained != show else ''}models shown: {CLASS_LABEL[t]} posts labelled as {CLASS_LABEL[p]} ({v:.0%} on average). "
         "Row percentages; each row sums to 100.")

# ---------------- Panel 3: threshold ----------------
st.subheader("3 · Alert threshold: what each kind of mistake costs")
st.markdown("<span class='small'>Each model's high-risk score is calibrated on 250 validation posts, then the threshold is set "
            "to minimise expected cost with your sidebar settings. Moving down the curve catches more high-risk messages but raises more false alarms.</span>",
            unsafe_allow_html=True)
f = go.Figure()
for k in chosen:
    c = curves[curves.model == k].sort_values("flagged_share")
    f.add_trace(go.Scatter(x=c.flagged_share * volume, y=c.recall, mode="lines", name=SHORT[k],
                           line=dict(color=colour(k), width=2, dash=dash(k)),
                           hovertemplate=f"{SHORT[k]}<br>%{{x:.0f}} alerts/day → recall %{{y:.0%}}<extra></extra>"))
    o = ops[k]
    f.add_trace(go.Scatter(x=[o.flagged_share * volume], y=[o.recall], mode="markers", showlegend=False,
                           marker=dict(size=11, color=colour(k), line=dict(width=2, color="white")),
                           hovertemplate=f"Cost-optimal point<br>{o.flagged_share * volume:.0f} alerts/day, recall {o.recall:.0%}<extra></extra>"))
f.update_layout(**LAYOUT, height=400)
f.update_xaxes(title=f"Alerts per day at {volume:,} messages/day")
f.update_yaxes(title="Share of high-risk messages caught", tickformat=".0%", range=[0, 1.02])
st.plotly_chart(f, width="stretch")
optab = pd.DataFrame([{"Model": SHORT[k], "Threshold (calibrated prob.)": round(ops[k].threshold, 3),
                       "Recall": f"{ops[k].recall:.0%}", "Alerts per day": round(ops[k].flagged_share * volume),
                       "Missed high-risk per day": round(ops[k].miss_share * volume, 1),
                       "False alarms per day": round(ops[k].false_alarm_share * volume)} for k in chosen])
st.dataframe(optab, hide_index=True, width="stretch")
take(f"With a miss costing {cost_miss} and a false alarm {cost_fa} reviewer-minutes, {SHORT[rec_model]} has the lowest expected cost: "
     f"it catches {op.recall:.0%} of high-risk messages with {op.flagged_share * volume:.0f} alerts a day.")

# ---------------- Panel 4: human band ----------------
st.subheader("4 · Send to a human when the model is unsure")
cut = st.slider("Route automatically only when the model's top-topic confidence is at least", 0.2, 0.95, 0.6, 0.05)
B = D["human_band"][(D["human_band"].set == eval_set) & D["human_band"].model.isin(chosen)]
f = go.Figure()
for k in chosen:
    b = B[B.model == k]
    f.add_trace(go.Scatter(x=b.share_to_human, y=b.error_rate_auto, mode="lines+markers", name=SHORT[k],
                           line=dict(color=colour(k), width=2, dash=dash(k)), marker=dict(size=6),
                           hovertemplate="%{x:.0%} to humans → %{y:.0%} errors on the rest<extra></extra>"))
    p = b[np.isclose(b.confidence_cutoff, cut)]
    if len(p):
        f.add_trace(go.Scatter(x=p.share_to_human, y=p.error_rate_auto, mode="markers", showlegend=False,
                               marker=dict(size=13, color=colour(k), line=dict(width=2, color="white")), hoverinfo="skip"))
f.update_layout(**LAYOUT, height=380)
f.update_xaxes(title="Share of messages sent to a human reviewer", tickformat=".0%")
f.update_yaxes(title="Topic error rate on auto-routed messages", tickformat=".0%")
st.plotly_chart(f, width="stretch")
pb = B[np.isclose(B.confidence_cutoff, cut)]
if len(pb):
    r = pb.sort_values("error_rate_auto").iloc[0]
    take(f"At a {cut:.0%} confidence cut-off, {SHORT[r.model]} sends {r.share_to_human:.0%} of messages to a person and gets "
         f"{r.error_rate_auto:.0%} of the rest wrong. Human review buys accuracy with staff time; the panel shows the exchange rate.")

# ---------------- Panel 5: safety tests ----------------
st.subheader("5 · Safety tests on tricky messages")
st.markdown("<span class='small'>58 short messages written by hand for this test (invented examples, not real posts). "
            "A model passes a message if it flags it when it should (e.g. coded language) and stays quiet when it should "
            "(e.g. jokes, recovery stories). Thresholds were set beforehand to catch 90% of high-risk validation posts.</span>",
            unsafe_allow_html=True)
S = D["safety_results"][D["safety_results"].model.isin(chosen)]
order = ["direct", "very_short", "text_speak", "coded_language", "third_person", "negation", "past_or_recovery", "hyperbole_or_jokes"]
pretty = {"direct": "Direct statements", "very_short": "Very short", "text_speak": "Text-speak & misspelling",
          "coded_language": "Coded language", "third_person": "Worry about someone else", "negation": "Negation (“I’m not…”)",
          "past_or_recovery": "Past or recovery", "hyperbole_or_jokes": "Jokes & exaggeration"}
piv = S.groupby(["test_type", "model"]).passed.mean().unstack().reindex(index=order, columns=chosen)
f = go.Figure(go.Heatmap(z=piv.values, x=[SHORT[k] for k in piv.columns], y=[pretty[t] for t in piv.index],
                         colorscale=[[0, "#f6efe6"], [1, "#1c5aa6"]], zmin=0, zmax=1, showscale=False,
                         text=(piv.values * 100).round(0), texttemplate="%{text:.0f}%",
                         hovertemplate="%{y} · %{x}: %{z:.0%} passed<extra></extra>"))
f.update_layout(**LAYOUT, height=360)
f.update_yaxes(autorange="reversed")
st.plotly_chart(f, width="stretch")
should = S[S.expected_high_risk == 1].groupby("model").passed.mean()
shouldnt = S[S.expected_high_risk == 0].groupby("model").passed.mean()
take("Flagging rate where it should flag: " + ", ".join(f"{SHORT[k]} {should.get(k, np.nan):.0%}" for k in chosen) +
     ". Staying quiet where it shouldn't flag: " + ", ".join(f"{SHORT[k]} {shouldnt.get(k, np.nan):.0%}" for k in chosen) + ".")
with st.expander("See the invented test messages and each model's answer"):
    ex = S.pivot_table(index=["test_type", "text", "expected_high_risk"], columns="model", values="flagged", aggfunc="first").reset_index()
    ex["test_type"] = ex.test_type.map(pretty)
    ex["expected_high_risk"] = ex.expected_high_risk.map({1: "should flag", 0: "should not flag"})
    for k in chosen:
        if k in ex: ex[k] = ex[k].map({True: "flagged", False: "—"})
    ex = ex.rename(columns={"test_type": "Test", "text": "Invented message", "expected_high_risk": "Expected", **SHORT})
    st.dataframe(ex, hide_index=True, width="stretch")

# ---------------- Panel 6: calibration ----------------
st.subheader("6 · Can the scores be trusted as probabilities?")
R = D["reliability"][(D["reliability"].set == eval_set) & D["reliability"].model.isin(chosen)]
kind = st.radio("Scores", ["raw", "calibrated"], horizontal=True, format_func=lambda x: {"raw": "Raw model scores", "calibrated": "After calibration on validation posts"}[x])
f = go.Figure()
f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#9a9a96", dash="dot", width=1), name="Perfectly calibrated", hoverinfo="skip"))
for k in chosen:
    r = R[(R.model == k) & (R.kind == kind)]
    f.add_trace(go.Scatter(x=r.mean_pred, y=r.frac_pos, mode="lines+markers", name=SHORT[k], line=dict(color=colour(k), width=2, dash=dash(k)),
                           marker=dict(size=8), customdata=r.n, hovertemplate="Predicted %{x:.0%} → actual %{y:.0%} (n=%{customdata})<extra></extra>"))
f.update_layout(**LAYOUT, height=400)
f.update_xaxes(title="Predicted probability of high risk", tickformat=".0%", range=[0, 1])
f.update_yaxes(title="Actual share high-risk", tickformat=".0%", range=[0, 1])
st.plotly_chart(f, width="stretch")
br = Ms.set_index("model")[["brier_raw", "brier_calibrated"]]
take("Brier score (lower is better), raw → calibrated: " + ", ".join(f"{SHORT[k]} {br.loc[k, 'brier_raw']:.3f} → {br.loc[k, 'brier_calibrated']:.3f}" for k in chosen if k in br.index) + ".")
llm_rows = M[(M.set == eval_set) & M.model.str.startswith("llm")].set_index("model")
if len(llm_rows) > 1:
    raw = llm_rows.loc["llm_qwen05"]
    best_llm = llm_rows.macro_f1.idxmax()
    st.markdown(f"**The zero-shot LLM, before and after calibration.** Raw, it put {raw.share_top_label:.0%} of posts into one topic "
                f"(macro F1 {raw.macro_f1:.2f}). Subtracting its built-in preference for each label "
                f"({SHORT[best_llm].split(', ')[1]}) lifts macro F1 to {llm_rows.loc[best_llm, 'macro_f1']:.2f}, with no training examples at all.")

# ---------------- Panel 7: drift and fairness ----------------
st.subheader("7 · Drift and fairness checks")
c1, c2 = st.columns(2)
dr = M[M.model.isin(chosen) & M.set.isin(["eval", "covid"])].pivot(index="model", columns="set", values="macro_f1").reindex(chosen)
f = go.Figure()
for setn, op_ in [("eval", 1.0), ("covid", 0.55)]:
    f.add_trace(go.Bar(y=[SHORT[k] for k in dr.index], x=dr[setn], orientation="h", name=SET_LABEL[setn],
                       marker=dict(color=[colour(k) for k in dr.index], opacity=op_, cornerradius=4),
                       hovertemplate="%{y}: macro F1 %{x:.2f}<extra>" + SET_LABEL[setn] + "</extra>"))
f.update_layout(**LAYOUT, height=360, barmode="group", bargap=0.3, bargroupgap=0.08, showlegend=False,
                title=dict(text="Macro F1: 2019 posts (solid) vs COVID-period posts (light)", font=dict(size=13)))
f.update_xaxes(title="Macro F1", range=[0, 1])
f.update_yaxes(autorange="reversed")
c1.plotly_chart(f, width="stretch")
drop = (dr["eval"] - dr["covid"]).dropna()
F = D["fairness"]
fset = "test" if eval_set == "eval" else "eval"
fm = c2.selectbox("Fairness check for", [k for k in chosen if k in set(F.model)] or chosen, format_func=SHORT.get)
var = c2.radio("Group posts by", ["length_group", "reading_group", "slang_group"], horizontal=True,
               format_func={"length_group": "Length", "reading_group": "Readability", "slang_group": "Text-speak"}.get)
ff = F[(F.model == fm) & (F.variable == var)]
ff = ff[ff.set == ("test" if "test" in set(ff.set) else "eval")]
f = go.Figure()
if len(ff):
    f.add_trace(go.Scatter(x=ff.recall, y=ff.group, mode="markers", marker=dict(size=11, color=colour(fm)),
                           error_x=dict(type="data", symmetric=False, array=ff.ci_hi - ff.recall, arrayminus=ff.recall - ff.ci_lo, color=colour(fm)),
                           customdata=ff.n_high_risk, hovertemplate="%{y}: recall %{x:.0%} (n=%{customdata} high-risk posts)<extra></extra>", showlegend=False))
    f.add_vline(x=ff.overall_recall.iloc[0], line=dict(color="#9a9a96", dash="dot"), annotation_text="overall", annotation_position="top")
f.update_layout(**LAYOUT, height=360, title=dict(text="High-risk recall by writing style (95% bootstrap CI)", font=dict(size=13)))
f.update_xaxes(title="Recall", tickformat=".0%", range=[0.4, 1.02])
c2.plotly_chart(f, width="stretch")
flagged = F[F.model.isin(chosen) & F.flag]
take((f"Largest drop on COVID-period posts: {SHORT[drop.idxmax()]} (−{drop.max():.2f} macro F1). " if len(drop) else "") +
     ("Groups where recall is clearly below overall: " + "; ".join(f"{SHORT[r.model]} · {r.group}" for _, r in flagged.drop_duplicates(['model', 'group']).iterrows())
      if len(flagged) else "No writing-style group has recall clearly below overall") +
     ". The data has no age or gender, so those checks need real service data.")

# ---------------- Panel 8: privacy ----------------
st.subheader("8 · Privacy step: personal details removed before any modelling")
c1, c2 = st.columns([1.3, 1])
c1.markdown("""
1. **Download** public-domain posts; the author field is never loaded.
2. **Redact** before modelling: regular expressions for emails, links, phone numbers and usernames; spaCy named-entity recognition for people and places.
3. **Train and score** on redacted text only.
4. **Publish aggregates only.** Predictions are keyed by a one-way hash; no post text, no model weights.
""")
pc = D["privacy_counts"].sort_values("count")
f = go.Figure(go.Bar(x=pc["count"], y=pc.entity.str.title(), orientation="h", marker=dict(color="#2a78d6", cornerradius=4),
                     hovertemplate="%{y}: %{x:,} replaced<extra></extra>"))
f.update_layout(**LAYOUT, height=260, title=dict(text=f"Details replaced with tags across {D['split_sizes'].query('set != "eval"').posts.sum():,} posts", font=dict(size=13)))
c2.plotly_chart(f, width="stretch")

# ---------------- Panel 9: recommendation ----------------
st.subheader("9 · What I'd deploy, and why")
ev = M[M.set == "eval"].set_index("model")
t = ev.loc["tfidf"] if "tfidf" in ev.index else None
tr_best = ev[ev.index.str.startswith("minilm")].macro_f1.idxmax() if ev.index.str.startswith("minilm").any() else None
llm_best = ev[ev.index.str.startswith("llm")].macro_f1.idxmax() if ev.index.str.startswith("llm").any() else None
lines = []
if t is not None and tr_best:
    tb = ev.loc[tr_best]
    better = tb.alerts_per_true_at_95 < t.alerts_per_true_at_95 - 0.1
    lines.append(f"<b>First-line screen: {'the fine-tuned transformer' if better else 'TF-IDF + calibrated logistic regression'}.</b> "
                 f"At 95% high-risk recall, TF-IDF needs {t.alerts_per_true_at_95:.1f} alerts per true case and the best transformer {tb.alerts_per_true_at_95:.1f}; "
                 f"the transformer {'earns its extra cost' if better else 'does not earn its extra cost and complexity'}. TF-IDF is also fully explainable (its weights can be read).")
lines.append("<b>Cost-based threshold plus human review:</b> set the alert threshold from what a missed message and a false alarm actually cost the service, "
             "and send low-confidence messages to a person rather than forcing an automatic route.")
if llm_best:
    lines.append(f"<b>Not the zero-shot LLM, yet.</b> Even calibrated (macro F1 {ev.loc[llm_best, 'macro_f1']:.2f}) it trails the trained models and is "
                 f"{ev.loc[llm_best, 'infer_secs_per_1000'] / max(ev.loc['tfidf', 'infer_secs_per_1000'], 1e-6):,.0f}× slower than TF-IDF on CPU. "
                 "Its value is elsewhere: summarising, drafting, or labelling help, validated against clinicians.")
lines.append("<b>Known weakness to fix before any use:</b> the safety tests show over-flagging of jokes, recovery stories and negations — "
             "acceptable for a screen that a human checks, not for anything automatic.")
st.markdown("<div class='card'>" + "<br><br>".join(lines) + "</div>", unsafe_allow_html=True)
st.markdown("""
**What I'd do next with real service data:** validate against clinical expert labels (not community names); measure how far
expert raters agree with each other and judge the model against that; run a silent prospective pilot; monitor drift and
false-negative audits monthly; and repeat the fairness checks with real age and gender, which this public data does not have.
""")

with st.expander("How to read this"):
    st.markdown(f"""
- **Data.** Posts from five Reddit communities in 2019 (anxiety, depression, lonely, relationships, SuicideWatch), plus Jan–Apr 2020 COVID-period posts for drift.
  “High risk” means the post came from r/SuicideWatch. That is a **stand-in label**: it is not a clinical judgement, and some posts elsewhere are high-risk too.
- **Models.** (1) TF-IDF word and word-pair features with logistic regression; (2) all-MiniLM-L6-v2, a small transformer, fine-tuned in PyTorch with a
  hand-written training loop; (3) Qwen2.5-0.5B-Instruct used zero-shot: each topic is scored by how likely the model is to say its name.
- **Fair comparison.** All models are scored on the same held-out posts (80 per topic). Thresholds and calibration are fitted on separate validation posts.
- **Costs** are compute time on a 2-vCPU machine without a GPU, converted at an assumed £0.08 per machine-hour. Real costs depend on hardware and volume; the ratios matter more than the pennies.
- **Why there is no box to type a message.** Someone in distress could use it and get a meaningless score. This is a demonstration of evaluation methods, not a triage tool.
- **Why no post text is shown.** The posts are public, but they are personal. Only aggregates and hand-written test messages appear here.
""")

st.divider()
st.markdown(f"""<div class='small'>
<b>Sources:</b> Low, D. M. et al. (2020), Reddit Mental Health Dataset, Zenodo record 3941387 (ODC-PDDL public domain).
Models: sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0), Qwen/Qwen2.5-0.5B-Instruct (Apache 2.0), spaCy en_core_web_sm (MIT).<br>
<b>Caveats:</b> stand-in labels; Reddit users are not the people who contact a crisis service; small evaluation sets (400 posts) so differences of a few points are noise.<br>
<b>Code and model card:</b> <a href="{REPO}">{REPO.replace('https://', '')}</a> · Portfolio demo.<br><br>
<b>If you need support now, text SHOUT to 85258</b> (UK, free, confidential, 24/7).
</div>""", unsafe_allow_html=True)
