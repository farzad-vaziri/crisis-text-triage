# Crisis text triage: simple vs complex models

**Live app:** https://crisis-text-triage.streamlit.app *(being deployed)*

Which model should sort messages to a mental-health support service by topic and risk? This project compares a **simple TF-IDF model**, a **transformer fine-tuned in PyTorch** and a **zero-shot LLM** on the same real posts. It looks at accuracy, missed high-risk messages, cost, calibration, safety tests, drift and fairness, and ends with which one I'd deploy and why.

> **If you need support now, text SHOUT to 85258** (UK, free, confidential, 24/7).

## Results

Same 400 held-out 2019 posts for every model (80 per topic). Trained and run on a 2-vCPU machine without a GPU.

| Model | Accuracy | Macro F1 | High-risk AUC | Alerts per true high-risk post at 95% recall | Training | Seconds per 1,000 messages |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.790 | 0.788 | **0.951** | **2.04** | 8 s, 19,859 posts | **0.2** |
| Fine-tuned MiniLM-L6, PyTorch (4k posts) | **0.815** | **0.812** | 0.924 | 2.29 | 522 s, 4,000 posts | 13 |
| Zero-shot Qwen2.5-0.5B, raw | 0.220 | 0.106 | 0.705 | 4.55 | none | 1,544 |
| Zero-shot Qwen2.5-0.5B + contextual calibration (content-free) | 0.488 | 0.478 | 0.773 | 4.22 | none | 1,544 |
| Zero-shot Qwen2.5-0.5B + calibration on 250 validation posts | 0.542 | 0.531 | 0.764 | 4.21 | none | 1,544 |

**What I'd deploy:** TF-IDF + calibrated logistic regression as a first-line screen, with a cost-based alert threshold and human review of low-confidence messages. The transformer is slightly better at telling topics apart, but worse at the job that matters most, ranking high-risk messages (more alerts per true case at the same recall), and it is about 60× slower. The zero-shot LLM is not fit for triage: raw, it put 98% of posts into one label. Calibration lifts it a lot, but it stays far behind the trained models. Its value lies elsewhere (summaries, drafting, labelling help), and only once validated against clinicians.

**Known weakness:** in the safety tests every model catches the direct, short, coded and text-speak messages. They over-flag negations ("I'm not going to hurt myself…"), recovery stories and jokes. That's fine for a screen a person checks, and not fine for anything automatic.

An earlier quick run (TF-IDF 0.772, MiniLM 0.755, LLM 0.215 accuracy) was made before the privacy step, with a fixed C and the last epoch kept. The pipeline here adds redaction, tunes C on validation and keeps the best epoch.

## What the app shows
1. Scoreboard: accuracy vs inference cost and training effort.
2. Confusion matrices: where models mix topics up (depression ↔ high risk is the main one).
3. Cost-based alert threshold: set the cost of a missed message and of a false alarm, and see recall, alerts and misses per day.
4. Send to a human: share sent to reviewers vs errors on the rest.
5. Safety tests on 58 invented tricky messages.
6. Calibration: reliability curves, Brier scores, and the LLM before and after calibration.
7. Drift (trained on 2019, tested on COVID-period posts) and fairness by writing style.
8. The privacy step.
9. What I'd deploy and why, plus what I'd do next with real service data.

## Data
- **Reddit Mental Health Dataset** (Low, D. M., Rumker, L., Talkar, T., Torous, J., Cecchi, G., & Ghosh, S. S., 2020), Zenodo record [3941387](https://zenodo.org/records/3941387), licence **ODC-PDDL (public domain)**.
- Communities: r/anxiety, r/depression, r/lonely, r/relationships, r/SuicideWatch. 2019 posts for training and testing (≤6,000 per community), Jan–Apr 2020 COVID-period posts for drift.
- **"High risk" = posted in r/SuicideWatch.** This is a stand-in label, not a clinical rating.
- **No post text is in this repo or on the page.** Raw data is downloaded to `data/raw/` (git-ignored). Committed predictions are keyed by a one-way hash. The only text in the repo is the 58 hand-written test messages.
- **Privacy step before modelling:** emails, links, phone numbers and usernames are removed by regular expression, and people and places by spaCy NER. 13,612 details were replaced (`data/privacy_counts.csv`). Author names are never loaded.

## Methods
- **TF-IDF:** word 1–2-grams, min_df 3, sublinear tf, 100k features; logistic regression with balanced classes, C chosen on validation (C = 2).
- **Transformer:** `sentence-transformers/all-MiniLM-L6-v2` + classification head, fine-tuned with a hand-written PyTorch loop (AdamW 5e-5, weight decay 0.01, linear decay, batch 16, max length 128, gradient clipping, best epoch kept on validation macro F1).
- **Zero-shot LLM:** `Qwen/Qwen2.5-0.5B-Instruct` run locally with Hugging Face, no API keys. Each topic is scored by the summed log-probability of its label after the prompt. Contextual calibration (Zhao et al., 2021) subtracts each label's average score on content-free inputs, or on 250 validation posts.
- **Calibration:** Platt scaling of each model's high-risk score on 250 validation posts. Alert thresholds minimise expected cost.
- **Safety tests:** thresholds fixed beforehand at 90% recall on validation.
- **Fairness proxy:** high-risk recall by post length, readability (own Flesch implementation) and text-speak rate, with bootstrap 95% CIs. The data has no age or gender.
- **Costs** use an assumed £0.08 per 2-vCPU machine-hour. The ratios matter more than the pennies.

Full details and the limits: [MODEL_CARD.md](MODEL_CARD.md). Original spec: [SPEC.md](SPEC.md).

## Reproduce
```bash
pip install -r requirements-train.txt
bash scripts/run_all.sh            # download → redact → train 3 models → evaluate (about 75 min on 2 CPUs)
pytest -q
```
Steps: `download.py` → `prepare.py` (sampling, redaction, splits) → `train_tfidf.py` → `train_transformer.py small|large` → `zeroshot_llm.py` → `evaluate.py`. Seeds are fixed (42). Model weights go to `models/` (git-ignored) and are **not published**; see the model card for why.

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app reads only the small precomputed CSVs in `data/`; no model runs at page load.

## Design choices
- **No box to type a message.** Someone in distress could use it and get a meaningless score.
- **No weights published.** A risk classifier on stand-in labels shouldn't be reused on real people.
- **Simplest method first.** Every complex model has to beat a strong, cheap baseline on the metric that matters.

## Next steps
A stronger transformer on a GPU (DistilRoBERTa / ModernBERT); a larger LLM (Qwen2.5-1.5B/3B) and few-shot prompting; validation against clinical expert labels and inter-rater agreement; real demographic fairness checks with service data.

Code: MIT licence. Data: public domain (ODC-PDDL). Models used: all-MiniLM-L6-v2 (Apache 2.0), Qwen2.5-0.5B-Instruct (Apache 2.0), spaCy en_core_web_sm (MIT).
