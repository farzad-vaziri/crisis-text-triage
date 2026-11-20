# crisis-text-triage – Build brief

Written by the ideas chat, 20 Nov 2026 (16:45). Built the same day. This file is the spec. Where the build departed from it, see "Build notes" at the end.

## 0. Why this exists, and one exception to the usual rules
- Written to show the skills that AI-scientist roles in digital mental-health services ask for: PyTorch or Hugging Face, LLMs on real free text, rigorous evaluation, safety testing for high-risk conversations, judgement about "the simplest method that solves the problem", and attention to bias, privacy and safety. Every panel should answer one of those points.
- **This repo is public** (an exception to the portfolio's usual private repos), because reproducible, tested code is part of the evidence for this role. Check that no secrets, emails or raw post text are committed.
- **Sensitive content rule:** no real post text from any community, and especially not r/SuicideWatch, may appear on the page, in the repo or in screenshots. Real posts are used only for training and for aggregate numbers. Every example message shown is short, written by hand and labelled "invented example".
- **Crisis signpost:** the page footer and README carry: "If you need support now, text SHOUT to 85258 (UK, free, 24/7)." Verified on giveusashout.org on 20 Nov 2026.
- **No live free-text scoring.** Visitors cannot type a message and get a risk score. Explained in the "How to read this" expander and in the model card.

## 1. Purpose
- **Story:** three ways to triage mental-health messages by topic and risk (TF-IDF, a fine-tuned transformer, a zero-shot LLM), tested on the same real posts for accuracy, missed high-risk messages, cost and safety, ending with which one I would deploy and why.
- **Audience:** recruiters and interviewers for data scientist / AI scientist roles. Skills: PyTorch and Hugging Face fine-tuning, LLMs on real text, evaluation design, safety testing, cost-based thresholds, calibration, drift, fairness checks, privacy-by-design, communication.
- **Visual standard:** polished and calm.
- **Not covered here:** ticket-triage-nlp; rater-agreement-lab; help-seeking-trends.

## 2. Data
- Reddit Mental Health Dataset (Low et al., 2020), Zenodo record 3941387, licence ODC-PDDL (public domain). Columns `subreddit`, `date`, `post` only; `author` never loaded.
- Classes: anxiety, depression, lonely, relationships, suicidewatch. Risk = suicidewatch vs rest (stand-in label, said on the page).
- Splits (seed 42): 2019 files, ≤6,000 posts per class, stratified 75/25; 250 validation posts (50 per class) taken from training; evaluation set 400 test posts (80 per class) shared by all models; drift set 400 posts (80 per class) from the COVID-period (`post`) files.
- `data/raw/` git-ignored; only text-free outputs and the invented safety tests are committed.
- Privacy step before modelling: regex for emails, URLs, phone numbers, usernames; spaCy `en_core_web_sm` NER for people and places; counts recorded.

## 3. Models and methods
Feasibility results (ideas chat, before the privacy step): TF-IDF 0.772 accuracy / 2.01 alerts per true case at 95% recall; MiniLM (800/class, 2 epochs) 0.755 / 2.47; zero-shot Qwen2.5-0.5B 0.215 / 3.75 (collapsed to "anxiety").
1. TF-IDF (1–2 grams, min_df 3, sublinear tf, 100k features) + logistic regression (balanced), C chosen on validation from {0.5, 1, 2, 4, 8}.
2. all-MiniLM-L6-v2 fine-tuned in PyTorch with an own training loop (AdamW 5e-5, weight decay 0.01, batch 16, max length 128): "small" 800/class × 2 epochs; "large" 2,000/class, up to 3 epochs with early stopping on validation macro F1.
3. Qwen2.5-0.5B-Instruct zero-shot, labels scored by summed token log-probability; contextual calibration (content-free inputs, and mean over validation posts).
- Platt calibration of each high-risk score on validation; reliability curves and Brier scores.
- Costs: measured inference time per 1,000 messages × an assumed £0.08 per 2-vCPU machine-hour.
- Safety tests: 58 invented messages across 8 types; thresholds set on validation for 90% recall.
- Fairness proxy by writing style (length quartile, readability band, text-speak rate) with bootstrap CIs; no age or gender in the data.
- Drift: trained on 2019, scored on COVID-period posts.
- Unit tests for threshold maths, calibration, readability, redaction, and a check that no committed file holds post text.

## 4. The page
Header and caveats; sidebar filters (evaluation set, models, alert costs and volume); KPI strip; 1 scoreboard; 2 confusion matrices; 3 cost-based alert threshold; 4 send-to-a-human band; 5 safety tests; 6 calibration (with LLM before/after); 7 drift and fairness; 8 privacy step; 9 what I'd deploy and why; "How to read this"; footer with sources, caveats, repo link and the Shout signpost.

## 5–6. Look, feel and performance
Fixed colour per model family (blue TF-IDF, aqua transformer, orange LLM; variants by dash/opacity); calm theme; precomputed CSVs only; no torch in the app requirements.

## 7. Checks
Unit tests; AppTest run of every filter; desktop and phone screenshots; scan for post text, author, emails, keys; README numbers match the app.

## 8. Files
`app.py`, `scripts/`, `data/`, `tests/`, `requirements.txt`, `requirements-train.txt`, `.streamlit/config.toml`, `README.md`, `MODEL_CARD.md`, `SPEC.md`, `.gitignore`, `LICENSE` (MIT). No model weights published.

## 9. Launch
- Repo `farzad-vaziri/crisis-text-triage` (public, created by Farzad 20 Nov), branch `main`.
- App URL: `crisis-text-triage.streamlit.app` (if taken, ask Farzad).
- Card text: "Which model should triage messages to a mental-health support service? Compare a simple text model, a fine-tuned transformer and an LLM on real posts, set the alert threshold by the cost of each kind of mistake, and see which one I'd deploy."

## Build notes (departures from the original brief)
- textstat needs downloaded NLTK data, so readability uses an own Flesch implementation (`scripts/metrics_lib.py`).
- A stronger transformer (DistilRoBERTa / ModernBERT on a GPU) and a larger Qwen were not run in the first build (CPU only); listed as next steps in the README.
- Files: training scripts are `train_tfidf.py`, `train_transformer.py`, `zeroshot_llm.py`; evaluation in `evaluate.py` (covers safety tests, drift and fairness).
