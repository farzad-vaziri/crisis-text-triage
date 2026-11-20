# Model card: crisis-text-triage

> **If you need support now, text SHOUT to 85258** (UK, free, confidential, 24/7).

## What this is
A **portfolio demonstration of how to evaluate text-classification models for a mental-health support service**. Three approaches are compared on the same posts:

| Key | Model | How it learns |
|---|---|---|
| `tfidf` | TF-IDF (word 1–2-grams) + logistic regression | Trained on ~19,900 redacted posts |
| `minilm_small`, `minilm_large` | `sentence-transformers/all-MiniLM-L6-v2` (22M parameters) fine-tuned in PyTorch | Own training loop; 4,000 and 10,000 posts |
| `llm_qwen05*` | `Qwen/Qwen2.5-0.5B-Instruct`, zero-shot | No training examples; topic scored by label log-probability, optionally with contextual calibration |

All models predict one of five topics (anxiety, depression, loneliness, relationships, high risk). The high-risk probability is re-calibrated (Platt scaling) on 250 validation posts.

## Intended use
- Showing evaluation methods: like-for-like comparison, cost-based thresholds, human-review bands, calibration, behavioural safety tests, drift and fairness checks, privacy-by-design.
- Discussion material for a job interview.

## Out of scope: do not use for
- **Any real triage, risk assessment or decision about a person.**
- Anyone in crisis. The app deliberately has no box to type a message.
- Any claim about clinical risk. The labels are not clinical.

## Training data
- Reddit Mental Health Dataset (Low et al., 2020), Zenodo record 3941387, licence ODC-PDDL (public domain).
- 2019 posts from r/anxiety, r/depression, r/lonely, r/relationships and r/SuicideWatch (≤6,000 per community, seed 42); Jan–Apr 2020 COVID-period posts for drift testing.
- Author names are never loaded. Names, places, emails, links, phone numbers and usernames are replaced with tags before any modelling (counts in `data/privacy_counts.csv`).

## Labels and their limits
- **Topic** = the community a post was written in. **High risk** = posted in r/SuicideWatch.
- These are **stand-in labels**. Posting in r/SuicideWatch is not a clinical rating; high-risk posts also appear in other communities, and some r/SuicideWatch posts are supportive replies or past experiences.
- Reddit users differ from people who contact a crisis text service (age, platform, writing style, situation).

## Evaluation
- Same 400 held-out posts (80 per topic) for every model; validation posts are used only for model selection, calibration and thresholds.
- Metrics: accuracy, macro F1, high-risk AUC, alerts per true high-risk post at 95% recall, Brier score before and after calibration, inference time.
- Results: see the table in `README.md` and `data/metrics.csv`.

## Safety tests
58 short **invented** messages (`data/safety_tests.csv`) in eight groups: direct statements, very short messages, text-speak and misspellings, coded language, worry about someone else, negation, past or recovery, jokes and exaggeration. Thresholds were fixed beforehand (90% recall on validation).
**Known failure:** the trained models over-flag negations ("I'm not going to hurt myself…"), recovery stories and jokes. That's acceptable for a screen checked by a person, and unacceptable for anything automatic.

## Fairness
The data has no age, gender or ethnicity, so those checks are **not possible here**. As a proxy, high-risk recall is compared across writing styles (post length, readability, text-speak) with bootstrap 95% CIs (`data/fairness.csv`). Long posts showed lower recall for TF-IDF in the first run. Real fairness checks need real service data with demographics.

## Known failures
- The raw zero-shot LLM puts almost every post into one label (label bias). Contextual calibration helps but does not close the gap to trained models.
- Performance drops on COVID-period posts (topic drift).
- Evaluation sets are small (400 posts); differences of a few points are within noise.

## Why the weights are not published
A suicide-risk classifier trained on stand-in labels could be picked up and used on real people, where it would be unvalidated and potentially harmful. The training code is public and reproducible, so anyone with a legitimate research purpose can retrain it and validate it properly.

## Contact and status
Portfolio project, November 2026. Not maintained as a product.
