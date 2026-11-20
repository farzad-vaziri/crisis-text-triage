"""Load, sample, redact and split. Writes text only to data/raw/ (git-ignored)."""
import pandas as pd, spacy, time
from sklearn.model_selection import train_test_split
from common import RAW, OUT, CLASSES, SEED, PER_CLASS_2019, EVAL_PER_CLASS, VAL_PER_CLASS, row_id
from redact import redact_all

def load(period, n=None):
    parts = []
    for c in CLASSES:
        d = pd.read_csv(RAW / f"{c}_{period}_features_tfidf_256.csv", usecols=["subreddit", "date", "post"])  # author never loaded
        d = d.dropna(subset=["post"])
        d["subreddit"] = d["subreddit"].str.lower()
        if n and len(d) > n:
            d = d.sample(n, random_state=SEED)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)

t0 = time.time()
y19 = load("2019", PER_CLASS_2019)
covid = load("post")
covid = pd.concat([g.sample(EVAL_PER_CLASS, random_state=SEED) for _, g in covid.groupby("subreddit")], ignore_index=True)

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler", "tok2vec"] if False else ["parser", "lemmatizer"])
nlp.max_length = 2_000_000
allposts = pd.concat([y19.assign(src="2019"), covid.assign(src="covid")], ignore_index=True)
allposts["post"] = allposts["post"].str.slice(0, 5000)
red, counts = redact_all(allposts["post"].tolist(), nlp)
allposts["text"] = red
allposts["id"] = [row_id(t) for t in allposts["post"]]
allposts = allposts.drop(columns=["post"]).drop_duplicates("id")
pd.Series(counts, name="count").rename_axis("entity").reset_index().to_csv(OUT / "privacy_counts.csv", index=False)
print("redaction", dict(counts), round(time.time() - t0), "s")

y19 = allposts[allposts.src == "2019"]
covid = allposts[allposts.src == "covid"]
train, test = train_test_split(y19, test_size=0.25, stratify=y19.subreddit, random_state=SEED)
val = pd.concat([g.sample(VAL_PER_CLASS, random_state=SEED) for _, g in train.groupby("subreddit")])
train = train[~train.id.isin(val.id)]
ev = pd.concat([g.sample(EVAL_PER_CLASS, random_state=SEED) for _, g in test.groupby("subreddit")])
for name, d in [("train", train), ("val", val), ("test", test), ("eval", ev), ("covid", covid)]:
    d[["id", "subreddit", "date", "text"]].to_parquet(RAW / f"{name}.parquet", index=False)
    print(name, len(d))
pd.DataFrame({"set": ["train", "val", "test", "eval", "covid"],
              "posts": [len(train), len(val), len(test), len(ev), len(covid)]}).to_csv(OUT / "split_sizes.csv", index=False)
