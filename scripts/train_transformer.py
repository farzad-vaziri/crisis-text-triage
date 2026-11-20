"""Model 2: fine-tuned Hugging Face transformer, own PyTorch training loop.

Usage: python train_transformer.py small   # 800 posts/class, 2 epochs (first run)
       python train_transformer.py large   # 2,000 posts/class, up to 3 epochs, early stopping on validation
"""
import sys, time, copy, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from common import CLASSES, SEED, MODELS
from predict_io import load_set, save_preds, save_meta

variant = sys.argv[1] if len(sys.argv) > 1 else "small"
per_class, max_epochs = {"small": (800, 2), "large": (2000, 3)}[variant]
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)
NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAXLEN, BATCH, LR = 128, 16, 5e-5
l2i = {c: i for i, c in enumerate(CLASSES)}

tr = load_set("train")
tr = pd.concat([g.sample(min(len(g), per_class), random_state=SEED) for _, g in tr.groupby("subreddit")]).sample(frac=1, random_state=SEED)
val = load_set("val")
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=len(CLASSES))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = max_epochs * int(np.ceil(len(tr) / BATCH))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))

def enc(texts):
    return tok(list(texts), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")

def predict(texts, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            out.append(torch.softmax(model(**enc(texts[i:i + bs])).logits, -1).numpy())
    return np.vstack(out)

t0 = time.time(); best, best_state, history = -1, None, []
for ep in range(max_epochs):
    model.train()
    for i in range(0, len(tr), BATCH):
        b = tr.iloc[i:i + BATCH]
        loss = model(**enc(b.text), labels=torch.tensor([l2i[s] for s in b.subreddit])).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
    f1 = f1_score(val.subreddit, np.array(CLASSES)[predict(val.text.tolist()).argmax(1)], average="macro")
    history.append({"epoch": ep + 1, "val_macro_f1": round(f1, 4), "secs": round(time.time() - t0)})
    print(history[-1], flush=True)
    if f1 > best:
        best, best_state = f1, copy.deepcopy(model.state_dict())
    elif variant == "large":
        break  # early stopping: validation F1 stopped improving
train_secs = time.time() - t0
model.load_state_dict(best_state)
MODELS.mkdir(exist_ok=True); model.save_pretrained(MODELS / f"minilm_{variant}")  # git-ignored, never published

key = f"minilm_{variant}"
for name in ["val", "eval", "covid", "safety"] + (["test"] if variant == "large" else []):
    d = load_set(name)
    t1 = time.time(); P = predict(d.text.tolist()); secs = time.time() - t1
    if name == "eval": per1000 = secs / len(d) * 1000
    save_preds(key, name, d, P)
save_meta(key, name=f"Fine-tuned MiniLM-L6 ({variant})", base_model=NAME, train_posts=len(tr), train_secs=round(train_secs),
          infer_secs_per_1000=round(per1000, 2), epochs_run=len(history), history=history,
          params=f"AdamW lr {LR}, weight decay 0.01, linear decay, batch {BATCH}, max length {MAXLEN}, grad clip 1.0, seed {SEED}",
          hardware="2 vCPU cloud workspace, no GPU")
print(key, "done", round(train_secs), "s")
