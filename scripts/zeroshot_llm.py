"""Model 3: zero-shot LLM (no training examples, no API keys), run locally via Hugging Face.

Each topic is scored by the sum of its token log-probabilities after the prompt.
Raw scores are saved so the evaluation can apply contextual calibration (Zhao et al., 2021).
"""
import sys, time, copy, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import CLASSES, OUT
from predict_io import load_set, PRED, save_meta

NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
KEY = "llm_" + NAME.split("/")[-1].split("-Instruct")[0].replace("Qwen2.5-", "qwen").replace(".", "").replace("B", "")  # e.g. llm_qwen05
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).eval()
LABEL_WORDS = {"anxiety": "anxiety", "depression": "depression", "lonely": "loneliness",
               "relationships": "relationships", "suicidewatch": "suicidal thoughts"}
SYSTEM = ("You classify messages sent to a mental-health support service. Answer with exactly one topic from: "
          "anxiety, depression, loneliness, relationships, suicidal thoughts.")


def label_logprobs(message: str) -> dict:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Message: " + message[:1200] + "\n\nTopic:"}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prefix, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, use_cache=True)
    first = out.logits[0, -1].log_softmax(-1)
    scores = {}
    for c, w in LABEL_WORDS.items():
        wid = tok(w, add_special_tokens=False).input_ids
        lp = first[wid[0]].item()
        if len(wid) > 1:
            with torch.no_grad():
                o = model(torch.tensor([wid[:-1]]), past_key_values=copy.deepcopy(out.past_key_values)).logits[0].log_softmax(-1)
            lp += sum(o[j, wid[j + 1]].item() for j in range(len(wid) - 1))
        scores[c] = lp
    return scores


if __name__ == "__main__":
    PRED.mkdir(exist_ok=True)
    cf = [label_logprobs(x) for x in ["N/A", "", "[MASK]"]]  # content-free inputs
    pd.DataFrame(cf).assign(input=["N/A", "", "[MASK]"]).to_csv(PRED / f"{KEY}_contentfree.csv", index=False)
    per1000 = None
    for name in ["safety", "val", "eval", "covid"]:
        d = load_set(name); t0 = time.time(); rows = []
        for i, t in enumerate(d.text):
            rows.append(label_logprobs(t))
            if i % 100 == 0: print(name, i, round(time.time() - t0), flush=True)
        if name == "eval": per1000 = (time.time() - t0) / len(d) * 1000
        S = pd.DataFrame(rows)[CLASSES]
        S.columns = [f"lp_{c}" for c in CLASSES]
        S.insert(0, "label", d["subreddit"].values); S.insert(0, "id", d["id"].values)
        S.round(4).to_csv(PRED / f"{KEY}_raw_{name}.csv", index=False)
    save_meta(KEY, name=f"Zero-shot {NAME.split('/')[-1]}", base_model=NAME, train_posts=0, train_secs=0,
              infer_secs_per_1000=round(per1000, 1), system_prompt=SYSTEM, label_words=LABEL_WORDS,
              hardware="2 vCPU cloud workspace, no GPU")
    print(KEY, "done")
