"""Core evaluation maths (unit-tested in tests/)."""
import numpy as np


def recall_alerts(y, score, threshold):
    """Recall on the high-risk class, alerts per true high-risk case, and share of messages flagged."""
    y = np.asarray(y).astype(bool); f = np.asarray(score) >= threshold
    tp = (f & y).sum()
    recall = tp / max(1, y.sum())
    alerts_per_true = f.sum() / tp if tp else np.inf
    return recall, alerts_per_true, f.mean()


def threshold_for_recall(y, score, target):
    """Highest threshold whose recall is at least `target`."""
    y = np.asarray(y).astype(bool); s = np.sort(np.asarray(score)[y])
    k = int(np.floor((1 - target) * len(s)))
    return s[min(k, len(s) - 1)]


def expected_cost(y, score, threshold, cost_miss, cost_false_alarm):
    """Average cost per message of missed high-risk cases plus false alarms."""
    y = np.asarray(y).astype(bool); f = np.asarray(score) >= threshold
    return (cost_miss * (y & ~f).sum() + cost_false_alarm * (~y & f).sum()) / len(y)


def bayes_threshold(cost_miss, cost_false_alarm):
    """Cost-optimal threshold for a calibrated probability: flag when p > C_fa / (C_fa + C_miss)."""
    return cost_false_alarm / (cost_false_alarm + cost_miss)


def softmax_rows(S):
    S = np.asarray(S, float); S = S - S.max(1, keepdims=True); E = np.exp(S)
    return E / E.sum(1, keepdims=True)


def contextual_calibrate(logprobs, reference):
    """Zhao et al. (2021) style: subtract each label's reference log-probability, then renormalise."""
    return softmax_rows(np.asarray(logprobs) - np.asarray(reference)[None, :])


def count_syllables(word: str) -> int:
    """Simple vowel-group heuristic (good enough for banding texts)."""
    import re
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith("le"):
        n -= 1
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    import re
    sentences = max(1, len(re.findall(r"[.!?]+", text)) or 1)
    words = [w for w in re.findall(r"[A-Za-z']+", text)]
    if not words:
        return 100.0
    syll = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syll / len(words))
