import sys, numpy as np, pandas as pd, pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics_lib import (recall_alerts, threshold_for_recall, expected_cost, bayes_threshold,
                         softmax_rows, contextual_calibrate, flesch_reading_ease)
from redact import redact_regex
from collections import Counter

y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
s = np.array([.9, .8, .4, .2, .7, .3, .1, .1, .05, .0])


def test_recall_alerts():
    r, apt, share = recall_alerts(y, s, 0.35)
    assert r == 0.75          # 3 of 4 high-risk caught (.9 .8 .4)
    assert apt == pytest.approx(4 / 3)  # flagged .9 .8 .4 .7 = 4 alerts for 3 true
    assert share == 0.4


def test_threshold_for_recall_reaches_target():
    for target in [0.5, 0.75, 0.9, 1.0]:
        t = threshold_for_recall(y, s, target)
        assert recall_alerts(y, s, t)[0] >= target


def test_expected_cost_and_bayes_threshold():
    assert expected_cost(y, s, 0.35, 10, 1) == pytest.approx((10 * 1 + 1 * 1) / 10)
    assert bayes_threshold(cost_miss=9, cost_false_alarm=1) == pytest.approx(0.1)


def test_softmax_and_contextual_calibration():
    L = np.log(np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]))
    assert np.allclose(softmax_rows(L).sum(1), 1)
    # if every input shares the same label bias, subtracting it removes the bias
    P = contextual_calibrate(L, np.log([0.7, 0.2, 0.1]))
    assert np.allclose(P[0], 1 / 3)
    assert P[1].argmax() != 0 or P[1][0] < 0.6


def test_flesch_orders_texts_sensibly():
    easy = "I am sad. I feel bad. I want help."
    hard = "Notwithstanding considerable psychotherapeutic intervention, persistent melancholic rumination continues."
    assert flesch_reading_ease(easy) > flesch_reading_ease(hard)


def test_redaction_on_invented_strings():
    c = Counter()
    out = redact_regex("mail me at jo.bloggs@example.com or see www.example.org, call +44 7700 900123, ask u/someone", c)
    assert "@" not in out and "example.org" not in out and "7700" not in out and "u/someone" not in out
    assert c["EMAIL"] == 1 and c["PHONE"] == 1 and c["USERNAME"] == 1


def test_no_post_text_or_author_committed():
    """Committed data must never hold post text or author names."""
    for f in (ROOT / "data").rglob("*.csv"):
        if "raw" in f.parts:
            continue
        cols = pd.read_csv(f, nrows=1).columns
        assert "post" not in cols and "author" not in cols, f
        if "text" in cols:  # only the hand-written safety tests may carry text
            assert f.name in {"safety_tests.csv", "safety_results.csv"}, f
