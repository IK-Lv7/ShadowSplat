import numpy as np

from workstation.eval.carla_threshold import auroc, fpr_at_95tpr, pick_threshold


def test_auroc_perfect_and_chance():
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.5  # ties count as 0.5


def test_fpr_at_95tpr_and_threshold():
    neg = np.arange(100, dtype=float)            # 0..99
    pos = np.arange(50, 150, dtype=float)        # 50..149
    assert fpr_at_95tpr(neg, pos) == (neg >= np.percentile(pos, 5)).mean()
    thr = pick_threshold(neg, 0.05)
    assert (neg > thr).mean() <= 0.05
