import numpy as np
from train import choose_threshold

def test_modes_respect_recall_constraint():
    y = np.array([0,0,0,1,1,1])
    prob = np.array([.01,.15,.45,.3,.65,.9])
    high, _ = choose_threshold(y, prob, min_recall=.85)
    precise, _ = choose_threshold(y, prob, min_recall=.65)
    assert high["recall"] >= .85
    assert precise["recall"] >= .65
    assert precise["threshold"] >= high["threshold"]

def test_returns_candidates():
    _, options = choose_threshold(np.array([0,1]), np.array([.1,.9]))
    assert len(options) > 1
