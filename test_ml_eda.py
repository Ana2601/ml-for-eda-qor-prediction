#!/usr/bin/env python3
"""Self-test: dataset determinism + held-out R^2 floor for the QoR model."""
import sys

import torch

from gen_data import FEATURES, TARGETS, make_dataset
from qor_model import r2, train_eval

x, y = make_dataset()
x2, y2 = make_dataset()
assert torch.equal(x, x2) and torch.equal(y, y2), "dataset must be deterministic"

assert x.shape[1] == len(FEATURES) == 5
assert y.shape[1] == len(TARGETS) == 3

r = train_eval(x, y, epochs=300, seed=0)   # quick check, fewer epochs
scores = {t: r2(r["y_te"][:, i], r["pred_te"][:, i]) for i, t in enumerate(TARGETS)}
print("held-out R^2:", {k: round(v, 3) for k, v in scores.items()})

ok = all(v > 0.80 for v in scores.values())
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)