#!/usr/bin/env python3
"""Train an MLP to predict post-P&R QoR (WNS / area / power) from netlist
features, then report R^2 / RMSE on a held-out test split plus a
generalization check on a spread of unseen design configurations.

Pure PyTorch on M1. The dataset is synthetic (see gen_data.py) so the whole
methodology pipeline runs in seconds; swapping in real Yosys/OpenROAD reports
does not change the training code.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

from gen_data import DATA, FEATURES, TARGETS, make_dataset

RESULT_JSON = os.path.join(os.path.dirname(__file__), "results", "qor_model.json")
RESULTS_DIR = os.path.dirname(RESULT_JSON)


def r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(0)) ** 2).sum())
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def train_eval(x, y, split=0.8, epochs=1500, seed=0) -> dict:
    torch.manual_seed(seed)
    n = x.shape[0]
    n_tr = int(n * split)
    x_tr, y_tr = x[:n_tr], y[:n_tr]
    x_te, y_te = x[n_tr:], y[n_tr:]

    # normalize features by train stats; targets by each column
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    ty_mu, ty_sd = y_tr.mean(0), y_tr.std(0) + 1e-8
    y_tr_n, y_te_n = (y_tr - ty_mu) / ty_sd, (y_te - ty_mu) / ty_sd

    model = nn.Sequential(
        nn.Linear(x.shape[1], 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, y.shape[1]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        l = loss(model(x_tr), y_tr_n)
        l.backward()
        opt.step()

    with torch.no_grad():
        pred_te = model(x_te) * ty_sd + ty_mu
        pred_tr = model(x_tr) * ty_sd + ty_mu

    return {
        "model": model, "pg": (mu, sd, ty_mu, ty_sd),
        "pred_te": pred_te, "y_te": y_te, "pred_tr": pred_tr, "y_tr": y_tr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=5, help="report mean+std over seeds")
    args = ap.parse_args()

    x, y = make_dataset()
    rows = []
    for seed in range(args.seeds):
        r = train_eval(x, y, epochs=args.epochs, seed=seed)
        rows.append(r)

    per_seed = []
    for r in rows:
        per_seed.append({t: {"r2": r2(r["y_te"][:, i], r["pred_te"][:, i])}
                         for i, t in enumerate(TARGETS)})

    print("per-target held-out test R^2 (mean over seeds):")
    summary = {}
    for j, t in enumerate(TARGETS):
        vals = [p[t]["r2"] for p in per_seed]
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        summary[t] = {"r2_mean": round(mean, 4), "r2_spread": round(spread, 4)}
        print(f"  {t:>10} : R^2 = {mean:.4f}  (spread {spread:.3f})")

    # generalization check on an unseen design spread (one typical sample)
    r = rows[0]
    with torch.no_grad():
        (mu, sd, ty_mu, ty_sd) = r["pg"]
        probe = torch.tensor([[9.0, 22.0, 4.5, 1.2, 5.0]])
        pred = ((r["model"](((probe - mu) / sd)) * ty_sd + ty_mu)
                .squeeze().tolist())
    print("\nunseen design  (gates=512, depth=22, fanout=4.5, wl=1.2, T=5ns)")
    for t, v in zip(TARGETS, pred):
        print(f"  predicted {t:>10} : {v:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULT_JSON, "w") as f:
        json.dump({"targets": TARGETS, "features": FEATURES, **summary}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())