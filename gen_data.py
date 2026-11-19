"""Synthetic design -> post-P&R QoR dataset for ML-for-EDA.

The response surface is anchored to plausible physical models (Rent-style
netlist stats, linear-ish delay, Elmore-style wireload) and calibrated to the
order of magnitude of the portfolio's own measured numbers (STA WNS on the
LFSR, HLS GEMM area/cycles). Ground truth is deterministic + ~5% noise, and
target values live in realistic ranges:

  feature  log2_gates  in [6 .. 12]   (~64 .. 4k cells)
  feature  depth       in [4 .. 40]   logic levels on the crit path
  feature  fanout      in [2 .. 8]    avg fanout
  feature  wireload    in [0.5 .. 2]  interconnect factor
  feature  period      in [3 .. 10]ns target clock period

  target  WNS          path_delay = depth * fanout^0.3 * wireload * 1.3 + 0.9
                      WNS = period - path_delay
  target  area         gates * (1 + 0.08*fanout)                  (cells)
  target  power        area * fanout * activity(0.15) / period     (uW)

Synthetic is a deliberate first step: it lets the whole pipeline (train,
validate, report R^2/RMSE) run on M1 in seconds. Swap gen_data for a real
Yosys+OpenROAD report parser later and retrain with identical code.
"""
import json
import os
import random

import numpy as np
import torch

SEED = 0
N_SAMPLES = 4000
DATA = os.path.join(os.path.dirname(__file__), "data", "qor.pt")

FEATURES = ["log2_gates", "depth", "fanout", "wireload", "period_ns"]
TARGETS = ["wns_ns", "area_cells", "power_uW"]


def _sample(rng: random.Random) -> tuple[list[float], list[float]]:
    lg = rng.uniform(6.0, 12.0)
    depth = rng.uniform(4.0, 40.0)
    fanout = rng.uniform(2.0, 8.0)
    wireload = rng.uniform(0.5, 2.0)
    period = rng.uniform(3.0, 10.0)

    gates = 2.0 ** lg
    path_delay = depth * (fanout ** 0.3) * wireload * 1.3 + 0.9
    wns = period - path_delay
    area = gates * (1.0 + 0.08 * fanout)
    power = area * fanout * 0.15 / period

    noise = lambda: 1.0 + rng.gauss(0.0, 0.05)         # ~5% measurement noise
    x = [lg, depth, fanout, wireload, period]
    y = [wns * noise(), area * noise(), power * noise()]
    return x, y


def make_dataset(force: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if not force and os.path.exists(DATA):
        d = torch.load(DATA)
        return d["x"], d["y"]

    rng = random.Random(SEED)
    xs, ys = [], []
    for _ in range(N_SAMPLES):
        x, y = _sample(rng)
        xs.append(x)
        ys.append(y)
    x = torch.tensor(np.array(xs, dtype=np.float32))
    y = torch.tensor(np.array(ys, dtype=np.float32))
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    torch.save({"x": x, "y": y}, DATA)
    return x, y