"""Causal Chambers light tunnel (Gamella, Peters, Buhlmann, Nat. Mach. Intell. 2025), dataset lt_walks_v1, experiment
actuators_white: 20,000 measurements, the 11 actuators drawn i.i.d. uniform before each one, sensor parameters fixed.
The 9 sensors are tested.

The file is cut into 10 blocks of 2,000 rows; one end block (drawn with a fixed generator) serves the threshold pool,
the other 18,000 rows the evaluation. Per seed, rows are shuffled into fit rows, calibration rows and a pool; windows
of consecutive rows and windows of random pool rows give the no-change cells, and the feature shifts are injected on
random pool rows. Seeds 1000-1019: threshold pool; seeds 3000-3019: evaluation."""
import os

import numpy as np

from .. import config

NAME = "light_tunnel"
KIND = "ridge"
DATASETS = ["white"]
POOL_SEEDS = list(range(1000, 1020))
EVAL_SEEDS = list(range(3000, 3020))
CSV = os.path.join(config.LT_WALKS, "actuators_white.csv")
CACHE = os.path.join(config.CACHE, "lt_walks_v1_actuators_white.npz")

SENSORS = ["current", "angle_1", "angle_2", "ir_1", "vis_1", "ir_2", "vis_2", "ir_3", "vis_3"]
N_ROWS, N_BLOCK = 20000, 2000
POOL_BLOCK = [0, 9][int(np.random.default_rng(20261125).integers(2))]
POOL_ROWS = (POOL_BLOCK * N_BLOCK, (POOL_BLOCK + 1) * N_BLOCK)
EVAL_ROWS = (N_BLOCK, N_ROWS) if POOL_BLOCK == 0 else (0, N_ROWS - N_BLOCK)
WS = (10, 50, 250)
EVAL_LAYOUT = dict(n_fit=2000, n_cal=1000, none_reps=10, iid_reps=10)
POOL_LAYOUT = dict(n_fit=800, n_cal=400, none_reps=1, iid_reps=4)
POOL_INJECT = dict(ws=(10, 50, 250), reps=4, types=("offset", "decoupled", "stuck", "half", "two", "many", "select"))
_C = {}


def _place(rng, n, lens):
    """Disjoint blocks of the given lengths at random positions in range(n)."""
    lens = np.asarray(lens); free = n - int(lens.sum()); assert free >= 0
    order = rng.permutation(len(lens)); gaps = np.sort(rng.integers(0, free + 1, size=len(lens)))
    starts = np.zeros(len(lens), int); used = 0
    for k, b in enumerate(order):
        starts[b] = gaps[k] + used; used += int(lens[b])
    return starts


def plan(seed):
    seed = int(seed)
    if seed not in POOL_SEEDS and seed not in EVAL_SEEDS:
        raise ValueError(f"unknown seed {seed}")
    in_pool = seed in POOL_SEEDS
    lay = POOL_LAYOUT if in_pool else EVAL_LAYOUT
    lo, hi = POOL_ROWS if in_pool else EVAL_ROWS; n = hi - lo
    rng = np.random.default_rng([20261125, seed, 0])
    lens = [w for w in WS for _ in range(lay["none_reps"])]
    starts = _place(rng, n, lens); taken = np.zeros(n, bool)
    for w, s in zip(lens, starts):
        assert not taken[s:s + w].any(); taken[s:s + w] = True
    rest = rng.permutation(np.nonzero(~taken)[0]); nf, nc = lay["n_fit"], lay["n_cal"]
    none = [(int(w), lo + int(s), k % lay["none_reps"]) for k, (w, s) in enumerate(zip(lens, starts))]
    return dict(in_pool=in_pool, fit=lo + rest[:nf], cal=lo + rest[nf:nf + nc], pool=lo + np.sort(rest[nf + nc:]), none=none)


def _sensors():
    if not _C:
        if not os.path.exists(CACHE):
            import pandas as pd
            df = pd.read_csv(CSV)
            assert len(df) == N_ROWS and (df["config"] == "standard").all()
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            np.savez_compressed(CACHE, X=df[SENSORS].to_numpy(float))
        _C["X"] = np.load(CACHE)["X"]
    return _C["X"]


def load(dataset, seed):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    p = plan(seed); X = _sensors(); d = len(SENSORS)
    z = np.zeros(d, bool); units = []
    for w, s, r in p["none"]:
        units.append(dict(W=X[s:s + w].copy(), src=z.copy(), mov=z.copy(), cnd=None, type="none", mag=0.0, w=w, rep=r,
                          note=f"rows {s}:{s + w}", t=np.arange(s, s + w, dtype=float), consecutive=True))
    lay = POOL_LAYOUT if p["in_pool"] else EVAL_LAYOUT; rr = np.random.default_rng([20261125, int(seed), 1])
    for w in WS:
        for r in range(lay["iid_reps"]):
            ix = np.sort(rr.choice(p["pool"], w, replace=False))
            units.append(dict(W=X[ix].copy(), src=z.copy(), mov=z.copy(), cnd=None, type="none_iid", mag=0.0, w=w,
                              rep=r, note=f"{w} random pool rows", t=ix.astype(float), consecutive=False))
    return dict(Xfit=X[p["fit"]].copy(), Xcal=X[p["cal"]].copy(), tested=list(range(d)), aux=[], names=SENSORS,
                units=units, pool=X[p["pool"]].copy(), tfit=p["fit"].astype(float), tcal=p["cal"].astype(float),
                tpool=p["pool"].astype(float))
