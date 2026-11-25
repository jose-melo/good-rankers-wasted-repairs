"""L-Town water network (BattLeDIM, Vrachimis et al. 2022), leak-free hydraulic simulations
(scripts/simulate_ltown.py): 33 pressure sensors tested; 3 flows, 1 tank level and hour-of-day / weekend inputs are
conditioning inputs. Values rounded to 0.01 as published, one row per hour.

One year per seed: fit = four whole weeks (days 0, 91, 182, 273), calibration = every other hour before the last week
(7,920 h, 330 blocks of 24 h), test = the last 7 days, one window of 24 consecutive hours per day and fault type.
Fault types: none, offset 0.25 and 1 sd, stuck, decoupled (replay of the sensor 168 h earlier), half (+1 sd from the
middle of the window); one sensor drawn per window.
Seeds 0-5: the published year (sim_published.npz) rolled by a seeded number of weeks (threshold pool).
Seeds 3000-3019: re-simulations with perturbed demands (sim/noise_<seed>.npz, evaluation)."""
import os

import numpy as np

from .. import config

NAME = "ltown"
KIND = "ridge"
D = 33; WK = 168; DAY = 24; NH = 8760
DATASETS = ["ltown"]
POOL_SEEDS = list(range(6)); EVAL_SEEDS = list(range(3000, 3020))
FIT_DAYS = (0, 91, 182, 273)
CELLS = (("none", 0.0), ("offset", 0.25), ("offset", 1.0), ("stuck", 0.0), ("decoupled", 0.0), ("half", 1.0))
_cache = {}


def layout():
    fit = np.concatenate([np.arange(d * DAY, d * DAY + WK) for d in FIT_DAYS])
    test = np.arange(NH - WK, NH)
    used = np.zeros(NH, bool); used[fit] = True; used[test] = True
    return fit, np.flatnonzero(~used), test


def hourly(P, F, L):
    P, F, L = (np.round(np.asarray(Z, float), 2)[::12] for Z in (P, F, L))
    h = np.arange(len(P)); hod = h % 24; dow = (h // 24) % 7
    return np.column_stack([P, F, L, np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
                            (dow >= 5).astype(float)])


def _series(seed):
    seed = int(seed)
    if seed in POOL_SEEDS:
        if "pub" not in _cache:
            z = np.load(os.path.join(config.LTOWN, "sim_published.npz"), allow_pickle=True)
            _cache["pub"] = hourly(z["P"], z["F"], z["L"])
        X = _cache["pub"]; assert len(X) == NH, X.shape
        k = int(np.random.default_rng([20261125, seed, 9]).integers(52))
        Xr = np.roll(X[:, :-3], -7 * k * DAY, axis=0)
        return np.column_stack([Xr, X[:, -3:]])
    if seed not in EVAL_SEEDS:
        raise ValueError(f"unknown seed {seed}")
    z = np.load(os.path.join(config.LTOWN, "sim", f"noise_{seed}.npz"), allow_pickle=True)
    assert int(z["seed"]) == seed and int(z["days"]) == 365
    X = hourly(z["P"], z["F"], z["L"]); assert len(X) == NH, X.shape
    return X


def load(dataset, seed):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    sim = _series(seed); seed = int(seed)
    fit, cal, test = layout(); rng = np.random.default_rng([500, seed])
    sd = sim[fit, :D].std(0); z = np.zeros(D, bool); units = []; t0 = test[0]
    for ty, mag in CELLS:
        w = DAY; reps = len(test) // w
        starts = t0 + rng.permutation(len(test) // w)[:reps] * w
        for rep, a in enumerate(starts):
            ix = np.arange(a, a + w); src = z.copy(); mov = z.copy(); s = int(rng.integers(D))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            W = sim[ix].copy()
            if ty == "offset":
                W[:, s] += sign * mag * sd[s]; src[s] = mov[s] = True
            elif ty == "stuck":
                W[:, s] = W[0, s]; src[s] = mov[s] = True
            elif ty == "decoupled":
                W[:, s] = sim[ix - WK, s]; src[s] = True
            elif ty == "half":
                W[w // 2:, s] += sign * mag * sd[s]; src[s] = mov[s] = True
            units.append(dict(W=W, src=src, mov=mov, cnd=None, type=ty, mag=mag, w=w, rep=rep, note="", grey=z.copy(),
                              ext=-1, t=ix.astype(float), consecutive=True))
    return dict(Xfit=sim[fit], Xcal=sim[cal], tested=list(range(D)), aux=list(range(D, sim.shape[1])),
                names=[f"s{j}" for j in range(D)], units=units, tfit=fit.astype(float), tcal=cal.astype(float))
