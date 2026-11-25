"""CATS, Controlled Anomalies Time Series v2 (Solenix, Zenodo 10.5281/zenodo.8338435): a simulated control process,
17 channels at 1 Hz, kept one row every 10 s (scripts/prepare_cats.py). The 10 telemetry and control channels (b*, c*)
are tested; the 7 command channels (a*) are conditioning inputs.

The 200 labelled anomalies, in time order, are cut into groups: 1000 = anomalies 0-9, 1001 = 10-19, study group
g = 0..19 = anomalies 20 + 9g .. 28 + 9g. A nominal row is usable when it is labelled 0 and lies outside
[start - 60 s, end + 600 s] of every anomaly. The initial nominal period (one run before the first anomaly) is cut into
a fit stretch (every 10th row kept), 30,000 consecutive calibration rows and 20,000 consecutive held-out rows, in an
order that rotates with the group index.

Datasets
  reference  seeds 1000, 1001, 0..19: no-change windows of 50 and 250 consecutive usable rows between the anomalies of
             the group (threshold pool of the classifier and FSL-Net).
  reserved   seeds 3000 + g: the reference of study group g; windows from the usable tiles of 250 rows that no
             reference window used, dealt in contiguous chunks of 43 tiles per seed. 20 tiles give no-change windows;
             the rest form the pool on which the feature shifts are injected, on random rows and on runs of
             consecutive rows (types <type>_c).
"""
import os

import numpy as np

from .. import config

NAME = "cats"
KIND = "lgbm"
CACHE = os.path.join(config.CATS, "cats_thin.npz")
CONSEC_CAL, CONSEC_HOLD, CONSEC_GAP, CONSEC_FIT_STEP = 30000, 20000, 250, 10
STUDY_SEEDS = list(range(20))
EVAL_SEEDS = list(range(3000, 3020))
N_FIRST, PER_SEED = 20, 9
NONE_WS = (50, 250)
N_NONE = 10
GUARD_BEFORE, GUARD_AFTER = 60, 600
POOL_INJECT = dict(ws=(50, 250), reps=5, types=("offset", "decoupled", "stuck", "half", "two", "many", "select"),
                   cons_types=("offset", "decoupled", "stuck", "half", "two", "many"))
_C = {}


def _cache():
    if not _C:
        z = np.load(CACHE, allow_pickle=False)
        X = z["X"]; y = z["y"]; row = z["row"]
        seg_lo, seg_hi = z["seg_lo"], z["seg_hi"]
        names = [str(c) for c in z["names"]]
        ok = (y == 0)
        for a, b in zip(seg_lo, seg_hi):
            i0 = np.searchsorted(row, a - GUARD_BEFORE, "left")
            i1 = np.searchsorted(row, b + GUARD_AFTER, "right")
            ok[i0:i1] = False
        usable = np.flatnonzero(ok)
        seg_rows = [np.flatnonzero((row >= a) & (row <= b)) for a, b in zip(seg_lo, seg_hi)]
        groups = [("first", 1000, list(range(0, 10))), ("first", 1001, list(range(10, N_FIRST)))]
        for s in STUDY_SEEDS:
            groups.append(("study", s, list(range(N_FIRST + PER_SEED * s, N_FIRST + PER_SEED * (s + 1)))))
        assert groups[-1][2][-1] == len(seg_lo) - 1 == 199
        starts = [int(seg_rows[g[2][0]][0]) for g in groups] + [len(row)]
        init = usable[usable < starts[0]]
        _C.update(X=X, row=row, names=names, usable=usable, groups=groups, starts=starts, init=init)
    return _C


def _group_index(seed):
    ks = [k for k, g in enumerate(_cache()["groups"]) if g[1] == seed]
    if len(ks) != 1:
        raise ValueError(f"unknown seed {seed!r}")
    return ks[0]


def reference_rows(k):
    """Fit, calibration and held-out rows of group index k (consecutive stretches of the initial nominal period)."""
    init = _cache()["init"]; assert (np.diff(init) == 1).all()
    n_fit_raw = len(init) - CONSEC_CAL - CONSEC_HOLD - 2 * CONSEC_GAP
    order = [("fit", "cal", "hold"), ("hold", "fit", "cal"), ("cal", "hold", "fit")][k % 3]
    size = dict(fit=n_fit_raw, cal=CONSEC_CAL, hold=CONSEC_HOLD); seg = {}; pos = 0
    for nm in order:
        seg[nm] = init[pos:pos + size[nm]]; pos += size[nm] + CONSEC_GAP
    return seg["fit"][::CONSEC_FIT_STEP], seg["cal"], seg["hold"]


def tiles_of_group(k):
    """Tiles of 250 consecutive usable rows between the anomalies of group k, and the indices of those used as
    reference windows."""
    C = _cache(); usable = C["usable"]
    free = usable[(usable >= C["starts"][k]) & (usable < C["starts"][k + 1])]
    wmax = max(NONE_WS); cuts = np.flatnonzero(np.diff(free) != 1) + 1; tiles = []
    for run in np.split(free, cuts):
        for q in range(len(run) // wmax):
            tiles.append(run[q * wmax:(q + 1) * wmax])
    pick = np.unique(np.round(np.linspace(0, len(tiles) - 1, N_NONE * len(NONE_WS))).astype(int))
    return tiles, pick


def reserved_chunks(n_seeds=len(EVAL_SEEDS)):
    C = _cache(); allt = []
    for k, (kind, _, _) in enumerate(C["groups"]):
        if kind != "study":
            continue
        tiles, pick = tiles_of_group(k); used = set(int(p) for p in pick)
        allt += [tiles[i] for i in range(len(tiles)) if i not in used]
    allt.sort(key=lambda t: int(t[0])); n_per = len(allt) // n_seeds
    return [allt[g * n_per:(g + 1) * n_per] for g in range(n_seeds)]


def _columns():
    names = _cache()["names"]
    tested = [j for j, c in enumerate(names) if c[0] in "bc"]
    aux = [j for j, c in enumerate(names) if c[0] == "a"]
    assert len(tested) == 10 and len(aux) == 7
    return tested, aux, names


def _none_unit(X, idx, rep, note):
    z = np.zeros(10, bool)
    return dict(W=X[idx].copy(), src=z.copy(), mov=z.copy(), cnd=z.copy(), type="none", mag=0.0, w=int(len(idx)),
                rep=rep, t=idx.astype(float), consecutive=True, note=note)


def load(dataset, seed):
    C = _cache(); X = C["X"]; tested, aux, names = _columns(); seed = int(seed)
    if dataset == "reference":
        k = _group_index(seed)
        fit_idx, cal_idx, _ = reference_rows(k)
        tiles, pick = tiles_of_group(k)
        units, reps = [], {w: 0 for w in NONE_WS}
        for i, p in enumerate(pick):
            w = NONE_WS[i % len(NONE_WS)]; idx = tiles[p][:w]
            units.append(_none_unit(X, idx, reps[w], "consecutive nominal rows between anomalies")); reps[w] += 1
        return dict(Xfit=X[fit_idx].copy(), Xcal=X[cal_idx].copy(), tested=tested, aux=aux, names=names, units=units,
                    tfit=fit_idx.astype(float), tcal=cal_idx.astype(float))
    if dataset != "reserved" or seed not in EVAL_SEEDS:
        raise ValueError(f"unknown dataset/seed {dataset!r} {seed}")
    g = seed - 3000
    fit_idx, cal_idx, _ = reference_rows(_group_index(g))
    avail = reserved_chunks()[g]
    allrows = set(np.concatenate(avail).tolist())
    assert not (allrows & (set(fit_idx.tolist()) | set(cal_idx.tolist())))
    n_need = N_NONE * len(NONE_WS)
    sel = np.unique(np.round(np.linspace(0, len(avail) - 1, n_need)).astype(int)); assert len(sel) == n_need
    units, reps = [], {w: 0 for w in NONE_WS}
    for i, p in enumerate(sel):
        w = NONE_WS[i % len(NONE_WS)]; idx = avail[p][:w]
        units.append(_none_unit(X, idx, reps[w], "consecutive nominal rows, reserved tile")); reps[w] += 1
    pool_idx = np.concatenate([avail[i] for i in range(len(avail)) if i not in set(sel.tolist())])
    return dict(Xfit=X[fit_idx].copy(), Xcal=X[cal_idx].copy(), tested=tested, aux=aux, names=names, units=units,
                pool=X[pool_idx].copy(), tfit=fit_idx.astype(float), tcal=cal_idx.astype(float),
                tpool=pool_idx.astype(float))
