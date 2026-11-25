"""Causal Chambers wind tunnel (Gamella, Peters, Buhlmann, Nat. Mach. Intell. 2025), dataset wt_walks_v1, experiment
regime_jumps_multi: one recording of 32 regimes of 10,000 measurements; in each regime some sensor parameters (or the
hatch) are set, as listed in the dataset's regimes_targets.py.

Tested: 7 sensors (current_in, current_out, rpm_in, rpm_out, mic, signal_1, signal_2). The four barometers are left
out: they follow the atmosphere, which drifts over the recording. Reference = regime 0 (rows 100..9999) cut into four
contiguous quarters: seed s holds out quarter s % 4 (its no-change windows and the pool for injected shifts), uses the
next as calibration rows and the other two as fit rows. In every other regime, windows of consecutive rows and of
random rows are drawn; a regime whose targets act on a tested sensor (in the published graph) gives labelled windows
with that sensor in the acted-on set. The feature shifts are injected on random pool rows and on runs of consecutive
pool rows (types <type>_c). Seeds 0-7."""
import ast
import os

import numpy as np

from .. import config

NAME = "wind_tunnel"
KIND = "lgbm"
DATASETS = ["multi"]
SEEDS = list(range(8))
CSV = os.path.join(config.WT_WALKS, "regime_jumps_multi.csv")
CACHE = os.path.join(config.CACHE, "wt_walks_v1_regime_jumps_multi.npz")

SENSORS = ["current_in", "current_out", "rpm_in", "rpm_out", "pressure_upwind", "pressure_downwind", "pressure_ambient",
           "pressure_intake", "mic", "signal_1", "signal_2"]
BARO = ["pressure_upwind", "pressure_downwind", "pressure_ambient", "pressure_intake"]
TESTED = [s for s in SENSORS if s not in BARO]
ACT = ["load_in", "load_out", "pot_1", "pot_2"]
PARAMS = ["hatch", "res_in", "res_out", "osr_upwind", "osr_downwind", "osr_ambient", "osr_intake", "v_in", "v_out", "v_1",
          "v_2"]
FIXED = ["osr_1", "osr_2", "osr_mic", "osr_in", "osr_out", "v_mic"]
COLS = SENSORS + ACT + PARAMS + FIXED + ["flag"]
TARGET_VALUES = {"hatch": (0, 45), "res_in": (1, 0), "res_out": (1, 0), "osr_upwind": (8, 1), "osr_downwind": (8, 1),
                 "osr_ambient": (8, 1), "osr_intake": (8, 1), "v_in": (1.1, 5), "v_out": (1.1, 5), "v_1": (5, 2.56),
                 "v_2": (5, 2.56)}
# published graph of the standard configuration (causalchamber package, ground_truth/main.py, _edges_wt_standard)
CHILDREN = {
    "load_in": ["rpm_in", "rpm_out", "current_in", "current_out", "pressure_intake", "pressure_upwind",
                "pressure_downwind", "mic"],
    "load_out": ["rpm_in", "rpm_out", "current_out", "current_in", "pressure_intake", "pressure_upwind",
                 "pressure_downwind", "mic"],
    "hatch": ["rpm_in", "rpm_out", "pressure_intake", "pressure_upwind", "pressure_downwind", "mic"],
    "res_in": ["rpm_in"], "res_out": ["rpm_out"],
    "osr_intake": ["pressure_intake"], "osr_upwind": ["pressure_upwind"], "osr_downwind": ["pressure_downwind"],
    "osr_ambient": ["pressure_ambient"],
    "osr_in": ["current_in"], "v_in": ["current_in"], "osr_out": ["current_out"], "v_out": ["current_out"],
    "pot_1": ["signal_1", "signal_2", "mic"], "osr_1": ["signal_1"], "v_1": ["signal_1"],
    "pot_2": ["signal_2"], "osr_2": ["signal_2"], "v_2": ["signal_2"],
    "osr_mic": ["mic"], "v_mic": ["mic"],
}
BURN = 100
N_REG = 10000
WS = (10, 50, 250)
R_LAB, R_NONE, R_REF = 4, 10, 3
POOL_INJECT = dict(ws=(10, 50, 250), reps=4, types=("offset", "decoupled", "stuck", "half", "two", "many", "select"),
                   cons_types=("offset", "decoupled", "stuck", "half", "two", "many", "none"))
_C = {}


def _check_graph():
    """CHILDREN must equal the package's _edges_wt_standard."""
    tree = ast.parse(open(os.path.join(config.WT_META, "ground_truth_main.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_edges_wt_standard":
            edges = set(ast.literal_eval(node.value))
            assert edges == {(a, b) for a, bs in CHILDREN.items() for b in bs}
            return
    raise RuntimeError("_edges_wt_standard not found")


def _targets():
    tree = ast.parse(open(os.path.join(config.WT_META, "regimes_targets.py")).read()); T = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            T[node.targets[0].id] = ast.literal_eval(node.value)
    return T["targets_multi"]


def _regimes():
    if not _C:
        if not os.path.exists(CACHE):
            import pandas as pd
            df = pd.read_csv(CSV)
            assert (df["config"] == "standard").all() and not df[COLS].isna().any().any()
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            np.savez_compressed(CACHE, X=df[COLS].to_numpy(float))
        X = np.load(CACHE)["X"]; c = {n: i for i, n in enumerate(COLS)}
        T = _targets(); n_reg = len(T)
        flag = X[:, c["flag"]].astype(int)
        assert X.shape[0] == n_reg * N_REG and (flag == np.repeat(np.arange(n_reg), N_REG)).all()
        for r in range(n_reg):
            blk = X[r * N_REG:(r + 1) * N_REG]
            for p in PARAMS:
                v = np.unique(blk[:, c[p]]); assert v.size == 1, (r, p)
                want = TARGET_VALUES[p][1] if p in T[r] else TARGET_VALUES[p][0]
                assert abs(v[0] - want) < 1e-3, (r, p, v[0], want)
        _check_graph()
        _C.update(X=X[:, [c[s] for s in TESTED]], T=T)
    return _C


def _place(rng, n, lens):
    lens = np.asarray(lens, int); free = n - int(lens.sum()); assert free >= 0
    order = rng.permutation(len(lens)); gaps = np.sort(rng.integers(0, free + 1, size=len(lens)))
    starts = np.zeros(len(lens), int); used = 0
    for k, b in enumerate(order):
        starts[b] = gaps[k] + used; used += int(lens[b])
    return starts


def _windows(rng, rows, reps):
    """reps windows of consecutive rows and reps windows of random rows per w inside rows, all disjoint; reps is
    lowered until they fit, and the largest w is dropped if even one of each does not."""
    n = len(rows); ws = list(WS); r = reps
    while ws:
        while r > 0 and 2 * r * sum(ws) > n:
            r -= 1
        if r > 0:
            break
        ws.pop(); r = reps
    reps_w = {w: r for w in ws}
    lens = [w for w in WS if w in reps_w for _ in range(reps_w[w])]
    st = _place(rng, n, lens); taken = np.zeros(n, bool); out = []; k = 0
    for w in WS:
        for rep in range(reps_w.get(w, 0)):
            s = st[k]; k += 1; assert not taken[s:s + w].any(); taken[s:s + w] = True
            out.append((w, rep, rows[s:s + w], True))
    free = rng.permutation(np.nonzero(~taken)[0]); pos = 0
    for w in WS:
        for rep in range(reps_w.get(w, 0)):
            out.append((w, rep, rows[np.sort(free[pos:pos + w])], False)); pos += w
    return out


def load(dataset, seed):
    if dataset not in DATASETS or int(seed) not in SEEDS:
        raise ValueError(f"unknown dataset/seed {dataset!r} {seed}")
    seed = int(seed); C = _regimes(); X, T = C["X"], C["T"]; d = len(TESTED); z = np.zeros(d, bool)
    q = (N_REG - BURN) // 4; quarters = [np.arange(BURN + i * q, BURN + (i + 1) * q) for i in range(4)]
    k = seed % 4; hold = quarters[k]; cal = quarters[(k + 1) % 4]
    fit = np.sort(np.concatenate([quarters[(k + 2) % 4], quarters[(k + 3) % 4]]))
    units = []

    def add(idx, src, ty, w, rep, note, consec):
        units.append(dict(W=X[idx].copy(), src=src, mov=src.copy(), cnd=None, type=ty, mag=float(src.sum()), w=int(w),
                          rep=int(rep), note=note, t=idx.astype(float), consecutive=bool(consec)))

    rng = np.random.default_rng([20261122, seed, 0])
    for w, rep, idx, cons in _windows(rng, hold, R_REF):
        add(idx, z.copy(), "none" if cons else "none_iid", w, rep, "exp=multi:r0", cons)
    for r in range(1, len(T)):
        rows = np.arange(r * N_REG + BURN, (r + 1) * N_REG)
        src = np.array([any(s in CHILDREN[t] for t in T[r]) for s in TESTED]); lab = bool(src.any())
        rr = np.random.default_rng([20261122, seed, 1, r])
        for w, rep, idx, cons in _windows(rr, rows, R_LAB if T[r] else R_NONE):
            ty = ("labelled" if cons else "labelled_iid") if lab else ("none" if cons else "none_iid")
            add(idx, src.copy(), ty, w, rep, f"exp=multi:r{r}", cons)
    return dict(Xfit=X[fit].copy(), Xcal=X[cal].copy(), tfit=fit.astype(float), tcal=cal.astype(float),
                tested=list(range(d)), aux=[], names=TESTED, units=units, pool=X[hold].copy(), tpool=hold.astype(float))
