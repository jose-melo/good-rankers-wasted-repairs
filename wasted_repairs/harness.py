"""Shared pieces of every run: rank p-values, the conditional predictors, and the injected feature shifts."""
import itertools

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu


def conformal_p(cal, s):
    """Conformal p-value of each value in s among the calibration values (larger = more extreme)."""
    cal = np.sort(np.asarray(cal)); n = len(cal)
    return (1 + n - np.searchsorted(cal, s, side="left")) / (n + 1)


def mw(cal, win):
    """One-sided rank test that the window values are larger than the calibration values."""
    if len(win) == 1:
        return float(conformal_p(cal, win)[0])
    return float(mannwhitneyu(win, cal, alternative="greater").pvalue)


def make_model(kind, seed):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1, random_state=seed, n_jobs=1)


class Cond:
    """Predictors of every tested feature x_j: one that reads all other features (key (j, ())) and, for each other
    tested feature k, one that leaves k out (key (j, (k,))). Auxiliary inputs are read by all of them."""

    def __init__(self, kind, Xf, Xc, d, aux, seed):
        self.d, self.aux, self.m, self.cal = d, list(aux), {}, {}
        for j in range(d):
            others = [c for c in range(d) if c != j]
            for T in [()] + [(t,) for t in others]:
                cols = [c for c in others if c not in T] + self.aux
                if not cols:
                    continue
                mod = make_model(kind, seed).fit(Xf[:, cols], Xf[:, j])
                self.m[(j, T)] = (mod, cols)
                self.cal[(j, T)] = np.abs(Xc[:, j] - mod.predict(Xc[:, cols]))
        self.byj = {j: [k for k in self.m if k[0] == j] for j in range(d)}

    def residuals(self, W):
        """Signed window residual of every predictor, keyed as self.m."""
        return {k: W[:, k[0]] - mod.predict(W[:, cols]) for k, (mod, cols) in self.m.items()}

    def full_pvals(self, resid):
        """Conditional test: rank of the window's absolute residuals of the predictor that reads all other features."""
        out = np.full(self.d, np.nan)
        for j in range(self.d):
            if (j, ()) in self.cal:
                out[j] = mw(self.cal[(j, ())], np.abs(resid[(j, ())]))
        return out


def ks_pvals(ref, W):
    """Per-feature two-sample Kolmogorov-Smirnov p-values, window against all reference rows."""
    return np.array([float(ks_2samp(ref[:, j], W[:, j]).pvalue) for j in range(W.shape[1])])


def holm(p, a):
    """Holm's step-down procedure; NaN counts as 1."""
    p = np.where(np.isnan(p), 1.0, p); o = np.argsort(p); m = len(p); rej = np.zeros(m, bool)
    for r, i in enumerate(o):
        if p[i] <= a / (m - r):
            rej[i] = True
        else:
            break
    return rej


# ---------------------------------------------------------------- injected feature shifts
def select_rows(rng, zt, beta, sign, w):
    """w rows drawn without replacement with probability proportional to exp(beta * sign * zt). Falls back to the
    Gumbel-top-k draw of the same law when the weights underflow."""
    zt = np.asarray(zt, float); n = len(zt)
    with np.errstate(all="ignore"):
        pr = np.exp(beta * sign * (zt - zt.mean())); pr /= pr.sum()
    try:
        return rng.choice(n, w, replace=False, p=pr), ""
    except ValueError:
        pass
    with np.errstate(all="ignore"):
        lw = beta * sign * zt
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    nfin = int(np.isfinite(lw).sum())
    key = lw + rng.gumbel(size=n)
    idx = np.argsort(-key, kind="stable")[:min(w, nfin)]
    note = "sel_draw=gumbel"
    if nfin < w:
        rest = np.setdiff1d(np.arange(n), idx)
        idx = np.r_[idx, rng.choice(rest, w - nfin, replace=False)]; note += f" sel_short={w - nfin}"
    return idx, note


def _windows(rng, n_pool, w, reps):
    """reps windows of w pool rows, disjoint while the pool lasts; returns (index arrays, overlap flag)."""
    idx, overlap, perm = [], False, rng.permutation(n_pool)
    pos = 0
    for _ in range(reps):
        if pos + w > n_pool:
            perm = rng.permutation(n_pool); pos = 0; overlap = True
        idx.append(perm[pos:pos + w]); pos += w
    return idx, overlap


def inject_units(pool, d, sd, corr, rng, ws, reps, types, blanket=None, off_mags=(1.0, 3.0), sel_beta=1.0, tpool=None,
                 cons_types=()):
    """Windows of clean pool rows with a known feature shift applied. The first d columns of pool are tested.

    Each unit holds the window W and the answer keys: src (the acted-on set S), mov (features whose marginal moves),
    cnd (acted-on features and their Markov blanket, when blanket is given), grey (features excluded from scoring).
    Types on random pool rows: none, offset (mags off_mags, in sd), decoupled (column replaced by other pool rows: its
    marginal is kept, its dependence is broken), stuck, half (+3 sd from the middle of the window), two and many
    (+-3 sd on 2 or d // 2 features), valdep, select (rows selected on one feature: nothing is acted on).
    cons_types repeat these shifts on runs of consecutive pool rows (types named <type>_c)."""
    U = []; z = np.zeros(d, bool); cur = {}

    def unit(W, src, mov, ty, mag, w, rep, note="", grey=None, ext=-1):
        if blanket is None:
            cnd = None
        else:
            cnd = z.copy()
            for s in np.nonzero(src)[0]:
                cnd[s] = True; cnd[list(blanket[s])] = True
        U.append(dict(W=W, src=src, mov=mov, cnd=cnd, type=ty, mag=float(mag), w=int(w), rep=int(rep), note=note,
                      grey=z.copy() if grey is None else grey, ext=int(ext),
                      t=None if tpool is None else np.asarray(tpool)[cur["ix"]].astype(float), consecutive=False))

    for ty in types:
        mags = off_mags if ty == "offset" else (0.0,)
        for mag in mags:
            for w in ws:
                if ty in ("stuck", "half", "select") and w < 10:
                    continue
                if ty == "valdep" and w != 1:
                    continue
                wins, ov = _windows(rng, len(pool), w, reps); note = "overlap" if ov else ""
                for rep, ix in enumerate(wins):
                    W = pool[ix].copy(); src = z.copy(); mov = z.copy(); s = int(rng.integers(d))
                    sign = 1.0 if rng.random() < 0.5 else -1.0
                    cur["ix"] = ix
                    if ty == "none":
                        pass
                    elif ty == "offset":
                        W[:, s] += sign * mag * sd[s]; src[s] = mov[s] = True
                    elif ty == "decoupled":
                        W[:, s] = pool[rng.integers(len(pool), size=w), s]; src[s] = True
                    elif ty == "stuck":
                        W[:, s] = W[0, s]; src[s] = mov[s] = True
                    elif ty == "half":
                        W[w // 2:, s] += sign * 3.0 * sd[s]; src[s] = mov[s] = True
                    elif ty in ("two", "many"):
                        k = 2 if ty == "two" else max(2, d // 2)
                        for t in rng.choice(d, k, replace=False):
                            W[:, t] += (1.0 if rng.random() < 0.5 else -1.0) * 3.0 * sd[t]; src[t] = mov[t] = True
                    elif ty == "valdep":
                        zrow = np.abs(W[0, :d] / sd); e = int(np.argmax(zrow))
                        c = np.abs(corr[e]).copy(); c[e] = -1; s = int(np.argmax(c))
                        W[:, s] += sign * 3.0 * sd[s]; src[s] = mov[s] = True
                        unit(W, src, mov, ty, 3.0, w, rep, note, ext=e); continue
                    elif ty == "select":
                        zt = pool[:, s] / sd[s]
                        ix2, snote = select_rows(rng, zt, sel_beta, sign, w)
                        W = pool[ix2].copy(); cur["ix"] = ix2
                        eff = np.abs(corr[s]) * sel_beta; mov = eff >= 0.2; grey = (eff >= 0.1) & (eff < 0.2)
                        nt = (note + f" sel={s} " + snote).strip() if snote else note + f" sel={s}"
                        unit(W, src, mov, ty, sel_beta, w, rep, nt, grey=grey); continue
                    unit(W, src, mov, ty, mag if ty == "offset" else (3.0 if ty in ("half", "two", "many") else 0.0), w,
                         rep, note)
    if cons_types:
        assert tpool is not None, "consecutive shifts need pool positions"
        tp = np.asarray(tpool, float); dts = np.diff(tp); delta = float(np.median(dts))
        runs = np.split(np.arange(len(tp)), np.flatnonzero(np.abs(dts - delta) > 1e-6 * max(1.0, delta)) + 1)

        def unit_c(W, src, mov, ty, mag, w, rep, note):
            unit(W, src, mov, ty + "_c", mag, w, rep, note); U[-1]["consecutive"] = True

        for ty in cons_types:
            mags = off_mags if ty == "offset" else (0.0,)
            for mag in mags:
                for w in ws:
                    if ty in ("stuck", "half") and w < 10:
                        continue
                    blocks = [r[q * w:(q + 1) * w] for r in runs for q in range(len(r) // w)]
                    assert len(blocks) >= 2, "pool holds fewer than two runs of w consecutive rows"
                    pick = rng.permutation(len(blocks)); ov = len(blocks) < reps
                    for rep in range(reps):
                        ix = blocks[pick[rep % len(blocks)]]; cur["ix"] = ix
                        W = pool[ix].copy(); src = z.copy(); mov = z.copy(); s = int(rng.integers(d))
                        sign = 1.0 if rng.random() < 0.5 else -1.0
                        if ty == "offset":
                            W[:, s] += sign * mag * sd[s]; src[s] = mov[s] = True
                        elif ty == "decoupled":
                            cand = [b for b in range(len(blocks)) if abs(int(blocks[b][0]) - int(ix[0])) >= w]
                            W[:, s] = pool[blocks[cand[int(rng.integers(len(cand)))]], s]; src[s] = True
                        elif ty == "stuck":
                            W[:, s] = W[0, s]; src[s] = mov[s] = True
                        elif ty == "half":
                            W[w // 2:, s] += sign * 3.0 * sd[s]; src[s] = mov[s] = True
                        elif ty in ("two", "many"):
                            k = 2 if ty == "two" else max(2, d // 2)
                            for t in rng.choice(d, k, replace=False):
                                W[:, t] += (1.0 if rng.random() < 0.5 else -1.0) * 3.0 * sd[t]; src[t] = mov[t] = True
                        elif ty != "none":
                            raise ValueError(f"no consecutive version of {ty!r}")
                        unit_c(W, src, mov, ty,
                               mag if ty == "offset" else (3.0 if ty in ("half", "two", "many") else 0.0), w, rep,
                               "overlap" if ov else "")
    return U


def consecutive_runs(tc, delta):
    """Index arrays (into time-sorted positions tc) of the maximal runs whose successive positions differ by delta."""
    if len(tc) == 0:
        return []
    br = np.flatnonzero(np.abs(np.diff(tc) - delta) > 1e-6 * max(1.0, delta)) + 1
    return np.split(np.arange(len(tc)), br)


def is_serial(consecutive, twin, tcal, w):
    """A window is ranked among calibration blocks when its rows are consecutive and positions are known."""
    return bool(consecutive) and twin is not None and tcal is not None and w > 1


def spacing(twin):
    return float(np.median(np.diff(np.sort(np.asarray(twin, float)))))
