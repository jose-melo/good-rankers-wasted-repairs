"""Per-feature KS and the conditional test calibrated on blocks of consecutive rows (KS_B, COND_B).

On a window of consecutive rows the statistic of the window is ranked among the same statistic over the disjoint blocks
of w consecutive calibration rows: p_j = (1 + #{b : S_b >= S_win}) / (1 + B). The test abstains (p = 1) when the
reference holds fewer than d / 0.10 blocks.
  KS_B:   S = two-sample KS distance between the rows (window or block) of x_j and the fit rows of x_j.
  COND_B: S = mean |r|, r the residual of the predictor of x_j from all other features.
On any other window both return the ordinary tests (KS against all reference rows; the conditional rank test).
"""
import math

import numpy as np
from scipy.stats import ks_2samp

from ..harness import consecutive_runs, is_serial, mw, spacing

Q = 0.10
ENTRANTS = {"KS_B": "p", "COND_B": "p"}


def ks_dist(ref_sorted, x):
    """Exact two-sample KS distance, computed in integers so equal distances compare exactly."""
    xs = np.sort(np.asarray(x, float)); m = len(xs); n = len(ref_sorted)
    cr_r = np.searchsorted(ref_sorted, xs, side="right").astype(np.int64)
    cr_l = np.searchsorted(ref_sorted, xs, side="left").astype(np.int64)
    cx_r = np.searchsorted(xs, xs, side="right").astype(np.int64)
    cx_l = np.searchsorted(xs, xs, side="left").astype(np.int64)
    num = max(int(np.max(np.abs(cr_r * m - cx_r * n))), int(np.max(np.abs(cr_l * m - cx_l * n))))
    return num / float(n * m)


def block_series(runs_series, w):
    out = []
    for r in runs_series:
        r = np.asarray(r, float); nb = len(r) // w
        if nb:
            out.append(np.ascontiguousarray(r[:nb * w]).reshape(nb, w))
    return out


def block_p(S_blocks, s_win):
    S = np.asarray(S_blocks, float)
    return float((1.0 + np.sum(S >= s_win)) / (len(S) + 1.0))


class BlockCalibrated:
    def __init__(self, ctx):
        self.d, self.names = int(ctx.d), tuple(ctx.entrants)
        d = self.d
        self.ref = np.vstack([ctx.Xf, ctx.Xc])[:, :d]
        self.fit_sorted = [np.sort(ctx.Xf[:, j]) for j in range(d)]
        self.cond = ctx.cond
        self.tcal = None if ctx.tcal is None else np.asarray(ctx.tcal, float)
        self.order = None if self.tcal is None else np.argsort(self.tcal, kind="stable")
        self.Xc = ctx.Xc; self._runs = {}; self._S = {}

    def _runs_for(self, delta):
        if delta not in self._runs:
            self._runs[delta] = consecutive_runs(self.tcal[self.order], delta)
        return self._runs[delta]

    def ordinary(self, W, resid):
        d = self.d; out = {}
        if "KS_B" in self.names:
            out["KS_B"] = np.array([float(ks_2samp(self.ref[:, j], W[:, j]).pvalue) for j in range(d)])
        if "COND_B" in self.names:
            P = np.full(d, np.nan)
            for j in range(d):
                k = (j, ())
                if k in self.cond.cal and k in resid:
                    P[j] = mw(self.cond.cal[k], np.abs(resid[k]))
            out["COND_B"] = P
        return out

    def _stats(self, kind, j, w, delta):
        key = (kind, j, w, float(delta))
        if key not in self._S:
            o = self.order; runs = [r for r in self._runs_for(delta) if len(r) >= w]
            if kind == "ks":
                c = self.Xc[o, j]
                self._S[key] = np.array([ks_dist(self.fit_sorted[j], b)
                                         for Bk in block_series([c[r] for r in runs], w) for b in Bk])
            else:
                c = self.cond.cal[(j, ())][o]
                self._S[key] = np.array([float(np.mean(b)) for Bk in block_series([c[r] for r in runs], w) for b in Bk])
        return self._S[key]

    def blocks(self, W, twin, resid):
        d, w = self.d, len(W)
        delta = spacing(twin)
        nblk = int(sum(len(r) // w for r in self._runs_for(delta)))
        ok = nblk >= math.ceil(d / Q - 1e-9)
        out = {}
        if "KS_B" in self.names:
            P = np.ones(d)
            if ok:
                for j in range(d):
                    P[j] = block_p(self._stats("ks", j, w, delta), ks_dist(self.fit_sorted[j], W[:, j]))
            out["KS_B"] = P
        if "COND_B" in self.names:
            P = np.ones(d)
            if ok:
                for j in range(d):
                    k = (j, ())
                    if k not in self.cond.cal or k not in resid:
                        P[j] = np.nan; continue
                    P[j] = block_p(self._stats("full", j, w, delta), float(np.mean(np.abs(resid[k]))))
            out["COND_B"] = P
        return out

    def window(self, W, twin, consecutive, resid):
        if is_serial(consecutive, twin, self.tcal, len(W)):
            return self.blocks(W, twin, resid)
        return self.ordinary(W, resid)


def fit(ctx):
    return BlockCalibrated(ctx)
