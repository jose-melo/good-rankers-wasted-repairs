"""Leave-one-out conformal localiser (ours).

Feature x_j is judged with d - 1 predictors, each fitted on all other features but one. Each predictor gives a
conformal p-value for the window's residuals (signed two-sided rank test and one-sided test of the absolute residuals,
combined by twice the smaller); x_j takes the largest over its predictors.

On rows that are not consecutive in time:
  stage 1   the p-values above; for w > 1 the calibration residuals are weighted by a density ratio estimated on the
            predictor's own inputs;
  re-test   each feature flagged by Holm's procedure at 0.10 is re-tested with predictors that leave out every other
            implicated feature and, in turn, one more (no input left: the marginal rank test of x_j);
  screen    u_j = m_j(x_-j) - g_j(x_j, aux), with m_j the predictor of x_j from the others and g_j a ridge regression
            of m_j on (x_j, aux), tested with the same pair (weighted for w > 1);
  p_j = max(stage 1, re-test, screen).
On consecutive rows the window is ranked among the disjoint blocks of w consecutive calibration rows (statistics mean |r|
and |mean r|, combined inside the rank by the lexicographic key of their exceedance counts); p_j = max(stage 1, re-test)
with block p-values. The test abstains (p = 1) when the reference holds fewer than d / 0.10 blocks.
"""
import math

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from ..harness import consecutive_runs, holm, is_serial, make_model, spacing

Q = 0.10
N_LOGIT_REF = 1000
C_LOGIT = 0.1
ENTRANTS = {"LOO": "p"}


def rank_p(cal, win, two, wc=None, wt=None):
    """Rank p-value of the window values among calibration values. two: two-sided location test of signed values,
    else one-sided (window larger). wc: calibration weights (weighted rank sum with effective sample size);
    wt: window weights (used at w = 1 as the test-point weight)."""
    n, m = len(cal), len(win)
    if wc is None:
        cs = np.sort(cal)
        if m == 1:
            if two:
                lo = np.searchsorted(cs, win[0], side="right"); hi = n - np.searchsorted(cs, win[0], side="left")
                return min(1.0, 2 * (1 + min(lo, hi)) / (n + 1))
            return (1 + n - np.searchsorted(cs, win[0], side="left")) / (n + 1)
        U = np.searchsorted(cs, win, side="left").sum(); s = np.sqrt(n * m * (n + m + 1) / 12)
        if two:
            p = 2 * min(norm.sf((U - n * m / 2 - 0.5) / s), norm.cdf((U - n * m / 2 + 0.5) / s))
            return float(min(1.0, p))
        return float(norm.sf((U - n * m / 2 - 0.5) / s))
    sc = wc.mean(); wc = wc / sc; o = np.argsort(cal); c = cal[o]; ww = wc[o]; cw = np.cumsum(ww); tot = cw[-1]
    if m == 1:
        t = float(wt[0]) / sc if wt is not None else 1.0
        lo_i = np.searchsorted(c, win[0], side="left"); hi_i = np.searchsorted(c, win[0], side="right")
        ge = tot - (cw[lo_i - 1] if lo_i > 0 else 0.0); le = cw[hi_i - 1] if hi_i > 0 else 0.0
        if two:
            return min(1.0, 2 * (min(ge, le) + t) / (tot + t))
        return (ge + t) / (tot + t)
    lo = np.searchsorted(c, win, side="left"); hi = np.searchsorted(c, win, side="right")
    below = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0); upto = np.where(hi > 0, cw[np.maximum(hi - 1, 0)], 0.0)
    F = (below + 0.5 * (upto - below)) / tot
    neff = tot ** 2 / np.sum(ww ** 2); sd = np.sqrt((1.0 / m + 1.0 / neff) / 12.0)
    z = (F.mean() - 0.5) / sd
    if two:
        return float(min(1.0, 2 * min(norm.sf(z), norm.cdf(z))))
    return float(norm.sf(z))


def pair_p(rc, r, wc=None, ww=None):
    """Signed two-sided test and one-sided test of absolute values, combined by twice the smaller p-value."""
    p = rank_p(rc, r, True, wc, ww)
    pa = rank_p(np.abs(rc), np.abs(r), False, wc, ww)
    return min(1.0, 2 * min(p, pa))


def ratio_weights(Zref, Zwin, Zapply):
    """Density ratio p_window / p_reference (logistic regression, class prior removed, log-odds clipped at +-10)."""
    if Zref.shape[1] == 0:
        return [np.ones(len(Z)) for Z in Zapply]
    X = np.vstack([Zref, Zwin]); y = np.r_[np.zeros(len(Zref)), np.ones(len(Zwin))]
    clf = LogisticRegression(C=C_LOGIT, max_iter=500).fit(X, y)
    return [np.exp(np.clip(clf.decision_function(Z) - np.log(len(Zwin) / len(Zref)), -10, 10)) for Z in Zapply]


def block_stats(runs_series, w):
    """Statistics (mean |x|, |mean x|) of the disjoint blocks of w rows starting at 0, w, 2w, ... of each run."""
    S1, S2 = [], []
    for r in runs_series:
        r = np.asarray(r, float); nb = len(r) // w
        if nb == 0:
            continue
        Bk = np.ascontiguousarray(r[:nb * w]).reshape(nb, w)
        S1.append(np.array([np.mean(np.abs(x)) for x in Bk])); S2.append(np.array([abs(np.mean(x)) for x in Bk]))
    if not S1:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(S1), np.concatenate(S2)


def _count_ge(v):
    s = np.sort(v)
    return len(v) - np.searchsorted(s, v, side="left")


def pair_block_p(S1, S2, win):
    """Conformal p-value of the window among the blocks for the pair (mean |x|, |mean x|): items are ordered by the
    lexicographic key (min, max) of their two exceedance counts."""
    win = np.asarray(win, float)
    a = np.r_[np.mean(np.abs(win)), S1]; b = np.r_[abs(np.mean(win)), S2]; n = len(a)
    ca, cb = _count_ge(a), _count_ge(b); lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    return int(np.sum((lo < lo[0]) | ((lo == lo[0]) & (hi <= hi[0])))) / n


class LOO:
    def __init__(self, ctx):
        cond, kind, Xf, Xc, d, seed = ctx.cond, ctx.kind, ctx.Xf, ctx.Xc, int(ctx.d), ctx.seed
        self.cond, self.kind, self.seed, self.d, self.aux, self.Xf, self.Xc = cond, kind, seed, d, list(ctx.A), Xf, Xc
        self.keys = [k for k in cond.m if len(k[1]) == 1]
        self.byj = {j: [k for k in self.keys if k[0] == j] for j in range(d)}
        self.tcal = None if ctx.tcal is None else np.asarray(ctx.tcal, float)
        self.order = None if self.tcal is None else np.argsort(self.tcal, kind="stable")
        self.rc = {}
        for k in self.keys:
            mod, cols = cond.m[k]; self.rc[k] = Xc[:, k[0]] - mod.predict(Xc[:, cols])
        self.lref = np.random.default_rng(seed + 4242).permutation(len(Xf))[:N_LOGIT_REF]
        n = len(Xf); fold = np.random.default_rng(seed + 77).permutation(n) % 2
        self.foc = {}
        for j in range(d):
            mod, cols = cond.m[(j, ())]; inp = [j] + self.aux
            mf = np.zeros(n)
            for f in (0, 1):
                mo = make_model(kind, seed).fit(Xf[fold != f][:, cols], Xf[fold != f, j])
                mf[fold == f] = mo.predict(Xf[fold == f][:, cols])
            mc = mod.predict(Xc[:, cols])
            g = make_model("ridge", seed).fit(Xf[:, inp], mf)
            self.foc[j] = (g, mc - g.predict(Xc[:, inp]))
        self.lazy, self._runs, self._bs = {}, {}, {}

    def _lazy_pred(self, j, cols):
        key = (j, tuple(cols))
        if key not in self.lazy:
            mod = make_model(self.kind, self.seed).fit(self.Xf[:, cols], self.Xf[:, j])
            self.lazy[key] = dict(mod=mod, rc=self.Xc[:, j] - mod.predict(self.Xc[:, cols]))
        return self.lazy[key]

    def _retest(self, W, P1, PM, single, marginal):
        """Re-test of the features flagged on stage 1. single(j, cols, rc, r): p of one predictor; marginal(j): p when
        no input is left."""
        d = self.d; P2 = np.full(d, np.nan); H1 = holm(P1, Q)
        if not H1.any():
            return P2
        X = H1 | holm(PM, Q)
        for j in np.flatnonzero(H1):
            E = X.copy(); E[j] = False
            if not E.any():
                continue
            R = [c for c in range(d) if not E[c] and c != j]
            sets = [[c for c in R if c != t] + self.aux for t in R] if R else [list(self.aux)]
            ps = []
            for cols in sets:
                if not cols:
                    ps.append(marginal(j))
                else:
                    L = self._lazy_pred(j, cols); r = W[:, j] - L["mod"].predict(W[:, cols])
                    ps.append(single(j, cols, L["rc"], r))
            P2[j] = max(ps)
        return P2

    # ---- rows that are not consecutive
    def rows(self, W, resid):
        d, w = self.d, len(W)
        P1 = np.full(d, np.nan)
        for j in range(d):
            ps = []
            for k in self.byj[j]:
                if w > 1:
                    cols = self.cond.m[k][1]
                    wc, ww = ratio_weights(self.Xf[self.lref][:, cols], W[:, cols], [self.Xc[:, cols], W[:, cols]])
                else:
                    wc, ww = None, None
                ps.append(pair_p(self.rc[k], resid[k], wc, ww))
            P1[j] = max(ps) if ps else np.nan
        PM = np.array([rank_p(self.Xc[:, j], W[:, j], True) for j in range(d)])

        def single(j, cols, rc, r):
            if w > 1:
                wc, ww = ratio_weights(self.Xf[self.lref][:, cols], W[:, cols], [self.Xc[:, cols], W[:, cols]])
            else:
                wc, ww = None, None
            return pair_p(rc, r, wc, ww)

        P2 = self._retest(W, P1, PM, single, lambda j: pair_p(self.Xc[:, j], W[:, j]))
        PF = np.full(d, np.nan)
        for j in range(d):
            inp = [j] + self.aux
            g, uc = self.foc[j]
            u = (W[:, j] - resid[(j, ())]) - g.predict(W[:, inp])
            if w > 1:
                wc, ww = ratio_weights(self.Xf[:, inp], W[:, inp], [self.Xc[:, inp], W[:, inp]])
            else:
                wc, ww = None, None
            PF[j] = pair_p(uc, u, wc, ww)
        return np.nanmax(np.vstack([P1, P2, PF]), 0)

    # ---- consecutive rows: calibration on blocks
    def _bp(self, key, cal, win, runs_w, delta):
        w = len(win); ck = (key, w, float(delta))
        if ck not in self._bs:
            c = np.asarray(cal, float)[self.order]; self._bs[ck] = block_stats([c[r] for r in runs_w], w)
        S1, S2 = self._bs[ck]
        return pair_block_p(S1, S2, win)

    def blocks(self, W, twin, resid):
        d, w = self.d, len(W)
        delta = spacing(twin)
        if delta not in self._runs:
            self._runs[delta] = consecutive_runs(self.tcal[self.order], delta)
        runs = self._runs[delta]; runs_w = [r for r in runs if len(r) >= w]
        nblk = int(sum(len(r) // w for r in runs))
        if nblk < math.ceil(d / Q - 1e-9):
            return np.ones(d)
        P1 = np.full(d, np.nan); PM = np.full(d, np.nan)
        for j in range(d):
            ps = [self._bp(("c", k), self.rc[k], resid[k], runs_w, delta) for k in self.byj[j]]
            P1[j] = max(ps) if ps else np.nan
            PM[j] = self._bp(("m", j), self.Xc[:, j], W[:, j], runs_w, delta)

        def single(j, cols, rc, r):
            return self._bp(("e", j, tuple(cols)), rc, r, runs_w, delta)

        P2 = self._retest(W, P1, PM, single, lambda j: PM[j])
        return np.nanmax(np.vstack([P1, P2]), 0)

    def window(self, W, twin, consecutive, resid):
        if is_serial(consecutive, twin, self.tcal, len(W)):
            return {"LOO": self.blocks(W, twin, resid)}
        return {"LOO": self.rows(W, resid)}


def fit(ctx):
    return LOO(ctx)
