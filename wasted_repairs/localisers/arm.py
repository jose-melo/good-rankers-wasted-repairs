"""ARM's rank statistic at the known split (reference | window), Peng et al., arXiv:2608.01691.

Per feature, Z_j = max(Wilcoxon rank-sum |z|, Mood squared-rank |z|) across the split, on within-column ranks of the
concatenated reference and window; p_j = (1 + #{null draws >= Z_j}) / (B + 1) with B = 9999 draws of the authors'
fixed-split null (random rank permutations, depending only on the two sample sizes). Ties are broken at random.
The authors' arm_core.py is downloaded by scripts/download_data.sh and loaded unmodified from config.ARM_CORE.
"""
import importlib.util
import os

import numpy as np

from .. import config

KIND = "p"
MIN_W = 1
B_NULL = 9999
BASE_SEED = 20261122
CACHE_DIR = os.path.join(config.CACHE, "arm")


class _NPCompat:
    """numpy proxy for arm_core.py: np.quantile(interpolation=...) is mapped to method=... (same rule)."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def quantile(self, a, q, *args, **kw):
        if "interpolation" in kw:
            kw["method"] = kw.pop("interpolation")
        return self._real.quantile(a, q, *args, **kw)


_core = None
_mem = {}


def core():
    global _core
    if _core is None:
        if not os.path.exists(config.ARM_CORE):
            raise FileNotFoundError(f"{config.ARM_CORE} missing: run scripts/download_data.sh")
        spec = importlib.util.spec_from_file_location("arm_core", config.ARM_CORE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        try:
            np.quantile(np.arange(3.0), 0.5, interpolation="higher")
        except TypeError:
            mod.np = _NPCompat(np)
        _core = mod
    return _core


def ranks_random_ties(X, rng):
    """Within-column ranks 1..n, ties broken uniformly at random."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    R = np.empty((n, d), dtype=float)
    ar = np.arange(1, n + 1)
    for j in range(d):
        x = X[:, j]
        if np.unique(x).size == n:
            order = np.argsort(x)
        else:
            order = np.lexsort((rng.random(n), x))
        R[order, j] = ar
    return R


def fixed_null(n, t, B):
    """Sorted null draws of the statistic at the fixed split t, cached on disk."""
    key = (n, t, B)
    if key not in _mem:
        path = os.path.join(CACHE_DIR, "fixed_n%d_t%d_B%d.npy" % (n, t, B))
        arr = None
        if os.path.exists(path):
            try:
                arr = np.load(path)
            except Exception:
                arr = None
        if arr is None:
            rng = np.random.default_rng([BASE_SEED, 1, n, t])
            arr = np.sort(core().fixed_split_reference(n, t, B, rng))
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = path + ".%d.tmp.npy" % os.getpid()
            np.save(tmp, arr); os.replace(tmp, path)
        _mem[key] = arr
    return _mem[key]


def run(ref, win, seed):
    ref = np.asarray(ref, dtype=float); win = np.asarray(win, dtype=float)
    if win.ndim == 1:
        win = win[None, :]
    d = ref.shape[1]
    X = np.vstack([ref, win])
    n, t = X.shape[0], ref.shape[0]
    bad = ~np.isfinite(X).all(axis=0)
    if bad.any():
        X = X.copy(); X[:, bad] = 0.0
    R = ranks_random_ties(X, np.random.default_rng([int(seed), n, d]))
    Z = core()._Z_at_split(R, t)
    ref_sorted = fixed_null(n, t, B_NULL)
    p = (1.0 + (ref_sorted.size - np.searchsorted(ref_sorted, Z, side="left"))) / (ref_sorted.size + 1.0)
    p[bad] = np.nan
    return p
