"""Random causal graphs: 8 features, linear (lin<g>) or nonlinear (nl<g>) structural equations, graph g drawn from
seed 7000 + g. Rows are independent. Evaluation datasets lin5-8 and nl5-8, seeds 4000-4019 (160 units); the threshold
pool of the classifier and FSL-Net uses lin0-4 and nl0-4, seeds 2000-2019."""
import numpy as np

from ..harness import inject_units

NAME = "graphs"
KIND = "lgbm"
D = 8
N_FIT, N_CAL, N_POOL, WS, REPS = 500, 500, 3000, (1, 10, 50, 250), 10
TYPES = ("none", "offset", "decoupled", "stuck", "half", "two", "many", "valdep", "select")


def graph(dataset):
    g = np.random.default_rng(7000 + int(dataset[-1]) + (100 if dataset.startswith("nl") else 0))
    B = np.zeros((D, D))
    for j in range(1, D):
        pa = g.choice(j, size=min(j, int(g.integers(1, 3))), replace=False)
        B[pa, j] = g.uniform(0.5, 1.5, len(pa)) * g.choice([-1, 1], len(pa))
    return B


def sample(dataset, B, n, rng, shift=None):
    X = np.zeros((n, D)); nl = dataset.startswith("nl")
    for j in range(D):
        pa = np.nonzero(B[:, j])[0]
        m = (np.tanh(X[:, pa]) * 2.0 if nl else X[:, pa]) @ B[pa, j] if len(pa) else 0.0
        X[:, j] = m + rng.standard_normal(n) + (shift[j] if shift is not None else 0.0)
    return X


def keys(B):
    pa = [set(np.nonzero(B[:, j])[0]) for j in range(D)]; ch = [set(np.nonzero(B[j])[0]) for j in range(D)]
    mb = [pa[j] | ch[j] | set().union(*[pa[c] for c in ch[j]]) - {j} if ch[j] else set(pa[j]) for j in range(D)]
    return pa, mb


def load(dataset, seed):
    B = graph(dataset); pa, mb = keys(B); rng = np.random.default_rng(10 * seed + 1)
    Xf, Xc, pool = (sample(dataset, B, n, rng) for n in (N_FIT, N_CAL, N_POOL))
    sd = Xf.std(0); corr = np.corrcoef(Xf.T)
    units = inject_units(pool, D, sd, corr, rng, WS, REPS, TYPES, blanket=mb)
    big0 = sample(dataset, B, 20000, np.random.default_rng(99))
    for mag in (1.0, 3.0):
        # propagating change: a shift of one node's mechanism, which moves its descendants
        for w in WS:
            for rep in range(REPS):
                s = int(rng.integers(D)); sh = np.zeros(D); sh[s] = (1 if rng.random() < 0.5 else -1) * mag * sd[s]
                W = sample(dataset, B, w, rng, shift=sh)
                eff = np.abs(sample(dataset, B, 20000, np.random.default_rng(99), shift=sh).mean(0) - big0.mean(0)) / sd
                src = np.zeros(D, bool); src[s] = True
                mov = eff >= 0.2; mov[s] = True; grey = (eff >= 0.1) & (eff < 0.2)
                cnd = np.zeros(D, bool); cnd[s] = True; cnd[list(pa[s])] = True
                units.append(dict(W=W, src=src, mov=mov, cnd=cnd, type="propagating", mag=mag, w=w, rep=rep, note="",
                                  grey=grey, ext=-1, t=None, consecutive=False))
    return dict(Xfit=Xf, Xcal=Xc, tested=list(range(D)), aux=[], names=[f"x{j}" for j in range(D)], units=units,
                tfit=None, tcal=None)
