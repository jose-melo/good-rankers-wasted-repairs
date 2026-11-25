"""Compute every requested localiser's per-feature output on every window of one (testbed, dataset, seed).

The reference of the seed is split into fit rows (predictors) and calibration rows; all columns are standardised by
the fit rows. Windows are the testbed's own units plus the feature shifts injected on its clean pool
(harness.inject_units, generator 777 + seed). Writes <out>/<testbed>/<dataset>_<seed>.npz with P_<localiser> [units, d].

  python -m wasted_repairs.run --testbed graphs --dataset lin5 --seed 4000 --localisers KS,COND,LOO,ARM,CLF,FSL
"""
import argparse
import importlib
import os
import time
import warnings

import numpy as np

from . import config, harness, localisers


class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run(tb, dataset, seed, kind, out_dir, names, types=None, ws=None, cond=True):
    warnings.filterwarnings("ignore"); t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"_w{'-'.join(map(str, ws))}" if ws else "") + (f"_t{'-'.join(types)}" if types else "")
    fn = os.path.join(out_dir, f"{dataset}_{seed}{tag}.npz")
    if os.path.exists(fn):
        print("exists", fn); return fn
    z = tb.load(dataset, seed); tested, aux = list(z["tested"]), list(z["aux"]); d = len(tested); cols = tested + aux
    tcal = z.get("tcal")
    Xf, Xc = z["Xfit"][:, cols], z["Xcal"][:, cols]
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
    Xf, Xc = (Xf - mu) / sd, (Xc - mu) / sd; A = list(range(d, len(cols)))
    need_cond = cond and any(n in names for n in ("COND", "LOO", "COND_B"))
    cnd = harness.Cond(kind, Xf, Xc, d, A, seed) if need_cond else None
    ref = np.vstack([Xf, Xc])[:, :d]
    units = list(z["units"])
    for u in units:
        u["W"] = (u["W"][:, cols] - mu) / sd
    if "pool" in z and getattr(tb, "POOL_INJECT", None):
        pool = z["pool"][:, cols]; spec = z.get("pool_inject") or tb.POOL_INJECT
        inj = harness.inject_units(pool, d, z["Xfit"][:, tested].std(0) + 1e-12,
                                   np.nan_to_num(np.corrcoef(z["Xfit"][:, tested].T)), np.random.default_rng(777 + seed),
                                   spec["ws"], spec["reps"], spec["types"], tpool=z.get("tpool"),
                                   cons_types=spec.get("cons_types", ()))
        for u in inj:
            u["W"] = (u["W"] - mu) / sd; u["note"] = (u["note"] + " injected on pool").strip()
        units += inj
    uidx = np.array([i for i, u in enumerate(units)
                     if (ws is None or int(u["w"]) in ws) and (types is None or str(u["type"]) in types)], int)
    units = [units[i] for i in uidx]; n = len(units)
    make_ctx = lambda ents: Ctx(dataset=dataset, seed=int(seed), kind=kind, d=d, A=A, Xf=Xf, Xc=Xc, cond=cnd,
                                tcal=tcal, entrants=list(ents))
    objs = localisers.build(names, make_ctx)
    P = {nm: np.full((n, d), np.nan) for nm in names}
    for i, u in enumerate(units):
        W = u["W"]; twin = u.get("t"); consecutive = bool(u.get("consecutive", False))
        if "KS" in names:
            P["KS"][i] = harness.ks_pvals(ref, W[:, :d])
        resid = cnd.residuals(W) if cnd is not None else {}
        if "COND" in names:
            P["COND"][i] = cnd.full_pvals(resid)
        for obj in objs:
            for k, v in obj.window(W, twin, consecutive, resid).items():
                if k in P:
                    P[k][i] = np.asarray(v, float)
    g = lambda k, dt: np.array([u.get(k, dt) for u in units])
    msk = lambda k: np.array([np.zeros(d, bool) if u.get(k) is None else np.asarray(u[k], bool)
                              for u in units]).reshape(n, d)
    np.savez_compressed(fn, testbed=tb.NAME, dataset=dataset, seed=seed, kind=kind, d=d,
                        type=g("type", ""), mag=g("mag", 0.0), w=g("w", 0), rep=g("rep", 0), note=g("note", ""),
                        src=msk("src"), mov=msk("mov"), grey=msk("grey"), uidx=uidx,
                        consecutive=np.array([bool(u.get("consecutive", False)) for u in units], bool),
                        entrants=np.array(names), kinds=np.array([localisers.kind(nm) for nm in names]),
                        **{"P_" + nm: P[nm] for nm in names})
    print(tb.NAME, dataset, seed, kind, tag, "units", n, "d", d, "seconds", round(time.time() - t0, 1), flush=True)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--kind", default=None, help="predictor family (default: the testbed's)")
    ap.add_argument("--localisers", default="KS,COND,LOO,ARM,CLF,FSL")
    ap.add_argument("--types", default=None, help="keep only these unit types (after all units are drawn)")
    ap.add_argument("--ws", default=None, help="keep only these window lengths")
    ap.add_argument("--out", default=os.path.join(config.OUT, "runs"))
    a = ap.parse_args()
    tb = importlib.import_module(f"wasted_repairs.testbeds.{a.testbed}")
    names = [x for x in a.localisers.split(",") if x]
    for nm in names:
        assert nm in localisers.ALL, f"unknown localiser {nm}; known: {localisers.ALL}"
    run(tb, a.dataset, a.seed, a.kind or tb.KIND, os.path.join(a.out, a.testbed), names,
        types=None if a.types is None else a.types.split(","),
        ws=None if a.ws is None else [int(x) for x in a.ws.split(",")])


if __name__ == "__main__":
    main()
