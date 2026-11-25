"""Score the per-window outputs of run.py: wasted-repair rate, power and hit-at-1 per setting, with a verdict at 0.10.

Per window, a localiser flags a set R: Holm's procedure at 0.10 on p-values, or scores above a threshold. The window
wastes a repair when R holds a feature outside the acted-on set S (features marked grey are not scored); power is the
share of S flagged; hit-at-1 is whether the top-ranked feature lies in S. A setting is (testbed, shift type, magnitude,
window length). Each unit (a seed, or a regime of the wind tunnel) contributes the mean over its windows; the verdict
uses one-sided 97.5 % t bounds over units: within margin (upper bound <= 0.12), above the level (lower bound > 0.10),
otherwise undetermined.

Score thresholds (classifier @cal, FSL-Net @cal) are the 0.9 quantile of the window maxima of no-change windows from
the threshold pool, other seeds only; FSL-Net is also scored at the authors' threshold 0.5 (FSL@0.5).

  python -m wasted_repairs.score --runs outputs/runs/graphs --pool outputs/pool/graphs --out outputs/cells/graphs.csv
"""
import argparse
import collections
import csv
import glob
import os
import re

import numpy as np
from scipy import stats

from .harness import holm

Q, MARGIN = 0.10, 0.02


def pool_seed(tb, seed):
    if tb == "graphs":
        return 2000 <= seed <= 2019
    if tb == "cats":
        return seed in (1000, 1001) or 0 <= seed <= 19
    if tb == "light_tunnel":
        return 1000 <= seed <= 1019
    if tb == "ltown":
        return 0 <= seed <= 9
    return False


def thr_dataset(tb, ds):
    if tb == "graphs":
        return re.sub(r"\d+$", "", ds)
    if tb == "cats" and ds == "reserved":
        return "reference"
    return ds


def unit_key(ds, seed, note):
    m = re.search(r"(?:exp|group)=(\S+)", note)
    return (ds, m.group(0)) if m else (ds, seed)


def load(d):
    return [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(d, "*.npz")))]


def thresholds(pool):
    cal = collections.defaultdict(list)
    for z in pool:
        tb, seed = str(z["testbed"]), int(z["seed"])
        if not pool_seed(tb, seed):
            continue
        for nm, kd in zip(z["entrants"], z["kinds"]):
            if kd != "score":
                continue
            P = z["P_" + nm]; ok = (z["type"] == "none") & ~np.isnan(P).any(1)
            for w in np.unique(z["w"][ok]):
                cal[(tb, thr_dataset(tb, str(z["dataset"])), str(nm), int(w))].append(
                    (seed, list(P[ok & (z["w"] == w)].max(1))))
    return cal


def threshold(cal, tb, ds, nm, w, seed):
    v = [x for s_, xs in cal.get((tb, thr_dataset(tb, ds), nm, w), []) if s_ != seed for x in xs]
    return float(np.quantile(v, 1 - Q)) if len(v) >= 20 else None


def summarise(units, m):
    per = np.array([np.nanmean([r[m] for r in rs]) if not np.all(np.isnan([r[m] for r in rs])) else np.nan
                    for rs in units.values()])
    per = per[~np.isnan(per)]; n = len(per)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(per.mean())
    h = float(stats.t.ppf(0.975, n - 1) * per.std(ddof=1) / np.sqrt(n)) if n > 1 and per.std() > 0 else \
        (0.0 if n > 1 else np.nan)
    return mean, mean - h, mean + h, n


def verdict(mean, lo, hi, n):
    if n < 2 or np.isnan(hi):
        return "undetermined"
    if lo > Q:
        return "above"
    if hi <= Q + MARGIN:
        return "within"
    return "undetermined"


def score(runs, pool):
    cal = thresholds(pool)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for z in runs:
        seed = int(z["seed"]); tb, ds = str(z["testbed"]), str(z["dataset"])
        for nm, kd in zip(z["entrants"], z["kinds"]):
            nm = str(nm); P = z["P_" + nm]
            variants = [(nm, None)] if kd != "score" else \
                [(f"{nm}@cal", "cal")] + ([(f"{nm}@0.5", 0.5)] if nm == "FSL" else [])
            for i in range(len(P)):
                v = P[i]
                if np.isnan(v).any():
                    continue
                src, grey = z["src"][i], z["grey"][i]
                ty, mag, w = str(z["type"][i]), float(z["mag"][i]), int(z["w"][i])
                for label, how in variants:
                    if kd == "p":
                        rej = holm(v, Q); order = np.argsort(v)
                    else:
                        t = threshold(cal, tb, ds, nm, w, seed) if how == "cal" else how
                        if t is None:
                            continue
                        rej = v > t; order = np.argsort(-v)
                    r = dict(rate=float(rej[~src & ~grey].any()),
                             power=float(rej[src].mean()) if src.any() else np.nan,
                             hit1=float(src[order[0]]) if src.any() else np.nan)
                    acc[(tb, ty, mag, w, label)][unit_key(ds, seed, str(z["note"][i]))].append(r)
    rows = []
    for (tb, ty, mag, w, ent), units in sorted(acc.items(), key=lambda kv: tuple(map(str, kv[0]))):
        rate = summarise(units, "rate")
        rows.append(dict(testbed=tb, type=ty, mag=mag, w=w, localiser=ent, n_units=len(units),
                         rate=rate[0], rate_lo=rate[1], rate_hi=rate[2], verdict=verdict(*rate),
                         power=summarise(units, "power")[0], hit1=summarise(units, "hit1")[0]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory of run.py outputs for one testbed")
    ap.add_argument("--pool", default=None, help="directory of run.py outputs of the threshold pool")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = score(load(a.runs), load(a.pool) if a.pool else [])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("wrote", a.out, len(rows), "cells")


if __name__ == "__main__":
    main()
