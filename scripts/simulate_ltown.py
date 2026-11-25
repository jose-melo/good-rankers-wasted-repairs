"""Leak-free L-Town simulations (BattLeDIM network L-TOWN_Real.inp, settings of the published dataset generator:
pressure-dependent demand, required pressure 25 m, minimum 0, 5-min step, EPANET 2.2, no leak).

  python scripts/simulate_ltown.py --published     -> data/ltown/sim_published.npz  (the published demand year)
  python scripts/simulate_ltown.py --seed 3000     -> data/ltown/sim/noise_3000.npz (perturbed demands)

Seed s perturbs every 365-day demand pattern (generator np.random.default_rng([20261125, s])): the year is rolled by
7k days, k uniform in 0..51, and each multiplier is multiplied by exp(a + e), with a ~ N(0, 0.10^2) per pattern and day
and e an AR(1) over 5-min steps (coefficient 0.99, stationary sd 0.05). Output: 33 pressures P, 3 flows F (m3/h),
1 tank level L, every 5 min. About 2 min and 1 GB of temporary files per simulated year."""
import argparse
import os
import shutil
import sys

import numpy as np
import wntr
import yaml
from scipy.signal import lfilter
from wntr.epanet import toolkit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wasted_repairs import config  # noqa: E402

EN_FLOW, EN_PRESSURE = 8, 11
SD_DAY, SD_STEP, PHI = 0.10, 0.05, 0.99
STEPS_DAY = 288


def perturb(wn, seed):
    rng = np.random.default_rng([20261125, int(seed)]); k = int(rng.integers(52)); shift = 7 * k * STEPS_DAY
    for name in sorted(wn.pattern_name_list):
        pat = wn.get_pattern(name); m = np.asarray(pat.multipliers, float)
        if m.size < 365 * STEPS_DAY:
            continue
        assert m.size == 365 * STEPS_DAY, (name, m.size)
        m = np.roll(m, -shift)
        a = np.repeat(rng.normal(0, SD_DAY, 365), STEPS_DAY)
        z = rng.normal(0, SD_STEP * np.sqrt(1 - PHI ** 2), m.size); z[0] = rng.normal(0, SD_STEP)
        e = lfilter([1.0], [1.0, -PHI], z)
        pat.multipliers = (m * np.exp(a + e)).tolist()
    return 7 * k


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--published", action="store_true")
    g.add_argument("--seed", type=int)
    ap.add_argument("--days", type=int, default=365)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(config.LTOWN, "dataset_configuration.yaml")))
    P, F, L = ([str(s) for s in cfg[k]] for k in ("pressure_sensors", "flow_sensors", "level_sensors"))
    wn = wntr.network.WaterNetworkModel(os.path.join(config.LTOWN, "L-TOWN_Real.inp"))
    step = round(wn.options.time.hydraulic_timestep); assert step == 300, step
    tag = "published" if a.published else f"noise_{a.seed}"
    tmp = os.path.join(config.CACHE, f"ltown_tmp_{tag}"); os.makedirs(tmp, exist_ok=True)
    if a.published:
        wn.options.time.duration = a.days * 86400 - step
        wn.options.hydraulic.demand_model = "PDA"; wn.options.hydraulic.required_pressure = 25
        wn.options.hydraulic.minimum_pressure = 0
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tmp, "t"))
        pr = res.node["pressure"]
        out = os.path.join(config.LTOWN, "sim_published.npz")
        np.savez_compressed(out, P=pr[P].values.astype("float32"),
                            F=res.link["flowrate"][F].values.astype("float32") * 3600, L=pr[L].values.astype("float32"))
    else:
        rot = perturb(wn, a.seed)
        wn.options.time.duration = a.days * 86400 - step
        wn.options.hydraulic.demand_model = "PDA"; wn.options.hydraulic.required_pressure = 25
        wn.options.hydraulic.minimum_pressure = 0
        inp = os.path.join(tmp, "t.inp"); wntr.network.io.write_inpfile(wn, inp, units="CMH", version=2.2)
        en = toolkit.ENepanet(); en.ENopen(inp, os.path.join(tmp, "t.rpt"), os.path.join(tmp, "t.bin"))
        ip = [en.ENgetnodeindex(x) for x in P]; il = [en.ENgetnodeindex(x) for x in L]
        jf = [en.ENgetlinkindex(x) for x in F]
        en.ENopenH(); en.ENinitH(0); rows = []
        while True:
            t = en.ENrunH()
            if t % step == 0 and (not rows or rows[-1][0] != t):
                rows.append([t] + [en.ENgetnodevalue(i, EN_PRESSURE) for i in ip]
                            + [en.ENgetlinkvalue(j, EN_FLOW) for j in jf] + [en.ENgetnodevalue(i, EN_PRESSURE) for i in il])
            if en.ENnextH() <= 0:
                break
        en.ENcloseH(); en.ENclose()
        R = np.array(rows); assert (np.diff(R[:, 0]) == step).all() and R[0, 0] == 0
        out = os.path.join(config.LTOWN, "sim", f"noise_{a.seed}.npz"); os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, P=R[:, 1:1 + len(P)].astype("float32"),
                            F=R[:, 1 + len(P):1 + len(P) + len(F)].astype("float32"),
                            L=R[:, 1 + len(P) + len(F):].astype("float32"), seed=a.seed, days=a.days, rotation_days=rot)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
