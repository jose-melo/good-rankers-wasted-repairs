"""Copy the scored cells of the localisers in the paper into results/<testbed>.csv (the files the figures read).
Usage: python scripts/collect_results.py outputs/cells"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTBEDS = ["graphs", "cats", "light_tunnel", "ltown", "wind_tunnel"]
LOCALISERS = ["KS", "ARM", "COND", "CLF@cal", "FSL@0.5", "KS_B", "COND_B", "LOO"]
# settings the paper reports: every graph setting, CATS and wind-tunnel shifts on consecutive rows, the light tunnel
# at w = 250, every L-Town setting (the classifier has no threshold pool on the wind tunnel)
KEEP = {"graphs": lambda r: True, "cats": lambda r: r["type"].endswith("_c") or r["type"] == "none",
        "light_tunnel": lambda r: int(r["w"]) == 250, "ltown": lambda r: True,
        "wind_tunnel": lambda r: r["type"].endswith("_c") and r["localiser"] != "CLF@cal"}
FIELDS = ["type", "mag", "w", "localiser", "n_units", "rate", "rate_lo", "rate_hi", "verdict", "power", "hit1"]


def main(cells):
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    for tb in TESTBEDS:
        rows = [r for r in csv.DictReader(open(os.path.join(cells, f"{tb}.csv"))) if r["localiser"] in LOCALISERS and KEEP[tb](r)]
        with open(os.path.join(ROOT, "results", f"{tb}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)
        print(tb, len(rows), "cells")


if __name__ == "__main__":
    main(sys.argv[1])
