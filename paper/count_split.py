"""Ranking does not decide repair: on the causal graphs and CATS, count the settings (shift type x magnitude x window
length) in which at least two localisers rank an acted-on feature first in more than 0.9 of windows, and those in
which one of them is within margin while another is above the level. Then attribute each such split setting: the
conditional test is the only one above the level, or the others above the level are already above it with no change.
Our test is left out under a propagating change, which its assumption excludes."""
import pandas as pd

from common import results

BASE = ["KS", "ARM", "COND", "CLF@cal", "FSL@0.5", "LOO"]


def split_settings(base=BASE, ours_under_propagation=False):
    c = pd.concat([results("graphs").assign(testbed="graphs"), results("cats").assign(testbed="cats")])
    acted = c[~c.type.isin(["none", "none_iid", "select"])]
    none = c[c.type == "none"].set_index(["testbed", "w", "localiser"]).verdict
    eligible, rows = 0, []
    for key, gr in acted.groupby(["testbed", "type", "mag", "w"]):
        g = gr[gr.localiser.isin(base)].set_index("localiser")
        if key[1].startswith("propagating") and not ours_under_propagation:
            g = g.drop("LOO", errors="ignore")
        good = g[g.hit1 > 0.9]
        if len(good) < 2:
            continue
        eligible += 1
        if (good.verdict == "within").any() and (good.verdict == "above").any():
            above = list(good.index[good.verdict == "above"])
            rows.append(dict(setting="/".join(map(str, key)), above=above,
                             above_with_no_change=[e for e in above if none.get((key[0], key[3], e), "") == "above"]))
    return eligible, rows


def main():
    eligible, rows = split_settings()
    cond = [r for r in rows if r["above"] == ["COND"]]
    other = [r for r in rows if r["above"] != ["COND"]]
    nochange = [r for r in other if set(r["above"]) - {"COND"} <= set(r["above_with_no_change"])]
    for r in rows:
        print(f"  {r['setting']:28s} above the level: {', '.join(r['above'])}")
    print(f"{len(rows)} of {eligible} settings: two localisers that both rank the acted-on feature first differ in "
          f"whether they waste repairs")
    print(f"{len(cond)} of the {len(rows)}: the conditional test is the only well-ranking localiser above the level")
    print(f"{len(nochange)} of the other {len(other)}: those above the level are above it with no change")
    for label, kw in [("our test kept under propagating changes", dict(ours_under_propagation=True)),
                      ("without our test", dict(base=[b for b in BASE if b != "LOO"]))]:
        e, r = split_settings(**kw)
        print(f"{len(r)} of {e} settings ({label})")
    return len(rows), eligible, len(cond), len(nochange), len(other)


if __name__ == "__main__":
    main()
