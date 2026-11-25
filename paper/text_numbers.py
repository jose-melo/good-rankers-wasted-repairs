"""Every number quoted in the text of the paper, recomputed from results/ and printed beside the sentence."""
from count_split import main as count_split
from common import fmt, get, results


def show(sentence, value):
    print(f"{value:>10}  {sentence}")


def main():
    print("== Introduction and Figure 1 (causal graphs, broken dependence, w = 250, 160 units)")
    cond, loo = get("graphs", "decoupled", 0.0, "COND"), get("graphs", "decoupled", 0.0, "LOO")
    show("rank x3 first: conditional test", fmt(cond["hit1"]))
    show("rank x3 first: leave-one-out test", fmt(loo["hit1"]))
    show("wasted-repair rate: conditional test", fmt(cond["rate"]))
    show("wasted-repair rate: leave-one-out test", fmt(loo["rate"]))

    print("== Results")
    n_split, n_set, n_cond, n_nochange, n_other = count_split()
    show("settings with two good rankers (graphs and CATS)", n_set)
    show("... of which one is within margin and another above the level", n_split)
    show("... the conditional null is the cause", n_cond)
    show(f"... of the other {n_other}, a level set on rows treated as independent", n_nochange)
    ks, ksb = get("cats", "none", 0.0, "KS"), get("cats", "none", 0.0, "KS_B")
    show("CATS, no change: per-feature KS names a feature in this share of windows", fmt(ks["rate"]))
    show("CATS, no change: KS calibrated on blocks", f"{fmt(ksb['rate'])} {ksb['verdict']}")
    cats = results("cats")
    condb = cats[(cats.localiser == "COND_B") & (cats.type != "none") & (cats.w == 250)]
    show("CATS faults (w = 250) where the block-calibrated conditional test is above the level",
         f"{(condb.verdict == 'above').sum()} of {len(condb)}")
    g = results("graphs"); lg = g[g.localiser == "LOO"]
    show("graph settings where our test is within margin", f"{(lg.verdict == 'within').sum()} of {len(lg)}")
    lc = cats[cats.localiser == "LOO"]
    show("CATS settings where our test is above the level", f"{(lc.verdict == 'above').sum()} of {len(lc)}")

    print("== Limitations")
    show("graphs, propagating 3 sd: wasted-repair rate of our test", fmt(get("graphs", "propagating", 3.0, "LOO")["rate"]))
    wt = results("wind_tunnel"); lw = wt[(wt.localiser == "LOO") & (wt.w >= 50)]
    show("wind tunnel, w >= 50: settings where our test abstains (rate 0, no feature named)",
         f"{(lw.rate == 0).sum()} of {len(lw)}")
    lt = results("ltown"); ll = lt[lt.localiser == "LOO"]
    show("L-Town settings where our test is within margin", f"{(ll.verdict == 'within').sum()} of {len(ll)}")


if __name__ == "__main__":
    main()
