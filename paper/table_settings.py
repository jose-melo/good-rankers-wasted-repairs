"""Per-setting table: wasted-repair rate (and power below it) on the five testbeds with no change, under an offset and
under a broken dependence. Writes paper/out/table_settings.tex (LaTeX, booktabs) and prints it as text."""
import os

from common import OURS, OUT, fmt, get

ENTS = [("KS", "KS"), ("ARM", "ARM"), ("COND", "Cond."), ("CLF@cal", "Clf.$^\\dagger$"), ("FSL@0.5", "FSL"),
        ("KS_B", "KS$_\\mathrm{B}$"), ("COND_B", "Cond.$_\\mathrm{B}$"), (OURS, "\\textbf{Ours}")]
TBS = ["graphs", "cats", "light_tunnel", "ltown", "wind_tunnel"]
SET = {"graphs": [("no change", "none", 0.0), ("offset 3\\,sd", "offset", 3.0), ("broken dep.", "decoupled", 0.0)],
       "cats": [("no change", "none", 0.0), ("offset 3\\,sd", "offset_c", 3.0), ("broken dep.", "decoupled_c", 0.0)],
       "light_tunnel": [("no change", "none_iid", 0.0), ("offset 3\\,sd", "offset", 3.0),
                        ("broken dep.", "decoupled", 0.0)],
       "ltown": [("no change", "none", 0.0), ("offset 1\\,sd", "offset", 1.0), ("broken dep.", "decoupled", 0.0)],
       "wind_tunnel": [("no change", "none_c", 0.0), ("offset 3\\,sd", "offset_c", 3.0),
                       ("broken dep.", "decoupled_c", 0.0)]}
TBHEAD = {"graphs": "Causal graphs ($w=250$, 160 units)", "cats": "CATS ($w=250$, 20 units)",
          "light_tunnel": "Light tunnel ($w=250$, 20 units)", "ltown": "L-Town ($w=24$\\,h, 20 years)",
          "wind_tunnel": "Wind tunnel ($w=10$, 8 units)"}
# not applicable: block calibration needs rows recorded in time order; the classifier and FSL-Net need w >= 20
NA = {("graphs", "KS_B"), ("graphs", "COND_B"), ("light_tunnel", "KS_B"), ("light_tunnel", "COND_B"),
      ("wind_tunnel", "CLF@cal"), ("wind_tunnel", "FSL@0.5")}
SUP = {"above": "$^{a}$", "undetermined": "$^{u}$", "within": ""}


def tex_num(s):
    return s.replace("<", "$<$")


def main():
    counts = {"within": 0, "undetermined": 0, "above": 0, "abstains": 0, "N/A": 0, "missing": 0}
    L = ["\\providecommand{\\na}{\\textcolor{black!45}{\\tiny N/A}}",
         "\\providecommand{\\pw}[1]{{\\tiny\\textcolor{black!55}{#1}}}",
         "\\begin{table}[t]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{0pt}",
         "\\renewcommand{\\arraystretch}{0.92}",
         "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * len(ENTS) + "@{}}", "\\toprule",
         "Setting & " + " & ".join(nm for _, nm in ENTS) + " \\\\", "\\midrule"]
    text = []
    for i, tb in enumerate(TBS):
        if i:
            L.append("\\addlinespace[2pt]")
        L.append("\\multicolumn{%d}{@{}l}{\\textit{%s}} \\\\" % (len(ENTS) + 1, TBHEAD[tb]))
        text.append(TBHEAD[tb].replace("$", "").replace("\\,", " "))
        for lab, typ, mag in SET[tb]:
            rates, pows, plain, anypow = [], [], [], False
            for ent, _ in ENTS:
                d = None if (tb, ent) in NA else get(tb, typ, mag, ent)
                if (tb, ent) in NA:
                    counts["N/A"] += 1; rates.append("\\na"); pows.append(""); plain.append("N/A"); continue
                if d is None:
                    counts["missing"] += 1; rates.append("\\textperiodcentered"); pows.append(""); plain.append("."); continue
                if d["abstains"]:
                    counts["abstains"] += 1; rates.append("\\textit{abst.}"); pows.append(""); plain.append("abst.")
                    continue
                counts[d["verdict"]] += 1
                r = tex_num(fmt(d["rate"]))
                rates.append(("\\textbf{" + r + "}" if ent == OURS else r) + SUP[d["verdict"]])
                p = ""
                if d["power"] is not None and not typ.startswith("none"):
                    anypow = True; pows.append("\\pw{" + tex_num(fmt(d["power"])) + "}"); p = "/" + fmt(d["power"])
                else:
                    pows.append("")
                plain.append(fmt(d["rate"]) + {"above": "a", "undetermined": "u", "within": ""}[d["verdict"]] + p)
            L.append("\\quad " + lab + " & " + " & ".join(rates) + " \\\\")
            if anypow:
                L.append(" & " + " & ".join(pows) + " \\\\[1pt]")
            text.append("  " + lab.replace("\\,", " ").ljust(13) + "".join(s.rjust(11) for s in plain))
    L += ["\\bottomrule", "\\end{tabular*}",
          "\\caption{Wasted-repair rate (top) and power (below, grey) on the five testbeds, with no change, under an "
          "offset and under a broken dependence. Superscript $a$: above the level; $u$: undetermined; no mark: within "
          "margin, from one-sided 97.5\\,\\% $t$ bounds over units. $\\dagger$: threshold calibrated on no-change "
          "windows of other units; B: calibrated on blocks of consecutive rows. \\na: not applicable: block "
          "calibration needs rows recorded in time order, which the causal graphs (independent rows) and the light "
          "tunnel (no consecutive calibration blocks) do not have; the classifier and FSL-Net need windows of at "
          "least 20 rows.}", "\\label{tab:settings}", "\\end{table}"]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "table_settings.tex"), "w").write("\n".join(L) + "\n")
    print(" " * 15 + "".join(nm.replace("$^\\dagger$", "").replace("$_\\mathrm{B}$", "_B").replace("\\textbf{", "")
                             .replace("}", "").rjust(11) for _, nm in ENTS))
    print("\n".join(text))
    print("rate/power; a: above the level, u: undetermined, no mark: within margin")
    print("cells:", counts)


if __name__ == "__main__":
    main()
