"""Shared loader and style for the figures and the per-setting table. Reads results/<testbed>.csv."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "out")

INK = "#1a1a1a"
GREY_TXT = "#4a4a4a"
RED = "#B2182B"
CM = 1 / 2.54
COLW = 12.2
COL = {"KS": "#0072B2", "ARM": "#56B4E9", "COND": "#F0E442", "CLF@cal": "#CC79A7", "FSL@0.5": "#009E73",
       "KS_B": "#0072B2", "COND_B": "#F0E442", "LOO": "#000000"}
EDGE = {"COND": "#7a7200"}
OURS = "LOO"

TESTBEDS = ["graphs", "cats", "light_tunnel", "ltown", "wind_tunnel"]
TB_W = {"graphs": 250, "cats": 250, "light_tunnel": 250, "ltown": 24, "wind_tunnel": 10}
TBNAME = {"graphs": "causal graphs", "cats": "CATS", "light_tunnel": "light tunnel", "ltown": "L-Town",
          "wind_tunnel": "wind tunnel"}
TBW = {"graphs": "w = 250", "cats": "w = 250", "light_tunnel": "w = 250", "ltown": "w = 24 h", "wind_tunnel": "w = 10"}
TBN = {"graphs": "160 units", "cats": "20 units", "light_tunnel": "20 units", "ltown": "20 years",
       "wind_tunnel": "8 units"}
BLOCK_TESTBEDS = ("cats", "ltown", "wind_tunnel")

# setting -> (type, magnitude) per testbed; None: not run on that testbed
FAULTS = [
    ("nothing changed", {"graphs": ("none", 0.0), "cats": ("none", 0.0), "light_tunnel": ("none_iid", 0.0),
                         "ltown": ("none", 0.0), "wind_tunnel": ("none_c", 0.0)}),
    ("offset of 1 sd", {"graphs": ("offset", 1.0), "cats": ("offset_c", 1.0), "light_tunnel": ("offset", 1.0),
                        "ltown": ("offset", 1.0), "wind_tunnel": ("offset_c", 1.0)}),
    ("offset of 3 sd", {"graphs": ("offset", 3.0), "cats": ("offset_c", 3.0), "light_tunnel": ("offset", 3.0),
                        "ltown": None, "wind_tunnel": ("offset_c", 3.0)}),
    ("stuck sensor", {"graphs": ("stuck", 0.0), "cats": ("stuck_c", 0.0), "light_tunnel": ("stuck", 0.0),
                      "ltown": ("stuck", 0.0), "wind_tunnel": ("stuck_c", 0.0)}),
    ("half-window fault", {"graphs": ("half", 3.0), "cats": ("half_c", 3.0), "light_tunnel": ("half", 3.0),
                           "ltown": ("half", 1.0), "wind_tunnel": ("half_c", 3.0)}),
    ("broken dependence", {"graphs": ("decoupled", 0.0), "cats": ("decoupled_c", 0.0),
                           "light_tunnel": ("decoupled", 0.0), "ltown": ("decoupled", 0.0),
                           "wind_tunnel": ("decoupled_c", 0.0)}),
    ("two acted on", {"graphs": ("two", 3.0), "cats": ("two_c", 3.0), "light_tunnel": ("two", 3.0), "ltown": None,
                      "wind_tunnel": ("two_c", 3.0)}),
    ("half of the features", {"graphs": ("many", 3.0), "cats": ("many_c", 3.0), "light_tunnel": ("many", 3.0),
                              "ltown": None, "wind_tunnel": ("many_c", 3.0)}),
    ("rows selected", {"graphs": ("select", 1.0), "cats": None, "light_tunnel": ("select", 1.0), "ltown": None,
                       "wind_tunnel": None}),
    ("propagating, 1 sd", {"graphs": ("propagating", 1.0), "cats": None, "light_tunnel": None, "ltown": None,
                           "wind_tunnel": None}),
    ("propagating, 3 sd", {"graphs": ("propagating", 3.0), "cats": None, "light_tunnel": None, "ltown": None,
                           "wind_tunnel": None}),
]

# rows of Figures 2 and 3
ENTS = [("KS", "per-feature KS"), ("ARM", "ARM's rank test"), ("COND", "conditional test"),
        ("CLF@cal", "classifier (cal.)"), ("FSL@0.5", "FSL-Net"), ("KS_B", "KS, block cal."),
        ("COND_B", "conditional, block cal."), (OURS, "leave-one-out (ours)")]
# cells that cannot exist by construction: block calibration on independent rows (graphs) or without consecutive
# calibration rows (light tunnel); the classifier and FSL-Net need windows of at least 20 rows (wind tunnel: w = 10)
NA = {("graphs", "KS_B"), ("graphs", "COND_B"), ("light_tunnel", "KS_B"), ("light_tunnel", "COND_B"),
      ("wind_tunnel", "CLF@cal"), ("wind_tunnel", "FSL@0.5")}

_DF = {}


def results(tb):
    if tb not in _DF:
        _DF[tb] = pd.read_csv(os.path.join(RESULTS, f"{tb}.csv"))
    return _DF[tb]


def get(tb, typ, mag, loc, w=None):
    """One setting of one localiser, or None if it was not run there."""
    if loc in ("KS_B", "COND_B") and tb not in BLOCK_TESTBEDS:
        return None
    c = results(tb); w = TB_W[tb] if w is None else w
    q = c[(c.type == typ) & (c.mag == mag) & (c.w == w) & (c.localiser == loc)]
    if len(q) != 1:
        return None
    r = q.iloc[0]
    d = dict(rate=float(r.rate), lo=float(r.rate_lo), hi=float(r.rate_hi), verdict=r.verdict, n=int(r.n_units),
             power=None if pd.isna(r.power) else float(r.power), hit1=None if pd.isna(r.hit1) else float(r.hit1))
    d["abstains"] = d["rate"] == 0 and d["power"] == 0
    return d


def fmt(x):
    """0 -> "0"; below .01 -> "<.01"; .995 or more -> "1.00"; else two decimals without the leading zero."""
    if x is None:
        return "–"
    if x == 0:
        return "0"
    if x < 0.01:
        return "<.01"
    if x >= 0.995:
        return "1.00"
    return f"{x:.2f}".replace("0.", ".", 1)


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.linewidth": 0.5,
        "axes.edgecolor": INK,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    print("wrote", os.path.join(OUT, name + ".pdf"))
