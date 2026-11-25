"""Figure 1. (1) A toy window: a fault replaces the recorded column of x3 in a chain x1 -> ... -> x5 (results/toy.json,
written by paper/toy/toy.js). (2) The p-value of each feature for KS, the conditional test and ours, and the features
Holm's procedure flags at .10. (3) Over all windows on the causal graphs (results/graphs.csv, broken dependence,
w = 250): how often x3 is ranked first, the wasted-repair rate and the power."""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from common import CM, COL, EDGE, GREY_TXT, INK, RESULTS, apply_style, get, save

apply_style()
TOY = json.load(open(os.path.join(RESULTS, "toy.json")))
ALPHA = TOY["meta"]["alpha"]
D = 5
ACTED = TOY["meta"]["acted"]
FAULT = INK
RED = "#B2182B"
RULE = "#c9c9c9"
SOFT = "#8a8a8a"
KEYS = [("ks", "KS"), ("full", "FULL"), ("loo", "LOO")]
ROWNAME = {"KS": "KS", "FULL": "conditional", "OURS": "ours", "LOO": "ours"}
SW = {"KS": (COL["KS"], COL["KS"]), "FULL": (COL["COND"], EDGE["COND"]), "OURS": (INK, INK), "LOO": (INK, INK)}


def fmt(x):
    if x == 0:
        return "0"
    if x < 0.01:
        return "<.01"
    if x >= 0.995:
        return "1.00"
    return f"{x:.2f}".lstrip("0")


def p2(p):
    return "<.01" if p < 0.01 else f"{p:.2f}".lstrip("0")


def holm(p, alpha=ALPHA):
    idx = np.argsort(p); out = np.zeros(len(p), bool)
    for r, i in enumerate(idx):
        if p[i] <= alpha / (len(p) - r):
            out[i] = True
        else:
            break
    return out


def one_window():
    res = {}
    for k, e in KEYS:
        o = TOY["one_window"][k]; p = np.array(o["p"]); named = holm(p)
        assert list(named) == list(o["named"]), (k, named, o["named"])
        res[e] = dict(p=p, named=named)
    return res


def metrics():
    out = {}
    for loc, rid in [("KS", "KS"), ("COND", "FULL"), ("LOO", "OURS")]:
        d = get("graphs", "decoupled", 0.0, loc)
        out[rid] = dict(rate=d["rate"], power=d["power"], hit1=d["hit1"])
    return out


def holm_sets(OW):
    out = {}
    for k, e in KEYS:
        nm = [j for j in range(D) if OW[e]["named"][j]]
        out[e] = "$\\emptyset$" if not nm else "$\\{" + ",".join(f"x_{j+1}" for j in nm) + "\\}$"
    return out


class Fig:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w * CM, h * CM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")

    def T(self, x, y, s, **k):
        k.setdefault("fontsize", 6.5); k.setdefault("color", INK); k.setdefault("va", "center"); k.setdefault("ha", "left")
        if k.get("fontweight") == "bold":
            k["fontfamily"] = "STIXGeneral"
        return self.ax.text(x, y, s, **k)

    def sub(self, x, y, w, h):
        return self.fig.add_axes([x / self.W, y / self.H, w / self.W, h / self.H])

    def panel(self, x0, y0, w, h, n, title, fs=7.5, ty=0.27, rule=None):
        if rule:
            self.ax.plot([x0, x0 + w], [y0 + h - rule] * 2, color=INK, lw=0.6, solid_capstyle="butt", zorder=1)
        cy = y0 + h - ty
        self.ax.add_patch(Circle((x0 + 0.25, cy), 0.135, fc=INK, ec="none", zorder=2))
        self.T(x0 + 0.25, cy - 0.006, str(n), color="white", ha="center", fontweight="bold", fontsize=6.5, zorder=3)
        self.T(x0 + 0.47, cy, title, fontsize=fs, zorder=3)

    def chevron(self, xa, xb, y, color):
        cx = (xa + xb) / 2
        t, hh, d = 0.075, 0.17, 0.10
        pts = [(cx - d / 2 - t / 2, y + hh), (cx + d / 2 + t / 2, y), (cx - d / 2 - t / 2, y - hh),
               (cx - d / 2 - t / 2 - t, y - hh), (cx + d / 2 - t / 2, y), (cx - d / 2 - t / 2 - t, y + hh)]
        self.ax.add_patch(Polygon([(px + t / 2, py) for px, py in pts], closed=True, fc=color, ec="none", zorder=5))

    def bbox_cm(self, t):
        self.fig.canvas.draw()
        e = t.get_window_extent(self.fig.canvas.get_renderer())
        (a, b), (c, d) = self.ax.transData.inverted().transform([[e.x0, e.y0], [e.x1, e.y1]])
        return a, b, c, d


def squiggle(F, cx, cy, half=0.1, amp=0.03, color="white", lw=0.8, seed=0.0):
    xs = np.linspace(cx - half, cx + half, 9)
    ys = cy + amp * np.sin(np.linspace(0, 4 * np.pi, 9) + seed)
    F.ax.plot(xs, ys, color=color, lw=lw, zorder=6, solid_capstyle="round")


def scatter_slope(F, x, y, sz, which, s=0.9):
    """x3 against x2 in the reference or the window, with the least-squares line; returns the correlation string."""
    xs, ys = (TOY["ref_x2"], TOY["ref_x3"]) if which == "ref" else (TOY["win_x2"], TOY["win_x3"])
    xs, ys = np.asarray(xs), np.asarray(ys)
    col, al = ("#7a7a7a", 0.55) if which == "ref" else (INK, 0.75)
    ax = F.sub(x, y, sz, sz)
    ax.scatter(xs, ys, s=s, lw=0, color=col, alpha=al, rasterized=True)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.45); sp.set_color("#7a7a7a")
    ax.set_facecolor("white")
    r = np.corrcoef(xs, ys)[0, 1]
    b, a = np.polyfit(xs, ys, 1)
    t = np.array([-3.0, 3.0])
    ax.plot(t, a + b * t, color=INK, lw=0.8, solid_capstyle="round")
    return f"{r:.2f}".replace("-0.00", "0.00").replace("0.", ".", 1)


def node(F, x, y, j, R=0.115, fs=6.5, ls="-"):
    F.ax.add_patch(Circle((x, y), R, fc="white", ec=INK, lw=0.6, ls=ls, zorder=4))
    F.T(x, y, f"$x_{j+1}$", ha="center", fontsize=fs, color=INK, zorder=5)


def edge(F, xa, ya, xb, yb, R=0.115, color="#555555", lw=0.65, hs=4.2, gapA=0.015, gapB=0.02):
    v = np.array([xb - xa, yb - ya]); u = v / np.linalg.norm(v)
    F.ax.add_patch(FancyArrowPatch((xa + u[0] * (R + gapA), ya + u[1] * (R + gapA)),
                                   (xb - u[0] * (R + gapB), yb - u[1] * (R + gapB)),
                                   arrowstyle=ArrowStyle("-|>", head_length=0.5, head_width=0.28),
                                   mutation_scale=hs * 1.6, lw=lw, color=color, shrinkA=0, shrinkB=0,
                                   connectionstyle="arc3,rad=0.0", zorder=3))


def x3_head(F, cx, cy, w=0.27, h=0.2, fs=6.5):
    F.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                  fc=FAULT, ec=FAULT, lw=0.4, zorder=4))
    F.T(cx, cy, "$x_3$", ha="center", fontsize=fs, color="white", zorder=5)


def table(F, x, y0, cw, rh, n_ref, n_win, gap, side_fs, side_dx, seed=4):
    """Reference rows, a dashed separator, window rows whose x3 cells are black with a white squiggle."""
    rng = np.random.default_rng(seed)
    F.ax.plot([x, x + D * cw], [y0, y0], color=INK, lw=0.6, solid_capstyle="butt")
    rows = n_ref + n_win
    for i in range(rows):
        win = i >= n_ref
        yy = y0 - (i + 0.5) * rh - (gap if win else 0)
        for j in range(D):
            cx = x + (j + 0.5) * cw
            if win and j in ACTED:
                F.ax.add_patch(Rectangle((x + j * cw + 0.03, yy - rh / 2 + 0.014), cw - 0.06, rh - 0.028,
                                         fc=FAULT, ec="none", zorder=3))
                squiggle(F, cx, yy, half=min(0.085, cw / 2 - 0.07), amp=min(0.025, rh * 0.15), lw=0.7,
                         seed=rng.random() * 3)
            else:
                L = 0.07 + 0.07 * rng.random()
                F.ax.plot([cx - L / 2, cx + L / 2], [yy, yy], color="#5c5c5c" if win else "#b5b5b5", lw=1.0,
                          solid_capstyle="round")
    y_ref_bot = y0 - n_ref * rh
    yb = y0 - rows * rh - gap
    F.ax.plot([x, x + D * cw], [y_ref_bot - gap / 2] * 2, color=SOFT, lw=0.45, ls=(0, (2, 1.5)))
    F.ax.plot([x, x + D * cw], [yb, yb], color=INK, lw=0.6, solid_capstyle="butt")
    for k in range(1, D):
        F.ax.plot([x + k * cw] * 2, [yb, y0], color="#dcdcdc", lw=0.3, zorder=0)
    xr = x + D * cw + side_dx
    F.T(xr, (y0 + y_ref_bot) / 2, "ref.", fontsize=side_fs, color=GREY_TXT)
    F.T(xr, (y_ref_bot - gap + yb) / 2, "window", fontsize=side_fs, color=GREY_TXT)
    return yb


def swatch_label(F, x, yc, rid, fs=6.5, s=0.13):
    fc, ec = SW[rid]
    F.ax.add_patch(Rectangle((x, yc - s / 2), s, s, fc=fc, ec=ec, lw=0.5))
    F.T(x + s + 0.07, yc, ROWNAME[rid], fontsize=fs, fontweight="bold" if rid in ("OURS", "LOO") else "normal")


def pcell_row(F, X0, yc, cw, rh, o, fs, inset):
    for j in range(D):
        cx = X0 + (j + 0.5) * cw
        fill = bool(o["named"][j])
        c = (FAULT if j in ACTED else RED) if fill else "white"
        F.ax.add_patch(FancyBboxPatch((X0 + j * cw + inset, yc - rh / 2), cw - 2 * inset, rh,
                                      boxstyle="round,pad=0,rounding_size=0.04", fc=c,
                                      ec=c if fill else "#b9b9b6", lw=0.35, zorder=3))
        F.T(cx, yc, p2(o["p"][j]), ha="center", fontsize=fs, color="white" if fill else INK, zorder=4)


def key_line(F, x, y, fs=7, sep=1.20, s=0.14):
    for i, (c, txt) in enumerate([(FAULT, "$x_3$ (faulty)"), (RED, "flagged, intact = wasted repair")]):
        xx = x + i * sep
        F.ax.add_patch(Rectangle((xx, y - s / 2), s, s, fc=c, ec="none"))
        F.T(xx + s + 0.06, y, txt, fontsize=fs, color=INK if c == FAULT else RED)


def main():
    OW, M = one_window(), metrics()
    SETS = holm_sets(OW)
    W, H = 12.2, 4.4
    F = Fig(W, H)
    y0, h = 0.03, H - 0.06
    g = 0.18
    widths = [4.42, 5.10]
    widths.append(W - 0.04 - sum(widths) - 2 * g)
    xs = [0.02]
    for w in widths:
        xs.append(xs[-1] + w + g)
    x1, x2, x3 = xs[:3]
    w1, w2, w3 = widths
    top = y0 + h
    Y_RULE = top - 1.55
    Y_HB = Y_RULE + 0.06
    Y_ROWS = [Y_RULE - 0.38 - i * 0.72 for i in range(3)]
    Y_FOOT = y0 + 0.30

    # 1 Data
    F.panel(x1, y0, w1, h, 1, "Data", rule=0.50, fs=8.5)
    cw, R = 0.385, 0.13
    tx = x1 + 0.16
    cx = [tx + (j + 0.5) * cw for j in range(D)]
    yh = top - 1.42
    gy = yh + 0.62
    R = 0.155
    dx = [cx[2] + (j - 2) * 0.47 for j in range(D)]
    for j in range(D):
        if j in ACTED:
            x3_head(F, dx[j], yh, w=0.32, h=0.25, fs=7.2)
        else:
            node(F, dx[j], yh, j, R=R, fs=6.8)
    node(F, dx[2], gy, 2, R=R, fs=6.8, ls=(0, (1.6, 1.0)))
    F.T(dx[2] + R + 0.07, gy, "true $x_3$", fontsize=7, color=GREY_TXT)
    for a, b in [(0, 1), (3, 4)]:
        edge(F, dx[a], yh, dx[b], yh, R=R, hs=3.4, lw=0.7, color=INK, gapA=0.01, gapB=0.01)
    edge(F, dx[1], yh, dx[2], gy, R=R, hs=3.2, color=INK)
    edge(F, dx[2], gy, dx[3], yh, R=R, hs=3.2, color=INK)
    s0, s1 = yh + 0.125 + 0.02, gy - R - 0.02
    F.ax.plot([cx[2]] * 2, [s0, s1], color=INK, lw=0.55, ls=(0, (1.4, 1.0)))
    ym = (s0 + s1) / 2
    F.ax.plot([cx[2] - 0.06, cx[2] + 0.06], [ym - 0.035, ym + 0.035], color=INK, lw=0.9, solid_capstyle="round")
    rh, tgap = 0.32, 0.14
    ttop = yh - 0.24
    yb = table(F, tx, ttop, cw, rh, 3, 3, gap=tgap, side_fs=7, side_dx=0.07)
    yc_ref = ttop - 1.5 * rh
    yc_win = ttop - 3 * rh - tgap - 1.5 * rh
    F.T(cx[2], yb - 0.24, "recorded $x_3$", ha="center", fontsize=7, color=GREY_TXT)
    sz = 0.80
    sx = x1 + 3.08
    for which, yc in [("ref", yc_ref), ("win", yc_win)]:
        rs = scatter_slope(F, sx, yc - sz / 2, sz, which, s=0.8)
        F.T(sx + sz + 0.06, yc, f"corr\n{rs}", ha="left", fontsize=7.2, linespacing=1.0)
    F.T(sx + sz / 2, yc_win - sz / 2 - 0.12, "$x_2$", ha="center", fontsize=7)
    F.T(sx - 0.04, yc_ref + sz / 2 - 0.09, "$x_3$", ha="right", fontsize=7)

    # 2 Computing p-values
    F.panel(x2, y0, w2, h, 2, "Computing p-values", rule=0.50, fs=8.5)
    cw2, rh2 = 0.455, 0.38
    X0 = x2 + 1.51
    grid_r = X0 + D * cw2
    yx = Y_HB + 0.10
    for j in range(D):
        if j in ACTED:
            x3_head(F, X0 + (j + 0.5) * cw2, yx, w=0.28, h=0.22, fs=7.5)
        else:
            F.T(X0 + (j + 0.5) * cw2, yx, f"$x_{j+1}$", ha="center", fontsize=7.5)
    F.ax.plot([x2 + 0.04, x2 + w2 - 0.04], [Y_RULE] * 2, color=RULE, lw=0.5)
    set_txt = {}
    for i, (k, e) in enumerate(KEYS):
        set_txt[e] = F.T(0, Y_ROWS[i], SETS[e], ha="center", fontsize=7)
    wmax = max(F.bbox_cm(t)[2] - F.bbox_cm(t)[0] for t in set_txt.values())
    RS = grid_r + 0.10 + wmax / 2
    for t in set_txt.values():
        t.set_x(RS)
    for i, (k, e) in enumerate(KEYS):
        nm = [j for j in range(D) if OW[e]["named"][j]]
        if not nm:
            continue
        set_txt[e].remove(); set_txt.pop(e)
        pieces = [("$\\{$", INK)]
        for n, j in enumerate(nm):
            if n:
                pieces.append(("$,$", INK))
            pieces.append((f"$x_{j+1}$", FAULT if j in ACTED else RED))
        pieces.append(("$\\}$", INK))
        objs = [F.T(0, Y_ROWS[i], p, ha="left", fontsize=7, color=c) for p, c in pieces]
        ws = []
        for (p, _), o in zip(pieces, objs):
            wb = F.bbox_cm(o)[2] - F.bbox_cm(o)[0]
            ws.append(wb - 0.045 if '_' in p else (wb + 0.012 if p == '$,$' else wb - 0.015))
        x = max(RS - sum(ws) / 2, grid_r + 0.10)
        for o, w_ in zip(objs, ws):
            o.set_x(x); x += w_
    F.T(RS, Y_HB, "flagged\nfeatures", ha="center", va="bottom", fontsize=7, color=GREY_TXT, linespacing=0.95)
    for i, (k, e) in enumerate(KEYS):
        yc = Y_ROWS[i]
        swatch_label(F, x2 + 0.04, yc, e, fs=7, s=0.15)
        pcell_row(F, X0, yc, cw2, rh2, OW[e], fs=6.3, inset=0.022)
    key_line(F, x2 + 0.12, Y_FOOT, sep=1.40)

    # 3 All windows
    F.panel(x3, y0, w3, h, 3, "All windows", rule=0.50, fs=8.5)
    lx = x3 + 0.04
    up, down = "$\\uparrow$", "$\\downarrow$"
    cols = [("$x_3$\nfirst", "hit1", up), ("wasted\nrepair", "rate", down), ("power", "power", up)]
    cx_ = [x3 + 0.44, x3 + 1.14, x3 + 1.90]
    yh2 = Y_HB + 0.30
    for (lab, _, arr), c in zip(cols, cx_):
        F.T(c, yh2, lab, ha="center", va="bottom", fontsize=7, color="#333333", linespacing=1.05)
        F.T(c, Y_HB - 0.02, arr, ha="center", va="bottom", fontsize=7, color="#333333")
    F.ax.plot([x3 + 0.04, x3 + w3 - 0.04], [Y_RULE] * 2, color=RULE, lw=0.5)
    for i, rid in enumerate(["KS", "FULL", "OURS"]):
        d = M[rid]
        yc = Y_ROWS[i]
        fc, ec = SW[rid]
        F.ax.add_patch(Rectangle((lx, yc - 0.075), 0.15, 0.15, fc=fc, ec=ec, lw=0.5))
        b = "bold" if rid == "OURS" else "normal"
        for (_, kk, _a), c in zip(cols, cx_):
            hot = (rid == "KS" and kk == "power") or (rid == "FULL" and kk == "rate")
            F.T(c, yc, fmt(d[kk]), ha="center", fontsize=8, fontweight=b, color=RED if hot else INK)

    for xa, xb in [(x1 + w1, x2), (x2 + w2, x3)]:
        F.chevron(xa, xb, Y_HB + 0.30, color="#9d9d98")
    yo = Y_ROWS[2]
    F.ax.add_patch(FancyBboxPatch((x2 + 0.04, yo - 0.26), x3 + w3 - 0.04 - (x2 + 0.04), 0.52,
                                  boxstyle="round,pad=0,rounding_size=0.08", fc="#efefec", ec="none", zorder=0))
    save(F.fig, "figure1")


if __name__ == "__main__":
    main()
