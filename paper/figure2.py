"""Figure 2: each localiser's wasted-repair rate on each testbed. One panel per testbed, one row per localiser, one
mark per setting on a square-root axis (diamond: no change, circle: a fault), filled by its verdict. Marks of one row
are spread vertically only so that equal rates stay visible."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import (CM, COLW, ENTS, FAULTS, GREY_TXT, INK, NA, OURS, RED, TBN, TBNAME, TBW, TESTBEDS, apply_style,
                    get, save)

apply_style()
plt.rcParams.update({"hatch.linewidth": 0.6, "hatch.color": "#4a4a4a"})

FC = {"within": "white", "undetermined": "#D9D9D9", "above": RED}
EC = {"within": INK, "undetermined": INK, "above": "#6e0f1a"}
FS_TICK, FS_ROW, FS_TTL, FS_SUB, FS_AX = 7, 7.5, 7.5, 7, 7
W = COLW
LABW, PG = 2.78, 0.10
PW = (W - LABW - 0.04 - PG * (len(TESTBEDS) - 1)) / len(TESTBEDS)
RH = 0.42
AXH = RH * len(ENTS)
TOPH = 0.90
H = TOPH + AXH + 0.42 + 0.30 + 0.32
fig = plt.figure(figsize=(W * CM, H * CM))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
AY = H - TOPH - AXH
sq = np.sqrt
off = np.linspace(-0.30, 0.30, len(FAULTS) - 1)
for g, tb in enumerate(TESTBEDS):
    px = LABW + g * (PW + PG)
    ax = fig.add_axes([px / W, AY / H, PW / W, AXH / H])
    ax.set_xlim(-0.08, 1.06); ax.set_ylim(len(ENTS) - 0.5, -0.5)
    ax.axvspan(0, sq(0.10), color="#EDEDED", lw=0, zorder=0)
    ax.axvline(sq(0.10), color=INK, lw=0.5, ls=(0, (2.5, 1.5)), zorder=1)
    for i in range(len(ENTS) - 1):
        ax.axhline(i + 0.5, color="#D8D8D8", lw=0.4, zorder=0)
    for r, (ent, _) in enumerate(ENTS):
        if (tb, ent) in NA:
            ax.text(0.68, r, "N/A", fontsize=FS_TICK, ha="center", va="center", color="#7a7a7a")
            continue
        anyrow = False
        for f, (fname, per) in enumerate(FAULTS):
            tm = per.get(tb)
            if tm is None:
                continue
            d = get(tb, tm[0], tm[1], ent)
            if d is None:
                continue
            anyrow = True
            if d["abstains"]:
                ax.text(0.40, r, "abstains", fontsize=FS_TICK, va="center", style="italic", color=GREY_TXT)
                continue
            yo = r if f == 0 else r + off[f - 1]
            ax.plot(sq(d["rate"]), yo, "D" if f == 0 else "o", ms=3.9 if f == 0 else 3.3, mfc=FC[d["verdict"]],
                    mec=EC[d["verdict"]], mew=0.45, zorder=4 if f == 0 else 3, clip_on=False)
        if not anyrow:
            ax.text(0.68, r, "not run", fontsize=FS_TICK, ha="center", va="center", color="#7a7a7a")
    ax.set_yticks([])
    ax.set_xticks([0, sq(0.1), sq(0.5), 1]); ax.set_xticklabels(["0", ".1", ".5", "1"], fontsize=FS_TICK)
    ax.tick_params(axis="x", pad=1.2, length=2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    bg.text(px + PW / 2, H - 0.20, TBNAME[tb], ha="center", va="center", fontsize=FS_TTL, weight="bold")
    bg.text(px + PW / 2, H - 0.47, TBW[tb], ha="center", va="center", fontsize=FS_SUB, color=GREY_TXT, style="italic")
    bg.text(px + PW / 2, H - 0.72, TBN[tb], ha="center", va="center", fontsize=FS_SUB, color=GREY_TXT, style="italic")
for r, (ent, nm) in enumerate(ENTS):
    y = AY + AXH - (r + 0.5) * RH
    bg.text(0.04, y, nm, fontsize=FS_ROW, va="center", weight="bold" if ent == OURS else "normal",
            family="STIXGeneral" if ent == OURS else None)
YAX = AY - 0.45
bg.text(LABW + (W - LABW) / 2, YAX, "wasted-repair rate (square-root axis); dashed: the level .10",
        ha="center", va="center", fontsize=FS_AX)
ly = YAX - 0.34
fig.canvas.draw(); rend = fig.canvas.get_renderer(); k = W / fig.bbox.width


def tw(t):
    return t.get_window_extent(rend).width * k


x = LABW - 0.50
t = bg.text(x, ly, "shape:", fontsize=FS_AX, va="center", color=GREY_TXT); x += tw(t) + 0.10
for mk, lab in [("D", "no change"), ("o", "a fault")]:
    bg.plot(x + 0.06, ly, mk, ms=3.4, mfc="white", mec=INK, mew=0.45)
    t = bg.text(x + 0.19, ly, lab, fontsize=FS_AX, va="center"); x += 0.19 + tw(t) + 0.22
x += 0.20
t = bg.text(x, ly, "fill:", fontsize=FS_AX, va="center", color=GREY_TXT); x += tw(t) + 0.10
for v, lab in [("within", "within margin"), ("undetermined", "undetermined"), ("above", "above the level")]:
    bg.add_patch(Rectangle((x, ly - 0.055), 0.11, 0.11, fc=FC[v], ec=EC[v], lw=0.45))
    t = bg.text(x + 0.19, ly, lab, fontsize=FS_AX, va="center"); x += 0.19 + tw(t) + 0.20
save(fig, "figure2")
