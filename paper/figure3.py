"""Figure 3: power per fault setting where the localiser is not above the level. One row per localiser, one mark per
setting, filled by its verdict as in Figure 2; settings with nothing acted on have no power and are not drawn. Right:
settings within margin, of those run."""
import numpy as np
import matplotlib.pyplot as plt

from common import CM, ENTS, FAULTS, GREY_TXT, INK, NA, OURS, TESTBEDS, apply_style, get, save

apply_style()
FC = {"within": "white", "undetermined": "#D9D9D9"}
NO_POWER = {"nothing changed", "rows selected"}
FS = 7
W, LABW, CNTW = 6.2, 2.62, 0.62
PW = W - LABW - CNTW - 0.10
RH = 0.47
AXH = RH * len(ENTS)
TOP, BOT = 0.30, 0.62
H = TOP + AXH + BOT
fig = plt.figure(figsize=(W * CM, H * CM))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")
ax = fig.add_axes([LABW / W, BOT / H, PW / W, AXH / H])
ax.set_xlim(-0.04, 1.04); ax.set_ylim(len(ENTS) - 0.5, -0.5)
for i in range(len(ENTS) - 1):
    ax.axhline(i + 0.5, color="#D8D8D8", lw=0.4, zorder=0)
OFF = np.array([0, -0.24, 0.24, -0.12, 0.12, -0.36, 0.36])
for r, (ent, nm) in enumerate(ENTS):
    cells = []
    for tb in TESTBEDS:
        if (tb, ent) in NA:
            continue
        for fname, per in FAULTS:
            tm = per.get(tb)
            d = get(tb, tm[0], tm[1], ent) if tm and fname not in NO_POWER else None
            if d is not None and not d["abstains"]:
                cells.append(d)
    keep = sorted((d for d in cells if d["verdict"] in FC), key=lambda d: d["power"])
    seen = {}
    for d in keep:
        b = round(d["power"] / 0.06)
        k = seen.get(b, 0); seen[b] = k + 1
        ax.plot(d["power"], r + OFF[k % len(OFF)], "o", ms=3.9, mfc=FC[d["verdict"]], mec=INK, mew=0.45, zorder=3,
                clip_on=False)
    if not keep:
        ax.text(0.5, r, "none", fontsize=FS, ha="center", va="center", color="#7a7a7a", style="italic")
    y = BOT + AXH - (r + 0.5) * RH
    kw = dict(fontsize=FS, va="center", weight="bold" if ent == OURS else "normal",
              family="STIXGeneral" if ent == OURS else None)
    bg.text(0.03, y, nm, **kw)
    n_within = sum(d["verdict"] == "within" for d in cells)
    bg.text(W - 0.03, y, f"{n_within}/{len(cells)}", ha="right", **kw)
    print(f"{nm}: {n_within}/{len(cells)} within margin")
bg.text(W - 0.03, H - 0.16, "within margin / run", fontsize=FS, ha="right", va="center", color=GREY_TXT, style="italic")
ax.set_yticks([])
ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"], fontsize=FS)
ax.tick_params(axis="x", pad=1.2, length=2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
bg.text(LABW + PW / 2, 0.17, "power", fontsize=FS, ha="center", va="center")
save(fig, "figure3")
