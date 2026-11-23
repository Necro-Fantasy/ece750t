"""Figures of the report.

Usage (from the repository folder, after the analysis scripts):
    python make_figures.py results/rq2_summary.csv results/rq3_summary.csv figures

Writes figures/fig_main.png (Figure 1, main text) and figures/fig_app_gaps.png
(Figure 2, appendix). All gaps are women minus men, in percentage points.
"""
import os
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rq2_csv, rq3_csv, outdir = sys.argv[1:4]
def load(p):
    V = {}
    for r in csv.DictReader(open(p)):
        V[(int(r["split"]), r["measure"])] = tuple(float(r[k]) if r[k] not in ("", "nan") else None
                                                   for k in ("point", "lo", "hi"))
    return V
V2, V3 = load(rq2_csv), load(rq3_csv)

INK, INK2, GRID, ZERO, SINGLE = "#1f1f1d", "#5b5b57", "#e4e4e0", "#3a3a37", "#a3a39e"
COL = {"resampling": "#2a78d6", "init": "#eb6834", "order": "#1baf7a"}
NAME = {"resampling": "Resampling", "init": "Initial weights", "order": "Data order"}
CONDS = ["resampling", "init", "order"]
PAIRS = [("resampling-init", "Resampling − initial weights"),
         ("resampling-order", "Resampling − data order"),
         ("init-order", "Initial weights − data order")]
M = lambda s: s.replace("-", "−")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK})

def layout(groups):
    """rows: (y, group_key, split); headers: (y, label, color)"""
    rows, heads, y = [], [], 0.0
    for key, label, color in groups:
        heads.append((y, label, color)); y += 0.85
        for s in (0, 1, 2):
            rows.append((y, key, s)); y += 1
        y += 0.35
    return rows, heads, y

def frame(ax, rows, heads, ymax, xlim, show_y):
    ax.axvline(0, color=ZERO, lw=0.9, zorder=1)
    ax.set_ylim(ymax - 0.2, -0.6)
    ax.set_xlim(*xlim)
    ax.set_yticks([y for y, _, _ in rows])
    ax.set_yticklabels([f"split {s}" for _, _, s in rows] if show_y else [])
    ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for y, label, color in heads:
        ax.text(xlim[0], y, ("● " if color else "") + label, ha="left", va="center",
                fontsize=7.3, fontweight="bold", color=INK, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.4))
        if color:
            ax.text(xlim[0], y, "●", ha="left", va="center", fontsize=7.3, color=color, zorder=6)

def lab(ax, x, y, v, ha, bold=False, extra=""):
    ax.text(x, y, M(f"{v:+.2f}") + extra, ha=ha, va="center", fontsize=6.6,
            color=INK if bold else INK2, fontweight="bold" if bold else "normal",
            bbox=dict(fc="white", ec="none", pad=0.3), zorder=4)

def forest(ax, rows, get, color_of, scale, pad, one_sided=False):
    for y, key, s in rows:
        p, lo, hi = (scale * v for v in get(key, s))
        c = color_of(key)
        filled = lo > 0 if one_sided else (lo > 0 or hi < 0)
        ax.plot([lo, hi], [y, y], color=c, lw=1.2, zorder=2)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.2, y + 0.2], color=c, lw=1.2, zorder=2)
        ax.plot(p, y, "o", ms=5, mfc=c if filled else "white", mec=c, mew=1.3, zorder=3)
        if p < 0 and one_sided:
            lab(ax, hi + pad, y, p, "left", bold=True, extra="  (gap widened)")
        else:
            lab(ax, hi + pad, y, p, "left")

def title(ax, t):
    ax.set_title(t, loc="left", fontsize=8, fontweight="bold", color=INK, pad=3)

key_fill = [Line2D([], [], marker="o", ls="none", ms=5, mfc=INK2, mec=INK2),
            Line2D([], [], marker="o", ls="none", ms=5, mfc="white", mec=INK2, mew=1.3)]

os.makedirs(outdir, exist_ok=True)

plt.rcParams.update({"font.size": 7})
leg = [Line2D([], [], marker="o", ls="none", ms=5, mfc=INK2, mec=INK2),
       Line2D([], [], marker="o", ls="none", ms=5, mfc="white", mec=INK2, mew=1.3)]

def panels(fig_w=6.7, fig_h=2.85, top=0.88, bottom=0.27):
    fig, axs = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=300)
    fig.subplots_adjust(left=0.06, right=0.985, top=top, bottom=bottom, wspace=0.21)
    return fig, axs

rows_p, heads_p, ymax_p = layout([(k, l, None) for k, l in PAIRS])
rows_c, heads_c, ymax_c = layout([(c, NAME[c], COL[c]) for c in CONDS])

# ---- main figure: decision quantities ----
fig, (a, b, c) = panels(fig_h=2.35, top=0.875, bottom=0.26)
forest(a, rows_p, lambda k, s: V2[(s, f"arb gapdiff {k}")], lambda k: INK, 100, 0.04)
frame(a, rows_p, heads_p, ymax_p, (-1.45, 1.05), True)
for ax, rate in ((b, "fpr"), (c, "fnr")):
    forest(ax, rows_c, lambda k, s, r=rate: V3[(s, f"{r} reduction {k}")], lambda k: COL[k], 100, 0.04,
           one_sided=True)
frame(b, rows_c, heads_c, ymax_c, (-0.85, 1.3), True)
frame(c, rows_c, heads_c, ymax_c, (-0.85, 1.3), False)
for ax in (b, c): ax.set_xticks([-0.5, 0, 0.5, 1.0])
a.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
title(a, "(a) RQ2: difference between gaps"); title(b, "(b) RQ3: FPR gap reduction"); title(c, "(c) RQ3: FNR gap reduction")
a.set_xlabel("Gap of first condition − gap of second (points)")
b.set_xlabel("Reduction (points)")
c.set_xlabel("Reduction (points)")
fig.legend(leg, ["(a) 95% interval excludes 0;  (b, c) lower end of 95% interval > 0", "otherwise"],
           loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.53, 0.0))
fig.savefig(f"{outdir}/fig_main.png", dpi=300, facecolor="white"); plt.close(fig)

# ---- appendix figure: the gaps themselves ----
fig, (a, b, c) = panels(fig_h=2.75)
forest(a, rows_c, lambda k, s: V2[(s, f"arb gap {k}")], lambda k: COL[k], 100, 0.04)
frame(a, rows_c, heads_c, ymax_c, (-1.6, 0.95), True)
def dumbbell(ax, rate):
    for y, k, s in rows_c:
        bse = 100 * V3[(s, f"{rate} gap base {k}")][0]; ens = 100 * V3[(s, f"{rate} gap ens {k}")][0]
        ax.annotate("", xy=(ens, y), xytext=(bse, y), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=COL[k], lw=1.0, mutation_scale=5, shrinkA=3, shrinkB=3))
        ax.plot(bse, y, "o", ms=4.5, color=SINGLE, zorder=3); ax.plot(ens, y, "o", ms=4.5, color=COL[k], zorder=3)
        DARK = {"resampling": "#1d5aa6", "init": "#b4481c", "order": "#127a56"}
        for v, col, other in ((bse, "#6b6b6b", ens), (ens, DARK[k], bse)):
            side = "right" if v < other or (v == other and col == "#6b6b6b") else "left"
            x = v - 0.07 if side == "right" else v + 0.07
            ax.text(x, y, M(f"{v:+.2f}"), ha=side, va="center", fontsize=6.6, color=col,
                    fontweight="bold" if col != "#6b6b6b" else "normal",
                    bbox=dict(fc="white", ec="none", pad=0.3), zorder=4)
        if abs(ens) > abs(bse):
            far = max(bse, ens) + 0.80
            ax.text(far, y, "(gap widened)", ha="left", va="center", fontsize=6.0, color=INK, fontweight="bold", zorder=4)
dumbbell(b, "fpr"); frame(b, rows_c, heads_c, ymax_c, (-3.4, 0.25), True)
dumbbell(c, "fnr"); frame(c, rows_c, heads_c, ymax_c, (-0.75, 3.2), False)
title(a, "(a) RQ2: gap in share flagged"); title(b, "(b) RQ3: FPR gap"); title(c, "(c) RQ3: FNR gap")
a.set_xlabel("Women − men (points)"); b.set_xlabel("Women − men (points)"); c.set_xlabel("Women − men (points)")
lg = [leg[0], leg[1], Line2D([], [], marker="o", ls="none", ms=5, color=SINGLE), Line2D([], [], marker="o", ls="none", ms=5, color=INK2)]
fig.legend(lg, ["(a) interval excludes 0", "(a) interval includes 0", "(b, c) single model", "(b, c) abstaining ensemble (colored; value in bold)"],
           loc="lower center", ncol=4, frameon=False, fontsize=6.4, bbox_to_anchor=(0.52, 0.0), columnspacing=1.0, handletextpad=0.3)
fig.savefig(f"{outdir}/fig_app_gaps.png", dpi=300, facecolor="white"); plt.close(fig)
print("ok")
