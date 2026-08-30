"""Generates the three figures used in the D-MT4SR paper.

All numbers are transcribed from the experiment logs and diagnostic scripts in
this repository (see paper Sec. V-VII for provenance).

Figures are drawn at their FINAL printed size so that no LaTeX rescaling is
applied and text renders at the intended point size:
  fig_saturation    -> 7.16 in (IEEEtran textwidth, used in a figure* float)
  fig_paired        -> 3.45 in (IEEEtran columnwidth)
  fig_concentration -> 3.45 in (IEEEtran columnwidth)

Run:  python figs/make_figs.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Palette -----------------------------------------------------------------
# Categorical slots assigned in fixed order, never cycled. Validated for
# colour-vision deficiency on a light (print) surface: worst all-pairs CVD
# dE 9.2, worst normal-vision dE 24.0. AQUA sits below 3:1 contrast on white,
# so every series drawn in AQUA also carries a direct text label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#b8b7b2"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "legend.handlelength": 1.9,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.35,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 0.45,
    "grid.linewidth": 0.4,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.9,
})

TEXTWIDTH = 7.16   # inches, IEEEtran two-column text width
COLWIDTH = 3.45    # inches, IEEEtran single column width


def save(fig, name):
    fig.savefig("figs/%s.pdf" % name, bbox_inches="tight", pad_inches=0.015)
    fig.savefig("figs/%s.png" % name, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


# =============================================================== Figure 1 ====
# Attention saturation: diagnose_saturation.py --data_name All_Beauty
epochs = [0, 1, 2]
rel_none = [852510.00, 909636.81, 928796.50]
qk_none = [2.785, 3.207, 3.032]
rel_std = [3.66, 4.08, 4.00]
ent_none = [0.0004, 0.0005, 0.0002]
ent_std = [3.9684, 3.9775, 3.9805]
ENT_MAX = float(np.log(100.0))

fig, ax = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.78))

# ---- (a) score magnitude, log scale
a = ax[0]
a.set_yscale("log")
a.plot(epochs, rel_none, "o-", color=ORANGE, label="relation scores, original")
a.plot(epochs, rel_std, "^-", color=AQUA, label="relation scores, normalized")
a.plot(epochs, qk_none, "s--", color=BLUE, label="query-key scores")
a.set_xlabel("Training epoch")
a.set_ylabel("Max. absolute score")
a.set_xticks(epochs)
a.set_xlim(-0.30, 2.95)
a.set_ylim(1.0, 6e6)
a.set_yticks([1e0, 1e2, 1e4, 1e6])
a.set_axisbelow(True)

# Direct annotation of the gap the panel exists to show.
a.annotate("", xy=(2.55, 9.1e5), xytext=(2.55, 3.6),
           arrowprops=dict(arrowstyle="<->", lw=0.8, color=INK2,
                           shrinkA=0, shrinkB=0))
a.text(2.45, 2.2e3, r"$3.1\times10^{5}$", rotation=90, ha="right",
       va="center", fontsize=7, color=INK)
# AQUA relief: the normalized series is named directly on the plot.
a.text(1.28, 14.0, "normalized relation and query-key at parity",
       fontsize=6.6, color=INK2, ha="center", va="center")
a.legend(loc="center left", bbox_to_anchor=(0.015, 0.55), framealpha=0.95,
         edgecolor=GRID, fancybox=False)
a.set_title("(a) Score magnitude", pad=4)

# ---- (b) attention entropy
b = ax[1]
b.plot(epochs, ent_std, "^-", color=AQUA, label="normalized")
b.plot(epochs, ent_none, "o-", color=ORANGE, label="original")
b.axhline(ENT_MAX, ls=(0, (2, 2)), lw=0.8, color=INK2)
b.text(2.90, ENT_MAX + 0.12, "maximum (uniform attention)", fontsize=6.6,
       color=INK2, ha="right", va="bottom")
b.set_xlabel("Training epoch")
b.set_ylabel("Attention entropy (nats)")
b.set_xticks(epochs)
b.set_xlim(-0.30, 2.95)
b.set_ylim(0, 5.5)
b.set_yticks([0, 1, 2, 3, 4, 5])
b.set_axisbelow(True)
# Direct labels: AQUA relief, and the number the reader needs.
b.text(0.10, 3.60, "86% of maximum", fontsize=6.8, color=INK, va="top")
b.text(0.10, 0.28, "0.01% of maximum", fontsize=6.8, color=INK, va="bottom")
b.legend(loc="center left", bbox_to_anchor=(0.015, 0.42), framealpha=0.95,
         edgecolor=GRID, fancybox=False)
b.set_title("(b) Attention entropy", pad=4)

fig.subplots_adjust(wspace=0.24)
save(fig, "fig_saturation")


# =============================================================== Figure 2 ====
# Per-seed paired deltas, D-MT4SR minus MT4SR, Appliances, 10 seeds.
# Form: one row per metric, one dot per seed. The question this figure answers
# is "how many seeds fall on the wrong side of zero", which a dot plot shows
# directly and a 30-bar grouped chart does not.
deltas = {
    "MRR":     [+0.0097, -0.0011, +0.0026, +0.0011, +0.0021,
                -0.0002, +0.0012, +0.0074, +0.0029, +0.0008],
    "NDCG@10": [+0.0095, +0.0007, +0.0020, +0.0024, +0.0006,
                +0.0007, +0.0010, +0.0090, +0.0058, +0.0041],
    "HIT@10":  [+0.0096, +0.0045, +0.0000, +0.0077, -0.0045,
                +0.0058, +0.0013, +0.0122, +0.0148, +0.0154],
}
records = ["8-2", "10-0", "8-1"]
colors = [BLUE, ORANGE, AQUA]          # fixed slot order, not by rank
rows = list(deltas.keys())

fig, a = plt.subplots(figsize=(COLWIDTH, 1.78))
for i, (name, color, rec) in enumerate(zip(rows, colors, records)):
    y = len(rows) - 1 - i
    vals = deltas[name]
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v < 0]
    ties = [v for v in vals if v == 0]
    # Filled = seed won, hollow = seed lost or tied: sign is encoded twice.
    a.plot(wins, [y] * len(wins), "o", color=color, mec="white", mew=0.8,
           ms=5.5, ls="none", zorder=3)
    a.plot(losses, [y] * len(losses), "o", mfc="white", mec=color, mew=1.2,
           ms=5.5, ls="none", zorder=3)
    a.plot(ties, [y] * len(ties), "o", mfc="white", mec=INK2, mew=1.2,
           ms=5.5, ls="none", zorder=3)
    a.plot([float(np.mean(vals))], [y - 0.21], "D", color=color, mec="white",
           mew=0.8, ms=5.0, ls="none", zorder=4)
    a.text(0.0212, y, rec, fontsize=7.5, va="center", ha="right", color=INK)

a.axvline(0, color=INK, lw=0.8, zorder=2)
a.set_yticks(range(len(rows)))
a.set_yticklabels(rows[::-1], color=INK)
a.set_ylim(-0.78, len(rows) - 0.42)
a.set_xlim(-0.0064, 0.0218)
a.set_xticks([-0.005, 0.000, 0.005, 0.010, 0.015, 0.020])
a.set_xticklabels([r"$-$0.005", "0", "0.005", "0.010", "0.015", "0.020"])
a.set_xlabel("Paired improvement of D-MT4SR over MT4SR")
a.grid(axis="x")
a.grid(axis="y", visible=False)
a.set_axisbelow(True)
a.text(0.0212, len(rows) - 0.70, "W-L", fontsize=7, va="center", ha="right",
       color=INK2)

# One legend, drawn from proxies, placed in the empty lower band.
proxies = [
    plt.Line2D([], [], ls="none", marker="o", color=INK2, mec="white",
               mew=0.8, ms=5.5, label="seed won"),
    plt.Line2D([], [], ls="none", marker="o", mfc="white", mec=INK2,
               mew=1.2, ms=5.5, label="lost or tied"),
    plt.Line2D([], [], ls="none", marker="D", color=INK2, mec="white",
               mew=0.8, ms=5.0, label="mean"),
]
a.legend(handles=proxies, loc="lower center", bbox_to_anchor=(0.52, -0.02),
         ncol=3, framealpha=0.95, edgecolor=GRID, fancybox=False,
         columnspacing=0.9, handletextpad=0.3, borderpad=0.32)
save(fig, "fig_paired")


# =============================================================== Figure 3 ====
# Held-out target concentration (check_target_concentration.py).
# Two datasets carry the colour; split is encoded by line style and marker fill,
# so the four series never rely on colour alone.
ks = [1, 5, 10, 20]
x = np.arange(len(ks))
series = [
    ("All Beauty, validation", [45.7, 69.2, 72.2, 74.7], BLUE, "--", "o", BLUE),
    ("All Beauty, test", [18.7, 45.6, 50.3, 55.4], BLUE, "-", "o", "white"),
    ("Appliances, validation", [1.4, 4.2, 6.4, 9.9], ORANGE, "--", "^", ORANGE),
    ("Appliances, test", [0.8, 3.0, 4.9, 7.8], ORANGE, "-", "^", "white"),
]

fig, a = plt.subplots(figsize=(COLWIDTH, 1.80))
for label, ys, color, ls, marker, mfc in series:
    a.plot(x, ys, ls=ls, marker=marker, color=color, mfc=mfc, mec=color,
           mew=1.1, ms=5.0, label=label, zorder=3)

a.set_xticks(x)
a.set_xticklabels([str(k) for k in ks])
a.set_xlim(-0.18, 3.28)
a.set_ylim(0, 82)
a.set_yticks([0, 20, 40, 60, 80])
a.set_xlabel("$k$ most frequent target items")
a.set_ylabel("Held-out targets covered (%)")
a.set_axisbelow(True)
a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
         frameon=False, columnspacing=1.2, handletextpad=0.45,
         labelspacing=0.3)
save(fig, "fig_concentration")

print("wrote fig_saturation, fig_paired, fig_concentration (.pdf and .png)")
