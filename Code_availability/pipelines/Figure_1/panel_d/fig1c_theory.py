"""
fig1c_theory.py
===============
Figure 1, right column (panels d, e) — grandma-simple versions.

(d) The real mixer is not an ideal multiplier: the exact diode-ring
    transfer law (blue, y ∝ sinh(u_RF)·sinh(u_LO), drawn at fixed
    u_LO = 0.5) bends away from the ideal product (red dashed).
    The gap is deterministic — i.e. PREDICTABLE — distortion (orange
    fill), and inside the green multiplicative window the mixer is
    nearly ideal. All math lives in the caption / Methods.

(e) The payoff of the PIML digital twin, as a before/after bar chart:
    same mixer, no hardware change — the model-hardware error drops
    from 16.5 dB (ideal-multiplier assumption) to 1.3 dB (with the
    PIML digital twin): 12.7x lower error.

Output: fig1c_theory.svg / .pdf / .png (600 dpi)
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

OUT = str(_data_dir(__file__))
MM = 1 / 25.4

# typography: the original (well-balanced) size family — annotations 5.8,
# ticks 6.0, axis labels 7.0; panel tags 8 pt bold lowercase (Nature style)
FS = 5.8
FS_TAG = 8.0

rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 7.0,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "mathtext.fontset": "stixsans",
    "svg.fonttype": "none",
})

C_IDEAL = "#C33D35"     # ideal-multiplier assumption
C_PHYS  = "#2458A6"     # real mixer / with PIML twin
C_DIST  = "#E28A2B"     # distortion
C_WIN   = "#1E8A78"     # multiplicative window (green)
C_WIN_T = "#0E6B5B"
C_GRAY  = "#6B7280"
C_INK   = "#1F2937"

fig = plt.figure(figsize=(56 * MM, 66 * MM))
# same grid as fig1b so rows align across columns
gs = fig.add_gridspec(2, 1, left=0.175, right=0.94, top=0.945, bottom=0.105,
                      hspace=0.62)

# =================== (d) real mixer vs ideal multiplier =====================
ax = fig.add_subplot(gs[0])
uLO = 0.5                      # fixed LO drive (stated in the caption)
u = np.linspace(-2.35, 2.35, 400)
y_exact = np.sinh(u) * np.sinh(uLO)
y_lin = u * uLO

# green: where the mixer is close to an ideal multiplier
ax.axvspan(-1.0, 1.0, color=C_WIN, alpha=0.15, lw=0)
# orange: the (deterministic, hence predictable) distortion
ax.fill_between(u, y_lin, y_exact, color=C_DIST, alpha=0.25, lw=0)

ax.plot(u, y_lin, ls=(0, (4, 2.4)), lw=1.1, color=C_IDEAL, zorder=3)
ax.plot(u, y_exact, lw=1.4, color=C_PHYS, zorder=4)

ax.text(0.0, -2.30, "multiplicative\nwindow", fontsize=FS, color=C_WIN_T,
        ha="center", va="center")
ax.text(-1.30, -0.22, "ideal multiplier", fontsize=FS, color=C_IDEAL,
        ha="center", va="center", rotation=13)
ax.text(1.03, 1.85, "real mixer", fontsize=FS, color=C_PHYS,
        ha="center", va="center")
ax.annotate("predictable\ndistortion", xy=(2.10, 1.85), xytext=(1.55, -1.35),
            fontsize=FS, color="#B4650A", ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=C_GRAY,
                            shrinkA=2, shrinkB=1))

ax.set_xlim(-2.35, 2.35)
ax.set_ylim(-3.0, 3.0)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([-2, 0, 2])
ax.set_xlabel(r"Input drive $u_{RF}$", labelpad=1.6)
ax.set_ylabel(r"Mixer output $y$", labelpad=1.0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ============= (e) PIML payoff: same hardware, 12.7x lower error ============
ax2 = fig.add_subplot(gs[1])

vals = [16.53, 1.30]
cols = [C_IDEAL, C_PHYS]
xpos = [0, 1]
ax2.bar(xpos, vals, width=0.52, color=cols, zorder=3)
ax2.text(0, 17.3, "16.5 dB", fontsize=6.2, ha="center", va="bottom",
         color=C_IDEAL, fontweight="bold")
ax2.text(1, 2.1, "1.3 dB", fontsize=6.2, ha="center", va="bottom",
         color=C_PHYS, fontweight="bold")

# 12.7x arrow between the two bar tops
ax2.annotate("", xy=(0.95, 2.9), xytext=(0.32, 16.2),
             arrowprops=dict(arrowstyle="-|>", mutation_scale=9, lw=1.1,
                             color=C_INK,
                             connectionstyle="arc3,rad=-0.25"))
ax2.text(0.46, 10.3, r"12.7$\times$" + "\nlower error", fontsize=6.6,
         ha="center", va="center", color=C_INK, fontweight="bold",
         linespacing=1.25)

ax2.set_xlim(-0.62, 1.62)
ax2.set_ylim(0, 23)
ax2.set_yticks([0, 5, 10, 15])
ax2.set_xticks(xpos)
ax2.set_xticklabels(["ideal-multiplier\nassumption", "with PIML\ndigital twin"],
                    fontsize=6.0, linespacing=1.15)
ax2.tick_params(axis="x", length=0)
ax2.set_ylabel("Model–hardware\nerror (dB)", labelpad=1.5)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# panel letters: Nature style — lowercase bold, no parentheses
fig.text(0.012, ax.get_position().y1 + 0.028, "d", fontsize=FS_TAG,
         fontweight="bold", va="top", ha="left", color=C_INK)
fig.text(0.012, ax2.get_position().y1 + 0.028, "e", fontsize=FS_TAG,
         fontweight="bold", va="top", ha="left", color=C_INK)

fig.savefig(os.path.join(OUT, "fig1c_theory.svg"))
fig.savefig(os.path.join(OUT, "fig1c_theory.pdf"))
fig.savefig(os.path.join(OUT, "fig1c_theory.png"), dpi=600)
print("wrote fig1c_theory.svg/.pdf/.png")
