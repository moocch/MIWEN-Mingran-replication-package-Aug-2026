# -*- coding: utf-8 -*-
"""
replot_from_npz.py
==================
Re-draws Fig. 5 from ``data/fig5_plot_data.npz`` alone -- no raw measurement
files, no instrument, no dependencies beyond numpy + matplotlib. Use this to
restyle the figure for a manuscript (fonts, size, colours, panel ratio)
without touching the physics.

    python code/replot_from_npz.py                       # default style
    python code/replot_from_npz.py --font "Arial Nova"   # pick a font
    python code/replot_from_npz.py --width 3.4           # single-column
    python code/replot_from_npz.py --out myfig           # myfig.pdf/.svg/.png

Output is true vector (PDF + SVG). Text is kept as text -- ``pdf.fonttype
42`` embeds editable TrueType and ``svg.fonttype none`` leaves SVG labels as
live <text>, so both open as editable objects in Illustrator or Inkscape.

Every quantity is read from the npz; the arrays are documented in
METHODS.md section 8.
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = str(_data_dir(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--npz", default=os.path.join(ROOT, "data", "fig5_plot_data.npz"))
ap.add_argument("--out", default=os.path.join(ROOT, "figures", "fig5_replot"),
                help="output path without extension")
ap.add_argument("--font", default="Arial", help="font family for all text")
ap.add_argument("--width", type=float, default=4.25, help="figure width, inch")
ap.add_argument("--height", type=float, default=4.55, help="figure height, inch")
ap.add_argument("--fontsize", type=float, default=8.0)
ap.add_argument("--dpi", type=int, default=600, help="dpi of the PNG copy")
ap.add_argument("--no-bottom", action="store_true",
                help="drop the Landauer / thermodynamic panel")
args = ap.parse_args()

z = np.load(args.npz, allow_pickle=True)
meta = json.loads(str(z["meta_json"]))

# ------------------------------------------------------------------ style --
rcParams.update({
    "font.family": [args.font, "DejaVu Sans"],
    "font.size": args.fontsize,
    "axes.linewidth": 0.7,
    "xtick.direction": "out", "ytick.direction": "out",
    "svg.fonttype": "none",     # SVG keeps live text
    "pdf.fonttype": 42,         # PDF embeds editable TrueType
})
C_BLUE, C_ORANGE, C_PINK, C_PURPLE = "#2458a6", "#e07a2d", "#c44e96", "#7a5aa6"
C_CYAN, C_INK, C_GRID = "#13a7b8", "#222222", "#a7adb5"

fig = plt.figure(figsize=(args.width, args.height))
if args.no_bottom:
    gs = fig.add_gridspec(1, 1, left=0.135, right=0.965, top=0.975, bottom=0.11)
    ax = fig.add_subplot(gs[0]); axb = None
else:
    gs = fig.add_gridspec(2, 1, height_ratios=[5.4, 1.0], hspace=0.10,
                          left=0.135, right=0.965, top=0.975, bottom=0.085)
    ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

xlim = z["xlim"]
for a in [a for a in (ax, axb) if a is not None]:
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(*xlim)
    a.grid(True, which="major", color=C_GRID, lw=0.55, ls=(0, (1.0, 1.65)),
           zorder=0)
    a.set_axisbelow(True)

# ------------------------------------------------------------- top panel ---
h100 = float(z["h100_line"])
ax.axhline(h100, color=C_INK, lw=1.15, zorder=3)
ax.text(0.985, h100 * 1.30, f"H100 GPU: {h100 * 1e15:.0f} fJ/MAC",
        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
        fontsize=args.fontsize, color=C_INK)

N = z["N_curve"]
ax.plot(N, z["e_ip_curve"], color=C_BLUE, lw=2.0, zorder=6,
        label="$e_{\\mathrm{ip}}(N) = e_1 + e_2 + e_3$")
ax.axhline(float(z["e1_floor"]), color=C_ORANGE, lw=1.2, ls=(0, (4.8, 1.9)),
           zorder=4, label="Waveform generation, $e_1$")
ax.plot(N, z["e2_curve"], color=C_PINK, lw=1.2, ls=(0, (3.0, 1.5, 1.0, 1.5)),
        zorder=4, label="I/Q sampling, $e_2$")
ax.plot(N, z["e3_curve"], color=C_PURPLE, lw=1.2, ls=(0, (1.0, 1.4)), zorder=4,
        label="Digital decoding, $e_3$")

pN, pe = z["points_N"], z["points_e_ip"]
speed = z["points_speedup_vs_h100"]
ax.plot(pN, pe, marker="v", ms=7.5, mfc="#f3b06c", mec=C_ORANGE, mew=1.2,
        ls="none", zorder=7, label="Experiment (this work)")
for k, (xoff, yoff) in enumerate([(0.115, 3.6), (0.10, 6.5)]):
    ax.annotate(f"$N = {int(pN[k])}$\n{pe[k] * 1e15:.2f} fJ/MAC\n"
                f"{speed[k]:.0f}$\\times$ vs H100", xy=(pN[k], pe[k]),
                xytext=(pN[k] * xoff, pe[k] * yoff), fontsize=args.fontsize - 0.5,
                style="italic",
                arrowprops=dict(arrowstyle="-", color="#666b73", lw=0.7))

ax.set_ylim(*z["ylim_top"])
ax.set_yticks([1e-13, 1e-14, 1e-15, 1e-16, 1e-17])
ax.legend(loc="lower left", fontsize=args.fontsize - 1.0, frameon=False,
          borderaxespad=0.4, handlelength=2.4, labelspacing=0.5)

# ---------------------------------------------------------- bottom panel ---
if axb is not None:
    ax.tick_params(labelbottom=False)
    b_star = float(z["b_star"])
    e_land, e_th = float(z["e_landauer"]), float(z["e_thermo"])
    # b^2 == 2b at b = 2, so at that label the two bounds are the same number
    # and share one line (black solid with cyan dashes on top).
    coincide = bool(z["limits_coincide"]) if "limits_coincide" in z.files \
        else abs(np.log10(e_land / e_th)) < 0.02
    if coincide:
        axb.axhline(e_land, color=C_INK, lw=1.3, zorder=3)
        axb.axhline(e_th, color=C_CYAN, lw=1.3, ls=(0, (3.2, 1.4)), zorder=4)
        axb.text(0.985, e_land * 1.30,
                 f"Landauer = thermodynamic limit ({b_star:g}-bit ENOB)",
                 transform=axb.get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=args.fontsize - 0.4, color=C_INK)
        axb.text(0.015, e_land * 0.72, f"{e_land * 1e21:.1f} zJ/MAC",
                 transform=axb.get_yaxis_transform(), ha="left", va="top",
                 fontsize=args.fontsize - 0.4, color=C_CYAN)
    else:
        axb.axhline(e_land, color=C_INK, lw=1.15, zorder=3)
        axb.text(0.985, e_land * 1.28, f"Landauer limit ({b_star:g}-bit ENOB)",
                 transform=axb.get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=args.fontsize, color=C_INK)
        axb.axhline(e_th, color=C_CYAN, lw=1.35, ls=(0, (3.2, 1.4)), zorder=3)
        axb.text(0.015, e_th * 0.74, "Thermodynamic limit",
                 transform=axb.get_yaxis_transform(), ha="left", va="top",
                 fontsize=args.fontsize, color=C_CYAN)
    axb.set_ylim(*z["ylim_bottom"])
    axb.set_yticks([1e-19, 1e-20])
    axb.set_xlabel(str(z["xlabel"]), fontsize=args.fontsize + 1)
    d = 0.012                                    # axis-break marks
    for a, ys in ((ax, 0.0), (axb, 1.0)):
        for x0 in (0.0, 1.0):
            a.plot([x0 - d, x0 + d], [ys - 2.2 * d, ys + 2.2 * d],
                   transform=a.transAxes, color="k", lw=0.8, clip_on=False)
else:
    ax.set_xlabel(str(z["xlabel"]), fontsize=args.fontsize + 1)

fig.supylabel(str(z["ylabel"]), fontsize=args.fontsize + 1, x=0.012)

for ext in ("pdf", "svg"):
    fig.savefig(f"{args.out}.{ext}")
fig.savefig(f"{args.out}.png", dpi=args.dpi)
plt.close(fig)

print(f"replotted from {os.path.basename(args.npz)}  "
      f"[{meta['methodology']}]")
for ext in ("pdf", "svg", "png"):
    print(f"  -> {args.out}.{ext}")
