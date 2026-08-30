"""
fig2_v3_twin.py
===============
Figure 2 (v3, after the Fig.2/Fig.3 content swap) — the PIML digital twin
and every level of its surface fidelity.  (The twin-anatomy schematic that
used to be panel a was removed; its content now lives in the caption.)

  a-e  The ladder: measured vs digital-twin IF-power surfaces at
       N = 1 / 2 / 4 / 8 / 4096 (viridis, per-column log scale).
  f-i  The N = 4096 power sweep vs the three model tiers (formerly Fig.2a-d):
       measured / ideal multiplier (10.3 dB) / physics only (1.80 dB) /
       full PIML twin (0.79 dB).
  j    Twin fidelity vs vector length (three tiers, published RMSEs).
  k    RMSE reduction vs the ideal multiplier per N
       (12.7 / 9.8 / 13.8 / 13.9 / 13.1x; ties to Fig. 1e).

Data: ../fig3/data/twin_predictions_N{1,2,4,8,4096}.npz (ladder),
      data/N4096_cascade_raw_data.npz (cascade).

Output: fig2_v3_preview.pdf / .png / .svg / _small.png  (180 x 135 mm, 600 dpi)
(f-i are square, 33 x 33 mm: both ports span -70..+10 dBm, so the axes
 box preserves the data's native 1:1 aspect -- no squashing; the row is
 edge-aligned with the ladder, 10 -> 176 mm, gaps distributed evenly.)
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
D3 = os.path.join(OUT, "..", "fig3", "data")
MM = 1 / 25.4
FIG_W, FIG_H = 180.0, 135.0

FS = 7.5
FS_TAG = 9.0

rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "axes.linewidth": 0.6,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "mathtext.fontset": "stixsans",
    "svg.fonttype": "none",
})

C_IDEAL  = "#E58C7D"    # ideal multiplier — fig2's salmon ("before") tone
C_IDEALT = "#C0503C"    # its text-label tone
C_PHYSB  = "#E2A33B"
C_PHYS   = "#2458A6"
C_VIOLET = "#6D28D9"
C_TEAL   = "#1E8A78"
C_GRAY   = "#6B7280"
C_INK    = "#1F2937"
C_BOX    = "#F3F4F6"
C_BLUEBG = "#E9F0F9"
C_VIOBG  = "#F2ECFB"

# ------------------------------------------------------------------- data ---
def w2dbm(p, floor=1e-30):
    return 10 * np.log10(np.maximum(np.asarray(p, float), floor)) + 30


ladder = []
for N, fn, rm in [(1, "twin_predictions_N1.npz", 1.30),
                  (2, "twin_predictions_N2.npz", 1.91),
                  (4, "twin_predictions_N4.npz", 1.67),
                  (8, "twin_predictions_N8.npz", 1.55)]:
    d = np.load(os.path.join(D3, fn))
    ladder.append(dict(N=N,
                       meas=w2dbm(d["p_if_meas_w"]),
                       twin=w2dbm(d["p_if_full_w"]),
                       p_lo=d["p_lo_dbm"], p_rf=d["p_rf_dbm"], rmse=rm))
d = np.load(os.path.join(D3, "twin_predictions_N4096.npz"))
ladder.append(dict(N=4096, meas=d["meas_db_arb"], twin=d["full_db_arb"],
                   p_lo=d["p_lo_dbm_grid"], p_rf=d["p_rf_dbm_grid"],
                   rmse=0.79))

cas = np.load(os.path.join(OUT, "data", "N4096_cascade_raw_data.npz"),
              allow_pickle=True)
p_lo_c, p_rf_c = cas["p_lo_dbm"], cas["p_rf_dbm"]
em = cas["error_metrics"]          # [ideal, physics, full] x [RMSE, ...]

# published model-tier RMSEs (Change_HW / manuscript):
RM_IDEAL = [16.53, 18.71, 23.12, 21.61, 10.35]
RM_PHYS  = [3.94, 3.37, 3.39, 2.81, 1.80]
RM_FULL  = [1.30, 1.91, 1.67, 1.55, 0.79]
IMPROVE  = [i / f for i, f in zip(RM_IDEAL, RM_FULL)]
N_LBL    = ["1", "2", "4", "8", "4096"]

# ------------------------------------------------------------------ figure --
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))


def axmm(x, y, w, h, **kw):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)


def tag(x_mm, y_mm, s):
    fig.text(x_mm / FIG_W, y_mm / FIG_H, s, fontsize=FS_TAG,
             fontweight="bold", va="top", ha="left", color=C_INK)


# =========================================================================
# a-e — the ladder: measured vs twin, viridis, per-column log scale
# =========================================================================
LX0, LW, LGAP = 10.0, 30.4, 2.6
ROW_M_Y, ROW_T_Y, LH = 106.0, 80.5, 23.0
col_titles = ["$N=1$ (scalar)", "$N=2$", "$N=4$", "$N=8$", "$N=4096$"]
tags_bf = ["a", "b", "c", "d", "e"]

cbar_im, cbar_ax = None, None
for k, dd in enumerate(ladder):
    x0 = LX0 + k * (LW + LGAP)
    vmin = float(np.percentile(dd["meas"], 0.5))
    vmax = float(np.percentile(dd["meas"], 99.5))
    ext = [dd["p_lo"].min(), dd["p_lo"].max(),
           dd["p_rf"].min(), dd["p_rf"].max()]
    axm = axmm(x0, ROW_M_Y, LW, LH)
    axt = axmm(x0, ROW_T_Y, LW, LH)
    im = axm.imshow(dd["meas"].T, origin="lower", aspect="auto", extent=ext,
                    vmin=vmin, vmax=vmax, cmap="viridis",
                    interpolation="nearest")
    axt.imshow(dd["twin"].T, origin="lower", aspect="auto", extent=ext,
               vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
    if k == 0:
        cbar_im, cbar_ax = im, axm
    axm.set_title(col_titles[k], pad=2.6, color=C_INK)
    for ax in (axm, axt):
        ax.set_xticks([-60, -30, 0])
        ax.set_yticks([-60, -30, 0])
        if k > 0:
            ax.set_yticklabels([])
    axm.set_xticklabels([])
    if k == 0:
        axm.set_ylabel("$P_{\\mathrm{RF}}$ (dBm)", labelpad=1.0)
        axt.set_ylabel("$P_{\\mathrm{RF}}$ (dBm)", labelpad=1.0)
    axt.text(0.05, 0.075, f"RMSE {dd['rmse']:.2f} dB",
             transform=axt.transAxes, fontsize=FS - 0.6, color="white",
             ha="left", va="bottom")
    if k == 0:
        axm.text(0.05, 0.935, "measured", transform=axm.transAxes,
                 fontsize=FS - 0.4, color="white", ha="left", va="top",
                 fontweight="bold")
        axt.text(0.05, 0.935, "digital twin", transform=axt.transAxes,
                 fontsize=FS - 0.4, color="white", ha="left", va="top",
                 fontweight="bold")
    tag(0.5 if k == 0 else x0 - 3.6, ROW_M_Y + LH + 5.4, tags_bf[k])

# inset colorbar (in panel b measured, dark noise-floor corner)
cax = cbar_ax.inset_axes((0.07, 0.09, 0.06, 0.52))
cb = fig.colorbar(cbar_im, cax=cax)
cb.set_ticks([-60, -30, 0])
cb.ax.tick_params(labelsize=FS - 1.8, length=1.6, width=0.5, pad=1.0,
                  colors="white")
cb.outline.set_edgecolor("white")
cb.outline.set_linewidth(0.5)
cax.set_title("$P_{\\mathrm{IF}}$\n(dBm)", fontsize=FS - 1.8, pad=1.6,
              color="white", loc="left")

# =========================================================================
# f-i — the N = 4096 sweep vs the three model tiers (formerly Fig. 2a-d)
# =========================================================================
surfs = [cas["measured_db"], cas["perfect_multiplier_db"],
         cas["physics_only_db"], cas["full_twin_db"]]
titles_c = ["measured, $N=4096$", "ideal multiplier", "physics only",
            "full PIML twin"]
notes_c = [None, f"RMSE {em[0, 0]:.1f} dB", f"RMSE {em[1, 0]:.2f} dB",
           f"RMSE {em[2, 0]:.2f} dB"]
tags_gj = ["f", "g", "h", "i"]
vmin_c = float(np.percentile(surfs[0], 0.5))
vmax_c = float(np.percentile(surfs[0], 99.5))
ext_c = [p_lo_c.min(), p_lo_c.max(), p_rf_c.min(), p_rf_c.max()]

CX0, CW = 10.0, 33.0         # square: native 1:1 aspect of the 33x33 sweep
R_F = LX0 + 4 * (LW + LGAP) + LW    # right edge of ladder column f (172.4)
CGAP = (R_F - CX0 - 4 * CW) / 3     # j's right edge aligns with f's
CY, CH = 36.0, 33.0
for k in range(4):
    x0 = CX0 + k * (CW + CGAP)
    ax = axmm(x0, CY, CW, CH)
    im = ax.imshow(surfs[k].T, origin="lower", aspect="auto", extent=ext_c,
                   vmin=vmin_c, vmax=vmax_c, cmap="viridis",
                   interpolation="nearest")
    ax.set_title(titles_c[k], pad=2.6, color=C_INK)
    ax.set_xticks([-60, -30, 0])
    ax.set_yticks([-60, -30, 0])
    if k == 0:
        ax.set_ylabel("$P_{\\mathrm{RF}}$ (dBm)", labelpad=1.0)
        cax2 = ax.inset_axes((0.07, 0.07, 0.055, 0.35))
        cb2 = fig.colorbar(im, cax=cax2)
        cb2.set_ticks([-100, -50])
        cb2.ax.tick_params(labelsize=FS - 1.8, length=1.6, width=0.5,
                           pad=1.0, colors="white")
        cb2.outline.set_edgecolor("white")
        cb2.outline.set_linewidth(0.5)
        cax2.set_title("$P_{\\mathrm{IF}}$\n(dB)", fontsize=FS - 1.8,
                       pad=1.6, color="white", loc="left")
    else:
        ax.set_yticklabels([])
    if notes_c[k]:
        ax.text(0.05, 0.075, notes_c[k], transform=ax.transAxes,
                fontsize=FS - 0.6, color="white", ha="left", va="bottom")
    tag(0.5 if k == 0 else x0 - 3.6, CY + CH + 5.4, tags_gj[k])

# one shared x-axis title for the ladder + cascade columns
fig.text((CX0 + (4 * CW + 3 * CGAP) / 2) / FIG_W, 30.6 / FIG_H,
         "$P_{\\mathrm{LO}}$ (dBm)", ha="center", va="center", fontsize=FS)

# =========================================================================
# j — twin fidelity vs vector length (published tier RMSEs)
# =========================================================================
AX_K = axmm(13.5, 7.3, 90, 18.7)
xs = np.arange(5)
AX_K.plot(xs, RM_IDEAL, ls=(0, (1.6, 1.4)), lw=0.9, color=C_IDEAL, zorder=3)
AX_K.plot(xs, RM_PHYS, ls=(0, (4, 2)), lw=0.9, color=C_PHYSB, zorder=3)
AX_K.plot(xs, RM_FULL, lw=1.0, color=C_PHYS, zorder=3)
AX_K.plot(xs, RM_IDEAL, "s", ms=3.6, mfc=C_IDEAL, mec="none", zorder=4,
          ls="none")
AX_K.plot(xs, RM_PHYS, "^", ms=3.8, mfc=C_PHYSB, mec="none", zorder=4,
          ls="none")
AX_K.plot(xs, RM_FULL, "o", ms=3.8, mfc=C_PHYS, mec="none", zorder=4,
          ls="none")
for x, v in zip(xs[:-1], RM_FULL[:-1]):
    AX_K.text(x + 0.06, v * 0.80, f"{v:.2f}", fontsize=FS - 1.2,
              color=C_PHYS, ha="left", va="top")
AX_K.text(4 + 0.08, RM_FULL[-1] * 1.18, f"{RM_FULL[-1]:.2f}",
          fontsize=FS - 1.2, color=C_PHYS, ha="left", va="bottom")
AX_K.text(4.25, 10.35, "ideal multiplier", fontsize=FS, color=C_IDEALT,
          ha="left", va="center")
AX_K.text(4.25, 1.80, "physics only", fontsize=FS, color="#C08A28",
          ha="left", va="center")
AX_K.text(4.25, 0.74, "full PIML twin", fontsize=FS, color=C_PHYS,
          ha="left", va="center")
AX_K.set_yscale("log")
AX_K.set_xlim(-0.3, 5.9)
AX_K.set_ylim(0.48, 42)
AX_K.set_xticks(xs)
AX_K.set_xticklabels(N_LBL)
AX_K.set_yticks([1, 3, 10, 30])
AX_K.set_yticklabels(["1", "3", "10", "30"])
AX_K.set_xlabel("vector length N", labelpad=1.0)
AX_K.set_ylabel("RMSE (dB)", labelpad=1.0)
for s in ("top", "right"):
    AX_K.spines[s].set_visible(False)

# =========================================================================
# k — per-N before/after bars with the improvement arrow (Fig. 1e style)
# =========================================================================
AX_L = axmm(122.0, 7.3, 54.0, 18.7)
xs = np.arange(5)
AX_L.bar(xs - 0.19, RM_IDEAL, width=0.34, color=C_IDEAL, zorder=3)
AX_L.bar(xs + 0.19, RM_FULL, width=0.34, color=C_PHYS, zorder=3)
for x, vi, vf, imp in zip(xs, RM_IDEAL, RM_FULL, IMPROVE):
    AX_L.annotate("", xy=(x + 0.19, vf + 1.0),
                  xytext=(x - 0.05, vi - 0.6),
                  arrowprops=dict(arrowstyle="-|>", mutation_scale=7,
                                  lw=0.9, color=C_INK,
                                  connectionstyle="arc3,rad=-0.28"))
    AX_L.text(x, vi + 0.9, f"{imp:.1f}$\\times$", fontsize=FS - 1.2,
              ha="center", va="bottom", color=C_INK, fontweight="bold")
AX_L.text(-0.55, 26.6, "ideal multiplier", fontsize=FS - 1.0,
          color=C_IDEALT, ha="left", va="center")
AX_L.text(-0.55, 23.9, "full PIML twin", fontsize=FS - 1.0, color=C_PHYS,
          ha="left", va="center")
AX_L.set_xlim(-0.65, 4.65)
AX_L.set_ylim(0, 31)
AX_L.set_xticks(xs)
AX_L.set_xticklabels(N_LBL)
AX_L.set_yticks([0, 10, 20])
AX_L.set_xlabel("vector length N", labelpad=1.0)
AX_L.set_ylabel("RMSE (dB)", labelpad=1.0)
for s in ("top", "right"):
    AX_L.spines[s].set_visible(False)

# ============================ panel tags =====================================
tag(0.5, 30.0, "j")
tag(110.0, 30.0, "k")

# ============================ save ==========================================
fig.savefig(os.path.join(OUT, "fig2_v3_preview.pdf"), dpi=600)
fig.savefig(os.path.join(OUT, "fig2_v3_preview.png"), dpi=600)
fig.savefig(os.path.join(OUT, "fig2_v3_preview.svg"))

from PIL import Image
im = Image.open(os.path.join(OUT, "fig2_v3_preview.png"))
im.resize((im.width // 3, im.height // 3), Image.LANCZOS).save(
    os.path.join(OUT, "fig2_v3_preview_small.png"))
print("wrote fig2_v3_preview.*  cascade "
      f"{em[0,0]:.2f}/{em[1,0]:.2f}/{em[2,0]:.2f} dB")
