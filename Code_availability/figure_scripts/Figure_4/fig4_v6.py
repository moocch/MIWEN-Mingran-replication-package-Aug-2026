"""
fig4_v6.py  (round 6 — aligned two-column layout)
=================================================
s41586-Fig.4 grammar, two claims, one column each. Confusion rows c/e
aligned (30-mm matrices at y 8-38). Right schematic d spans y 46-136
(two side-by-side portrait step boxes, vertical flow inside each).
Column headers are direct claim titles in ink.
Confusion values carry +/- 1 sigma and %.

Left  — in-physics CNN inference (comb, N = 1,200):
  a  implementation schematic (real photo/patch/kernels/feature maps),
     classifier outside the recirculation loop.
  b  bars 98.42 / 98.92 / 99.50.
  c  confusions: measured vs clean-trained digital.
Right — hardware-aware training (serial, 0 dBm, N = 600):
  d  two framed steps: measure once -> fit f -> validate; train with
     f inside every product (labels live in the caption).
  e  confusions on hardware: clean-trained vs hardware-aware.

Data: ../fig4_v5/data/ + data/comb_assets.npz (all gate-checked; see
README.md).  Output: fig4_v6_preview.pdf/.png/.svg/_small.png
(180 x 142 mm)
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = str(_data_dir(__file__))
D6 = OUT
D5 = OUT   # Fig. 4 serial-campaign assets live beside the comb assets
MM = 1 / 25.4
FIG_W, FIG_H = 180.0, 142.0

FS = 7.5
FS_S = 6.3
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

# house palette only (figs 1-3): blue = measured/twin, salmon = ideal/
# failure, violet = learned/twin-in-training, teal = outputs, gold =
# mixer/analog, viridis = measured power surfaces, grays = digital
C_PHYSBG = "#FBF3E2"      # gold tint  (analog domain, fig3a mixer box)
C_PHYS = "#9C6E1B"        # gold label
C_DIGBG = "#F3F4F6"       # gray       (digital/software, fig4_v3 box)
C_DIG = "#374151"         # ink-gray label
C_SALM = "#E58C7D"
C_SALMT = "#C0503C"       # ideal-assumption failure
C_PINK = "#2458A6"        # twin accent = house blue (full-twin color of
                          # fig2k/l and fig3c)
C_PINKBG = "#E9F0F9"      # light-blue tint (fig2a physics-block fill)
C_ORANGE = "#F5DFC0"      # fig1 saturation tint
C_WIN = "#DEEFE2"         # fig1 multiplicative-window tint
C_STEPBG = "#F4F5F7"
C_BLUE = "#2458A6"        # measured
C_LBLUE = "#C9CDD3"       # light gray-blue (digital bar 1)
C_MGRAY = "#8D97A5"       # medium gray     (digital bar 2)
C_VIOLET = "#6D28D9"      # data/patch stream (RF) — deep violet, as in
                          # fig1a's data path
C_AMBT = "#B87A0F"        # dark amber (predictable-distortion label)
C_TEAL = "#1E8A78"
C_GOLD = "#B8860B"
C_GRAY = "#6B7280"
C_INK = "#1F2937"

# ------------------------------------------------------------------- data ---
R = np.load(os.path.join(D5, "fig4_serial_results.npz"), allow_pickle=True)
A = np.load(os.path.join(D5, "fig4_panel_assets.npz"))
CB = np.load(os.path.join(D6, "comb_assets.npz"))

sc = A["scatter_captures"]
conf_meas = CB["conf_measured"].astype(float)
conf_cdig = CB["conf_clean_digital"].astype(float)
conf_serial_clean = R["confusion_clean"].astype(float)
conf_serial_twin = R["confusion_twin"].astype(float)

B1_VAL = [98.42, 98.92, 99.50]
B1_SIG = [0.36, 0.30, 0.20]

# ------------------------------------------------------------------ figure --
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))


def axmm(x, y, w, h, **kw):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)


def tag(x_mm, y_mm, s):
    t = fig.text(x_mm / FIG_W, y_mm / FIG_H, s, fontsize=FS_TAG,
                 fontweight="bold", va="top", ha="left", color=C_INK)
    t.set_gid("tag")


def band(ax, x, y, w, h, fc, r=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec="none", zorder=1))


def box(ax, x, y, w, h, fc="white", ec=C_GRAY, lw=0.8, r=1.0, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arr(ax, p0, p1, color=C_INK, lw=0.9, ls="-", z=5, ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=1, shrinkB=1, zorder=z))


def polyarr(ax, pts, color=C_INK, lw=0.9, z=5, ms=6, ls="-"):
    pts = [tuple(p) for p in pts]
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=z, linestyle=ls)
    arr(ax, pts[-2], pts[-1], color=color, lw=lw, z=z, ms=ms, ls=ls)


def comb(ax, x0, y0, w, h, heights, color, lw=1.0):
    ax.plot([x0, x0 + w], [y0, y0], color=C_GRAY, lw=0.5, zorder=4)
    n = len(heights)
    for i, v in enumerate(heights):
        cx = x0 + (i + 0.5) * w / n
        ax.plot([cx, cx], [y0, y0 + v * h], color=color, lw=lw, zorder=5)
        ax.plot(cx, y0 + v * h, ".", ms=1.8, color=color, zorder=5)


def mixer(ax, x, y, rr):
    ax.add_patch(Circle((x, y), rr, fc="white", ec=C_INK, lw=1.1, zorder=6))
    d = rr / np.sqrt(2)
    ax.plot([x - d, x + d], [y - d, y + d], color=C_INK, lw=0.9, zorder=7)
    ax.plot([x - d, x + d], [y + d, y - d], color=C_INK, lw=0.9, zorder=7)


def confusion(ax, M, title, tcolor, val, vcolor, ylab):
    Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1, origin="upper",
              interpolation="nearest", aspect="equal", rasterized=True)
    ax.set_xticks([0, 14, 28, 42])
    ax.set_yticks([0, 14, 28, 42])
    ax.set_title(title, pad=2.4, color=tcolor, fontsize=FS - 0.4)
    ax.set_xlabel("predicted class", labelpad=1.0)
    if ylab:
        ax.set_ylabel("true class", labelpad=1.0)
    else:
        ax.set_yticklabels([])
    ax.text(41.5, 4.6, val, fontsize=FS - 0.6, fontweight="bold",
            color=vcolor, ha="right", va="center")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def stepnum(ax, x, y, n, color):
    ax.add_patch(Circle((x, y), 1.55, fc="white", ec=color, lw=0.8,
                        zorder=6))
    ax.text(x, y - 0.06, n, fontsize=FS_S - 0.4, color=color, ha="center",
            va="center", fontweight="bold", zorder=7)


# column headers (direct claim titles) + separator
fig.text(44.0 / FIG_W, 139.4 / FIG_H, "CNN inference on a passive mixer",
         fontsize=FS, color=C_INK, ha="center", va="center",
         fontweight="bold")
fig.text(136.0 / FIG_W, 139.4 / FIG_H, "hardware-aware training",
         fontsize=FS, color=C_INK, ha="center", va="center",
         fontweight="bold")
sep = plt.Line2D([90.0 / FIG_W, 90.0 / FIG_W],
                 [4.0 / FIG_H, 140.0 / FIG_H], transform=fig.transFigure,
                 color="#C9CDD3", lw=0.7, ls=(0, (4, 3)))
fig.add_artist(sep)

# ============================ a: conv on the mixer ==========================
AX_A = axmm(2.0, 74.0, 86.5, 62)
AX_A.set_xlim(0, 86.5)
AX_A.set_ylim(0, 62)
AX_A.axis("off")

photo = CB["photo_u8"]
patch = CB["patch_u8"]
pr, pc = [int(v) for v in CB["patch_rc"]]

# photo (small) with patch marker + zoomed patch (label ABOVE the zoom)
PX, PY, PW = 1.5, 49.5, 10.0
AX_A.imshow(photo, extent=[PX, PX + PW, PY, PY + PW], origin="upper",
            interpolation="nearest", zorder=3)
rx = PX + pc / 32 * PW
ry = PY + PW - (pr + 5) / 32 * PW
AX_A.add_patch(plt.Rectangle((rx, ry), 5 / 32 * PW, 5 / 32 * PW,
                             fc="none", ec=C_VIOLET, lw=0.9, zorder=5))
AX_A.text(7.5, 47.2, "GTSRB", fontsize=FS, color=C_INK,
          ha="center", va="center")
AX_A.text(8.3, 44.8, "real photographs", fontsize=FS - 0.5,
          color=C_INK, ha="center", va="center")

ZX, ZY, ZW = 17.8, 50.5, 8.0
AX_A.imshow(patch, extent=[ZX, ZX + ZW, ZY, ZY + ZW], origin="upper",
            interpolation="nearest", zorder=4)
AX_A.add_patch(plt.Rectangle((ZX, ZY), ZW, ZW, fc="none", ec=C_VIOLET,
                             lw=0.9, zorder=5))
for cnr in ((rx + 5 / 32 * PW, ry + 5 / 32 * PW, ZX, ZY + ZW),
            (rx + 5 / 32 * PW, ry, ZX, ZY)):
    AX_A.plot([cnr[0], cnr[2]], [cnr[1], cnr[3]], color=C_GRAY, lw=0.5,
              zorder=4)
AX_A.text(21.0, 60.4, "$5{\\times}5{\\times}3$ patch",
          fontsize=FS - 0.5, color=C_VIOLET, ha="center", va="center")

# Physical band
band(AX_A, 27.5, 19.0, 31.0, 41.5, C_PHYSBG)
AX_A.text(44.0, 58.4, "Physical — passive mixer", fontsize=FS,
          color=C_PHYS, ha="center", va="center", fontweight="bold")

KT = CB["kern_tiles"]
KS = 3.8
for j in range(4):
    kx = 33.5 + (j % 2) * (KS + 0.6)
    ky = 49.4 - (j // 2) * (KS + 0.6)
    AX_A.imshow(KT[j].mean(-1), extent=[kx, kx + KS, ky, ky + KS],
                origin="upper", cmap="Blues", vmin=0, vmax=1,
                interpolation="nearest", zorder=4)
AX_A.text(38.6, 55.0, "kernel bank (LO)", fontsize=FS - 0.5,
          color=C_BLUE, ha="center", va="center", fontweight="bold")

comb(AX_A, 33.0, 25.5, 11.0, 5.0, [0.8, 0.45, 1.0, 0.6, 0.9], C_VIOLET)
AX_A.text(38.6, 23.0, "patch stream (RF)", fontsize=FS - 0.5,
          color=C_VIOLET, ha="center", va="center", fontweight="bold")
polyarr(AX_A, [(ZX + ZW / 2, 50.1), (ZX + ZW / 2, 28.0), (32.3, 28.0)],
        color=C_VIOLET, lw=0.8)

mixer(AX_A, 48.5, 38.5, 3.1)
arr(AX_A, (41.9, 45.4), (46.6, 40.7), color=C_BLUE, lw=0.9)
arr(AX_A, (44.8, 29.5), (46.8, 35.5), color=C_VIOLET, lw=0.9)
comb(AX_A, 52.4, 35.4, 5.4, 4.4, [0.5, 1.0, 0.7], C_TEAL)
AX_A.text(54.6, 43.6, "$\\mathbf{y} = \\mathbf{W}\\mathbf{x}$",
          fontsize=FS, color=C_TEAL, ha="center", va="center")

# Digital band (activation only — the loop body)
band(AX_A, 59.8, 19.0, 15.2, 41.5, C_DIGBG)
AX_A.text(67.4, 58.4, "Digital", fontsize=FS, color=C_DIG,
          ha="center", va="center", fontweight="bold")
arr(AX_A, (58.9, 38.5), (59.6, 38.5), color=C_INK, lw=0.9, ms=5)

AX_A.text(67.4, 55.2, "activation", fontsize=FS, color=C_INK,
          ha="center", va="center")

FMs = CB["featmaps"]
FT = 6.0
for j in range(3):
    fx = 62.2 + j * 1.6
    fy = 44.0 - j * 1.6
    AX_A.imshow(FMs[2 - j], extent=[fx, fx + FT, fy, fy + FT],
                origin="upper", cmap="Blues", zorder=4 + j,
                interpolation="nearest")
AX_A.text(67.4, 35.9, "feature\nmaps", fontsize=FS, color=C_INK,
          ha="center", va="center", linespacing=1.15)

# recirculate loop (excludes the classifier)
polyarr(AX_A, [(67.5, 18.6), (67.5, 14.2), (33.0, 14.2), (33.0, 18.6)],
        color=C_INK, lw=0.8)
AX_A.text(31.4, 16.4, "next layer, $\\ell < 4$", fontsize=FS - 0.5,
          color=C_INK, ha="right", va="center")
AX_A.text(50.5, 11.2, "conv 32 $\\rightarrow$ conv 64 $\\rightarrow$ "
          "dense 128 $\\rightarrow$ dense 43", fontsize=FS - 0.5,
          color=C_GRAY, ha="center", va="center")

# classifier box OUTSIDE the loop (same vertical extent as the bands)
box(AX_A, 75.6, 19.0, 10.6, 41.5, fc="white", ec=C_INK, lw=0.9, r=1.6)
arr(AX_A, (75.0, 38.5), (75.6, 38.5), color=C_INK, lw=0.9, ms=5)
AX_A.text(80.9, 61.4, "$\\ell = 4$ only", fontsize=FS, color=C_INK,
          ha="center", va="center")
bx0, by0 = 76.9, 23.5
AX_A.plot([bx0, bx0 + 8.0], [by0, by0], color=C_GRAY, lw=0.5, zorder=4)
for i, v in enumerate([0.3, 0.45, 1.0, 0.35]):
    AX_A.add_patch(plt.Rectangle((bx0 + 0.5 + i * 1.9, by0), 1.3,
                                 v * 9.0, fc=C_TEAL if i == 2 else C_LBLUE,
                                 ec="none", zorder=5))
AX_A.text(80.9, 46.5, "argmax", fontsize=FS, color=C_INK,
          ha="center", va="center")
AX_A.text(80.9, 54.5, "“Stop”", fontsize=FS, color=C_TEAL,
          ha="center", va="center", fontweight="bold")

# ============================ b: claim-1 bars ===============================
AX_B = axmm(13.5, 53.0, 62.5, 27)
xs = np.arange(3)
AX_B.bar(xs, B1_VAL, 0.6, color=[C_SALM, C_BLUE, C_MGRAY], zorder=3)
AX_B.errorbar(xs, B1_VAL, yerr=B1_SIG, fmt="none", ecolor=C_INK,
              elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=4)
for x, v, s in zip(xs, B1_VAL, B1_SIG):
    AX_B.text(x, v + s + 0.08, f"{v:.2f}", fontsize=FS_S,
              ha="center", va="bottom", color=C_INK,
              fontweight="bold" if x == 1 else "normal")
AX_B.set_ylim(97.7, 100.1)
AX_B.set_yticks([98, 99, 100])
AX_B.set_xticks(xs)
AX_B.set_xticklabels(["digital,\nsame weights", "measured,\nin-physics",
                      "digital,\nclean-trained"], fontsize=FS_S,
                     linespacing=1.15)
AX_B.tick_params(axis="x", length=0, pad=2.0)
for i, lab in enumerate(AX_B.get_xticklabels()):
    lab.set_color([C_SALMT, C_BLUE, C_GRAY][i])
AX_B.set_ylabel("accuracy (%)", labelpad=1.5)
for s in ("top", "right"):
    AX_B.spines[s].set_visible(False)

# ============================ c: comb confusions ============================
AX_C1 = axmm(11.0, 8.0, 30, 30)
confusion(AX_C1, conf_meas, "measured, in-physics", C_BLUE,
          "98.92 $\\pm$ 0.30 %", C_BLUE, True)
AX_C2 = axmm(45.0, 8.0, 30, 30)
confusion(AX_C2, conf_cdig, "digital, clean-trained", C_GRAY,
          "99.50 $\\pm$ 0.20 %", C_INK, False)
cax1 = axmm(77.5, 8.0, 1.5, 30)
cb1 = plt.colorbar(plt.cm.ScalarMappable(cmap="Blues"), cax=cax1)
cb1.set_ticks([0, 1])
cb1.ax.tick_params(labelsize=FS_S, width=0.6, length=2.0)
cb1.set_label("recall", labelpad=-4.0, fontsize=FS_S)

# ============================ d: the twin inside weight training ============
AX_D = axmm(94.0, 46.0, 84, 90)
AX_D.set_xlim(0, 84)
AX_D.set_ylim(0, 90)
AX_D.axis("off")

# ---- step 1 (left portrait box): measure once -> fit f -> validate ---------
band(AX_D, 1.5, 1.5, 39.5, 87.0, C_STEPBG, r=1.8)
stepnum(AX_D, 4.5, 85.5, "1", C_INK)

AX_DM = axmm(102.25, 108.0, 26, 20)
lo_db = R["cw_p_lo_dbm"]
rf_db = R["cw_p_rf_dbm"]
amp_db = 20 * np.log10(np.maximum(R["cw_amp_uv_mean"], 1e-3))
AX_DM.imshow(amp_db, origin="lower", cmap="viridis", aspect="auto",
             extent=[rf_db[0], rf_db[-1], lo_db[0], lo_db[-1]],
             rasterized=True)
AX_DM.set_xticks([])
AX_DM.set_yticks([])
for s in AX_DM.spines.values():
    s.set_linewidth(0.6)

arr(AX_D, (21.25, 61.3), (21.25, 57.2), color=C_INK, lw=0.9)

box(AX_D, 14.75, 43.5, 13.0, 13.0, fc="white", ec=C_INK, lw=1.1, r=0.9,
    z=5)
uu = np.linspace(-1, 1, 50)
AX_D.plot(21.25 + uu * 4.6, 50.0 + uu * 3.4, color=C_GRAY, lw=0.5,
          ls=(0, (1.6, 1.4)), zorder=6)
AX_D.plot(21.25 + uu * 4.6, 50.0 + np.tanh(2.6 * uu) / np.tanh(2.6) * 3.4,
          color=C_INK, lw=1.2, zorder=6)

arr(AX_D, (21.25, 42.8), (21.25, 38.7), color=C_INK, lw=0.9)

AX_DV = axmm(104.25, 62.0, 22, 22)
xw0, nv0 = sc[0, 0], sc[0, 1]
AX_DV.axvspan(-1.3, 1.3, color=C_WIN, lw=0, zorder=1)
AX_DV.axvspan(-7, -1.3, color=C_ORANGE, alpha=0.45, lw=0, zorder=1)
AX_DV.axvspan(1.3, 7, color=C_ORANGE, alpha=0.45, lw=0, zorder=1)
AX_DV.plot([-7, 7], [-7, 7], color=C_INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
AX_DV.scatter(xw0, nv0, s=1.2, c=C_BLUE, alpha=0.6, lw=0, zorder=3,
              rasterized=True)
AX_DV.set_xlim(-7, 7)
AX_DV.set_ylim(-7, 7)
AX_DV.set_xticks([])
AX_DV.set_yticks([])
for s in AX_DV.spines.values():
    s.set_linewidth(0.6)
AX_D.text(21.25, 10.0, "predictable\ndistortion", fontsize=FS_S - 0.2,
          color="#B87A0F", ha="center", va="center", linespacing=1.35)

# ---- step 2 (right portrait box): train the weights through f --------------
band(AX_D, 43.5, 1.5, 39.0, 87.0, C_PINKBG, r=1.8)
stepnum(AX_D, 46.5, 85.5, "2", C_PINK)

box(AX_D, 47.5, 70.5, 10.0, 10.0, fc="white", ec=C_BLUE, lw=0.9, r=0.9,
    z=4)
AX_D.text(52.5, 78.3, "$\\mathbf{W}$", fontsize=FS, color=C_BLUE,
          ha="center", va="center", fontweight="bold", zorder=6)
comb(AX_D, 49.0, 72.3, 7.0, 3.2, [0.6, 1.0, 0.45, 0.8], C_BLUE)

comb(AX_D, 62.5, 75.5, 7.0, 3.5, [0.8, 0.5, 1.0, 0.6], C_VIOLET)
AX_D.text(60.4, 77.4, "$x$", fontsize=FS, color=C_VIOLET, ha="center",
          va="center")
arr(AX_D, (66.0, 75.2), (66.0, 64.35), color=C_VIOLET, lw=0.9)

box(AX_D, 58.5, 55.0, 9.0, 9.0, fc="white", ec=C_INK, lw=1.0, r=0.8, z=5)
AX_D.plot(63.0 + uu * 3.2, 59.5 + uu * 2.3, color=C_GRAY, lw=0.4,
          ls=(0, (1.6, 1.4)), zorder=6)
AX_D.plot(63.0 + uu * 3.2,
          59.5 + np.tanh(2.6 * uu) / np.tanh(2.6) * 2.3, color=C_INK,
          lw=1.1, zorder=6)
AX_D.text(69.9, 56.0, "$f$", fontsize=FS_S + 0.4, color=C_INK,
          ha="center", va="center", fontstyle="italic")
polyarr(AX_D, [(52.5, 70.2), (52.5, 59.5), (58.2, 59.5)],
        color=C_BLUE, lw=0.9)

AX_D.add_patch(Circle((63.0, 48.0), 2.2, fc="white", ec=C_TEAL, lw=0.9,
                      zorder=5))
AX_D.text(63.0, 47.95, "$\\Sigma$", fontsize=FS_S + 0.4, color=C_TEAL,
          ha="center", va="center", zorder=6)
arr(AX_D, (63.0, 54.7), (63.0, 50.55), color=C_PINK, lw=0.9)

box(AX_D, 56.5, 35.0, 13.0, 7.0, fc="white", ec=C_GRAY, lw=0.8, r=1.0)
AX_D.text(63.0, 38.5, "$|\\cdot|$, act.", fontsize=FS_S, color=C_INK,
          ha="center", va="center")
arr(AX_D, (63.0, 45.5), (63.0, 42.35), color=C_TEAL, lw=0.9)
for j in range(4):
    lx = 55.4 + j * 4.0
    box(AX_D, lx, 23.0, 3.2, 6.1, fc="white" if j else "#E9F0F9",
        ec=C_GRAY, lw=0.7, r=0.6)
    AX_D.text(lx + 1.6, 26.05, str(j + 1), fontsize=FS_S - 0.4,
              color=C_GRAY, ha="center", va="center")
arr(AX_D, (63.0, 34.7), (63.0, 29.45), color=C_INK, lw=0.8)
AX_D.text(75.5, 26.05, "layers", fontsize=FS_S - 0.6, color=C_GRAY,
          ha="center", va="center")
box(AX_D, 57.5, 9.5, 11.0, 7.0, fc="white", ec=C_GRAY, lw=0.8, r=1.0)
AX_D.text(63.0, 13.0, "loss", fontsize=FS_S, color=C_INK, ha="center",
          va="center")
arr(AX_D, (63.0, 22.7), (63.0, 16.85), color=C_INK, lw=0.8)

polyarr(AX_D, [(63.0, 9.2), (63.0, 4.8), (45.8, 4.8), (45.8, 75.5),
               (47.4, 75.5)],
        color=C_PINK, lw=0.9, ls=(0, (2.6, 1.8)))
AX_D.text(48.0, 40.0, "update $\\mathbf{W}$", fontsize=FS, color=C_PINK,
          ha="center", va="center", rotation=90, zorder=6)

# ============================ e: serial confusions ==========================
AX_F1 = axmm(100.5, 8.0, 30, 30)
confusion(AX_F1, conf_serial_clean, "clean-trained (measured)", C_INK,
          "5.7 $\\pm$ 0.9 %", C_SALMT, True)
AX_F2 = axmm(134.5, 8.0, 30, 30)
confusion(AX_F2, conf_serial_twin, "hardware-aware (measured)", C_BLUE,
          "98.5 $\\pm$ 0.5 %", C_BLUE, False)
cax2 = axmm(167.0, 8.0, 1.5, 30)
cb2 = plt.colorbar(plt.cm.ScalarMappable(cmap="Blues"), cax=cax2)
cb2.set_ticks([0, 1])
cb2.ax.tick_params(labelsize=FS_S, width=0.6, length=2.0)
cb2.set_label("recall", labelpad=-4.0, fontsize=FS_S)

# ============================ panel tags ====================================
tag(0.6, 140.8, "a")
tag(0.6, 84.0, "b")
tag(0.6, 41.5, "c")
tag(92.0, 140.8, "d")
tag(92.0, 41.5, "e")

# ---- readability: no gray TEXT anywhere (lines/rules may stay gray) --------
import matplotlib.text as mtext
for t in fig.findobj(mtext.Text):
    if t.get_color() == C_GRAY:
        t.set_color(C_INK)

# ---- uniform typography: every text at the body size FS --------------------
# (exceptions, per standard typesetting: panel letters stay at FS_TAG,
#  the one manual subscript stays reduced)
for t in fig.findobj(mtext.Text):
    if t.get_gid() in ("tag", "sub"):
        continue
    t.set_fontsize(FS)

# ============================ save ==========================================
fig.savefig(os.path.join(OUT, "fig4_v6_preview.pdf"), dpi=600)
fig.savefig(os.path.join(OUT, "fig4_v6_preview.png"), dpi=600)
fig.savefig(os.path.join(OUT, "fig4_v6_preview.svg"))

from PIL import Image
im = Image.open(os.path.join(OUT, "fig4_v6_preview.png"))
im.resize((im.width // 3, im.height // 3), Image.LANCZOS).save(
    os.path.join(OUT, "fig4_v6_preview_small.png"))
print("wrote fig4_v6_preview.* (round 6, aligned)")
