"""
fig5_analog_v3.py
=================
Figure 5 — the fully analog client.  This is fig5_analog_v2 with ONE
change: the whole type ladder is multiplied by SCALE, so the labels read
larger against the 10 pt body text without touching the composition.

  * layout, panels, strings, numbers, gates, palette, strokes and the
    180 x 92 mm canvas are v2's, unchanged;
  * v2's ladder (7.5 / 6.3 pt with a 5.2-9.0 pt spread) is preserved as
    written -- every fontsize is wrapped in S(), which multiplies it by
    SCALE, so the relative hierarchy is exactly v2's, just bigger.

History of this request (author, 2026-08-28):
  1. "make every font the body size" -> a uniform 10 pt inside the
     180 x 92 mm canvas.  Reverted the same day: 10 pt does not fit a
     layout budgeted for 5.2-7.5 pt.
  2. a uniform 10 pt three-band re-cut at 180 x 151 mm (kept for
     reference as rejected_2026-08-28_uniform10pt_threeband.py).
     Rejected: "too big, and it changed the layout".
  3. this file -- v2's layout back, type scaled by SCALE.

  a  the unrolled fully analog cascade: DAC -> [mixer -> LPF -> passive
     block] x2 -> mixer -> LPF -> ADC; weight combs broadcast on
     disjoint LO bands; pass drives -10 / -34 / -58 dBm.
  b  the passive activation block, enlarged: 1:n matching network
     (+18.5 dB measured), zero-bias anti-phase pair (squarer; the CMOS
     pseudo-balun ED of Wang et al., SSC-L 2018), baseband LPF.
  c  link budget: per-tone SNR of each pass (MNIST + GTSRB plans);
     pass 4 out of budget -> the client supports L = 3.
  d  one checkpoint per task, executed twice: digital vs the analog
     chain at L = 3.

Data: data/*.json -- verbatim copies of the archive results
(V2/fully analog/files (1).zip -> miwen_fully_analog_archive.zip ->
code/results/).  Contract unchanged from v2: every number in the data
panels (c/d) and every settings constant shown in b is loaded from
these files and gate-checked (zero tolerance).
Output: fig5_analog_v3_preview.pdf/.png/.svg/_small.png  (180 x 92 mm)
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                Arc, Polygon)

OUT = str(_data_dir(__file__))
DD = OUT
MM = 1 / 25.4
FIG_W, FIG_H = 180.0, 92.0

# ---------------------------------------------------------------------------
# The single typography knob.  v2's ladder is kept verbatim below; SCALE
# multiplies all of it, so the hierarchy is untouched and only the overall
# size moves.  SCALE = 1.0 reproduces fig5_analog_v2 exactly.
# For reference on the page: the manuscript body is 10 pt and the figure is
# included at width=\textwidth with a 0.9995 scale, so S(x) lands on the
# page at essentially x pt.
# ---------------------------------------------------------------------------
# 1.20 chosen because it is the smallest bump that lifts the SMALLEST label
# on the figure (v2's 5.2 pt port tags) above the 6 pt floor journals ask
# for -- 5.2 -> 6.24 pt -- while the largest body text on the artwork stays
# at 9.0 pt, under the manuscript's 10 pt. Verified collision-free over
# SCALE = 1.00-1.30, so this is a safe one-number dial.
SCALE = float(os.environ.get("FIG5_SCALE", 1.20))

FS = 7.5
FS_S = 6.3
FS_TAG = 9.0


def S(pt):
    """v2 type size -> this figure's type size."""
    return pt * SCALE


rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": S(FS),
    "axes.labelsize": S(FS),
    "axes.titlesize": S(FS),
    "axes.linewidth": 0.6,
    "xtick.labelsize": S(FS),
    "ytick.labelsize": S(FS),
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "mathtext.fontset": "stixsans",
    "svg.fonttype": "none",
})

# house palette (fig1-4); assignment per author, 2026-08-28:
#   band = fig4d's light-blue step tint; activation block + panel-b frame =
#   the amber/cream of the earlier rounds; c/d data = fig4b's salmon + blue
C_PHYSBG = "#E9F0F9"      # analog-domain band (fig4d band-2 / fig2a
                          # physics-block light-blue tint)
C_AMBT = "#B87A0F"        # activation-block + panel-b frame border/text
C_SALM = "#E58C7D"        # fig4b's "digital, same weights" salmon
C_SALMT = "#C0503C"       # ideal-assumption failure / darker salmon text
C_BLUE = "#2458A6"        # fig4b's in-physics blue = the chain's data
C_BLUE2 = "#7FA3D1"       # light house blue (panel-c second series)
C_LBLUE = "#C9CDD3"
C_MGRAY = "#8D97A5"
C_VIOLET = "#6D28D9"      # data stream
C_TEAL = "#1E8A78"
C_GRAY = "#6B7280"
C_SLATE = "#5B6470"       # small subtext (kept, unlike C_GRAY)
C_INK = "#1F2937"
C_STEPBG = "#F4F5F7"
C_BLOCKFC = "#FDF1E3"     # activation-block / panel-b frame fill (cream)

# ------------------------------------------------------------------- data ---
LB = json.load(open(os.path.join(DD, "link_budget.json")))
RS = json.load(open(os.path.join(DD, "results_summary.json")))
G = np.load(os.path.join(DD, "input_glyphs.npz"))
# data/twin_2x2.json stays archived in data/ but is no longer displayed
# (the 2x2 panel was dropped on 2026-08-27)

# ---- gates: every displayed number must equal the archive value ------------
def _g(cond, msg):
    if not cond:
        raise AssertionError("GATE: " + msg)

st = LB["settings"]
for k, v in dict(cl_pass_dB=7.47, boost_dB=18.5, k_ED_per_V=208.7,
                 R_v_ohm=3500.0, step_down_loss_dB=1.0,
                 first_pass_dBm=-10.0).items():
    _g(abs(st[k] - v) < 1e-9, f"settings.{k} = {st[k]} != {v}")

# constants displayed in panels b/f that have no JSON anchor of their own;
# pinned here against the archive code (comb_analog_sim.py) and tied to the
# gated settings where possible
CL_MIXER_DB = 6.65     # ZEM-4300+ datasheet CL at rated LO drive
IL_FILT_DB = 0.82      # LFCN-490+ datasheet IL at 0.49 GHz
P_N_DBM = -73.6        # per-bin noise floor, archive P_N_DBM (main text Fig. 2)
_g(abs(CL_MIXER_DB + IL_FILT_DB - st["cl_pass_dB"]) < 1e-9,
   "CL split 6.65 + 0.82 != cl_pass_dB")

MN_ROWS = LB["MNIST"]["rows"]
GT_ROWS = LB["GTSRB"]["rows"]
_g([round(r["drive_dBm"], 2) for r in MN_ROWS] ==
   [-10.0, -33.94, -58.02, -84.46], "MNIST drives")
_g([round(r["tone_snr_dB"], 2) for r in MN_ROWS] ==
   [48.97, 33.97, 11.83, -14.61], "MNIST SNRs")
_g([round(r["drive_dBm"], 2) for r in GT_ROWS] ==
   [-10.0, -34.27, -58.48, -85.16], "GTSRB drives")
_g([round(r["tone_snr_dB"], 2) for r in GT_ROWS] ==
   [43.04, 32.57, 8.36, -18.31], "GTSRB SNRs")
_g(round(LB["MNIST"]["block_efficiency"][0], 3) == 2.255, "block eff")

def _acc(tag):
    return RS[tag]["accuracy_pct"], RS[tag]["std_pct"]

for tag, a, s in [("MZT_L3d", 94.3, 0.0), ("MZH_L3", 93.9, 0.1),
                  ("ZT_L3d", 88.83, 0.0), ("ZH_L3", 87.53, 0.14)]:
    aa, ss = _acc(tag)
    _g(abs(aa - a) < 1e-9 and abs(ss - s) < 1e-9, f"{tag} {aa}+-{ss}")

# ------------------------------------------------------------------ figure --
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))


def axmm(x, y, w, h, **kw):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)


def tag(x_mm, y_mm, s):
    # panel letters stay OUTSIDE the SCALE knob: 9.0 pt flat, matching
    # the tags of figs. 2-4 (author request 2026-08-29)
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


def wire(ax, pts, color=C_INK, lw=0.9, z=4, ls="-"):
    pts = np.asarray(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls,
            solid_capstyle="round", zorder=z)


def mixer(ax, x, y, rr):
    ax.add_patch(Circle((x, y), rr, fc="white", ec=C_INK, lw=1.1, zorder=6))
    d = rr / np.sqrt(2)
    ax.plot([x - d, x + d], [y - d, y + d], color=C_INK, lw=0.9, zorder=7)
    ax.plot([x - d, x + d], [y + d, y - d], color=C_INK, lw=0.9, zorder=7)


def antenna(ax, x, y, h=3.2, color=C_BLUE):
    ax.plot([x, x], [y, y + h * 0.55], color=color, lw=0.9, zorder=6)
    ax.plot([x - h * 0.38, x, x + h * 0.38],
            [y + h, y + h * 0.55, y + h], color=color, lw=0.9, zorder=6)


def ground(ax, x, y, s=1.0, color=C_INK):
    for i, w in enumerate((2.4, 1.6, 0.8)):
        ax.plot([x - w * s / 2, x + w * s / 2],
                [y - i * 0.7 * s, y - i * 0.7 * s], color=color, lw=0.8,
                zorder=5)


def diode(ax, x0, x1, y, color=C_INK, s=1.15):
    xm = (x0 + x1) / 2
    ax.plot([x0, xm - s], [y, y], color=color, lw=0.9, zorder=5)
    ax.plot([xm + s, x1], [y, y], color=color, lw=0.9, zorder=5)
    ax.add_patch(Polygon([(xm - s, y + s), (xm - s, y - s), (xm + s, y)],
                         closed=True, fc="white", ec=color, lw=0.9, zorder=6))
    ax.plot([xm + s, xm + s], [y - s, y + s], color=color, lw=0.9, zorder=6)


def vcoil(ax, x, y0, y1, n=3, r=None, side=1, color=C_INK):
    """Vertical winding: n semicircular arcs bulging toward side*x."""
    r = abs(y1 - y0) / (2 * n) if r is None else r
    ys = np.linspace(min(y0, y1), max(y0, y1), n + 1)
    for a, b in zip(ys[:-1], ys[1:]):
        ax.add_patch(Arc(((x), (a + b) / 2), 2 * r, b - a,
                         theta1=-90 if side > 0 else 90,
                         theta2=90 if side > 0 else 270,
                         ec=color, lw=0.9, zorder=5))


def hcoil(ax, x0, x1, y, n=3, color=C_INK):
    xs = np.linspace(x0, x1, n + 1)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(Arc(((a + b) / 2, y), b - a, b - a, theta1=0,
                         theta2=180, ec=color, lw=0.9, zorder=5))


def cap(ax, x, y, w=2.2, gap=0.7, color=C_INK, horiz=False):
    if horiz:
        ax.plot([x - gap / 2, x - gap / 2], [y - w / 2, y + w / 2],
                color=color, lw=1.0, zorder=5)
        ax.plot([x + gap / 2, x + gap / 2], [y - w / 2, y + w / 2],
                color=color, lw=1.0, zorder=5)
    else:
        ax.plot([x - w / 2, x + w / 2], [y + gap / 2, y + gap / 2],
                color=color, lw=1.0, zorder=5)
        ax.plot([x - w / 2, x + w / 2], [y - gap / 2, y - gap / 2],
                color=color, lw=1.0, zorder=5)


# ============================ a: the cascade ================================
AX_A = axmm(2.0, 54.0, 176, 34)
AX_A.set_xlim(-1.0, 175.0)      # shifted so the composition sits centred
AX_A.set_ylim(0, 34)
AX_A.axis("off")

YC = 17.0          # chain centre line
digit = G["digit_u8"]
photo = G["photo_u8"]

# the two inputs, stacked and aligned on one x-centre
AX_A.imshow(digit, extent=[3.5, 10.5, 19.5, 26.5], origin="upper",
            cmap="gray_r", interpolation="nearest", zorder=4)
AX_A.add_patch(plt.Rectangle((3.5, 19.5), 7.0, 7.0, fc="none", ec=C_INK,
                             lw=0.8, zorder=5))
AX_A.imshow(photo, extent=[3.5, 10.5, 10.5, 17.5], origin="upper",
            interpolation="nearest", zorder=5)
AX_A.add_patch(plt.Rectangle((3.5, 10.5), 7.0, 7.0, fc="none", ec=C_INK,
                             lw=0.8, zorder=6))
AX_A.text(7.0, 7.6, "input $x$", fontsize=S(FS_S), color=C_VIOLET,
          ha="center", va="center")

arr(AX_A, (11.3, YC), (14.2, YC), color=C_VIOLET, lw=0.9, ms=5)
box(AX_A, 14.5, YC - 4.0, 10.0, 8.0, fc="white", ec=C_INK, lw=0.9, r=1.2)
AX_A.text(19.5, YC, "DAC", fontsize=S(FS_S + 0.3), color=C_INK, ha="center",
          va="center")

# analog-domain band + its caption removed 2026-08-28 (moved to the LaTeX
# caption)

XM = [37.0, 80.5, 124.0]          # mixer centres
FLO = ["0.30", "0.35", "0.40"]

arr(AX_A, (24.7, YC), (33.7, YC), color=C_INK, lw=0.9, ms=5)
for i, xm in enumerate(XM):
    mixer(AX_A, xm, YC, 3.0)
    # broadcast weight comb on the LO port
    antenna(AX_A, xm, 26.8, h=3.4)
    wire(AX_A, [(xm, 26.8), (xm, 21.4)], color=C_BLUE, lw=0.9)
    arr(AX_A, (xm, 21.9), (xm, 20.3), color=C_BLUE, lw=0.9, ms=5)
    AX_A.text(xm + 1.6, 28.4, f"$w^{{({i + 1})}}$, {FLO[i]} GHz",
              fontsize=S(FS_S - 0.3), color=C_BLUE, ha="left", va="center")
    # selection filter
    wire(AX_A, [(xm + 3.0, YC), (xm + 6.5, YC)])
    box(AX_A, xm + 6.5, YC - 3.5, 9.0, 7.0, fc="white", ec=C_INK, lw=0.9,
        r=1.0)
    AX_A.text(xm + 11.0, YC, "LPF", fontsize=S(FS_S + 0.3), color=C_INK,
              ha="center", va="center")

# port convention (archive fig:chain): weight comb -> LO, activation comb ->
# DC-coupled IF, product band selected at the RF port by the LPF
# (part numbers ZEM-4300+/LFCN-490+ live in the caption, not on the figure)
AX_A.text(31.6, 19.4, "IF", fontsize=S(FS_S - 1.1), color=C_SLATE,
          ha="center", va="center")
AX_A.text(35.0, 22.1, "LO", fontsize=S(FS_S - 1.1), color=C_SLATE,
          ha="center", va="center")
AX_A.text(41.7, 14.4, "RF", fontsize=S(FS_S - 1.1), color=C_SLATE,
          ha="center", va="center")

# the two passive activation blocks
for xb in (55.5, 99.0):
    arr(AX_A, (xb - 2.8, YC), (xb - 0.3, YC), color=C_INK, lw=0.9, ms=5)
    box(AX_A, xb, YC - 6.0, 19.0, 12.0, fc="white", ec=C_INK, lw=1.0,
        r=1.4)
    for dy, txt in ((2.6 * SCALE, "analog"), (0.0, "activation"),
                    (-2.6 * SCALE, "block (b)")):
        AX_A.text(xb + 9.5, YC + dy, txt, fontsize=S(FS_S - 0.2),
                  color=C_INK, ha="center", va="center", fontweight="bold",
                  zorder=6)
    arr(AX_A, (xb + 19.3, YC), (xb + 21.9, YC), color=C_INK, lw=0.9, ms=5)

# pass-drive line removed 2026-08-28 (values stay gate-checked above and
# move to the LaTeX caption)

# readout (the band-to-ADC gap mirrors the DAC-to-band gap)
arr(AX_A, (139.5, YC), (147.4, YC), color=C_INK, lw=0.9, ms=5)
box(AX_A, 147.8, YC - 4.0, 10.0, 8.0, fc="white", ec=C_INK, lw=0.9,
    r=1.2)
AX_A.text(152.8, YC, "ADC", fontsize=S(FS_S + 0.3), color=C_INK,
          ha="center", va="center")
arr(AX_A, (157.8, YC), (160.6, YC), color=C_TEAL, lw=0.9, ms=5)
AX_A.text(164.0, YC + 0.1, "class", fontsize=S(FS_S + 0.3), color=C_TEAL,
          ha="center", va="center")

# ============================ b: the passive block ==========================
AX_B = axmm(2.0, 3.0, 64, 44)
AX_B.set_xlim(0, 64)
AX_B.set_ylim(0, 44)
AX_B.axis("off")

# background frame patch removed 2026-08-28

YT, YB = 31.8, 15.2          # top / bottom branch
YM = (YT + YB) / 2           # 23.5

# input, into the top of the primary
arr(AX_B, (1.2, YT), (5.2, YT), color=C_INK, lw=0.9, ms=5)
AX_B.text(1.0, YT + 2.4, "from LPF (a)", fontsize=S(FS_S - 1.0), color=C_INK,
          ha="left", va="center")
wire(AX_B, [(5.2, YT), (7.5, YT)])
vcoil(AX_B, 7.5, YB, YT, n=3, side=-1, r=2.0)
wire(AX_B, [(7.5, YB), (7.5, 13.6)])
ground(AX_B, 7.5, 13.6, s=0.85)
# core
AX_B.plot([9.3, 9.3], [YB - 0.8, YT + 0.8], color=C_INK, lw=1.0, zorder=5)
AX_B.plot([10.3, 10.3], [YB - 0.8, YT + 0.8], color=C_INK, lw=1.0,
          zorder=5)
# secondary, centre-tapped
vcoil(AX_B, 12.1, YB, YT, n=4, side=1)
wire(AX_B, [(12.1, YT), (20.0, YT)])
wire(AX_B, [(12.1, YB), (20.0, YB)])
wire(AX_B, [(12.1, YM), (16.5, YM), (16.5, YM - 2.6)])
ground(AX_B, 16.5, YM - 2.6, s=0.85)
AX_B.text(18.9, YM + 1.5, "c.t.", fontsize=S(FS_S - 1.0), color=C_SLATE,
          ha="center", va="center")
AX_B.text(10.6, 12.0, "1 : $n$", fontsize=S(FS_S - 0.6), color=C_INK,
          ha="center", va="center")
# anti-phase pair (drawn as junction symbols; device identity below)
diode(AX_B, 20.0, 28.0, YT)
diode(AX_B, 20.0, 28.0, YB)
AX_B.text(24.0, YT + 2.4, "$D_1$", fontsize=S(FS_S - 0.6), color=C_INK,
          ha="center", va="center")
AX_B.text(24.0, YB - 2.5, "$D_2$", fontsize=S(FS_S - 0.6), color=C_INK,
          ha="center", va="center")
wire(AX_B, [(28.0, YT), (31.0, YT), (31.0, YB), (28.0, YB)])
AX_B.plot(31.0, YM, ".", ms=3.2, color=C_INK, zorder=6)
# baseband low-pass
wire(AX_B, [(31.0, YM), (35.0, YM)])
wire(AX_B, [(35.0, YM), (35.0, YM - 1.9)])
cap(AX_B, 35.0, YM - 2.8)
wire(AX_B, [(35.0, YM - 3.7), (35.0, YM - 4.9)])
ground(AX_B, 35.0, YM - 4.9, s=0.85)
wire(AX_B, [(35.0, YM), (37.0, YM)])
hcoil(AX_B, 37.0, 44.0, YM, n=3)
wire(AX_B, [(44.0, YM), (46.0, YM)])
wire(AX_B, [(46.0, YM), (46.0, YM - 1.9)])
cap(AX_B, 46.0, YM - 2.8)
wire(AX_B, [(46.0, YM - 3.7), (46.0, YM - 4.9)])
ground(AX_B, 46.0, YM - 4.9, s=0.85)
# video step-down network into the 50-ohm IF port
wire(AX_B, [(46.0, YM), (48.6, YM)])
box(AX_B, 48.6, YM - 2.2, 6.0, 4.4, fc="white", ec=C_INK, lw=0.8, r=0.6)
AX_B.text(51.6, YM, "$n$:1", fontsize=S(FS_S - 1.0), color=C_INK,
          ha="center", va="center")
arr(AX_B, (54.6, YM), (58.6, YM), color=C_INK, lw=0.9, ms=5)
# output-port labels and the step-down loss figure removed 2026-08-28
# (moved to the LaTeX caption); arrow black and 4.0 mm to mirror the
# input arrow (author request 2026-08-29)
AX_B.text(39.5, YM + 3.2, "LPF", fontsize=S(FS_S - 0.8),
          color=C_INK, ha="center", va="center")

# annotations (subtitle, self-mixer line and equation removed 2026-08-28;
# panel title removed 2026-08-29; wording lives in the LaTeX caption)

# ============================ c: link budget ================================
AX_C = axmm(78.0, 12.0, 44, 32)
p = np.arange(1, 5)
mn_snr = [r["tone_snr_dB"] for r in MN_ROWS]
gt_snr = [r["tone_snr_dB"] for r in GT_ROWS]

AX_C.axhspan(-27, 0, color=C_SALM, alpha=0.14, lw=0, zorder=1)
AX_C.axhline(0, color=C_SLATE, lw=0.7, ls=(0, (4, 2)), zorder=2)
AX_C.text(1.02, 2.0, "SNR = 0 dB", fontsize=S(FS_S - 0.8), color=C_INK,
          ha="left", va="bottom")

# panel c keeps its own pair (deep + light house blue, 2026-08-28) so the
# salmon/blue of panel d stays reserved for digital-vs-analog
for snr, col, mk in [(mn_snr, C_BLUE2, "o"), (gt_snr, C_BLUE, "s")]:
    AX_C.plot(p[:3], snr[:3], color=col, lw=1.3, marker=mk, ms=3.6,
              zorder=4)
    AX_C.plot(p[2:], snr[2:], color=col, lw=1.0, ls=(0, (3, 2)),
              zorder=3, alpha=0.75)
    AX_C.plot(p[3], snr[3], marker=mk, ms=3.6, mfc="white", mec=col,
              mew=1.0, lw=0, zorder=4)
AX_C.text(1.14, 51.5, "MNIST", fontsize=S(FS_S - 0.4), color="#5E87BC",
          ha="left", va="center", fontweight="bold")
AX_C.text(1.02, 33.0, "GTSRB", fontsize=S(FS_S - 0.4), color=C_BLUE,
          ha="left", va="center", fontweight="bold")

AX_C.text(3.30, 21.0, "$L = 3$", fontsize=S(FS_S + 0.3), color=C_BLUE,
          ha="left", va="center", fontweight="bold")
AX_C.plot([3.27, 3.06], [18.3, 12.6], color=C_SLATE, lw=0.5, zorder=3)
AX_C.text(0.84, -22.6, "pass 4: out of budget", fontsize=S(FS_S - 0.4),
          color=C_SALMT, ha="left", va="center")

AX_C.set_xlim(0.72, 4.28)
AX_C.set_ylim(-27, 56)
AX_C.set_xticks(p)
AX_C.set_xticklabels([f"pass {i}\n$-${abs(r['drive_dBm']):.0f} dBm"
                      for i, r in zip(p, MN_ROWS)], fontsize=S(FS_S - 0.4),
                     linespacing=1.15)
AX_C.tick_params(axis="x", length=0, pad=2.0)
AX_C.set_yticks([-20, 0, 20, 40])
AX_C.set_ylabel("per-tone SNR (dB)", labelpad=1.5)
for s in ("top", "right"):
    AX_C.spines[s].set_visible(False)

# ============================ d: two executions =============================
# L = 4 group removed 2026-08-28 (author decision); its numbers stay in
# data/results_summary.json for the text.
AX_D = axmm(134.0, 12.0, 44, 32)
pairs = [("MZT_L3d", "MZH_L3"), ("ZT_L3d", "ZH_L3")]
xs = np.arange(2)
for j, (td, ta) in enumerate(pairs):
    ad, sd = _acc(td)
    aa, sa = _acc(ta)
    AX_D.bar(j - 0.17, ad, 0.32, color=C_SALM, zorder=3)
    AX_D.bar(j + 0.17, aa, 0.32, color=C_BLUE, zorder=3)
    AX_D.errorbar(j + 0.17, aa, yerr=sa, fmt="none", ecolor=C_INK,
                  elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=4)
    AX_D.text(j - 0.17, ad + 1.5, f"{ad:.1f}", fontsize=S(FS_S - 0.8),
              ha="center", va="bottom", color=C_INK)
    AX_D.text(j + 0.17, aa + sa + 1.5, f"{aa:.1f}", fontsize=S(FS_S - 0.8),
              ha="center", va="bottom", color=C_INK, fontweight="bold")

AX_D.set_xlim(-0.55, 1.55)
AX_D.set_ylim(0, 110)
AX_D.set_xticks(xs)
AX_D.set_xticklabels(["MNIST\n$L=3$", "GTSRB\n$L=3$"],
                     fontsize=S(FS_S - 0.4), linespacing=1.15)
AX_D.tick_params(axis="x", length=0, pad=2.0)
AX_D.set_yticks([0, 50, 100])
AX_D.set_ylabel("accuracy (%)", labelpad=1.5)
for s in ("top", "right"):
    AX_D.spines[s].set_visible(False)
# direct labels inside the first bar pair (house style, no legend box)
AX_D.text(-0.17, 47.0, "digital, same weights", fontsize=S(FS_S - 0.8),
          color="white", ha="center", va="center", rotation=90, zorder=5)
AX_D.text(0.17, 47.0, "fully analog chain", fontsize=S(FS_S - 0.8),
          color="white", ha="center", va="center", rotation=90, zorder=5)

# ============================ panel tags ====================================
tag(0.6, 91.4, "a")
tag(0.6, 49.8, "b")
tag(70.0, 49.8, "c")
tag(127.5, 49.8, "d")

# ---- readability: no gray TEXT anywhere (lines/rules may stay gray) --------
import matplotlib.text as mtext
for t in fig.findobj(mtext.Text):
    if t.get_color() == C_GRAY:
        t.set_color(C_INK)

# ============================ save ==========================================
fig.savefig(os.path.join(OUT, "fig5_analog_v3_preview.pdf"), dpi=600)
fig.savefig(os.path.join(OUT, "fig5_analog_v3_preview.png"), dpi=600)
fig.savefig(os.path.join(OUT, "fig5_analog_v3_preview.svg"))

from PIL import Image
im = Image.open(os.path.join(OUT, "fig5_analog_v3_preview.png"))
im.resize((im.width // 3, im.height // 3), Image.LANCZOS).save(
    os.path.join(OUT, "fig5_analog_v3_preview_small.png"))
print(f"wrote fig5_analog_v3_preview.* — v2 layout, type x{SCALE:g} (main {S(FS):.2f} pt), 180 x 92 mm")
