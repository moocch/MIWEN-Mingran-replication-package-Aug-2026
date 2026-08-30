# -*- coding: utf-8 -*-
"""
Regenerate the two Comment-4 figures of the reviewer-response letter as *_v2
(never overwriting the originals), with annotations consistent with the frozen
revised manuscript (main_PANS.tex):
  - benchtop baseband bandwidth 10 MHz (10 MS/s), no fictitious single-layer
    MNIST demo point (7,840 params @ 25 MHz does not exist in the revision)
  - "this work" scale anchored to the ten-layer GTSRB network (~5.3e5 weights)
  - reviewer-case (1e8 params) markers matched to the letter's Table rows:
    10 MHz -> 10 s, 100 MHz -> 1 s, 400 MHz -> 250 ms, 2 GHz -> 50 ms,
    10 GHz -> 10 ms
  - literal "\\," escapes of the v1 figures removed (proper mathtext)
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
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join(str(_data_dir(__file__)), "reproduced")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})

# --- restrained, CVD-safe colors (Okabe-Ito blue/vermillion + neutral grays) ---
C_BLUE = "#0072B2"      # this work
C_RED  = "#C4402A"      # reviewer case emphasis (vermillion, slightly deepened)
C_G1   = "#9A9A9A"      # context line (lighter)
C_G2   = "#4A4A4A"      # context line (darker)
BAND_RT = "#E7F1E7"     # real-time band (light green tint)
BAND_IA = "#F7F0DC"     # interactive band (light warm tint)
INK_MUT = "#6B6B6B"

# ============================================================================
# Figure 1: sequential download vs parallel frequency-multiplexed broadcast
# ============================================================================
fig, ax = plt.subplots(figsize=(11.4, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def bs_box(x, y, w=10.5, h=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc="#DCE9F7", ec=C_BLUE, lw=1.6))
    ax.text(x + w / 2, y + h / 2, "BS", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#1A3A5C")


# ---------------- panel (a): sequential download -----------------------------
ax.text(1, 96, "(a)  Sequential weight download — model underlying the estimate",
        fontsize=12.5, fontweight="bold", color="black")

bs_box(1.5, 74.5)
labels = [r"$w_1$", r"$w_2$", r"$w_3$", r"$\cdots$", r"$w_{N_\mathrm{p}}$"]
x0, bw, gap = 17.5, 13.4, 3.2
for i, lab in enumerate(labels):
    x = x0 + i * (bw + gap)
    ax.add_patch(Rectangle((x, 76), bw, 8, fc="#FBE4E0", ec=C_RED, lw=1.1))
    ax.text(x + bw / 2, 80, lab, ha="center", va="center",
            fontsize=11, color="#7A2417")
# time arrow
ax.annotate("", xy=(99, 71.5), xytext=(4, 71.5),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2))
ax.text(98.5, 68.3, "time", ha="right", va="top", fontsize=11,
        color="#888888", style="italic")
# duration annotation
ax.annotate("", xy=(x0 + 4 * (bw + gap) + bw, 64.5), xytext=(x0, 64.5),
            arrowprops=dict(arrowstyle="<|-|>", color=C_RED, lw=1.4))
ax.text((2 * x0 + 4 * (bw + gap) + bw) / 2, 60.2,
        r"$T_\mathrm{seq} = b\,N_\mathrm{p}/R_\mathrm{serial}$"
        r"  ($\sim$ tens of seconds for $10^8$ params at 100 Mbit s$^{-1}$,"
        r" $b = 16$–$32$ bit/param)",
        ha="center", va="top", fontsize=11.2, color=C_RED)

# ---------------- panel (b): parallel broadcast -------------------------------
ax.text(1, 48, "(b)  Parallel frequency-multiplexed broadcast — MIWEN (this work)",
        fontsize=12.5, fontweight="bold", color="black")

bs_box(1.5, 21)
# time-frequency tile: N_p subcarriers stacked in frequency, one period long
tx, ty, tw, th = 26, 15.5, 65, 22
ax.add_patch(Rectangle((tx, ty), tw, th, fc="#EAF4EA", ec="#2E7D32", lw=1.5))
n_stripes = 20
for i in range(n_stripes):
    yy = ty + (i + 0.5) * th / n_stripes
    ax.plot([tx + 1.2, tx + tw - 1.2], [yy, yy], color="#57A05B",
            lw=1.5, solid_capstyle="butt", alpha=0.85)
# frequency axis (bandwidth) on the left of the tile
ax.annotate("", xy=(23.8, ty + th), xytext=(23.8, ty),
            arrowprops=dict(arrowstyle="<|-|>", color="#2E7D32", lw=1.3))
ax.text(22.3, ty + th / 2, r"bandwidth $B$",
        ha="center", va="center", fontsize=11, color="#2E7D32", rotation=90)
ax.text(tx + tw / 2, ty + th + 2.2,
        r"$N_\mathrm{p}$ orthogonal subcarriers, one per weight "
        r"— all transmitted in parallel".replace(r"—", "—"),
        ha="center", va="bottom", fontsize=11, color="#2E7D32")
# one waveform period below the tile
ax.annotate("", xy=(tx + tw, ty - 2.8), xytext=(tx, ty - 2.8),
            arrowprops=dict(arrowstyle="<|-|>", color="#2E7D32", lw=1.4))
ax.text(tx + tw / 2, ty - 4.4,
        r"one waveform period   "
        r"$T_\mathrm{inf} = 1/\Delta f = N_\mathrm{p}/B$",
        ha="center", va="top", fontsize=12, color="#2E7D32")
# deployment scaling line (numbers = letter Table rows)
ax.text(50, 0.2,
        r"$10^8$ params:  benchtop $B$ = 10 MHz $\Rightarrow$ 10 s   $\vert$"
        r"   5G NR mmWave, 400 MHz $\Rightarrow$ 250 ms   $\vert$"
        r"   802.11ay / 6G D-band, 10 GHz $\Rightarrow$ 10 ms",
        ha="center", va="bottom", fontsize=11.2, color="#1A3A5C")

fig.savefig(os.path.join(OUT, "c4_parallel_schematic_v2.pdf"),
            bbox_inches="tight")
plt.close(fig)

# ============================================================================
# Figure 2: latency N_p/B and throughput B/N_p versus occupied bandwidth
# ============================================================================
B = np.logspace(1, np.log10(2e4), 400)            # MHz
series = [
    (5.3e5, C_BLUE, 1.9, r"this work: GTSRB network ($5.3{\times}10^5$ params)"),
    (1e7,   C_G1,   1.5, r"$10^7$ params"),
    (1e8,   C_RED,  2.2, r"$10^8$ params"),
    (1e9,   C_G2,   1.5, r"$10^9$ params (LLM scale)"),
]
tri_B  = np.array([10, 100, 400, 2000, 10000])     # MHz, = Table rows
tri_lat = ["10 s", "1 s", "250 ms", "50 ms", "10 ms"]
tri_thr = ["0.1", "1", "4", "20", "100"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 5.1))

std_ticks = [(10, "10 MHz\n(this work)"), (100, "100 MHz\n5G FR1"),
             (400, "400 MHz\n5G FR2"), (2000, "2 GHz\n802.11ad"),
             (10000, "10 GHz\n802.11ay / 6G")]

# ---------------- panel (a): latency ------------------------------------------
lat = lambda Np: Np / (B * 1e3)                    # ms
a1.axhspan(1e-3, 10, color=BAND_RT, zorder=0)
a1.axhspan(10, 100, color=BAND_IA, zorder=0)
a1.text(1.6e4, 4.2, "real-time  (< 10 ms)", ha="right", fontsize=9.5,
        color="#3E6B40")
a1.text(1.6e4, 42, "interactive  (< 100 ms)", ha="right", fontsize=9.5,
        color="#8A7434")
for Np, c, lw, lab in series:
    a1.loglog(B, lat(Np), color=c, lw=lw, label=lab, zorder=3)
a1.loglog(tri_B, 1e8 / (tri_B * 1e3), "^", ms=8, mfc=C_RED, mec="white",
          mew=0.8, zorder=5)
for bb, t in zip(tri_B, tri_lat):
    a1.annotate(t, (bb, 1e8 / (bb * 1e3)), textcoords="offset points",
                xytext=(7, 6), fontsize=9.5, color=C_RED, fontweight="bold")
a1.axvline(10, color=INK_MUT, lw=0.9, ls=(0, (4, 3)), zorder=1)
a1.set_xlim(10, 2e4); a1.set_ylim(1e-3, 1e5)
a1.set_xlabel(r"Occupied RF bandwidth  $B$  (MHz)")
a1.set_ylabel(r"Single-inference latency  $T_\mathrm{inf} = N_\mathrm{p}/B$  (ms)")
a1.set_title(r"(a)  Latency scales as $N_\mathrm{p}/B$", fontsize=12)
a1.legend(loc="lower left", fontsize=9, framealpha=0.95)
sec = a1.secondary_xaxis("top")
sec.set_xticks([t for t, _ in std_ticks])
sec.set_xticklabels([l for _, l in std_ticks], fontsize=8, color=INK_MUT)
sec.tick_params(length=2, color=INK_MUT)

# ---------------- panel (b): throughput ---------------------------------------
thr = lambda Np: B * 1e6 / Np                      # inferences per second
a2.axhspan(30, 1e6, color=BAND_RT, zorder=0)
a2.text(13, 45, "> 30 inf. s$^{-1}$  (video rate)", ha="left",
        fontsize=9.5, color="#3E6B40")
for Np, c, lw, lab in series:
    a2.loglog(B, thr(Np), color=c, lw=lw, label=lab, zorder=3)
a2.loglog(tri_B, tri_B * 1e6 / 1e8, "^", ms=8, mfc=C_RED, mec="white",
          mew=0.8, zorder=5)
for bb, t in zip(tri_B, tri_thr):
    a2.annotate(t, (bb, bb * 1e6 / 1e8), textcoords="offset points",
                xytext=(7, -11), fontsize=9.5, color=C_RED, fontweight="bold")
a2.axvline(10, color=INK_MUT, lw=0.9, ls=(0, (4, 3)), zorder=1)
a2.set_xlim(10, 2e4); a2.set_ylim(1e-2, 1e6)
a2.set_xlabel(r"Occupied RF bandwidth  $B$  (MHz)")
a2.set_ylabel(r"Inference throughput  $R_\mathrm{inf} = B/N_\mathrm{p}$  (inf. s$^{-1}$)")
a2.set_title(r"(b)  Throughput scales as $B/N_\mathrm{p}$", fontsize=12)
a2.legend(loc="upper left", fontsize=9, framealpha=0.95)
sec2 = a2.secondary_xaxis("top")
sec2.set_xticks([t for t, _ in std_ticks])
sec2.set_xticklabels([l for _, l in std_ticks], fontsize=8, color=INK_MUT)
sec2.tick_params(length=2, color=INK_MUT)

for a in (a1, a2):
    a.grid(True, which="major", color="#DDDDDD", lw=0.6, zorder=0)
    a.grid(True, which="minor", color="#EEEEEE", lw=0.4, zorder=0)
    a.set_axisbelow(True)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "c4_latency_bandwidth_v2.pdf"),
            bbox_inches="tight")
plt.close(fig)

print("written:")
print(os.path.join(OUT, "c4_parallel_schematic_v2.pdf"))
print(os.path.join(OUT, "c4_latency_bandwidth_v2.pdf"))
