"""
fig1b_motivation.py
===================
Figure 1, middle column (panels b, c): MOTIVATION.

(b) The physical diode-ring mixer is not an ideal multiplier.
    IF output power vs LO (weight) drive, simulated with the calibrated
    physics block of the PIML digital twin (paper-fitted parameters:
    G = -1.84 dB, P_sat = +1.73 dBm, beta_LO = 1.045, P_comp = +4.47 dBm,
    beta_RF = 2.13, P_n = -73.6 dBm, L_leak = -84.3 dB), compared with the
    ideal-multiplier assumption. Three regimes: thermal noise floor /
    multiplicative window / LO saturation.

(c) Consequence for analog computation precision: thermal noise and mixer
    nonlinearity create an optimal energy window (SNDR-composition
    simulation). The ideal-multiplier assumption predicts monotonically
    increasing precision; the physical mixer does not.

Output: fig1b_motivation.svg / .png (600 dpi)
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

# ----------------------------------------------------------------- style ----
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
    "svg.fonttype": "none",          # keep text editable in the SVG
})

C_IDEAL = "#C33D35"     # ideal-multiplier assumption
C_PHYS  = "#2458A6"     # calibrated mixer model (simulation)
C_DIST  = "#E28A2B"     # distortion
C_WIN   = "#1E8A78"     # good multiplicative window (green, must pop)
C_WIN_T = "#0E6B5B"
C_GRAY  = "#6B7280"
C_INK   = "#1F2937"

# ------------------------------------------------- calibrated physics block --
G_DB, PSAT_DBM, BETA_LO = -1.84, 1.73, 1.045
PCOMP_DBM, BETA_RF = 4.47, 2.13
PN_DBM, LEAK_DB = -73.6, -84.3


def dbm2w(p):
    return 1e-3 * 10 ** (np.asarray(p, float) / 10.0)


def w2dbm(p):
    return 10.0 * np.log10(np.maximum(p, 1e-30) / 1e-3)


def twin_pif_dbm(p_lo_dbm, p_rf_dbm):
    """Forward model of the calibrated physics block (per PIMLMixer.forward)."""
    p_lo_w, p_rf_w = dbm2w(p_lo_dbm), dbm2w(p_rf_dbm)
    alpha = -np.expm1(-(p_lo_w / dbm2w(PSAT_DBM)) ** BETA_LO)
    c = (p_rf_w / dbm2w(PCOMP_DBM)) ** BETA_RF
    p_rf_eff = p_rf_w * (1.0 + c) ** (-1.0 / BETA_RF)
    g = 10 ** (G_DB / 10)
    leak = 10 ** (LEAK_DB / 10)
    pif = g * alpha * p_rf_eff + leak * p_rf_w + dbm2w(PN_DBM)
    return w2dbm(pif)


def ideal_pif_dbm(p_lo_dbm, p_rf_dbm):
    """Ideal multiplier: no turn-on, no compression, no floor (slope 1/1)."""
    return G_DB + (p_lo_dbm - PSAT_DBM) + p_rf_dbm


# ------------------------------------------------------------------ figure --
fig = plt.figure(figsize=(56 * MM, 66 * MM))
gs = fig.add_gridspec(2, 1, left=0.175, right=0.94, top=0.945, bottom=0.105,
                      hspace=0.62)

# ============================== (b) transfer curve ===========================
ax = fig.add_subplot(gs[0])
P_RF0 = -35.0
plo = np.linspace(-70, 14, 400)
y_phys = twin_pif_dbm(plo, P_RF0)
y_ideal = ideal_pif_dbm(plo, P_RF0)

# regime boundaries
lo_noise = -35.0           # ideal signal crosses the noise floor
lo_sat = PSAT_DBM          # LO saturation knee
ax.axvspan(-70, lo_noise, color="#6C8EBF", alpha=0.15, lw=0)   # thermal noise
ax.axvspan(lo_noise, lo_sat, color=C_WIN, alpha=0.15, lw=0)
ax.axvspan(lo_sat, 14, color=C_DIST, alpha=0.16, lw=0)

ax.plot(plo, y_ideal, ls=(0, (4, 2.4)), lw=1.0, color=C_IDEAL, zorder=3)
ax.plot(plo, y_phys, lw=1.3, color=C_PHYS, zorder=4)

# operating point of the network experiments (P_LO = -3 dBm: soft knee)
ax.plot([-3], [twin_pif_dbm(-3, P_RF0)], "o", ms=3.4, mfc="white",
        mec=C_PHYS, mew=0.9, zorder=6)
ax.annotate("op. point", xy=(-3.2, twin_pif_dbm(-3, P_RF0) - 1.4),
            xytext=(-5.5, -57), fontsize=FS, color=C_INK, ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=C_GRAY,
                            shrinkA=0, shrinkB=2))

# labels
ax.text(-52.5, -78.0, "thermal noise", fontsize=FS, color="#3E618F",
        ha="center", va="center")
ax.text(-52.5, -67.8, "calibrated\nmixer model", fontsize=FS, color=C_PHYS,
        ha="center", va="center")
ax.text(-18.0, -32.5, "multiplicative\nwindow", fontsize=FS, color=C_WIN_T,
        ha="center", va="center")
ax.text(8.6, -47.5, "LO\nsatur.", fontsize=FS, color="#B4650A",
        ha="center", va="center")
ax.text(-16.5, -46.5, "ideal multiplier", fontsize=FS, color=C_IDEAL,
        ha="center", va="center", rotation=40)

ax.set_xlim(-70, 14)
ax.set_ylim(-80, -25)
ax.set_xticks([-60, -40, -20, 0])
ax.set_yticks([-75, -55, -35])
ax.set_xlabel(r"Weight $P_{\mathrm{LO}}$ (dBm)", labelpad=1.6)
ax.set_ylabel(r"$P_{\mathrm{IF}}$ (dBm)", labelpad=1.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ================== (c) computing-accuracy window (WISE convention) =========
# accuracy = -log2(RMSE/2) with RMSE of the normalized inner product
# (WISE, Gao et al.); RMSE_noise = 1/sqrt(27*SNR_tone) per the AWGN analysis,
# with the calibrated noise floor P_n split over N = 4096 tones and the
# calibrated compression point shifted by the multi-tone PAPR (~12 dB).
ax2 = fig.add_subplot(gs[1])
B = 25e6                        # occupied bandwidth (Hz): E/MAC = P / B
E = np.logspace(-16, -11, 400)  # J per MAC
P = E * B

N_TONES = 4096
N0 = dbm2w(PN_DBM)              # calibrated noise floor (-73.6 dBm)
P3 = dbm2w(PCOMP_DBM - 12.0)    # PAPR-shifted compression (multi-tone frame)

snr_tone = P / (N_TONES * N0)
rmse_n = 1.0 / np.sqrt(27.0 * snr_tone)
rmse_d = P / P3
rmse = np.sqrt(rmse_n ** 2 + rmse_d ** 2)
acc = -np.log2(np.clip(rmse, 1e-9, 2.0) / 2.0)
acc_ideal = -np.log2(np.clip(rmse_n, 1e-9, 2.0) / 2.0)

# optimal window: within 1 bit of the peak
mask = acc >= acc.max() - 1.0
ax2.axvspan(E[mask][0], E[mask][-1], color=C_WIN, alpha=0.15, lw=0)

ax2.plot(E, acc_ideal, ls=(0, (4, 2.4)), lw=1.0, color=C_IDEAL, zorder=3)
ax2.plot(E, acc, lw=1.3, color=C_PHYS, zorder=4)
ax2.set_xscale("log")

ax2.text(1.35e-13, 7.1, "optimal\nwindow", fontsize=FS, color=C_WIN_T,
         ha="center", va="center")
ax2.text(2.3e-12, 9.1, "ideal: noise only", fontsize=FS, color=C_IDEAL,
         ha="center", va="center", rotation=26)
ax2.text(7.0e-16, 4.6, "noise-\nlimited", fontsize=FS, color="#4B5563",
         ha="left", va="center")
ax2.text(9.0e-12, 4.3, "distortion-\nlimited", fontsize=FS, color="#B4650A",
         ha="right", va="center")

ax2.set_xlim(1e-16, 1e-11)
ax2.set_ylim(0, 10.3)
ax2.set_yticks([0, 5, 10])
ax2.set_xticks([1e-16, 1e-14, 1e-12])
ax2.set_xlabel("Client energy per MAC (J)", labelpad=1.6)
ax2.set_ylabel("Accuracy (bit)", labelpad=1.5)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# panel letters: Nature style — lowercase bold, no parentheses
fig.text(0.012, ax.get_position().y1 + 0.028, "b", fontsize=FS_TAG,
         fontweight="bold", va="top", ha="left", color=C_INK)
fig.text(0.012, ax2.get_position().y1 + 0.028, "c", fontsize=FS_TAG,
         fontweight="bold", va="top", ha="left", color=C_INK)

fig.savefig(os.path.join(OUT, "fig1b_motivation.svg"))
fig.savefig(os.path.join(OUT, "fig1b_motivation.pdf"))
fig.savefig(os.path.join(OUT, "fig1b_motivation.png"), dpi=600)
print("wrote fig1b_motivation.svg/.pdf/.png")
