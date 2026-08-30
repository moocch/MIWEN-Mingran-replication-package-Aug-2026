"""
fig3_v12_payoff.py
==================
Figure 3 (v12) — TRIAL layout, print-width variant of v11: the same
a-f + two-row energy panel g, repacked into the standard 180-mm figure*
canvas so nothing is scaled down at include time (fonts stay 7.5 pt on
the page). Panels a-f sit on an aligned grid of 31.5-mm columns (panel a
re-laid natively for the narrow column: titles unbolded, one font step
down); panel g
(33.5 mm wide) is the V2/energy fig5_energy_package energy-per-real-MAC
figure (same 2026-08-10 campaign, 25-dB operating points, raw decode,
criterion RMSE < 1/16). Energy is computed directly from the raw
campaign files already in fig3/data/ — verified identical to the
package's raw files — with the package's assertion guards retained,
following Gao et al., Sci. Adv. 12, eadz0817 (2026), Eqs. 1-3.
fig3_v10 (published) and fig3_v11 (237.5-mm trial) are untouched.

  a    Correction scheme (flow).
  b    Mechanism: the twin's compression law reshapes the known w(t)
       (rebuilt N = 65,536 transmitted LO symbol, 10-dB PAPR clip).
  c    Residual distributions of the three model tiers (N = 1 grid,
       as in v8 — ties the correction back to the Fig. 2 model ladder).
  d,e  THE PAYOFF (large): measured N = 65,536 inner products
       before/after correction at 15 dB (d) and 25 dB (e):
       0.064->0.036 (1.8x) and 0.056->0.017 (3.3x); RMSE and ENOB
       annotated as mean +/- 1 s.d. over the three acquisitions.
  f    Inner-product RMSE vs SNR and the measured guard-bin receiver
       noise floor 1/sqrt(27*SNR); capped +/-1 s.d. error bars.
       N = 65,536 is stamped on d, e and f (panel titles).
  g    NEW (two rows tall, right of c and f): client-side energy per
       real MAC vs N — e_ip = e1 + e2 + e3 model curve, the flat radio
       floor e1, the 1/N read-out fees e2 and e3, the H100 reference
       (70 fJ/MAC), the two measured triangles (1.47 fJ/MAC at N = 4096,
       48x below H100; 0.59 fJ/MAC at N = 65,536, 119x below), and the
       coincident Landauer = thermodynamic floor at 2-bit ENOB
       (11.5 zJ/MAC) in a broken-axis strip.

Data (fig3/data/):
  gr_ip_scatter_N65536_20260810.npz  raw campaign, N = 65,536 (also g)
  gr_ip_scatter_N4096_20260810.npz   raw campaign, N = 4096 (g only)
  ip_optimized_N65536_20260826.npz   twin-corrected, N = 65,536
  twin_predictions_N1.npz            scalar three-tier residuals (v8)

Output: fig3_v12_preview.pdf / .png / .svg / _small.png  (184.9 x 102 mm;
ink bounds symmetric at ~4.1 mm per side, prints at 96.6 % of drawn size
at \textwidth)
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
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle,
                                PathPatch)
from matplotlib.path import Path
import json
import math
import os

OUT = str(_data_dir(__file__))
D = os.path.join(OUT, "data")
MM = 1 / 25.4
# 184.9-mm canvas: panel content spans ink 4.1..180.6 mm, so the extra
# right margin makes the ink bounds symmetric (~4.1 mm each side) and
# keeps panel g's last x tick label (10^5) clear of the canvas edge.
FIG_W, FIG_H = 184.9, 102.0

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

C_BEFORE = "#E58C7D"          # also the ideal-multiplier tier color
C_BEFORT = "#C0503C"
C_PHYS   = "#2458A6"
C_LEARN  = "#E2A33B"          # also the physics-only tier color
C_VIOLET = "#6D28D9"
C_TEAL   = "#1E8A78"
C_GOLD   = "#B8860B"
C_GRAY   = "#6B7280"
C_INK    = "#1F2937"
C_BOX    = "#F3F4F6"
C_VIOBG  = "#F2ECFB"

# ------------------------------------------------------------------- data ---
gr64 = np.load(os.path.join(D, "gr_ip_scatter_N65536_20260810.npz"),
               allow_pickle=True)
o64 = np.load(os.path.join(D, "ip_optimized_N65536_20260826.npz"))
d1 = np.load(os.path.join(D, "twin_predictions_N1.npz"))

SNR64 = [float(np.mean(gr64["snr3_db_reps"][p])) for p in range(2)]
RB = o64["rmse_before_mean"]; RBE = o64["rmse_before_err"]
RA = o64["rmse_after_mean"];  RAE = o64["rmse_after_err"]
EB = o64["enob_before_mean"]; EBE = o64["enob_before_err"]
EA = o64["enob_after_mean"];  EAE = o64["enob_after_err"]

# the caption states "mean +/- 1 s.d., n = 3"; the *_err fields are read,
# not recomputed here, so gate their semantics — an upstream rerun with
# --errorbar sem, or with a different repeat count, must fail loudly
# rather than silently falsify the caption
assert int(o64["n_repeats"]) == 3, f'n_repeats = {o64["n_repeats"]}, caption says n = 3'
assert str(o64["errorbar"]) == "std", f'errorbar = {o64["errorbar"]}, caption says s.d.'
for key, mean, err in (("rmse_before", RB, RBE), ("rmse_after", RA, RAE),
                       ("enob_before", EB, EBE), ("enob_after", EA, EAE)):
    reps = o64[key + "_reps"]
    assert reps.shape[1] == 3, f"{key}: {reps.shape[1]} repeats, caption says 3"
    assert np.allclose(mean, reps.mean(axis=1)), f"{key}: mean != mean of repeats"
    assert np.allclose(err, reps.std(axis=1, ddof=1)), f"{key}: err is not the ddof=1 s.d."

# scalar residual densities of the three model tiers (fig2 convention:
# ideal floored at 1e-15 W) — identical to v8 panel c
def w2dbm(p, floor=1e-15):
    return 10 * np.log10(np.maximum(np.asarray(p, float), floor)) + 30


P_LO1, P_RF1 = np.meshgrid(d1["p_lo_dbm"], d1["p_rf_dbm"], indexing="ij")
ideal_w = (1e-3 * 10 ** (P_LO1 / 10) * 1e-3 * 10 ** (P_RF1 / 10)) / 1e-3
dbm_meas1 = w2dbm(d1["p_if_meas_w"])
res_ideal = (w2dbm(ideal_w) - dbm_meas1).ravel()
res_phys = (w2dbm(d1["p_if_phys_w"]) - dbm_meas1).ravel()
res_full = (w2dbm(d1["p_if_full_w"]) - dbm_meas1).ravel()

# rebuild one transmitted N = 65,536 LO data symbol for the waveform panel
# (per-symbol RMS normalization approximates the frame RMS to <0.1%; the
#  10-dB digital PAPR clip of the campaign is reproduced exactly)
NV = int(gr64["vec_N"]); LF = int(gr64["fft_len"])
seed = int(gr64["seed"]); sidx = gr64["slot_seed_idx"]
kinds = gr64["slot_kinds"]
slot0 = int(sidx[np.where(kinds == "D")[0][0]])
rr = np.random.default_rng([seed, slot0])
a_vec = rr.uniform(0, 1, NV) * np.exp(1j * rr.uniform(0, 2 * np.pi, NV))
bins = 3 * (np.arange(NV) - NV // 2) + 1
XSPEC = np.zeros(LF, np.complex128)
XSPEC[np.mod(bins, LF)] = a_vec
sA = np.fft.ifft(XSPEC)
sA = sA / np.sqrt(np.mean(np.abs(sA) ** 2))
thr = 10.0 ** (10.0 / 20.0)                       # campaign papr_clip_db = 10
mA = np.abs(sA)
sA = np.where(mA > thr, sA * (thr / np.maximum(mA, 1e-30)), sA)
sA = sA / np.sqrt(np.mean(np.abs(sA) ** 2))

PCOMP_LO, BETA_C = -5.1990871823, 0.6101367971    # calibrated N = 4096 twin
p_lo0 = float(np.mean(gr64["p_lo_dbm_tx"]))       # -3 dBm at the LO port


def comp_ratio_lo(p_dbm):
    pw = 1e-3 * 10.0 ** (np.asarray(p_dbm, float) / 10.0)
    pcw = 1e-3 * 10.0 ** (PCOMP_LO / 10.0)
    cw = pw * (1.0 + (pw / pcw) ** BETA_C) ** (-1.0 / BETA_C)
    return cw / np.maximum(pw, 1e-300)


env = np.abs(sA)
p_inst = p_lo0 + 20 * np.log10(np.maximum(env, 1e-12))
env_c = env * np.sqrt(np.maximum(comp_ratio_lo(p_inst), 0.0))
env_c = env_c * (np.sqrt(np.mean(env ** 2)) / np.sqrt(np.mean(env_c ** 2)))
i0 = int(np.argmax(env))
w_half = 110
sl = slice(max(0, i0 - w_half), min(LF, i0 + w_half))
tt = np.arange(sl.start, sl.stop) - i0

# --------------------------------------------- g: energy accounting --------
# Client-side energy per real MAC of the same 2026-08-10 campaign (25-dB
# panels, raw decode, criterion RMSE < 1/16 met by every repeat); formulas,
# benchmark constants and assertion guards identical to the V2/energy
# fig5_energy_package (Gao et al., Sci. Adv. 12, eadz0817 (2026), Eqs. 1-3).
ETA_RADIO = 0.1                       # transmit-chain efficiency, 10 %
EADC = 1e-12                          # J per ADC conversion
EDIG = 1e-12                          # J per digital real MAC
N_ADC, N_DIG = 6, 8                   # 3 complex samples; 2 complex rescales
H100 = 70e-15                         # GPU reference, J per real MAC
KBT0 = 1.380649e-23 * 300.0
B_STAR = 2.0                          # limit label: criterion ENOB, rounded down


def energy_point(z):
    """One measured operating point (25-dB panel) -> N, e1, e_ip."""
    meta_g = json.loads(str(z["meta_json"]))
    i = int(np.where(z["labels"] == "25 dB")[0][0])
    N = int(z["vec_N"])
    slot_len = int(z["fft_len"]) + int(z["cp_len"])
    n_slots = (int(z["frame_len"]) - int(z["gap0"]) - int(z["gap1"])) / slot_len
    assert abs(n_slots - round(n_slots)) < 1e-9, "frame not slot-aligned"
    T_ip = round(n_slots) * slot_len / float(z["fs_hz"]) / int(z["n_data"])
    px_dbm = float(z["p_rf_dbm_tx"][i]) - float(meta_g["rf_atten_db"])
    e1 = 1e-3 * 10.0 ** (px_dbm / 10.0) * T_ip / (4.0 * N * ETA_RADIO)
    e23 = (N_ADC * EADC + N_DIG * EDIG) / (4.0 * N)
    return N, e1, e1 + e23


gr40 = np.load(os.path.join(D, "gr_ip_scatter_N4096_20260810.npz"),
               allow_pickle=True)
EN0, E1_0, EIP0 = energy_point(gr40)          # N = 4096
EN1, E1_1, EIP1 = energy_point(gr64)          # N = 65,536

E1_FLOOR = E1_1                               # radio floor pinned at N = 65,536
N_grid_g = np.logspace(np.log10(30.0), 5.0, 600)
e2_curve = N_ADC * EADC / (4.0 * N_grid_g)
e3_curve = N_DIG * EDIG / (4.0 * N_grid_g)
eip_curve = E1_FLOOR + e2_curve + e3_curve
# at b* = 2 the Landauer (b^2) and thermodynamic (2b) bounds coincide
E_LIMIT = B_STAR ** 2 * math.log(2.0) * KBT0

for name, got, want in [
        ("e1  (N=65536) [fJ]", E1_1 * 1e15, 0.534),
        ("e_ip(N=65536) [fJ]", EIP1 * 1e15, 0.587),
        ("x vs H100 (N=65536)", H100 / EIP1, 119.2),
        ("e1  (N=4096)  [fJ]", E1_0 * 1e15, 0.613),
        ("e_ip(N=4096)  [fJ]", EIP0 * 1e15, 1.467),
        ("x vs H100 (N=4096)", H100 / EIP0, 47.7),
        ("limit @2 bit [zJ]", E_LIMIT * 1e21, 11.48)]:
    assert abs(got - want) <= 0.01 * want, f"energy check drifted: {name}"

# ------------------------------------------------------------------ figure --
fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))


def axmm(x, y, w, h, **kw):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], **kw)


def tag(x_mm, y_mm, s):
    fig.text(x_mm / FIG_W, y_mm / FIG_H, s, fontsize=FS_TAG,
             fontweight="bold", va="top", ha="left", color=C_INK)


def schembox(ax, x, y, w, h, fc=C_BOX, ec=C_GRAY, lw=0.8, r=1.2):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(b)
    return b


def sarrow(ax, p0, p1, color=C_INK, lw=1.0, style="-|>", ls="-",
           connect=None, z=4, ms=7):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=connect or "arc3,rad=0",
                        shrinkA=1, shrinkB=1, zorder=z)
    ax.add_patch(a)


def figarrow(p0_mm, p1_mm, color=C_INK, lw=1.1, connect="arc3,rad=0", ms=9):
    a = FancyArrowPatch((p0_mm[0] / FIG_W, p0_mm[1] / FIG_H),
                        (p1_mm[0] / FIG_W, p1_mm[1] / FIG_H),
                        transform=fig.transFigure, arrowstyle="-|>",
                        mutation_scale=ms, lw=lw, color=color,
                        connectionstyle=connect, shrinkA=0, shrinkB=0)
    fig.add_artist(a)


def hbrace(x_left, x_right, y_arm, y_spine, y_cusp, r=3.0, lw=1.1,
           color=C_INK):
    """Horizontal brace gathering [x_left, x_right]: hooked ends at
    y_arm, spine at y_spine, center cusp pointing down to y_cusp."""
    xm = (x_left + x_right) / 2.0
    W, H = FIG_W, FIG_H

    def p(x, y):
        return (x / W, y / H)

    verts = [p(x_left, y_arm),
             p(x_left, y_spine), p(x_left + r, y_spine),
             p(xm - r, y_spine),
             p(xm, y_spine), p(xm, y_cusp),
             p(xm, y_spine), p(xm + r, y_spine),
             p(x_right - r, y_spine),
             p(x_right, y_spine), p(x_right, y_arm)]
    codes = [Path.MOVETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3]
    patch = PathPatch(Path(verts, codes), transform=fig.transFigure,
                      fc="none", ec=color, lw=lw, capstyle="round",
                      joinstyle="round")
    fig.add_artist(patch)


# ---------------- a: correction scheme (vertical flow) ----------------------
# 36.5-mm axes (extends into the a-b gap, which panel a can use because it
# has no y-axis; panels b-g are untouched). The branch boxes keep their
# gray sublabels; the free-floating gray annotations stay dropped.
AX_A = axmm(10.5, 8.0 + 55.5, 36.5, 34)
AX_A.set_xlim(0, 36.5); AX_A.set_ylim(0, 34); AX_A.axis("off")

# input chip (top)
schembox(AX_A, 12.25, 28.9, 12.0, 4.6, fc="white", ec=C_INK, lw=0.8)
AX_A.text(18.25, 31.3, r"$\mathbf{a},\ \mathbf{b}$", fontsize=FS,
          ha="center", va="center", color=C_INK)

# two branches (mixer analog | PIML twin digital)
schembox(AX_A, 1.0, 17.8, 16.5, 8.2, fc="#FBF3E2", ec=C_GOLD, lw=0.9)
AX_A.text(9.25, 24.2, "mixer (analog)", fontsize=FS - 0.5,
          ha="center", va="center", color=C_GOLD)
AX_A.text(9.25, 20.4, "measured $\\hat{c}$,\nerror $r$", fontsize=FS - 2,
          ha="center", va="center", color=C_GRAY, linespacing=1.2)

schembox(AX_A, 19.0, 17.8, 16.5, 8.2, fc=C_VIOBG, ec=C_VIOLET, lw=0.9)
AX_A.text(27.25, 24.2, "PIML twin", fontsize=FS - 0.5,
          ha="center", va="center", color=C_VIOLET)
AX_A.text(27.25, 20.4, "distortion\nfeatures $\\widehat{\\delta}_m$",
          fontsize=FS - 2, ha="center", va="center", color=C_GRAY,
          linespacing=1.2)

sarrow(AX_A, (15.5, 28.7), (9.9, 26.8), color=C_INK)
sarrow(AX_A, (21.0, 28.7), (26.6, 26.8), color=C_INK)

# subtract node and corrected output (bottom)
MDX, MDY = 18.25, 12.2
sarrow(AX_A, (9.25, 17.5), (MDX - 2.0, MDY), color=C_GOLD,
       connect="angle,angleA=-90,angleB=180,rad=2.0")
sarrow(AX_A, (27.25, 17.5), (MDX + 2.0, MDY), color=C_VIOLET,
       connect="angle,angleA=-90,angleB=0,rad=2.0")
circ2 = Circle((MDX, MDY), 1.7, fc="white", ec=C_INK, lw=0.9, zorder=5)
AX_A.add_patch(circ2)
AX_A.plot([MDX - 0.9, MDX + 0.9], [MDY, MDY], color=C_INK, lw=0.9,
          zorder=6)

schembox(AX_A, 11.75, 1.4, 13.0, 6.6, fc="white", ec=C_TEAL, lw=0.9)
AX_A.text(18.25, 6.2, "corrected", fontsize=FS - 0.5, ha="center",
          va="center", color=C_TEAL)
AX_A.text(18.25, 3.4, r"$\langle\mathbf{a},\mathbf{b}\rangle$",
          fontsize=FS, ha="center", va="center", color=C_TEAL)
sarrow(AX_A, (MDX, MDY - 1.9), (MDX, 8.2), color=C_TEAL)

# ---------------- b: the twin predicts the compression of w(t) --------------
AX_B = axmm(54.0, 63.5, 31.5, 34)
AX_B.plot(tt, env[sl], lw=0.9, color=C_PHYS, zorder=3)
AX_B.plot(tt, env_c[sl], lw=0.9, color=C_LEARN, zorder=4)
AX_B.fill_between(tt, env_c[sl], env[sl], color=C_LEARN, alpha=0.25, lw=0,
                  zorder=2)
AX_B.text(-104, 4.02, "transmitted $w(t)$", fontsize=FS, color=C_PHYS,
          ha="left", va="center")
AX_B.text(-104, 3.55, "after twin compression", fontsize=FS,
          color="#B87A0F", ha="left", va="center")
AX_B.set_xlim(tt[0], tt[-1])
AX_B.set_ylim(0, 4.35)
AX_B.set_yticks([0, 1, 2, 3])
AX_B.set_xticks([-100, 0, 100])
AX_B.set_xlabel("sample (zoom)", labelpad=1.0)
AX_B.set_ylabel("|envelope| (norm.)", labelpad=1.0)
for s in ("top", "right"):
    AX_B.spines[s].set_visible(False)

hbrace(13.5, 84.0, 56.4, 54.4, 52.1)                  # gather a + b
figarrow((48.75, 52.15), (48.75, 47.9))               # down into d,e

# ---------------- c: residual densities of the three model tiers ------------
AX_C = axmm(97.5, 63.5, 31.5, 34)
bins_c = np.arange(-60, 60.5, 1.0)
for res, col, z in [(res_ideal, C_BEFORE, 3), (res_phys, C_LEARN, 4),
                    (res_full, C_PHYS, 5)]:
    AX_C.hist(res, bins=bins_c, density=True, color=col, alpha=0.55,
              zorder=z)
AX_C.axvline(0, color=C_INK, lw=0.7, zorder=6)
AX_C.text(-8.5, 0.345, "full twin", fontsize=FS, color=C_PHYS,
          ha="right", va="center")
AX_C.annotate("", xy=(-0.9, 0.32), xytext=(-8.0, 0.335),
              arrowprops=dict(arrowstyle="-", lw=0.5, color=C_GRAY,
                              shrinkA=0, shrinkB=1))
AX_C.text(-14.5, 0.10, "physics\nonly", fontsize=FS, color="#C08A28",
          ha="center", va="center", linespacing=1.15)
AX_C.annotate("", xy=(-4.5, 0.055), xytext=(-11.5, 0.082),
              arrowprops=dict(arrowstyle="-", lw=0.5, color=C_GRAY,
                              shrinkA=0, shrinkB=1))
AX_C.text(17.0, 0.10, "ideal\nmultiplier", fontsize=FS,
          color=C_BEFORT, ha="center", va="center", linespacing=1.15)
AX_C.annotate("", xy=(11.5, 0.045), xytext=(15.0, 0.082),
              arrowprops=dict(arrowstyle="-", lw=0.5, color=C_GRAY,
                              shrinkA=0, shrinkB=1))
AX_C.set_xlim(-30, 30)
AX_C.set_ylim(0, 0.45)
AX_C.set_xticks([-20, 0, 20])
AX_C.set_yticks([0, 0.2, 0.4])
AX_C.set_xlabel("residual, model $-$ meas. (dB)", labelpad=1.0)
AX_C.set_ylabel("density", labelpad=1.0)
for s in ("top", "right"):
    AX_C.spines[s].set_visible(False)

# ---------------- d,e: THE PAYOFF — N = 65,536 scatters ---------------------
def ip_panel(ax, p, title):
    ct = np.asarray(o64["c_true"])
    R = o64["c_hat_before_reps"].shape[1]
    tr = np.concatenate([np.tile(ct.real, R), np.tile(ct.imag, R)])
    bf_c = o64["c_hat_before_reps"][p].reshape(-1)
    af_c = o64["c_hat_after_reps"][p].reshape(-1)
    bf = np.concatenate([bf_c.real, bf_c.imag])
    af = np.concatenate([af_c.real, af_c.imag])
    ax.plot([-1, 1], [-1, 1], color=C_INK, lw=0.7, zorder=2)
    ax.scatter(tr, bf, s=2.6, c=C_BEFORE, alpha=0.55, lw=0, zorder=3,
               rasterized=True)
    ax.scatter(tr, af, s=2.6, c=C_PHYS, alpha=0.55, lw=0, zorder=4,
               rasterized=True)
    ax.text(-0.93, 0.86, "before",
            fontsize=FS - 0.8, color=C_BEFORT, ha="left", va="center")
    ax.text(-0.93, 0.68, "after",
            fontsize=FS - 0.8, color=C_PHYS, ha="left", va="center")
    ax.set_title(title, pad=2.6, color=C_INK)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("analog estimate", labelpad=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


AX_D = axmm(12.0, 8.0, 31.5, 34)
ip_panel(AX_D, 0, "$N$ = 65,536,  SNR = 15 dB")
AX_D.set_xlabel("true inner product", labelpad=1.2)

AX_E = axmm(54.0, 8.0, 31.5, 34)
ip_panel(AX_E, 1, "$N$ = 65,536,  SNR = 25 dB")
AX_E.set_xlabel("true inner product", labelpad=1.2)
AX_E.set_yticklabels([])
AX_E.set_ylabel("")

# ---------------- f: approach to the receiver noise floor -------------------
AX_F = axmm(97.5, 8.0, 31.5, 34)
xs_f = np.linspace(12.5, 27.5, 120)
floor_line = 1.0 / np.sqrt(27.0 * 10.0 ** (xs_f / 10.0))
AX_F.plot(xs_f, floor_line, ls=(0, (5, 2)), lw=1.0, color=C_INK, zorder=2)
AX_F.plot(SNR64, RB, ls=(0, (1.5, 1.5)), lw=0.9, color=C_BEFORE, zorder=3)
AX_F.plot(SNR64, RA, lw=0.9, color=C_PHYS, zorder=3)
AX_F.errorbar(SNR64, RB, yerr=RBE, fmt="s", ms=4.6, mfc=C_BEFORE,
              mec="none", ecolor=C_BEFORT, elinewidth=0.8, capsize=1.6,
              capthick=0.8, zorder=5, ls="none")
AX_F.errorbar(SNR64, RA, yerr=RAE, fmt="o", ms=4.6, mfc=C_PHYS,
              mec="none", ecolor=C_PHYS, elinewidth=0.8, capsize=1.6,
              capthick=0.8, zorder=5, ls="none")
AX_F.set_title("$N$ = 65,536", pad=2.6, color=C_INK)
AX_F.text(19.9, 0.076, "before", fontsize=FS, color=C_BEFORT,
          ha="center", va="center")
AX_F.text(16.3, 0.0245, "after", fontsize=FS, color=C_PHYS,
          ha="center", va="center")
AX_F.text(19.8, 0.0130, "measured receiver\nnoise floor", fontsize=FS,
          color=C_INK, ha="center", va="center", linespacing=1.2)
AX_F.set_yscale("log")
AX_F.set_xlim(12.5, 27.5)
AX_F.set_ylim(0.0085, 0.115)
AX_F.set_xticks([15, 20, 25])
AX_F.set_yticks([0.01, 0.03, 0.1])
AX_F.set_yticklabels(["0.01", "0.03", "0.1"])
AX_F.set_xlabel("SNR (dB)", labelpad=1.0)
AX_F.set_ylabel("inner-product RMSE", labelpad=1.0)
for s in ("top", "right"):
    AX_F.spines[s].set_visible(False)

# ---------------- g: client-side energy per real MAC (two rows tall) --------
C_E_OR, C_E_PK, C_E_PU, C_E_CY = "#E07A2D", "#C44E96", "#7A5AA6", "#13A7B8"
AX_G = axmm(145.0, 22.5, 33.5, 75)    # energy curves
AX_G2 = axmm(145.0, 8.0, 33.5, 12)    # broken-axis limit strip
for a in (AX_G, AX_G2):
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(30, 1e5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

AX_G.axhline(H100, color=C_INK, lw=0.9, zorder=2)
AX_G.text(0.97, H100 * 1.32, "H100 GPU: 70 fJ/MAC",
          transform=AX_G.get_yaxis_transform(), ha="right", va="bottom",
          fontsize=FS - 1.3, color=C_INK)

AX_G.plot(N_grid_g, eip_curve, color=C_PHYS, lw=1.5, zorder=6,
          label="$e_{\\mathrm{ip}}(N)=e_1{+}e_2{+}e_3$")
AX_G.axhline(E1_FLOOR, color=C_E_OR, lw=1.0, ls=(0, (4.8, 1.9)), zorder=4,
             label="waveform generation, $e_1$")
AX_G.plot(N_grid_g, e2_curve, color=C_E_PK, lw=1.0,
          ls=(0, (3.0, 1.5, 1.0, 1.5)), zorder=4,
          label="I/Q sampling, $e_2$")
AX_G.plot(N_grid_g, e3_curve, color=C_E_PU, lw=1.0, ls=(0, (1.0, 1.4)),
          zorder=4, label="digital decoding, $e_3$")
AX_G.plot([EN0, EN1], [EIP0, EIP1], marker="v", ms=5.5, mfc="#F3B06C",
          mec=C_E_OR, mew=1.0, ls="none", zorder=7,
          label="experiment (this work)")

AX_G.annotate(f"$N$ = 4096\n{EIP0 * 1e15:.2f} fJ/MAC\n"
              f"{H100 / EIP0:.0f}$\\times$ vs H100",
              xy=(EN0, EIP0), xytext=(1900, 1.35e-14), fontsize=FS - 1.3,
              color=C_INK, ha="center", va="center", linespacing=1.25,
              arrowprops=dict(arrowstyle="-", lw=0.5, color=C_GRAY,
                              shrinkA=10, shrinkB=2))
AX_G.annotate(f"$N$ = 65,536\n{EIP1 * 1e15:.2f} fJ/MAC\n"
              f"{H100 / EIP1:.0f}$\\times$ vs H100",
              xy=(EN1, EIP1), xytext=(1.6e4, 4.5e-15), fontsize=FS - 1.3,
              color=C_INK, ha="center", va="center", linespacing=1.25,
              arrowprops=dict(arrowstyle="-", lw=0.5, color=C_GRAY,
                              shrinkA=10, shrinkB=2))

AX_G.legend(loc="lower left", fontsize=FS - 1.5, frameon=False,
            borderaxespad=0.2, handlelength=2.0, labelspacing=0.45,
            handletextpad=0.6)
AX_G.set_ylim(3.2e-18, 2.3e-13)
AX_G.set_yticks([1e-13, 1e-14, 1e-15, 1e-16, 1e-17])
AX_G.tick_params(labelbottom=False)
AX_G.set_ylabel("energy per real MAC (J/MAC)", labelpad=1.5)

AX_G2.axhline(E_LIMIT, color=C_INK, lw=1.1, zorder=3)
AX_G2.axhline(E_LIMIT, color=C_E_CY, lw=1.1, ls=(0, (3.2, 1.4)), zorder=4)
AX_G2.text(0.97, E_LIMIT * 1.35, "Landauer = thermodynamic\nlimit (2-bit ENOB)",
           transform=AX_G2.get_yaxis_transform(), ha="right", va="bottom",
           fontsize=FS - 1.5, color=C_INK, linespacing=1.2)
AX_G2.text(0.04, E_LIMIT * 0.70, f"{E_LIMIT * 1e21:.1f} zJ/MAC",
           transform=AX_G2.get_yaxis_transform(), ha="left", va="top",
           fontsize=FS - 1.5, color=C_E_CY)
AX_G2.set_ylim(3e-21, 2.1e-19)
AX_G2.set_yticks([1e-19, 1e-20])
AX_G2.set_xticks([1e2, 1e3, 1e4, 1e5])
AX_G2.set_xlabel("input size, $N$", labelpad=1.0)

# y-axis break marks on the shared left spine
for a, y0 in ((AX_G, 0.0), (AX_G2, 1.0)):
    dy = 1.3 / (75.0 if a is AX_G else 12.0)
    a.plot([-0.018, 0.018], [y0 - dy, y0 + dy], transform=a.transAxes,
           color="k", lw=0.8, clip_on=False, zorder=10)

# ============================ panel tags =====================================
tag(5.0, 100.9, "a")
tag(47.0, 100.9, "b")
tag(90.5, 100.9, "c")
tag(5.0, 46.8, "d")
tag(47.0, 46.8, "e")
tag(90.5, 46.8, "f")
tag(138.0, 100.9, "g")

# ============================ save ==========================================
fig.savefig(os.path.join(OUT, "fig3_v12_preview.pdf"), dpi=600)
fig.savefig(os.path.join(OUT, "fig3_v12_preview.png"), dpi=600)
fig.savefig(os.path.join(OUT, "fig3_v12_preview.svg"))

from PIL import Image
im = Image.open(os.path.join(OUT, "fig3_v12_preview.png"))
im.resize((im.width // 3, im.height // 3), Image.LANCZOS).save(
    os.path.join(OUT, "fig3_v12_preview_small.png"))
print("wrote fig3_v12_preview.*")
print(f"  N=65536  before {RB[0]:.4f}/{RB[1]:.4f} -> after "
      f"{RA[0]:.4f}/{RA[1]:.4f}  ENOB {EB[0]:.2f}/{EB[1]:.2f} -> "
      f"{EA[0]:.2f}/{EA[1]:.2f}  SNR {SNR64[0]:.2f}/{SNR64[1]:.2f} dB")
print(f"  energy   N=4096  e_ip={EIP0 * 1e15:.3f} fJ/MAC "
      f"({H100 / EIP0:.1f}x below H100)   N=65536  "
      f"e_ip={EIP1 * 1e15:.3f} fJ/MAC ({H100 / EIP1:.1f}x below H100)  "
      f"floor e1={E1_FLOOR * 1e15:.3f} fJ  limit={E_LIMIT * 1e21:.2f} zJ")
