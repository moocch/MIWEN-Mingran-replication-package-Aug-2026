# -*- coding: utf-8 -*-
"""
fig5_energy_scaling.py
======================
Builds Fig. 5 (client-side energy per real MAC vs. inner-product size N)
*directly from the raw measurement files*, and exports everything that ends
up on the canvas into ``data/fig5_plot_data.npz`` so the figure can be
re-drawn as a vector graphic without re-reading the raw data.

Run from anywhere:

    python code/fig5_energy_scaling.py

Outputs
-------
figures/fig5_energy_scaling.pdf     vector (for the manuscript)
figures/fig5_energy_scaling.svg     vector (for Illustrator / Inkscape)
figures/fig5_energy_scaling.png     600 dpi raster (for slides)
data/fig5_plot_data.npz             every curve, point and constant plotted
data/fig5_plot_data_curves.csv      the same curves as plain text

Nothing about the operating points is hard-coded: transmit power, waveform
length, RMSE and ENOB are all read out of the two raw ``.npz`` files. The
literals in CHECKS below are assertions, not inputs -- if the raw data
changes, the script fails loudly instead of silently drawing stale numbers.

See METHODS.md for the derivation of every equation used here.
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------- paths ----
HERE = str(_data_dir(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
DATA_OUT = os.path.join(ROOT, "data")
FIG_OUT = os.path.join(ROOT, "figures")
os.makedirs(FIG_OUT, exist_ok=True)

RUN_N4096 = os.path.join(RAW, "gr_fig3c_ip_scatter_20260810_011915_N4096",
                         "gr_fig3c_ip_scatter.npz")
RUN_N65536 = os.path.join(RAW, "gr_fig3c_ip_scatter_20260810_002043_N65536",
                          "gr_fig3c_ip_scatter.npz")

# ------------------------------------------------- benchmark constants -----
# Energy-accounting methodology: Gao et al., Sci. Adv. 12, eadz0817 (2026).
kB, T0 = 1.380649e-23, 300.0          # Boltzmann constant, room temperature
kBT0 = kB * T0
H100 = 70e-15                         # NVIDIA H100 reference, J per real MAC
ETA_RADIO = 0.1                       # transmit-chain efficiency, 10 %
EADC = 1e-12                          # ADC energy, J per conversion
EDIG = 1e-12                          # digital MAC energy, J
N_ADC = 6                             # 3 complex samples  = 6 real conversions
N_DIG = 8                             # 2 complex rescales = 8 real MACs
PANEL = "25 dB"                       # operating panel used for the figure
RMSE_CRITERION = 0.0625               # accuracy criterion on normalised output

# Precision label for the reference limits. The criterion RMSE < 0.0625 was
# quoted as "5 bit" in WISE (error vs. full scale, width 2). Under the ENOB
# convention used here -- ENOB = log2(std[y]/RMSE), error vs. the actual
# spread of the answers -- the same criterion is ~2.4 bit (measured 2.42 and
# 2.44; analytic 2.415). The offset is log2(2/std[y]) ~ log2(6) = 2.585 bit,
# because std[y] = 1/3 analytically for U(0,1)-amplitude random vectors.
# See METHODS.md, sections 5-6.
#
# The reference limits are drawn at the round value B_STAR = 2 bit, which is
# the criterion-equivalent precision rounded down (conservative: a lower b
# means a lower floor, so the quoted headroom is not overstated).
# NOTE: b^2 = 2b exactly at b = 2, so at this label the Landauer and the
# thermodynamic bound COINCIDE at 11.48 zJ/MAC. The plotting code detects
# this and draws a single merged line rather than two superimposed ones.
B_STAR = 2.0


def load_point(npz_path):
    """Read one operating point (the PANEL column) out of a raw run file."""
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    i = int(np.where(z["labels"] == PANEL)[0][0])

    N = int(z["vec_N"])
    fft_len, cp_len = int(z["fft_len"]), int(z["cp_len"])
    fs = float(z["fs_hz"])
    q_data = int(z["n_data"])
    frame_len = int(z["frame_len"])
    gaps = int(z["gap0"]) + int(z["gap1"])

    # Airtime amortised per inner product. The frame holds n_slots bursts of
    # (fft_len + cp_len) samples -- pilots included, since they are genuinely
    # transmitted -- of which q_data carry data inner products.
    slot_len = fft_len + cp_len
    n_slots = (frame_len - gaps) / slot_len
    assert abs(n_slots - round(n_slots)) < 1e-9, "frame length not slot-aligned"
    n_slots = int(round(n_slots))
    T_ip = n_slots * slot_len / fs / q_data
    # Variant that also charges the two inter-frame guard gaps (+0.4 %); see
    # METHODS.md section 3.2. Not used for the published numbers.
    T_ip_with_gaps = frame_len / fs / q_data

    px_dbm = float(z["p_rf_dbm_tx"][i]) - float(meta["rf_atten_db"])
    px_w = 1e-3 * 10 ** (px_dbm / 10.0)

    return dict(
        N=N, px_dbm=px_dbm, px_w=px_w, T_ip=T_ip, T_ip_with_gaps=T_ip_with_gaps,
        n_slots=n_slots, q_data=q_data, fs=fs, fft_len=fft_len, cp_len=cp_len,
        p_tx_dbm=float(z["p_rf_dbm_tx"][i]), atten_db=float(meta["rf_atten_db"]),
        rmse_mean=float(z["rmse_mean"][i]), rmse_sd=float(z["rmse_sd"][i]),
        enob_mean=float(z["enob_mean"][i]), enob_sd=float(z["enob_sd"][i]),
        std_y=float(z["std_y"][i]), snr3_db=float(z["snr3_db"][i]),
        n_repeats=int(z["n_repeats"][i]),
        rmse_reps=np.array(z["rmse_reps"][i]),
        enob_reps=np.array(z["enob_reps"][i]),
        run=os.path.basename(os.path.dirname(npz_path)),
    )


def e1_of(pt):
    """Waveform generation, J per real MAC."""
    return pt["px_w"] * pt["T_ip"] / (4.0 * pt["N"] * ETA_RADIO)


def e23_of(N):
    """Read-out (ADC + digital decode), J per real MAC. Fixed fee / (4N)."""
    return (N_ADC * EADC + N_DIG * EDIG) / (4.0 * N)


# ------------------------------------------------------------- assemble ----
p0 = load_point(RUN_N4096)      # N = 4096
p1 = load_point(RUN_N65536)     # N = 65536
for p in (p0, p1):
    p["e1"] = e1_of(p)
    p["e2"] = N_ADC * EADC / (4.0 * p["N"])
    p["e3"] = N_DIG * EDIG / (4.0 * p["N"])
    p["e_ip"] = p["e1"] + p["e2"] + p["e3"]
    p["speedup"] = H100 / p["e_ip"]
    # every repeat must satisfy the accuracy criterion
    assert np.all(p["rmse_reps"] < RMSE_CRITERION), \
        f"{p['run']}: a repeat misses RMSE < {RMSE_CRITERION}"

# Model curve: e1 is a floor set by the required RF power (it does not depend
# on N -- see METHODS.md section 3.3); the read-out fee falls as 1/N. The
# floor is pinned to the largest measured N, where the radio term dominates
# and the estimate is least contaminated by read-out overhead.
E1_FLOOR = p1["e1"]
N_curve = np.logspace(np.log10(30.0), 5.0, 600)
e2_curve = N_ADC * EADC / (4.0 * N_curve)
e3_curve = N_DIG * EDIG / (4.0 * N_curve)
e_curve = E1_FLOOR + e2_curve + e3_curve
N_bend = (N_ADC * EADC + N_DIG * EDIG) / (4.0 * E1_FLOOR)   # read-out == floor

# Reference limits at the label precision B_STAR (METHODS 6.3). At b = 2
# these two expressions are algebraically identical.
e_landauer = B_STAR ** 2 * np.log(2.0) * kBT0
e_thermo = 2.0 * B_STAR * np.log(2.0) * kBT0

# -------------------------------------------------------------- checks -----
CHECKS = [
    ("e1  (N=65536) [fJ]", p1["e1"] * 1e15, 0.534, 0.01),
    ("e_ip(N=65536) [fJ]", p1["e_ip"] * 1e15, 0.587, 0.01),
    ("x vs H100 (N=65536)", p1["speedup"], 119.2, 0.01),
    ("e1  (N=4096)  [fJ]", p0["e1"] * 1e15, 0.613, 0.01),
    ("e_ip(N=4096)  [fJ]", p0["e_ip"] * 1e15, 1.467, 0.01),
    ("x vs H100 (N=4096)", p0["speedup"], 47.7, 0.01),
    ("Px (N=4096)  [dBm]", p0["px_dbm"], -62.81, 0.01),
    ("Px (N=65536) [dBm]", p1["px_dbm"], -63.41, 0.01),
    ("T_ip (N=65536) [ms]", p1["T_ip"] * 1e3, 30.68, 0.01),
    ("bend N (read-out = floor)", N_bend, 6553.0, 0.01),
    ("ENOB equiv of criterion, N=4096",
     np.log2(p0["std_y"] / RMSE_CRITERION), 2.442, 0.01),
    ("ENOB equiv of criterion, N=65536",
     np.log2(p1["std_y"] / RMSE_CRITERION), 2.419, 0.01),
    ("Landauer @2 bit [zJ]", e_landauer * 1e21, 11.48, 0.01),
    ("Thermodynamic @2 bit [zJ]", e_thermo * 1e21, 11.48, 0.01),
    ("headroom above the floor", p1["e_ip"] / e_landauer, 51185.0, 0.01),
]
print("=== consistency checks (values derived from raw .npz) ===")
ok = True
for name, got, want, tol in CHECKS:
    good = abs(got - want) <= tol * max(1.0, abs(want))
    ok &= good
    print(f"  {name:34s} {got:12.4f}   expect {want:<10} [{'OK' if good else 'FAIL'}]")
assert ok, "a derived number no longer matches the published value"

print("\n=== operating points (panel '%s', criterion RMSE < %.4f) ===" % (PANEL, RMSE_CRITERION))
for p in (p0, p1):
    print(f"  N={p['N']:<6d} Px={p['px_dbm']:+.2f} dBm ({p['px_w'] * 1e9:.3f} nW)  "
          f"T_ip={p['T_ip'] * 1e3:.3f} ms")
    print(f"           e1={p['e1'] * 1e15:.3f}  e2={p['e2'] * 1e15:.3f}  "
          f"e3={p['e3'] * 1e15:.3f}  ->  e_ip={p['e_ip'] * 1e15:.3f} fJ/MAC "
          f"({p['speedup']:.1f}x below H100)")
    print(f"           RMSE={p['rmse_mean']:.5f}+-{p['rmse_sd']:.5f}   "
          f"ENOB={p['enob_mean']:.3f}+-{p['enob_sd']:.3f} bit  (n={p['n_repeats']})")

# ------------------------------------------------------- export plot data --
meta = dict(
    created_by="fig5_energy_scaling.py",
    methodology="Gao et al., Sci. Adv. 12, eadz0817 (2026), Eqs. 1-3",
    panel=PANEL, rmse_criterion=RMSE_CRITERION,
    eta_radio=ETA_RADIO, e_adc_J=EADC, e_dig_J=EDIG,
    n_adc_conversions=N_ADC, n_digital_macs=N_DIG,
    h100_J_per_mac=H100, kB_T0_J=kBT0, b_star_bit=B_STAR,
    e1_floor_source="N=65536 operating point",
    units="all energies in J per real MAC",
)
np.savez_compressed(
    os.path.join(DATA_OUT, "fig5_plot_data.npz"),
    # --- curves (top panel) ---
    N_curve=N_curve, e_ip_curve=e_curve,
    e1_curve=np.full_like(N_curve, E1_FLOOR),
    e2_curve=e2_curve, e3_curve=e3_curve,
    # --- horizontal reference lines ---
    h100_line=H100, e1_floor=E1_FLOOR,
    e_landauer=e_landauer, e_thermo=e_thermo, b_star=B_STAR,
    limits_coincide=bool(abs(np.log10(e_landauer / e_thermo)) < 0.02),
    N_bend=N_bend,
    # --- measured points (the two triangles) ---
    points_N=np.array([p0["N"], p1["N"]]),
    points_e_ip=np.array([p0["e_ip"], p1["e_ip"]]),
    points_e1=np.array([p0["e1"], p1["e1"]]),
    points_e2=np.array([p0["e2"], p1["e2"]]),
    points_e3=np.array([p0["e3"], p1["e3"]]),
    points_px_dbm=np.array([p0["px_dbm"], p1["px_dbm"]]),
    points_px_W=np.array([p0["px_w"], p1["px_w"]]),
    points_T_ip_s=np.array([p0["T_ip"], p1["T_ip"]]),
    points_speedup_vs_h100=np.array([p0["speedup"], p1["speedup"]]),
    points_rmse_mean=np.array([p0["rmse_mean"], p1["rmse_mean"]]),
    points_rmse_sd=np.array([p0["rmse_sd"], p1["rmse_sd"]]),
    points_enob_mean=np.array([p0["enob_mean"], p1["enob_mean"]]),
    points_enob_sd=np.array([p0["enob_sd"], p1["enob_sd"]]),
    points_std_y=np.array([p0["std_y"], p1["std_y"]]),
    points_n_repeats=np.array([p0["n_repeats"], p1["n_repeats"]]),
    points_run=np.array([p0["run"], p1["run"]]),
    # --- axes ---
    xlim=np.array([30.0, 1e5]),
    ylim_top=np.array([3.2e-18, 2.3e-13]),
    ylim_bottom=np.array([3e-21, 2.1e-19]),
    xlabel="Input size, N", ylabel="Energy per real MAC (J/MAC)",
    meta_json=json.dumps(meta, indent=2),
)

with open(os.path.join(DATA_OUT, "fig5_plot_data_curves.csv"), "w",
          newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["# energy per real MAC (J/MAC); see METHODS.md"])
    w.writerow(["N", "e_ip_total", "e1_waveform", "e2_adc", "e3_digital"])
    for k in range(len(N_curve)):
        w.writerow([f"{N_curve[k]:.6g}", f"{e_curve[k]:.6e}",
                    f"{E1_FLOOR:.6e}", f"{e2_curve[k]:.6e}",
                    f"{e3_curve[k]:.6e}"])

# ------------------------------------------------------------- the figure --
rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.linewidth": 0.7,
    "xtick.direction": "out", "ytick.direction": "out",
    "svg.fonttype": "none",      # keep text as text in the SVG
    "pdf.fonttype": 42,          # embed TrueType, editable in Illustrator
})
C_BLUE, C_ORANGE, C_PINK, C_PURPLE = "#2458a6", "#e07a2d", "#c44e96", "#7a5aa6"
C_CYAN, C_INK, C_GRID = "#13a7b8", "#222222", "#a7adb5"

fig = plt.figure(figsize=(4.25, 4.55))
gs = fig.add_gridspec(2, 1, height_ratios=[5.4, 1.0], hspace=0.10,
                      left=0.135, right=0.965, top=0.975, bottom=0.085)
ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for a in (ax, axb):
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(30, 1e5)
    a.grid(True, which="major", color=C_GRID, lw=0.55, ls=(0, (1.0, 1.65)),
           zorder=0)
    a.set_axisbelow(True)

ax.axhline(H100, color=C_INK, lw=1.15, zorder=3)
ax.text(0.985, H100 * 1.30, "H100 GPU: 70 fJ/MAC",
        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
        fontsize=8, color=C_INK)

ax.plot(N_curve, e_curve, color=C_BLUE, lw=2.0, zorder=6,
        label="$e_{\\mathrm{ip}}(N) = e_1 + e_2 + e_3$")
ax.axhline(E1_FLOOR, color=C_ORANGE, lw=1.2, ls=(0, (4.8, 1.9)), zorder=4,
           label="Waveform generation, $e_1$")
ax.plot(N_curve, e2_curve, color=C_PINK, lw=1.2,
        ls=(0, (3.0, 1.5, 1.0, 1.5)), zorder=4, label="I/Q sampling, $e_2$")
ax.plot(N_curve, e3_curve, color=C_PURPLE, lw=1.2, ls=(0, (1.0, 1.4)),
        zorder=4, label="Digital decoding, $e_3$")

ax.plot([p0["N"], p1["N"]], [p0["e_ip"], p1["e_ip"]], marker="v", ms=7.5,
        mfc="#f3b06c", mec=C_ORANGE, mew=1.2, ls="none", zorder=7,
        label="Experiment (this work)")
ax.annotate(f"$N = {p0['N']}$\n{p0['e_ip'] * 1e15:.2f} fJ/MAC\n"
            f"{p0['speedup']:.0f}$\\times$ vs H100", xy=(p0["N"], p0["e_ip"]),
            xytext=(p0["N"] * 0.115, p0["e_ip"] * 3.6), fontsize=7.5,
            style="italic",
            arrowprops=dict(arrowstyle="-", color="#666b73", lw=0.7))
ax.annotate(f"$N = {p1['N']}$\n{p1['e_ip'] * 1e15:.2f} fJ/MAC\n"
            f"{p1['speedup']:.0f}$\\times$ vs H100", xy=(p1["N"], p1["e_ip"]),
            xytext=(p1["N"] * 0.10, p1["e_ip"] * 6.5), fontsize=7.5,
            style="italic",
            arrowprops=dict(arrowstyle="-", color="#666b73", lw=0.7))

ax.set_ylim(3.2e-18, 2.3e-13)
ax.set_yticks([1e-13, 1e-14, 1e-15, 1e-16, 1e-17])
ax.legend(loc="lower left", fontsize=7.0, frameon=False, borderaxespad=0.4,
          handlelength=2.4, labelspacing=0.5)
ax.tick_params(labelbottom=False)

# At b = 2 the Landauer (b^2) and thermodynamic (2b) bounds are equal, so a
# single line carries both identities: black solid with cyan dashes on top.
COINCIDE = abs(np.log10(e_landauer / e_thermo)) < 0.02
if COINCIDE:
    axb.axhline(e_landauer, color=C_INK, lw=1.3, zorder=3)
    axb.axhline(e_thermo, color=C_CYAN, lw=1.3, ls=(0, (3.2, 1.4)), zorder=4)
    axb.text(0.985, e_landauer * 1.30,
             f"Landauer = thermodynamic limit ({B_STAR:g}-bit ENOB)",
             transform=axb.get_yaxis_transform(), ha="right", va="bottom",
             fontsize=7.6, color=C_INK)
    axb.text(0.015, e_landauer * 0.72, f"{e_landauer * 1e21:.1f} zJ/MAC",
             transform=axb.get_yaxis_transform(), ha="left", va="top",
             fontsize=7.6, color=C_CYAN)
else:
    axb.axhline(e_landauer, color=C_INK, lw=1.15, zorder=3)
    axb.text(0.985, e_landauer * 1.28,
             f"Landauer limit ({B_STAR:g}-bit ENOB)",
             transform=axb.get_yaxis_transform(), ha="right", va="bottom",
             fontsize=8, color=C_INK)
    axb.axhline(e_thermo, color=C_CYAN, lw=1.35, ls=(0, (3.2, 1.4)), zorder=3)
    axb.text(0.015, e_thermo * 0.74, "Thermodynamic limit",
             transform=axb.get_yaxis_transform(), ha="left", va="top",
             fontsize=8, color=C_CYAN)
axb.set_ylim(3e-21, 2.1e-19)
axb.set_yticks([1e-19, 1e-20])
axb.set_xlabel("Input size, $N$", fontsize=9)

d = 0.012          # axis-break marks
for a, ys in ((ax, 0.0), (axb, 1.0)):
    for x0 in (0.0, 1.0):
        a.plot([x0 - d, x0 + d], [ys - 2.2 * d, ys + 2.2 * d],
               transform=a.transAxes, color="k", lw=0.8, clip_on=False)
fig.supylabel("Energy per real MAC (J/MAC)", fontsize=9, x=0.012)

for ext in ("pdf", "svg"):
    fig.savefig(os.path.join(FIG_OUT, f"fig5_energy_scaling.{ext}"))
fig.savefig(os.path.join(FIG_OUT, "fig5_energy_scaling.png"), dpi=600)
plt.close(fig)

print("\nwrote:")
for f in ("figures/fig5_energy_scaling.pdf", "figures/fig5_energy_scaling.svg",
          "figures/fig5_energy_scaling.png", "data/fig5_plot_data.npz",
          "data/fig5_plot_data_curves.csv"):
    print("  " + f)
