# -*- coding: utf-8 -*-
"""
verify_note5.py -- asserts every derivable number of Supplementary Note 5
(the analog activation block: booster, detector, squaring derivation
Eqs. S15-S18, level plan S5.6, per-layer fidelity S5.8).

Sources
-------
tex     : 202607_MIWEN_Manuscript/Supp_M/Supplementary_Information.tex,
          Supplementary Note 5 (lines ~819-1039).
archive : ../../_shared_fully_analog_simulation  (READ-ONLY; byte-identical
          to code/results/* inside miwen_fully_analog_archive.zip of
          202607_MIWEN_Manuscript/V2/fully analog/files (1).zip -- verified
          by sha256 for results_summary.json and link_budget.json).
          - results/link_budget.json    : drive ladder, tone SNRs, block
                                          efficiencies, physics settings
          - results/results_summary.json: per-tag accuracies (ZD_L3, ZI_L3,
                                          ZH_L3, ZT_L3d, ...)
          - comb_analog_sim.py          : the constants' in-code provenance
            (CL_MIXER=6.65 ZEM-4300+, IL_FILT=0.82 LFCN-490+, BOOST_DB=18.5,
             K_ED=208.7, R_V=3500, IL_STEP=1.0, P_N_DBM=-73.6, NOISE_C=27)

Published (non-derivable) constants asserted only for identity with the
archive settings; their datasheet/paper provenance is in README.md:
  6.65 dB  ZEM-4300+ conversion loss          (Mini-Circuits datasheet)
  0.82 dB  LFCN-490+ insertion loss           (Mini-Circuits datasheet)
  18.5 dB  passive boost at 400 MHz           (Wang et al. ESSCIRC 2017,
                                               bib key mercier2017esscirc)
  208.7/V  pseudo-balun detector k_ED         (Wang et al. SSC-L 2018,
                                               bib key wang2018sscl)
  -73.6 dBm per-bin noise floor               (calibrated twin, as published)

Run:  python verify_note5.py
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import json
import math
import sys
from pathlib import Path

HERE = _data_dir(__file__)
ARCHIVE = HERE.parents[1] / "_shared_fully_analog_simulation"
OLD_LETTER_JSON = Path(
    r"E:\archive\Manuscript\Overleaf"
    r"\202607_MIWEN_v2\Response_Letter_Reproductivity\Reviewer1_Comment7"
    r"\b_activation_block_note\c7_activation_numbers.json")

results = []


def check(label, got, want, tol=0.0):
    ok = abs(got - want) <= tol if tol else got == want
    results.append((ok, label, got, want, tol))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}\n       got {got!r}  vs tex/archive {want!r}"
          + (f"  (tol {tol})" if tol else ""))
    return ok


def round_half_up(x, nd=1):
    """Manuscript-style rounding (42.15 -> 42.2), immune to float round()."""
    f = 10 ** nd
    return math.floor(x * f + 0.5) / f


print("=" * 72)
print("Supplementary Note 5 -- numeric verification")
print("archive:", ARCHIVE)
print("=" * 72)

lb = json.load(open(ARCHIVE / "results" / "link_budget.json"))
rs = json.load(open(ARCHIVE / "results" / "results_summary.json"))
S = lb["settings"]

# ---------------------------------------------------------------- S5.1 ----
print("\n--- S5.1  per-pass loss (datasheet constants) ---")
CL_MIXER, IL_FILT = 6.65, 0.82          # ZEM-4300+ CL, LFCN-490+ IL
check("6.65 + 0.82 = 7.47 dB per-pass loss", CL_MIXER + IL_FILT, 7.47, 1e-9)
check("archive settings cl_pass_dB", S["cl_pass_dB"], 7.47)

# ---------------------------------------------------------------- S5.2 ----
print("\n--- S5.2  Eq. S15 passive boost bound ---")
Q_IND, C_IN, F0, R_S, IL_XFMR = 40.0, 1.5e-12, 0.40e9, 50.0, 1.0
R_Q = Q_IND / (2 * math.pi * F0 * C_IN)                     # R_in <= Q/(w C)
check("R_in <= Q/(w C) = 10.6 kOhm", R_Q / 1e3, 10.6, 0.05)
boost_bound = 10 * math.log10(R_Q / (2 * R_S)) - IL_XFMR    # A_v^2 = Rin/2Rs
check("boost bound 19.3 dB after 1 dB IL", boost_bound, 19.3, 0.05)
check("adopted boost (archive) = 18.5 dB <= bound",
      float(S["boost_dB"] <= boost_bound), 1.0)
check("archive settings boost_dB", S["boost_dB"], 18.5)

# ---------------------------------------------------------------- S5.3/4 --
print("\n--- S5.3/S5.4  Eq. S16 detector transfer and knee ---")
K_ED = S["k_ED_per_V"]
check("archive k_ED_per_V", K_ED, 208.7)
v_knee = 1.0 / (2 * K_ED)
check("knee v* = 1/(2 k_ED) = 2.4 mV", v_knee * 1e3, 2.4, 0.005)
# small-signal limit of Eq. S16: 0.5 v (1 - exp(-2 k v)) -> k v^2
v = 1e-6
ratio = 0.5 * v * (1 - math.exp(-2 * K_ED * v)) / (K_ED * v * v)
check("Eq. S16 small-signal limit -> k_ED v^2 (ratio -> 1)", ratio, 1.0, 1e-3)

# ---------------------------------------------------------------- S5.5 ----
print("\n--- S5.5  Eq. S18 output interface ---")
R_0 = 50.0
R_v_formula = R_0 * 10 ** 1.85          # 50 * 10^(18.5/10)
check("R_v = 50 * 10^1.85 ~ 3.5 kOhm", R_v_formula / 1e3, 3.5, 0.05)
R_V = S["R_v_ohm"]
check("archive R_v_ohm = 3500 (conservative ~3.5 kOhm)", R_V, 3500.0)
eta = 4 * R_0 * R_V / (R_0 + R_V) ** 2
check("eta = 4 R0 Rv/(R0+Rv)^2 ~ 0.056", eta, 0.056, 5e-4)
check("eta in dB ~ -12.5", 10 * math.log10(eta), -12.5, 0.06)
check("n^2 ~ R_v/R_0 ~ 70", R_V / R_0, 70.0, 0.0)
check("archive step_down_loss_dB (the 'further 1 dB')",
      S["step_down_loss_dB"], 1.0)

# ---------------------------------------------------------------- S5.6 ----
print("\n--- S5.6  block efficiency and the level ladder ---")
effM = lb["MNIST"]["block_efficiency"]
effG = lb["GTSRB"]["block_efficiency"]
check("MNIST first-pass block efficiency 2.255% (tex '2.3%')",
      round_half_up(effM[0]), 2.3, 0.0)
check("GTSRB first-pass block efficiency 2.091% (tex '2.1%')",
      round_half_up(effG[0]), 2.1, 0.0)
check("archived MNIST block_efficiency[0]", effM[0], 2.255)
check("archived GTSRB block_efficiency[0]", effG[0], 2.091)

drivesM = [r["drive_dBm"] for r in lb["MNIST"]["rows"]]
drivesG = [r["drive_dBm"] for r in lb["GTSRB"]["rows"]]
snrG = [r["tone_snr_dB"] for r in lb["GTSRB"]["rows"]]
# main-text ladder -10 -> -34 -> -58 dBm (both planning dimensions round so)
for name, d in (("MNIST", drivesM), ("GTSRB", drivesG)):
    check(f"{name} ladder rounds to -10/-34/-58 dBm",
          (round(d[0]), round(d[1]), round(d[2])) == (-10, -34, -58), True)
check("fourth pass near -84 dBm (MNIST row 4)", drivesM[3], -84.0, 0.5)
P_FLOOR = -73.6
check("MNIST 4th pass below -73.6 dBm floor",
      float(drivesM[3] < P_FLOOR), 1.0)
check("GTSRB 4th pass below -73.6 dBm floor",
      float(drivesG[3] < P_FLOOR), 1.0)

# block-efficiency arithmetic is exactly eta_block = P_{l+1}/(P_l * 10^-.747)
for name, d, eff in (("MNIST", drivesM, effM), ("GTSRB", drivesG, effG)):
    for i in range(3):
        e = 100 * 10 ** ((d[i + 1] - d[i] + 7.47) / 10)
        check(f"{name} eff[{i}] = next/(prev * 10^(-7.47/10))",
              e, eff[i], 0.005)

# the -73.6 dBm floor is the constant of the archived tone-SNR law:
# SNR = 10 log10(27 * P / (D * P_n)),  P_n = 1 mW * 10^(-7.36)
P_n = 10 ** (P_FLOOR / 10)
for name in ("MNIST", "GTSRB"):
    for r in lb[name]["rows"]:
        snr = 10 * math.log10(27 * 10 ** (r["drive_dBm"] / 10)
                              / (r["dim_in"] * P_n))
        check(f"{name} pass {r['pass_index']} SNR law with -73.6 dBm floor",
              snr, r["tone_snr_dB"], 0.01)

# ---------------------------------------------------------------- S5.7 ----
print("\n--- S5.7  discrete-Schottky boundary (SMS7630) ---")
n_ideal, nVT = 1.05, 0.027              # tex: n V_T ~ 27 mV
k_schottky = 1.0 / (4 * nVT)            # 1/(4 n V_T)
check("SMS7630 k_ED = 1/(4 n V_T) ~ 9.3 /V (n V_T = 27 mV)",
      k_schottky, 9.3, 0.05)
VT300 = 1.380649e-23 * 300.0 / 1.602176634e-19
check("n V_T at n = 1.05, T = 300 K ~ 27 mV", n_ideal * VT300 * 1e3,
      27.0, 0.2)
check("video deficit 20 log10(208.7/9.3) = 27 dB",
      20 * math.log10(208.7 / 9.3), 27.0, 0.05)
# 5.1-kOhm junction resistance caps Eq. S15 near 16 dB regardless of Q
R_j = 5.1e3
cap = 10 * math.log10(min(R_Q, R_j) / (2 * R_S)) - IL_XFMR
check("R_j = 5.1 kOhm caps boost near 16 dB", cap, 16.0, 0.1)

# ---------------------------------------------------------------- S5.8 ----
print("\n--- S5.8  per-layer fidelity: quoted drives/SNRs/accuracies ---")
for i, (d_tex, s_tex) in enumerate([(-10.0, 43.0), (-34.3, 32.6),
                                    (-58.5, 8.4)]):
    check(f"GTSRB pass {i+1} drive {d_tex} dBm", drivesG[i], d_tex, 0.05)
    check(f"GTSRB pass {i+1} tone SNR {s_tex} dB", snrG[i], s_tex, 0.05)

check("ZD_L3 (chain-trained Z_L3 weights, digital execution) = 88.29 %",
      rs["ZD_L3"]["accuracy_pct"], 88.29)
check("  ... rounds to the tex's 88.3 %",
      round_half_up(rs["ZD_L3"]["accuracy_pct"]), 88.3)
check("ZI_L3 (same weights, ideal square law) = 42.15 %",
      rs["ZI_L3"]["accuracy_pct"], 42.15)
check("  ... rounds to the tex's 42.2 % (half-up)",
      round_half_up(rs["ZI_L3"]["accuracy_pct"]), 42.2)
check("ZD_L3 and ZI_L3 run the same weights (Z_L3)",
      (rs["ZD_L3"]["weights"], rs["ZI_L3"]["weights"]) == ("Z_L3", "Z_L3"),
      True)
check("'within 1.30 points of digital execution': ZT_L3d - ZH_L3",
      rs["ZT_L3d"]["accuracy_pct"] - rs["ZH_L3"]["accuracy_pct"], 1.30, 1e-9)

# ------------------------------------------------- old-letter cross-check --
if OLD_LETTER_JSON.exists():
    print("\n--- cross-check vs Reviewer1_Comment7 c7_activation_numbers ---")
    c7 = json.load(open(OLD_LETTER_JSON))
    check("c7 v_knee_mV", c7["v_knee_mV"], v_knee * 1e3, 1e-9)
    check("c7 boost_ceiling_dB", c7["boost_ceiling_dB"], boost_bound, 1e-9)
    check("c7 R_Q_kOhm", c7["R_Q_kOhm"], R_Q / 1e3, 1e-9)
    check("c7 GTSRB ladder == archived ladder",
          [round(x, 2) for x in c7["ladder_L3_dbm"]] == drivesG[:3], True)
    check("c7 block_efficiency == archived",
          [round(x, 3) for x in c7["block_efficiency_pct"]] == effG, True)
else:
    print("\n(old-letter c7_activation_numbers.json not found -- skipped)")

# ------------------------------------------------------------------ tally --
print("\n" + "=" * 72)
n_fail = sum(1 for ok, *_ in results if not ok)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
if n_fail:
    for ok, label, got, want, tol in results:
        if not ok:
            print(f"  FAILED: {label}: got {got!r}, want {want!r}")
    sys.exit(1)
print("ALL CHECKS PASSED")
