# -*- coding: utf-8 -*-
"""Verify every number of Supplementary Note 3 (client-side energy accounting
of the measured operating points, main-text Fig. 3g) from the archived raw
campaign bytes in ./raw/ plus the documented package conventions.

Data: raw/gr_fig3c_ip_scatter_20260810_011915_N4096/gr_fig3c_ip_scatter.npz
      raw/gr_fig3c_ip_scatter_20260810_002043_N65536/gr_fig3c_ip_scatter.npz
(byte-identical to fig5_energy_package/data/raw/<run>/ in
 V3_Manu/Manuscript_Reproductivity/fig3/g, MD5 f9b38769... / 3ff3c54f...).

Conventions (fig5_energy_package METHODS.md, benchmark of Gao et al. 2026):
  P_x = p_rf_dbm_tx - rf_atten_db (30-dB instrumentation pad, meta_json)
  T_ip = n_slots*(fft_len+cp_len)/fs/200,  n_slots = (frame-gaps)/(fft+cp)
  e1 = P_x*T_ip/(4N*0.1); e2 = 6 pJ/(4N); e3 = 8 pJ/(4N);  H100 = 70 fJ/MAC
Panel index 1 = the 25-dB captures (the priced operating points).

Run:  python verify_note3.py     (numpy only; exits nonzero on any mismatch)
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
import os

import numpy as np

HERE = str(_data_dir(__file__))
RUNS = {
    4096: os.path.join(HERE, "raw",
                       "gr_fig3c_ip_scatter_20260810_011915_N4096",
                       "gr_fig3c_ip_scatter.npz"),
    65536: os.path.join(HERE, "raw",
                        "gr_fig3c_ip_scatter_20260810_002043_N65536",
                        "gr_fig3c_ip_scatter.npz"),
}

ETA_RADIO = 0.10          # benchmark radiated-to-wall-plug efficiency
E_ADC = 1e-12             # J per ADC conversion (benchmark constant)
E_DIG = 1e-12             # J per digital real MAC (benchmark constant)
H100 = 70e-15             # J per real MAC, GPU arithmetic-energy reference
R5 = 0.0625               # five-bit criterion RMSE < 2^(1-5)
KB = 1.380649e-23
T0 = 300.0

n_pass = 0


def check(name, ok, detail=""):
    global n_pass
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    assert ok, name
    n_pass += 1


# ------------------------------------------------------------------ expected
EXP = {  # per N: quoted SI Note 3 values
    4096: dict(px_dbm=-62.81, px_nw=0.524, t_ip_ms=1.918,
               e1=0.613, e2=0.366, e3=0.488, tot=1.467, x=47.7,
               rmse_m=0.0609, rmse_sd=0.0003, std_y=0.3395,
               rmse15=0.0672, corr_tot=2.444, corr_x=28.6),
    65536: dict(px_dbm=-63.41, px_nw=0.457, t_ip_ms=30.68,
                e1=0.534, e2=0.023, e3=0.031, tot=0.588, x=119.0,
                rmse_m=0.0561, rmse_sd=0.0006, std_y=0.3343,
                rmse15=0.0643, corr_tot=0.649, corr_x=108.0),
}

enob_crit = {}
e1_by_n = {}
tot_by_n = {}

for N, path in RUNS.items():
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    E = EXP[N]
    print(f"\n===== N = {N}  ({os.path.relpath(path, HERE)}) =====")

    check(f"N={N}: vec_N field", int(d["vec_N"]) == N)
    check(f"N={N}: 25-dB panel is index 1", str(d["labels"][1]) == "25 dB")

    # --- P_x: transmit-port power minus the 30-dB instrumentation pad -----
    pad = float(meta["rf_atten_db"])
    check(f"N={N}: rf_atten_db = 30 in meta_json", pad == 30.0)
    px_dbm = float(d["p_rf_dbm_tx"][1]) - pad
    px_w = 1e-3 * 10.0 ** (px_dbm / 10.0)
    check(f"N={N}: P_x = {E['px_dbm']} dBm", abs(px_dbm - E["px_dbm"]) < 5e-3,
          f"measured {px_dbm:.4f} dBm")
    check(f"N={N}: P_x = {E['px_nw']} nW", abs(px_w * 1e9 - E["px_nw"]) < 5e-4,
          f"measured {px_w*1e9:.4f} nW")
    check(f"N={N}: LO (weight) comb at -3 dBm",
          np.allclose(d["p_lo_dbm_tx"], -3.0))

    # --- T_ip from the archived frame geometry ----------------------------
    fft, cp = int(d["fft_len"]), int(d["cp_len"])
    gap0, gap1 = int(d["gap0"]), int(d["gap1"])
    frame, fs = int(d["frame_len"]), float(d["fs_hz"])
    n_slots = (frame - gap0 - gap1) // (fft + cp)
    check(f"N={N}: frame carries 227 (fft+cp) bursts", n_slots == 227,
          f"(1 preamble + {int(d['n_pilot'])} pilot + {int(d['n_data'])} data slots)")
    t_ip = n_slots * (fft + cp) / fs / 200.0
    check(f"N={N}: T_ip = {E['t_ip_ms']} ms", abs(t_ip * 1e3 - E["t_ip_ms"]) < 5e-3,
          f"computed {t_ip*1e3:.4f} ms")
    t_gap = frame / fs / 200.0     # stricter variant charging both guard gaps
    check(f"N={N}: gap-charging variant +0.43%, no quoted change",
          abs(t_gap / t_ip - 1.00427) < 2e-4,
          f"{t_gap*1e3:.3f} ms vs {t_ip*1e3:.3f} ms")

    # --- 117 ns of airtime per real MAC, identical at both sizes ----------
    per_mac_ns = t_ip / (4 * N) * 1e9
    check(f"N={N}: airtime per real MAC = 117 ns", abs(per_mac_ns - 117.05) < 0.1,
          f"computed {per_mac_ns:.2f} ns")

    # --- the three bills e1/e2/e3 and the total ---------------------------
    e1 = px_w * t_ip / (4 * N * ETA_RADIO)
    e2 = 6 * E_ADC / (4 * N)
    e3 = 8 * E_DIG / (4 * N)
    tot = e1 + e2 + e3
    for tag, val, exp in (("e1", e1, E["e1"]), ("e2", e2, E["e2"]),
                          ("e3", e3, E["e3"]), ("e_ip", tot, E["tot"])):
        check(f"N={N}: {tag} = {exp} fJ/MAC", abs(val * 1e15 - exp) < 1e-3,
              f"computed {val*1e15:.4f} fJ")
    check(f"N={N}: {E['x']}x below the 70-fJ H100 reference",
          abs(H100 / tot - E["x"]) < 0.1, f"computed {H100/tot:.2f}x")
    e1_by_n[N], tot_by_n[N] = e1, tot

    # --- accuracy criterion (25-dB repeats meet it; 15-dB points miss) ----
    rm, rs = float(d["rmse_mean"][1]), float(d["rmse_sd"][1])
    check(f"N={N}: RMSE mean = {E['rmse_m']}", abs(rm - E["rmse_m"]) < 5e-5,
          f"measured {rm:.5f}")
    check(f"N={N}: RMSE s.d. = {E['rmse_sd']}", abs(rs - E["rmse_sd"]) < 5e-5,
          f"measured {rs:.5f}")
    check(f"N={N}: all three 25-dB repeats satisfy RMSE < 0.0625",
          bool(np.all(d["rmse_reps"][1] < R5)),
          "reps " + np.array2string(d["rmse_reps"][1], precision=4))
    check(f"N={N}: 15-dB point misses the criterion (RMSE {E['rmse15']})",
          float(d["rmse_mean"][0]) > R5
          and abs(float(d["rmse_mean"][0]) - E["rmse15"]) < 5e-4,
          f"measured {float(d['rmse_mean'][0]):.4f}")

    # --- std_y and the criterion-equivalent ENOB --------------------------
    sy = float(d["std_y"][1])
    check(f"N={N}: std_y = {E['std_y']}", abs(sy - E["std_y"]) < 5e-5,
          f"measured {sy:.5f}")
    enob_crit[N] = math.log2(sy / R5)

    # --- closed-loop tuner: 0.4-dB tolerance, 2-3 iteration convergence ---
    # (verifiable from archived bytes: meta_json carries the tolerance, and
    #  tune_history_json is the per-iteration trace of the session itself)
    check(f"N={N}: tuner tolerance 0.4 dB in meta_json",
          float(meta["tune_tol_db"]) == 0.4)
    hist = json.loads(str(d["tune_history_json"]))
    iters = [len(h) for h in hist]
    check(f"N={N}: tuner converged within 2-3 iterations (both panels)",
          all(2 <= it <= 3 for it in iters), f"iterations per panel {iters}")
    finals = [abs(h[-1]["err_db"]) for h in hist]
    check(f"N={N}: final captures within the 0.4-dB tolerance",
          all(e <= 0.4 for e in finals),
          "final |err_db| " + str([f"{e:.3f}" for e in finals]))
    check(f"N={N}: three repeats at frozen power",
          bool(np.all(d["n_repeats"] == 3)))

print("\n===== cross-size and reference-line quantities =====")
check("criterion-equivalent ENOB range 2.42-2.44 bits",
      round(min(enob_crit.values()), 2) == 2.42
      and round(max(enob_crit.values()), 2) == 2.44,
      f"log2(std_y/0.0625) = {enob_crit[65536]:.4f} (N=65536), "
      f"{enob_crit[4096]:.4f} (N=4096)")

n_bend = (6 * E_ADC + 8 * E_DIG) / (4 * e1_by_n[65536])
check("N_bend = 14 pJ / (4 e1) ~ 6.5e3", 6.4e3 < n_bend < 6.6e3,
      f"computed {n_bend:.0f}")

model_floor = e1_by_n[65536]
model_4096 = model_floor + 14e-12 / (4 * 4096)   # Eq. S10: e1 + 3.5 pJ/N
check("Eq. S10 model curve: N=4096 point sits ~6% above it",
      0.04 < tot_by_n[4096] / model_4096 - 1 < 0.08,
      f"excess {100*(tot_by_n[4096]/model_4096-1):.1f}%")

e_land = 2 ** 2 * KB * T0 * math.log(2)
e_thermo = 2 * 2 * KB * T0 * math.log(2)
check("Landauer = thermodynamic bound = 11.48 zJ at b=2 (algebraic identity)",
      abs(e_land * 1e21 - 11.48) < 5e-3 and e_land == e_thermo,
      f"computed {e_land*1e21:.4f} zJ")
headroom = tot_by_n[65536] / e_land
check("measured N=65536 point sits ~5.1e4 above the b=2 floor",
      5.0e4 < headroom < 5.2e4, f"computed {headroom:.3e}")

# --- deploying the twin correction: +4 complex coeffs = 16 real MACs -----
for N, (exp_tot, exp_x) in ((65536, (0.649, 108.0)), (4096, (2.444, 28.6))):
    tot_c = tot_by_n[N] + 16 * E_DIG / (4 * N)
    check(f"N={N}: with correction deployed e_ip = {exp_tot} fJ/MAC",
          abs(tot_c * 1e15 - exp_tot) < 1e-3, f"computed {tot_c*1e15:.4f} fJ")
    check(f"N={N}: corrected point {exp_x}x below H100",
          abs(H100 / tot_c - exp_x) < 0.15, f"computed {H100/tot_c:.2f}x")

print(f"\nAll {n_pass} checks passed: every Supplementary Note 3 number is"
      "\nreproduced from the archived raw npz bytes + package conventions.")
