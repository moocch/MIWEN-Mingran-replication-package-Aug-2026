#!/usr/bin/env python3
"""verify_table1.py -- Supplementary Table 1 (scalar PIML-twin parameters).

Asserts every displayed value of Supplementary Table 1 (identical to the red
table in the response letter, Main.tex) and the four quoted text numbers
(3.94 dB physics-only RMSE, 1.30 dB full-twin RMSE, R^2 = 0.9969, 90.4%
within +/-2 dB) against the archived calibration chain in
upstream_scalar_PIML_calibration/.

Sources used:
  * twin_training_run_PIML_Digital_Twin_2/summary.json
      -> the 7 trainable physics parameters, the structural turn-on shape,
         and the two RMSE metrics.
  * twin_predictions.npz  (measured + predicted surfaces exported by the
      training run) -> R^2 and the within-2-dB fraction, which the training
      run did not store in summary.json and are recomputed here, plus the
      1,849-point / 43x43 grid claim.

"Rounded as displayed" means: format the full-precision source value with
the same number of decimals as printed in the table/text and require an
exact string match.

Requires: numpy.  Run:  python verify_table1.py
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import json
import sys
from pathlib import Path

import numpy as np

HERE = _data_dir(__file__)
CHAIN = HERE / "upstream_scalar_PIML_calibration"

summary_path = CHAIN / "twin_training_run_PIML_Digital_Twin_2" / "summary.json"
S = json.loads(summary_path.read_text())
P = S["physics_params"]
M = S["metrics_db"]

results = []


def check(label, actual, displayed, nd):
    """Pass iff `actual`, rounded to nd decimals as displayed, equals `displayed`."""
    got = f"{actual:.{nd}f}"
    want = f"{displayed:.{nd}f}"
    ok = got == want
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: displayed {want}, "
          f"source {actual!r} -> rounds to {got}")


def check_str(label, actual, expected):
    ok = actual == expected
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: displayed {expected!r}, source {actual!r}")


print("== Supplementary Table 1: the 7 trainable physics parameters + 1 structural choice ==")
check("#1 G_max,dB      = -1.84 dB", P["G_max_dB"], -1.84, 2)
check("#2 P_LO,sat      = +1.73 dBm", P["P_LO_sat_dBm"], 1.73, 2)
check("#3 beta_LO       = 1.045", P["beta_LO"], 1.045, 3)
check("#4 P_RF,comp     = +4.47 dBm", P["P_RF_comp_dBm"], 4.47, 2)
check("#5 beta_RF       = 2.13", P["beta_RF"], 2.13, 2)
check("#6 P_noise       = -73.62 dBm", P["P_noise_dBm"], -73.62, 2)
check("#7 leak_dB       = -84.25 dB", P["leak_dB"], -84.25, 2)
check_str("#8 turn-on shape = Weibull", P["lo_shape"], "weibull")

print()
print("== Quoted text numbers (Supplementary Note 1 / response letter) ==")
check("physics-only RMSE = 3.94 dB", M["rmse_physics_only"], 3.94, 2)
check("full-twin RMSE    = 1.30 dB", M["rmse_physics_plus_nn"], 1.30, 2)

# R^2 and the within-2-dB fraction are not stored in summary.json; recompute
# them from the exported measured/predicted surfaces (dB domain, all points).
D = np.load(CHAIN / "twin_predictions.npz")
meas_db = 10.0 * np.log10(D["p_if_meas_w"] * 1e3)          # dBm, measured
full_db = 10.0 * np.log10(D["p_if_full_w"].astype(np.float64) * 1e3)
res = meas_db - full_db

n_points = meas_db.size
ok = n_points == 1849 and meas_db.shape == (43, 43)
results.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  grid = 43 x 43 = 1,849 points: "
      f"shape {meas_db.shape}, n = {n_points}")

rmse_npz = float(np.sqrt(np.mean(res ** 2)))
ok = abs(rmse_npz - M["rmse_physics_plus_nn"]) < 1e-6
results.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  consistency: RMSE from twin_predictions.npz "
      f"({rmse_npz:.6f}) == summary.json ({M['rmse_physics_plus_nn']:.6f})")

r2 = float(1.0 - np.sum(res ** 2) / np.sum((meas_db - meas_db.mean()) ** 2))
check("R^2 = 0.9969 (dB domain)", r2, 0.9969, 4)

within2 = float(100.0 * np.mean(np.abs(res) <= 2.0))
check("within +/-2 dB = 90.4 %", within2, 90.4, 1)

# Informational (not asserted): supporting numbers quoted in the letter.
print()
print("-- informational --")
print(f"      within +/-1 dB          : {100.0 * np.mean(np.abs(res) <= 1.0):.1f} % "
      "(letter quotes 67%)")
print(f"      measured dynamic range  : {meas_db.max() - meas_db.min():.2f} dB "
      "(text quotes 77 dB)")

print()
n_fail = results.count(False)
if n_fail:
    print(f"RESULT: {n_fail} of {len(results)} checks FAILED")
    sys.exit(1)
print(f"RESULT: all {len(results)} checks PASSED")
