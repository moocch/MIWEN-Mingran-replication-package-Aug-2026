#!/usr/bin/env python3
"""verify_table2.py -- Supplementary Table 2 (N = 4096 PIML-twin parameters).

Asserts the 15 displayed physics-parameter values of Supplementary Table 2
(identical to the red table in the response letter, Main.tex) and the four
quoted text metrics (0.79 dB full-twin RMSE, 1.80 dB physics-only,
10.35 dB ideal multiplier, 96% within +/-2 dB) against the archived
calibration record in upstream_measured_data_and_twin_code/.

Sources used:
  * upstream_measured_data_and_twin_code/summary.json
      (byte-identical to twin_training_run_4096DT_model/summary.json,
       which this script also verifies) -> physics_params + metrics_db.

"Rounded as displayed" means: format the full-precision source value with
the same number of decimals as printed and require an exact string match.

Requires: Python 3 standard library only.  Run:  python verify_table2.py
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import hashlib
import json
import sys
from pathlib import Path

HERE = _data_dir(__file__)
CHAIN = HERE / "upstream_measured_data_and_twin_code"

top = CHAIN / "summary.json"
model = CHAIN / "twin_training_run_4096DT_model" / "summary.json"
S = json.loads(top.read_text())
P = S["physics_params"]
M = S["metrics_db"]

results = []


def check(label, actual, displayed, nd):
    got = f"{actual:.{nd}f}"
    want = f"{displayed:.{nd}f}"
    ok = got == want
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: displayed {want}, "
          f"source {actual!r} -> rounds to {got}")


md5_top = hashlib.md5(top.read_bytes()).hexdigest()
md5_model = hashlib.md5(model.read_bytes()).hexdigest()
ok = md5_top == md5_model
results.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  summary.json (top level) is byte-identical to "
      f"twin_training_run_4096DT_model/summary.json (md5 {md5_top})")
print()

print("== Supplementary Table 2: the 15 calibrated physics parameters ==")
check("#1  G_max,dB   (G)         = +2.73 dB", P["G"], 2.73, 2)
check("#2  P_LO,sat   (PsatL)     = -2.80 dBm", P["PsatL"], -2.80, 2)
check("#3  beta_LO    (betaL)     = 1.85", P["betaL"], 1.85, 2)
check("#4  P_RF,sat   (PsatR)     = -8.02 dBm", P["PsatR"], -8.02, 2)
check("#5  beta_RF    (betaR)     = 2.92", P["betaR"], 2.92, 2)
check("#6  w          (w_hill)    = 1.00", P["w_hill"], 1.00, 2)
check("#7  P_RF,comp  (PcompRF)   = -3.94 dBm", P["PcompRF"], -3.94, 2)
check("#8  P_LO,comp  (PcompLO)   = -5.20 dBm", P["PcompLO"], -5.20, 2)
check("#9  beta_C     (betaC)     = 0.61", P["betaC"], 0.61, 2)
check("#10 kappa                  = -1.49 dB", P["kappa"], -1.49, 2)
check("#11 c_papr,LO  (c_papr_lo) = 0.54", P["c_papr_lo"], 0.54, 2)
check("#12 c_papr,RF  (c_papr_rf) = 0.07", P["c_papr_rf"], 0.07, 2)
check("#13 leak_dB    (leak)      = -97.7 dB", P["leak"], -97.7, 1)
check("#14 C_cal                  = +13.2 dB", P["C_cal"], 13.2, 1)
check("#15 P_floor    (floor)     = -97.6 dB", P["floor"], -97.6, 1)

print()
print("== Quoted text metrics (Supplementary Note 1, S1.4) ==")
check("full-twin RMSE       = 0.79 dB", M["rmse_physics_plus_nn"], 0.79, 2)
check("physics-only RMSE    = 1.80 dB", M["rmse_physics_only"], 1.80, 2)
check("ideal-multiplier RMSE = 10.35 dB", M["rmse_ideal_shape"], 10.35, 2)
check("within +/-2 dB       = 96 %", 100.0 * M["frac_abs_err_le_2db"], 96.0, 0)

print()
print("== Note 1 S1.5: the two frozen constants reused at N = 65,536 ==")
check("P_LO,comp = -5.20 dBm (frozen)", P["PcompLO"], -5.20, 2)
check("beta_C    = 0.610     (frozen)", P["betaC"], 0.610, 3)

print()
n_fail = results.count(False)
if n_fail:
    print(f"RESULT: {n_fail} of {len(results)} checks FAILED")
    sys.exit(1)
print(f"RESULT: all {len(results)} checks PASSED")
