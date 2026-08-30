#!/usr/bin/env python3
"""verify_note1.py -- vector-twin numbers of Supplementary Note 1 (S1.3).

Asserts the numbers quoted in Supplementary Note 1, S1.3 (transfer to
frequency-multiplexed vectors, N = 2-8) against the archived training record
vector_run_summary.json (copied from
V3_Manu/Manuscript_Reproducibility/fig2/j/):

  c_LO = 0.885, c_RF = -0.040, |delta_NN| <= 6 dB,
  RMSE 1.91 / 1.67 / 1.55 dB at N = 2 / 4 / 8.

It also checks the S1.3 claim that the seven scalar physics parameters are
frozen: the n1_physics_params block carried in vector_run_summary.json must
round to exactly the Supplementary Table 1 values.

The Table 1 and Table 2 numbers themselves are asserted by
../../supp_tables/table1_scalar_twin_parameters/verify_table1.py and
../../supp_tables/table2_twin4096_parameters/verify_table2.py.

Requires: Python 3 standard library only.  Run:  python verify_note1.py
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

HERE = _data_dir(__file__)
S = json.loads((HERE / "vector_run_summary.json").read_text())

results = []


def check(label, actual, displayed, nd):
    got = f"{actual:.{nd}f}"
    want = f"{displayed:.{nd}f}"
    ok = got == want
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}: displayed {want}, "
          f"source {actual!r} -> rounds to {got}")


V = S["vector_papr_params"]
M = S["metrics_db"]
C = S["config"]

print("== Note 1, S1.3: vector twin (N = 2-8) ==")
check("c_LO  = 0.885", V["c_papr_lo"], 0.885, 3)
check("c_RF  = -0.040", V["c_papr_rf"], -0.040, 3)
check("|delta_NN| cap = 6 dB (config nn_max_db)", C["nn_max_db"], 6.0, 1)
check("RMSE(N=2) = 1.91 dB (full twin)", M["2"]["full"]["RMSE_dB"], 1.91, 2)
check("RMSE(N=4) = 1.67 dB (full twin)", M["4"]["full"]["RMSE_dB"], 1.67, 2)
check("RMSE(N=8) = 1.55 dB (full twin)", M["8"]["full"]["RMSE_dB"], 1.55, 2)

print()
print("== S1.3 'seven parameters frozen': n1_physics_params == Table 1 values ==")
P = S["n1_physics_params"]
check("frozen G_max,dB   = -1.84 dB", P["G_max_dB"], -1.84, 2)
check("frozen P_LO,sat   = +1.73 dBm", P["P_LO_sat_dBm"], 1.73, 2)
check("frozen beta_LO    = 1.045", P["beta_LO"], 1.045, 3)
check("frozen P_RF,comp  = +4.47 dBm", P["P_RF_comp_dBm"], 4.47, 2)
check("frozen beta_RF    = 2.13", P["beta_RF"], 2.13, 2)
check("frozen P_noise    = -73.62 dBm", P["P_noise_dBm"], -73.62, 2)
check("frozen leak_dB    = -84.25 dB", P["leak_dB"], -84.25, 2)
ok = P["lo_shape"] == "weibull"
results.append(ok)
print(f"{'PASS' if ok else 'FAIL'}  frozen turn-on shape = Weibull: source {P['lo_shape']!r}")

print()
n_fail = results.count(False)
if n_fail:
    print(f"RESULT: {n_fail} of {len(results)} checks FAILED")
    sys.exit(1)
print(f"RESULT: all {len(results)} checks PASSED")
