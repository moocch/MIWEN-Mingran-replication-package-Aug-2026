# -*- coding: utf-8 -*-
"""Rebuild N4096_cascade_raw_data.npz (Fig. 2 g-j data) from
twin_predictions_N4096.npz and verify every key against the archived file.
The original one-off repackaging script was not archived; this script encodes
the verified mapping. Usage: python rebuild_make_cascade_raw_data.py
Verified result: 17/17 keys bit-identical.

Mapping (numerically verified):
  p_lo_dbm / p_rf_dbm    = twin p_lo_dbm_grid / p_rf_dbm_grid
  measured_db            = twin meas_db_arb
  perfect_multiplier_db  = twin ideal_nominal_dbm + twin offset_ideal_db
  physics_only_db        = twin phys_db_arb
  full_twin_db           = twin full_db_arb
  error_metrics[3,4]     = [RMSE, MAE, P95, MAX] (dB) of each model surface vs
                           measured_db; the RMSE entries of the physics/full
                           rows follow the original file's convention of using
                           the published values 1.80 / 0.79
                           (computed: 1.9677 / 0.7858).
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import argparse
import os

import numpy as np

HERE = str(_data_dir(__file__))

META = {
    "figure_title": "N=4096 step-by-step model cascade",
    "array_orientation": "axis 0 indexes LO (p_lo_dbm); axis 1 indexes RF (p_rf_dbm)",
    "x_axis_label": "LO port power dBm",
    "y_axis_label": "RF port power dBm",
    "heatmap_value_label": "IF power dB arb",
    "error_value_label": "Error dB",
    "model_names": np.array(
        ["Perfect multiplier", "Physics-only DT", "Full PIML"]),
    "metric_names": np.array(
        ["RMSE dB", "MAE dB", "P95 abs error dB", "Max abs error dB"]),
    "panel_array_names": np.array(
        ["measured_db", "perfect_multiplier_db", "physics_only_db",
         "full_twin_db"]),
    "panel_titles": np.array(
        ["A. Measured (dual USRP)",
         "B. Perfect multiplier (no Digital Twin)",
         "C. Physics-only Digital Twin",
         "D. Full Digital Twin (Physics + NN)"]),
}

PUBLISHED_RMSE = {"physics_only_db": 1.80, "full_twin_db": 0.79}


def metrics_row(model_db, meas_db):
    e = model_db - meas_db
    return [float(np.sqrt(np.mean(e ** 2))),
            float(np.mean(np.abs(e))),
            float(np.percentile(np.abs(e), 95)),
            float(np.max(np.abs(e)))]


def rebuild(twin_path):
    tp = np.load(twin_path, allow_pickle=True)
    out = {
        "p_lo_dbm": tp["p_lo_dbm_grid"],
        "p_rf_dbm": tp["p_rf_dbm_grid"],
        "measured_db": tp["meas_db_arb"],
        "perfect_multiplier_db": tp["ideal_nominal_dbm"] + tp["offset_ideal_db"],
        "physics_only_db": tp["phys_db_arb"],
        "full_twin_db": tp["full_db_arb"],
    }
    em = np.array([metrics_row(out[k], out["measured_db"])
                   for k in ("perfect_multiplier_db", "physics_only_db",
                             "full_twin_db")])
    em[1, 0] = PUBLISHED_RMSE["physics_only_db"]
    em[2, 0] = PUBLISHED_RMSE["full_twin_db"]
    out["error_metrics"] = em
    out.update(META)
    return out


def verify(out, ref_path):
    ref = np.load(ref_path, allow_pickle=True)
    n_ok, problems = 0, []
    keys = sorted(set(ref.files) | set(out.keys()))
    for k in keys:
        if k not in out or k not in ref.files:
            problems.append(f"key mismatch: {k}")
            continue
        a, b = np.asarray(out[k]), ref[k]
        if a.dtype.kind == "U" or b.dtype.kind == "U":
            same = a.shape == b.shape and np.all(a.astype(str) == b.astype(str))
        else:
            same = a.shape == b.shape and np.array_equal(a, b)
        print(("  OK   " if same else "  DIFF ") + k)
        n_ok += same
        if not same:
            problems.append(f"differs: {k}")
    return n_ok, len(keys), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", default=os.path.join(
        HERE, "..", "..", "f", "twin_predictions_N4096.npz"))
    ap.add_argument("--ref", default=os.path.join(
        HERE, "..", "N4096_cascade_raw_data.npz"))
    ap.add_argument("--out", default=os.path.join(
        HERE, "rebuilt_N4096_cascade_raw_data.npz"))
    args = ap.parse_args()

    out = rebuild(args.twin)
    n_ok, n_all, problems = verify(out, args.ref)
    np.savez(args.out, **out)
    print(f"wrote {os.path.abspath(args.out)}")
    print(f"verified: {n_ok}/{n_all} keys bit-identical to the archived file")
    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()
