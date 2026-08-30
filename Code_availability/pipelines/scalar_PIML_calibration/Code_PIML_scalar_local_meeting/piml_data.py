# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

from pathlib import Path
import numpy as np

NPZ_PATH = _data_dir(__file__) / "twin_predictions.npz"

PHYSICS_PARAMS = {
    "G_max_dB":      -1.8438,
    "P_LO_sat_dBm":   1.7347,
    "beta_LO":        1.0453,
    "P_RF_comp_dBm":  4.4746,
    "beta_RF":        2.1329,
    "P_noise_dBm":  -73.6208,
    "leak_dB":      -84.2515,
    "LO_shape":      "weibull",
}


def _w_to_dbm(p_w, floor_w=1e-15):
    p = np.maximum(np.asarray(p_w, dtype=np.float64), floor_w)
    return 10.0 * np.log10(p) + 30.0


def load_data(npz_path=None):
    npz = np.load(npz_path or NPZ_PATH)

    p_lo = npz["p_lo_dbm"]
    p_rf = npz["p_rf_dbm"]
    P_LO, P_RF = np.meshgrid(p_lo, p_rf, indexing="ij")

    p_lo_w = 1e-3 * 10.0 ** (P_LO / 10.0)
    p_rf_w = 1e-3 * 10.0 ** (P_RF / 10.0)
    p_if_ideal_w = (p_lo_w * p_rf_w) / 1e-3

    dbm_meas  = _w_to_dbm(npz["p_if_meas_w"])
    dbm_ideal = _w_to_dbm(p_if_ideal_w)
    dbm_phys  = _w_to_dbm(npz["p_if_phys_w"])
    dbm_full  = _w_to_dbm(npz["p_if_full_w"])

    return dict(
        p_lo_axis  = p_lo,
        p_rf_axis  = p_rf,
        P_LO       = P_LO,
        P_RF       = P_RF,
        dbm_meas   = dbm_meas,
        dbm_ideal  = dbm_ideal,
        dbm_phys   = dbm_phys,
        dbm_full   = dbm_full,
        res_ideal  = dbm_ideal - dbm_meas,
        res_phys   = dbm_phys  - dbm_meas,
        res_full   = dbm_full  - dbm_meas,
        delta_NN   = np.asarray(npz["delta_db"]),
    )


def metrics(y_pred_dbm, y_meas_dbm):
    yp = np.asarray(y_pred_dbm).ravel().astype(np.float64)
    yt = np.asarray(y_meas_dbm).ravel().astype(np.float64)
    err = yp - yt
    yt_mean = yt.mean()
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    return dict(
        N              = int(err.size),
        RMSE_dB        = float(np.sqrt(np.mean(err ** 2))),
        MAE_dB         = float(np.mean(np.abs(err))),
        Bias_dB        = float(np.mean(err)),
        sigma_dB       = float(np.std(err)),
        max_abs_err_dB = float(np.max(np.abs(err))),
        p95_abs_err_dB = float(np.percentile(np.abs(err), 95)),
        p99_abs_err_dB = float(np.percentile(np.abs(err), 99)),
        R2             = 1.0 - ss_res / ss_tot,
        Pearson_rho    = float(np.corrcoef(yp, yt)[0, 1]),
        within_1dB_pct = 100.0 * float(np.mean(np.abs(err) <= 1.0)),
        within_2dB_pct = 100.0 * float(np.mean(np.abs(err) <= 2.0)),
        within_3dB_pct = 100.0 * float(np.mean(np.abs(err) <= 3.0)),
        within_5dB_pct = 100.0 * float(np.mean(np.abs(err) <= 5.0)),
    )


if __name__ == "__main__":
    d = load_data()
    print("Loaded twin_predictions.npz")
    print(f"  grid:           {d['dbm_meas'].shape}  ({d['dbm_meas'].size} cells)")
    print(f"  LO range:       {d['p_lo_axis'].min()} .. {d['p_lo_axis'].max()} dBm")
    print(f"  RF range:       {d['p_rf_axis'].min()} .. {d['p_rf_axis'].max()} dBm")
    print()
    for label, pred in [("ideal multiplier", d["dbm_ideal"]),
                        ("physics-only",     d["dbm_phys"]),
                        ("full PIML",        d["dbm_full"])]:
        m = metrics(pred, d["dbm_meas"])
        print(f"  {label:<20}  RMSE = {m['RMSE_dB']:6.2f} dB    "
              f"R^2 = {m['R2']:7.4f}    |err|<=2dB = {m['within_2dB_pct']:4.1f}%")
