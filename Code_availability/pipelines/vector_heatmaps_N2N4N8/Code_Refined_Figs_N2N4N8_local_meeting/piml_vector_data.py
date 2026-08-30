# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

from pathlib import Path
import numpy as np

RUN_DIR = _data_dir(__file__) / "run"


N1_PHYSICS_PARAMS = {
    "G_max_dB":      -1.8438,
    "P_LO_sat_dBm":   1.7347,
    "beta_LO":        1.0453,
    "P_RF_comp_dBm":  4.4746,
    "beta_RF":        2.1329,
    "P_noise_dBm":  -73.6208,
    "leak_dB":      -84.2515,
    "LO_shape":      "weibull",
}

VECTOR_PARAMS = {
    "c_papr_lo":  None,
    "c_papr_rf":  None,
}


def _w_to_dbm(p_w, floor_w=1e-30):
    p = np.maximum(np.asarray(p_w, dtype=np.float64), floor_w)
    return 10.0 * np.log10(p) + 30.0


def load_vector_params(summary_json=None):
    import json
    path = Path(summary_json) if summary_json else (RUN_DIR / "summary.json")
    with path.open() as f:
        s = json.load(f)
    VECTOR_PARAMS["c_papr_lo"] = float(s["vector_papr_params"]["c_papr_lo"])
    VECTOR_PARAMS["c_papr_rf"] = float(s["vector_papr_params"]["c_papr_rf"])
    return VECTOR_PARAMS


def load_one_N(N, run_dir=None):
    base = Path(run_dir) if run_dir else RUN_DIR
    path = base / f"twin_predictions_N{N}.npz"
    npz = np.load(path)

    p_lo = npz["p_lo_dbm"]
    p_rf = npz["p_rf_dbm"]
    P_LO, P_RF = np.meshgrid(p_lo, p_rf, indexing="ij")

    dbm_meas  = _w_to_dbm(npz["p_if_meas_w"])
    dbm_ideal = _w_to_dbm(npz["p_if_ideal_w"])
    dbm_naive = _w_to_dbm(npz["p_if_naive_w"])
    dbm_phys  = _w_to_dbm(npz["p_if_phys_w"])
    dbm_full  = _w_to_dbm(npz["p_if_full_w"])

    return dict(
        N          = int(npz["N"]) if "N" in npz.files else N,
        ip_sq      = float(npz["ip_sq"]),
        papr_lo    = float(npz["papr_lo"]),
        papr_rf    = float(npz["papr_rf"]),
        p_lo_axis  = p_lo,
        p_rf_axis  = p_rf,
        P_LO       = P_LO,
        P_RF       = P_RF,
        dbm_meas   = dbm_meas,
        dbm_ideal  = dbm_ideal,
        dbm_naive  = dbm_naive,
        dbm_phys   = dbm_phys,
        dbm_full   = dbm_full,
        res_ideal  = dbm_ideal - dbm_meas,
        res_naive  = dbm_naive - dbm_meas,
        res_phys   = dbm_phys  - dbm_meas,
        res_full   = dbm_full  - dbm_meas,
        delta_NN   = np.asarray(npz["delta_db"]),
    )


def load_all(run_dir=None, Ns=(2, 4, 8)):
    return {N: load_one_N(N, run_dir=run_dir) for N in Ns}


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
    vp = load_vector_params()
    print("Vector extension PAPR-shift scalars (trained, on top of frozen N=1):")
    print(f"  c_papr_lo = {vp['c_papr_lo']:+.4f}")
    print(f"  c_papr_rf = {vp['c_papr_rf']:+.4f}")
    print()
    data = load_all()
    print(f"{'N':>3} {'|<a,b>|^2':>10} {'PAPR_LO':>9} {'PAPR_RF':>9}  "
          f"{'grid':>8}  RMSE per model (dB)")
    for N, d in data.items():
        m_ideal = metrics(d["dbm_ideal"], d["dbm_meas"])
        m_naive = metrics(d["dbm_naive"], d["dbm_meas"])
        m_phys  = metrics(d["dbm_phys"],  d["dbm_meas"])
        m_full  = metrics(d["dbm_full"],  d["dbm_meas"])
        print(f"{N:3d} {d['ip_sq']:10.4f} {d['papr_lo']:9.2f} {d['papr_rf']:9.2f}  "
              f"{str(d['dbm_meas'].shape):>8}  "
              f"ideal={m_ideal['RMSE_dB']:5.2f}  "
              f"naive={m_naive['RMSE_dB']:5.2f}  "
              f"phys={m_phys['RMSE_dB']:5.2f}  "
              f"full={m_full['RMSE_dB']:5.2f}")
