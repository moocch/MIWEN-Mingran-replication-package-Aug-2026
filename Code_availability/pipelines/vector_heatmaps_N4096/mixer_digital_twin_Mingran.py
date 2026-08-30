from __future__ import annotations
import base64, zlib
import numpy as np

IP_EFF   = 0.01397973060960252
PAPR_LO  = 5.357
PAPR_RF  = 5.311
MAX_DB   = 5.0

P_MAX_DBM, GAIN_MIN_DB, GAIN_MAX_DB = 18.0, 0.0, 31.5
AMP_TARGET, AMP_MAX, R_OHM = 0.6, 0.95, 50.0

PHYS = {
    "G": 2.7293514785998596,
    "PsatL": -2.8049513064337788,
    "betaL": 1.8507810921098646,
    "PsatR": -8.022814238429426,
    "betaR": 2.9163471944551613,
    "w_hill": 0.9998751648148396,
    "PcompRF": -3.944236460733695,
    "PcompLO": -5.199087182330942,
    "betaC": 0.6101367971342286,
    "kappa": -1.4910963469974874,
    "c_papr_lo": 0.5383216348803388,
    "c_papr_rf": 0.0681181224296381,
    "leak": -97.70295030590809,
    "C_cal": 13.231423935543141,
    "floor": -97.58202830211204,
}

_NN_SHAPES = {
    "net.0.weight": (48, 6),
    "net.0.bias": (48,),
    "net.2.weight": (48, 48),
    "net.2.bias": (48,),
    "net.4.weight": (1, 48),
    "net.4.bias": (1,),
}
_NN_BLOB = (
    "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"
)


def _load_nn():
    raw = zlib.decompress(base64.b64decode(_NN_BLOB.encode()))
    out, off = {}, 0
    for k, shp in _NN_SHAPES.items():
        n = int(np.prod(shp))
        out[k] = np.frombuffer(raw, dtype="<f8", count=n, offset=off).reshape(shp).copy()
        off += n * 8
    return out

_NN = _load_nn()

def _power_to_gain_amp(t):
    t = np.asarray(t, dtype=np.float64)
    gain = GAIN_MAX_DB + t - P_MAX_DBM - 20.0 * np.log10(AMP_TARGET)
    amp = np.full_like(t, AMP_TARGET)
    hi = gain > GAIN_MAX_DB
    gain = np.where(hi, GAIN_MAX_DB, gain)
    amp = np.where(hi, np.minimum(10.0 ** ((t - P_MAX_DBM) / 20.0), AMP_MAX), amp)
    lo = gain < GAIN_MIN_DB
    gain = np.where(lo, GAIN_MIN_DB, gain)
    amp = np.where(lo, 10.0 ** ((t - (P_MAX_DBM - GAIN_MAX_DB)) / 20.0), amp)
    return gain, amp

def _alpha_mix(p_dbm, psat, beta, w):
    l = np.clip((p_dbm - psat) / 10.0, -30.0, 5.0)
    weib = -np.expm1(-(10.0 ** np.clip(beta * l, -30.0, 10.0)))
    xh = 10.0 ** np.clip(beta * l, -30.0, 30.0)
    return (1.0 - w) * weib + w * (xh / (1.0 + xh))

def _comp_w(p_dbm, pc_dbm, beta_c):
    pw = 1e-3 * 10.0 ** (p_dbm / 10.0)
    pcw = 1e-3 * 10.0 ** (pc_dbm / 10.0)
    r = np.clip(pw / pcw, 1e-30, None)
    return pw * (1.0 + r ** beta_c) ** (-1.0 / beta_c)

def _phys_log10(PLO, PRF):
    v = PHYS
    ple = PLO + v["c_papr_lo"] * PAPR_LO
    pre = PRF + v["c_papr_rf"] * PAPR_RF
    a_l = _alpha_mix(ple, v["PsatL"], v["betaL"], v["w_hill"])
    a_r = _alpha_mix(pre, v["PsatR"], v["betaR"], v["w_hill"])
    c_r = _comp_w(pre, v["PcompRF"], v["betaC"])
    c_l = _comp_w(ple, v["PcompLO"], v["betaC"])
    sig = (10.0 ** (v["G"] / 10.0) * IP_EFF *
           (a_l * c_r + 10.0 ** (v["kappa"] / 10.0) * a_r * c_l))
    leak = 10.0 ** (v["leak"] / 10.0) * 1e-3 * 10.0 ** (PRF / 10.0)
    arb = 10.0 ** (v["C_cal"] / 10.0) * (sig + leak) + 1e-3 * 10.0 ** (v["floor"] / 10.0)
    return np.log10(np.clip(arb, 1e-32, None))

def _nn_delta_db(PLO, PRF):
    g_lo, a_lo = _power_to_gain_amp(PLO)
    g_rf, a_rf = _power_to_gain_amp(PRF)
    f = np.stack([PLO / 30.0, PRF / 30.0,
                  a_lo / AMP_TARGET, a_rf / AMP_TARGET,
                  g_lo / GAIN_MAX_DB, g_rf / GAIN_MAX_DB], axis=-1)
    x = np.tanh(f @ _NN["net.0.weight"].T + _NN["net.0.bias"])
    x = np.tanh(x @ _NN["net.2.weight"].T + _NN["net.2.bias"])
    x = x @ _NN["net.4.weight"].T + _NN["net.4.bias"]
    return MAX_DB * np.tanh(x[..., 0])

def predict(p_lo_dbm, p_rf_dbm, use_nn: bool = True):
    PLO = np.asarray(p_lo_dbm, dtype=np.float64)
    PRF = np.asarray(p_rf_dbm, dtype=np.float64)
    y = 10.0 * _phys_log10(PLO, PRF)
    if use_nn:
        y = y + _nn_delta_db(PLO, PRF)
    return y if y.ndim else float(y)

def predict_uv(p_lo_dbm, p_rf_dbm, use_nn: bool = True):
    y = predict(p_lo_dbm, p_rf_dbm, use_nn=use_nn)
    return 1e6 * np.sqrt(R_OHM * 10.0 ** (np.asarray(y) / 10.0))


if __name__ == "__main__":
    for plo, prf in [(-10.0, -5.0), (0.0, 0.0), (-40.0, -20.0), (10.0, 10.0)]:
        print("P_LO=%+6.1f dBm  P_RF=%+6.1f dBm  ->  y=%8.3f dB-arb  (%12.2f uV)"
              % (plo, prf, predict(plo, prf), predict_uv(plo, prf)))
