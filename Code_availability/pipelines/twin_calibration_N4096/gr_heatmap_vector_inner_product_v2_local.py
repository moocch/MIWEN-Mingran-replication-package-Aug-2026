from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import argparse
import json
import os
import sys
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm


os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
try:
    from gnuradio import gr, uhd, blocks
    _GR_OK = True
    _GR_ERR = None
except Exception as _exc:
    gr = None
    uhd = None
    blocks = None
    _GR_OK = False
    _GR_ERR = _exc


SCRIPT_DIR = _data_dir(__file__)
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_STEM = "gr_usrp_mixer_vector_heatmap"

TX_ARGS = "addr=192.168.30.2,second_addr=192.168.40.2,master_clock_rate=200e6"
USRP_SAMPLE_RATE = 10e6
USRP_LO_OFFSET = 5e6
TX_ANTENNA = "TX/RX"

FREQ_RF_HZ = 1.20e9
FREQ_LO_HZ = 0.90e9
IF_FREQ_HZ = abs(FREQ_RF_HZ - FREQ_LO_HZ)
PORT_R_OHM = 50.0

P_MAX_DBM_DEFAULT = 18.0
GAIN_MIN_DB = 0.0
GAIN_MAX_DB = 31.5
AMP_TARGET = 0.6
AMP_MAX = 0.95
AMP_WARN = 1e-3

RX_ARGS = "addr=192.168.60.2,second_addr=192.168.50.2,master_clock_rate=200e6"
RX_ANTENNA = "RX2"
RX_GAIN_DB_DEFAULT = 30.0
RX_CLOCK_SOURCE_DEFAULT = "external"
RX_TIME_SOURCE_DEFAULT = "internal"

RX_N_CAPTURE_DEFAULT = 65536
RX_RING_CAPACITY = 1 << 21
RX_UV_SCALE_DEFAULT = 1e6
RX_GRAB_TIMEOUT_S = 3.0
RX_CLIP_THRESH = 0.98
IF_ATTEN_DB_DEFAULT = 30.0

N_DEFAULT = 2
DF_TONE_HZ_DEFAULT = 1.0e6
VECTOR_SEED_DEFAULT = 0
TX_TEMPLATE_MIN_SAMPLES = 8192

P_LO_DBM_MIN_DEFAULT = -70.0
P_LO_DBM_MAX_DEFAULT = +10.0
N_LO_DEFAULT = 33
P_RF_DBM_MIN_DEFAULT = -70.0
P_RF_DBM_MAX_DEFAULT = +10.0
N_RF_DEFAULT = 33

REPEATS_EACH_DEFAULT = 3
SETTLE_S_DEFAULT = 0.05


@dataclass
class GainAmp:
    gain_db: float
    amp: float
    predicted_dbm: float


def power_to_gain_amp(
    target_dbm: float,
    p_max_dbm: float = P_MAX_DBM_DEFAULT,
    amp_target: float = AMP_TARGET,
) -> GainAmp:
    gain = GAIN_MAX_DB + target_dbm - p_max_dbm - 20.0 * np.log10(amp_target)
    amp = amp_target
    if gain > GAIN_MAX_DB:
        gain = GAIN_MAX_DB
        amp = 10.0 ** ((target_dbm - p_max_dbm) / 20.0)
        if amp > AMP_MAX:
            warnings.warn(
                f"target {target_dbm:+.2f} dBm exceeds device max "
                f"(~+{p_max_dbm:.1f} dBm); clipping amp to {AMP_MAX}.",
                RuntimeWarning,
            )
            amp = AMP_MAX
    elif gain < GAIN_MIN_DB:
        gain = GAIN_MIN_DB
        amp = 10.0 ** ((target_dbm - (p_max_dbm - GAIN_MAX_DB)) / 20.0)
        if amp < AMP_WARN:
            warnings.warn(
                f"target {target_dbm:+.2f} dBm requires amp={amp:.2e} "
                f"(< {AMP_WARN:.0e}); may be below DAC SFDR.",
                RuntimeWarning,
            )
    predicted = p_max_dbm + (gain - GAIN_MAX_DB) + 20.0 * np.log10(amp)
    return GainAmp(gain_db=gain, amp=float(amp), predicted_dbm=float(predicted))


def tone_offsets_hz(N: int, df_tone_hz: float) -> np.ndarray:
    if N <= 1:
        return np.zeros(1, dtype=float)
    return (np.arange(N, dtype=float) - (N - 1) / 2.0) * float(df_tone_hz)


def period_samples_for_vector(N: int, df_tone_hz: float, fs_hz: float) -> int:
    if N <= 1:
        return 1
    if N % 2 == 0:
        T_s = 2.0 / float(df_tone_hz)
    else:
        T_s = 1.0 / float(df_tone_hz)
    period = int(round(T_s * float(fs_hz)))
    if period < 1:
        raise ValueError(
            f"period_samples = {period} (df_tone={df_tone_hz} Hz, fs={fs_hz} Hz). "
            "df_tone too large or fs too small; baseband period is less than 1 sample."
        )
    return period


def make_vector_baseband(
    vec: np.ndarray,
    df_tone_hz: float,
    fs_hz: float,
    n_samples: int,
) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.complex128)
    N = int(vec.shape[0])
    if N == 1:
        return np.full(n_samples, vec[0], dtype=np.complex64)
    f_offsets = tone_offsets_hz(N, df_tone_hz)
    n = np.arange(n_samples, dtype=np.float64)
    phase = (2.0 * np.pi / float(fs_hz)) * np.outer(f_offsets, n)
    expj = np.exp(1j * phase)
    s_bb = vec @ expj
    return s_bb.astype(np.complex64)


def rms_normalize_to_amp(
    s_bb: np.ndarray,
    amp_rms_target: float,
    amp_max: float = AMP_MAX,
) -> tuple:
    s = np.asarray(s_bb, dtype=np.complex128)
    cur_rms = float(np.sqrt(np.mean(np.abs(s) ** 2)))
    if cur_rms <= 0.0 or not np.isfinite(cur_rms):
        return np.zeros_like(s, dtype=np.complex64), 0.0, 0.0, False
    s_unit = s / cur_rms
    s_target = s_unit * float(amp_rms_target)
    peak = float(np.max(np.abs(s_target)))
    peak_limited = False
    if peak > amp_max:
        scale = float(amp_max) / peak
        s_target = s_target * scale
        peak = amp_max
        peak_limited = True
    actual_rms = float(np.sqrt(np.mean(np.abs(s_target) ** 2)))
    return s_target.astype(np.complex64), actual_rms, peak, peak_limited


def make_unit_rms_template(
    vec: np.ndarray,
    df_tone_hz: float,
    fs_hz: float,
    min_samples: int = TX_TEMPLATE_MIN_SAMPLES,
) -> tuple:
    N = int(np.asarray(vec).reshape(-1).shape[0])
    period = period_samples_for_vector(N, df_tone_hz, fs_hz)
    k = max(1, int(np.ceil(float(min_samples) / float(period))))
    n_samples = k * period
    s = make_vector_baseband(vec, df_tone_hz, fs_hz, n_samples)
    s_unit, _, peak_unit, _ = rms_normalize_to_amp(s, amp_rms_target=1.0, amp_max=np.inf)
    return s_unit.astype(np.complex64), int(period), float(peak_unit)


def baseband_papr_db(
    vec: np.ndarray,
    df_tone_hz: float,
    fs_hz: float,
    oversample: int = 8,
) -> float:
    vec = np.asarray(vec, dtype=np.complex128).reshape(-1)
    N = int(vec.shape[0])
    if N <= 1:
        return 0.0
    period = period_samples_for_vector(N, df_tone_hz, fs_hz)
    osf = max(int(oversample), 1)
    n_eval = max(int(period) * osf, 64)
    fs_eval = float(fs_hz) * osf
    f_offsets = tone_offsets_hz(N, df_tone_hz)
    n_idx = np.arange(n_eval, dtype=np.float64)
    phase_grid = (2.0 * np.pi / fs_eval) * np.outer(f_offsets, n_idx)
    basis = np.exp(1j * phase_grid)
    s = vec @ basis
    rms = float(np.sqrt(np.mean(np.abs(s) ** 2)))
    peak = float(np.max(np.abs(s)))
    if rms <= 0.0:
        return float("inf")
    return 20.0 * np.log10(peak / rms)


def reduce_papr_by_phase_rotation(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    df_tone_hz: float,
    fs_hz: float,
    n_trials: int = 512,
    oversample: int = 8,
    seed: int = 12345,
    verbose: bool = True,
) -> tuple:
    a = np.asarray(vec_a, dtype=np.complex128).reshape(-1)
    b = np.asarray(vec_b, dtype=np.complex128).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"vec_a/vec_b shape mismatch: {a.shape} vs {b.shape}")
    N = int(a.shape[0])

    papr_a_before = baseband_papr_db(a, df_tone_hz, fs_hz, oversample)
    papr_b_before = baseband_papr_db(b, df_tone_hz, fs_hz, oversample)

    if N <= 2 or n_trials <= 0:
        reason = ("N<=2 (PAPR independent of phase)" if N <= 2 else "n_trials<=0")
        if verbose:
            print(f"[papr] N={N}, skip PAPR reduction ({reason}). "
                  f"PAPR_a={papr_a_before:+.2f} dB, PAPR_b={papr_b_before:+.2f} dB")
        return a, b, {
            "applied": False,
            "phi": np.zeros(N, dtype=float),
            "papr_a_before_db": float(papr_a_before),
            "papr_b_before_db": float(papr_b_before),
            "papr_a_after_db": float(papr_a_before),
            "papr_b_after_db": float(papr_b_before),
            "n_trials": int(n_trials),
            "oversample": int(oversample),
            "seed": int(seed),
        }

    period = period_samples_for_vector(N, df_tone_hz, fs_hz)
    osf = max(int(oversample), 1)
    n_eval = max(int(period) * osf, 64)
    fs_eval = float(fs_hz) * osf
    f_offsets = tone_offsets_hz(N, df_tone_hz)
    n_idx = np.arange(n_eval, dtype=np.float64)
    phase_grid = (2.0 * np.pi / fs_eval) * np.outer(f_offsets, n_idx)
    tone_basis = np.exp(1j * phase_grid).astype(np.complex128)

    rng = np.random.default_rng(int(seed))

    candidates = [np.zeros(N, dtype=float)]
    k_arr = np.arange(N, dtype=float)
    candidates.append(-np.pi * k_arr * (k_arr - 1.0) / float(N))
    candidates.append(np.pi * (k_arr ** 2) / float(N))
    K = int(n_trials)
    candidates.extend(list(rng.uniform(-np.pi, np.pi, size=(K, N))))

    phis = np.stack([np.asarray(p, dtype=np.float64) for p in candidates], axis=0)
    d_mat = np.exp(1j * phis)
    sa_mat = (d_mat * a[None, :]) @ tone_basis
    sb_mat = (d_mat * b[None, :]) @ tone_basis
    rms_a_arr = np.sqrt(np.mean(np.abs(sa_mat) ** 2, axis=1))
    rms_b_arr = np.sqrt(np.mean(np.abs(sb_mat) ** 2, axis=1))
    peak_a_arr = np.max(np.abs(sa_mat), axis=1)
    peak_b_arr = np.max(np.abs(sb_mat), axis=1)
    papr_a_arr = peak_a_arr / np.maximum(rms_a_arr, 1e-30)
    papr_b_arr = peak_b_arr / np.maximum(rms_b_arr, 1e-30)
    metric_arr = np.maximum(papr_a_arr, papr_b_arr)
    k_star = int(np.argmin(metric_arr))
    best_phi = phis[k_star].copy()

    d_best = np.exp(1j * best_phi)
    a_new = (a * d_best).astype(np.complex128)
    b_new = (b * d_best).astype(np.complex128)

    papr_a_after = baseband_papr_db(a_new, df_tone_hz, fs_hz, oversample)
    papr_b_after = baseband_papr_db(b_new, df_tone_hz, fs_hz, oversample)

    if verbose:
        ip_before = np.vdot(a, b)
        ip_after = np.vdot(a_new, b_new)
        cand_tag = ("zero" if k_star == 0
                    else "Schroeder" if k_star == 1
                    else "Newman" if k_star == 2
                    else f"SLM#{k_star - 3}")
        print(f"[papr] N={N}, candidates: 3 structured + {K} SLM trials -> picked '{cand_tag}'")
        print(f"[papr] PAPR_a: {papr_a_before:+.2f} -> {papr_a_after:+.2f} dB "
              f"({papr_a_before - papr_a_after:+.2f} dB reduction)")
        print(f"[papr] PAPR_b: {papr_b_before:+.2f} -> {papr_b_after:+.2f} dB "
              f"({papr_b_before - papr_b_after:+.2f} dB reduction)")
        ceil_a_before = AMP_MAX / (10.0 ** (papr_a_before / 20.0))
        ceil_a_after  = AMP_MAX / (10.0 ** (papr_a_after  / 20.0))
        ceil_b_before = AMP_MAX / (10.0 ** (papr_b_before / 20.0))
        ceil_b_after  = AMP_MAX / (10.0 ** (papr_b_after  / 20.0))
        print(f"[papr] amp_rms ceiling (before peak-limit, AMP_MAX={AMP_MAX}):")
        print(f"[papr]   LO: {ceil_a_before:.4f} -> {ceil_a_after:.4f} "
              f"(+{20*np.log10(max(ceil_a_after/max(ceil_a_before,1e-12), 1e-12)):.2f} dB headroom)")
        print(f"[papr]   RF: {ceil_b_before:.4f} -> {ceil_b_after:.4f} "
              f"(+{20*np.log10(max(ceil_b_after/max(ceil_b_before,1e-12), 1e-12)):.2f} dB headroom)")
        print(f"[papr] <a,b> preservation check: "
              f"|<a,b>|^2 before={np.abs(ip_before)**2:.6e}, "
              f"after={np.abs(ip_after)**2:.6e}, "
              f"|Δ<a,b>|={np.abs(ip_after - ip_before):.2e} (should be ~1e-16)")

    return a_new, b_new, {
        "applied": True,
        "phi": best_phi,
        "papr_a_before_db": float(papr_a_before),
        "papr_b_before_db": float(papr_b_before),
        "papr_a_after_db": float(papr_a_after),
        "papr_b_after_db": float(papr_b_after),
        "n_trials": int(n_trials),
        "oversample": int(oversample),
        "seed": int(seed),
        "selected_candidate_idx": int(k_star),
    }


def generate_unit_norm_vectors(N: int, seed: int) -> tuple:
    if N == 1:
        return (np.array([1.0 + 0j], dtype=np.complex128),
                np.array([1.0 + 0j], dtype=np.complex128))
    rng = np.random.default_rng(int(seed))
    a = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
    b = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2.0)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return a.astype(np.complex128), b.astype(np.complex128)


def ip_magnitude_squared(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, b)) ** 2)


if _GR_OK:

    class IQGrabber(gr.sync_block):

        def __init__(self, capacity: int = RX_RING_CAPACITY):
            gr.sync_block.__init__(
                self, name="IQGrabber",
                in_sig=[np.complex64], out_sig=None,
            )
            self._cap = int(capacity)
            self._buf = np.zeros(self._cap, dtype=np.complex64)
            self._widx = 0
            self._total = 0
            self._lock = threading.Lock()

        def work(self, input_items, output_items):
            x = input_items[0]
            n = len(x)
            if n:
                with self._lock:
                    end = self._widx + n
                    if end <= self._cap:
                        self._buf[self._widx:end] = x
                    else:
                        first = self._cap - self._widx
                        self._buf[self._widx:] = x[:first]
                        self._buf[:n - first] = x[first:]
                    self._widx = end % self._cap
                    self._total += n
            return n

        def total_samples(self) -> int:
            with self._lock:
                return int(self._total)

        def grab_fresh(self, n: int, timeout_s: float = RX_GRAB_TIMEOUT_S) -> np.ndarray:
            n = int(min(int(n), self._cap))
            start = self.total_samples()
            t0 = time.time()
            while self.total_samples() - start < n:
                if time.time() - t0 > timeout_s:
                    break
                time.sleep(0.001)
            with self._lock:
                idx = (self._widx - n) % self._cap
                if idx + n <= self._cap:
                    out = self._buf[idx:idx + n].copy()
                else:
                    first = self._cap - idx
                    out = np.concatenate([self._buf[idx:], self._buf[:n - first]])
            return out

    class DualUSRPVectorFlowgraph(gr.top_block):

        def __init__(
            self,
            vec_a: np.ndarray,
            vec_b: np.ndarray,
            df_tone_hz: float,
            *,
            tx_args: str = TX_ARGS,
            rx_args: str = RX_ARGS,
            sample_rate: float = USRP_SAMPLE_RATE,
            freq_rf: float = FREQ_RF_HZ,
            freq_lo: float = FREQ_LO_HZ,
            lo_offset: float = USRP_LO_OFFSET,
            tx_antenna: str = TX_ANTENNA,
            rx_antenna: str = RX_ANTENNA,
            rx_gain_db: float = RX_GAIN_DB_DEFAULT,
            rx_clock_source: str = RX_CLOCK_SOURCE_DEFAULT,
            rx_time_source: str = RX_TIME_SOURCE_DEFAULT,
            ring_capacity: int = RX_RING_CAPACITY,
            p_max_dbm: float = P_MAX_DBM_DEFAULT,
            initial_p_rf_dbm: float = -30.0,
            initial_p_lo_dbm: float = -30.0,
            verbose: bool = True,
        ):
            gr.top_block.__init__(self, "Dual USRP vector inner product sweep")

            self._verbose = verbose
            self._sr = float(sample_rate)
            self._p_max_dbm = float(p_max_dbm)
            self._df_tone_hz = float(df_tone_hz)

            vec_a = np.asarray(vec_a, dtype=np.complex128).reshape(-1)
            vec_b = np.asarray(vec_b, dtype=np.complex128).reshape(-1)
            if vec_a.shape[0] != vec_b.shape[0]:
                raise ValueError(
                    f"vec_a (len={vec_a.shape[0]}) and vec_b (len={vec_b.shape[0]}) must match."
                )
            self._N = int(vec_a.shape[0])
            self._vec_a = vec_a
            self._vec_b = vec_b

            s_a_unit, period_a, self._peak_unit_a = make_unit_rms_template(
                vec_a, self._df_tone_hz, self._sr)
            s_b_unit, period_b, self._peak_unit_b = make_unit_rms_template(
                vec_b, self._df_tone_hz, self._sr)
            self._period = int(period_a)
            self._s_a_unit = s_a_unit
            self._s_b_unit = s_b_unit

            if verbose:
                print(f"[gr-tx] connecting TX USRP: {tx_args}")

            self.usrp_sink = uhd.usrp_sink(
                ",".join((tx_args, "")),
                uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0, 1]),
            )
            self.usrp_sink.set_samp_rate(self._sr)
            for ch, freq in [(0, freq_rf), (1, freq_lo)]:
                self.usrp_sink.set_center_freq(uhd.tune_request(freq, lo_offset), ch)
                self.usrp_sink.set_antenna(tx_antenna, ch)
                self.usrp_sink.set_gain(15.0, ch)
                if verbose:
                    print(f"[gr-tx] CH{ch}: freq~{freq/1e6:9.3f} MHz ant={tx_antenna}")

            self.src_rf = blocks.vector_source_c(self._s_b_unit.tolist(), True, 1, [])
            self.src_lo = blocks.vector_source_c(self._s_a_unit.tolist(), True, 1, [])
            self.mul_rf = blocks.multiply_const_cc(complex(0.0, 0.0))
            self.mul_lo = blocks.multiply_const_cc(complex(0.0, 0.0))

            self.connect((self.src_rf, 0), (self.mul_rf, 0), (self.usrp_sink, 0))
            self.connect((self.src_lo, 0), (self.mul_lo, 0), (self.usrp_sink, 1))

            if verbose:
                print(f"[gr-rx] connecting RX USRP: {rx_args}")
            self.usrp_source = uhd.usrp_source(
                ",".join((rx_args, "")),
                uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]),
            )
            self.usrp_source.set_clock_source(rx_clock_source, 0)
            self.usrp_source.set_time_source(rx_time_source, 0)
            self.usrp_source.set_samp_rate(self._sr)
            self.usrp_source.set_center_freq(uhd.tune_request(IF_FREQ_HZ, lo_offset), 0)
            self.usrp_source.set_gain(float(rx_gain_db), 0)
            self.usrp_source.set_antenna(rx_antenna, 0)
            if verbose:
                print(f"[gr-rx] CH0: freq~{IF_FREQ_HZ/1e6:.3f} MHz ant={rx_antenna} "
                      f"gain={rx_gain_db:.1f} dB clk={rx_clock_source}")

            self.grabber = IQGrabber(capacity=int(ring_capacity))
            self.connect((self.usrp_source, 0), (self.grabber, 0))

            if verbose:
                print(
                    f"[gr] N={self._N}, df_tone={self._df_tone_hz/1e6:.3f} MHz, "
                    f"baseband period={self._period} samples "
                    f"({self._period/self._sr*1e6:.3f} us), "
                    f"template len: RF={self._s_b_unit.size}, LO={self._s_a_unit.size}"
                )
                print(
                    f"[gr] unit-RMS template peak: LO={self._peak_unit_a:.3f} "
                    f"({20*np.log10(max(self._peak_unit_a,1e-12)):+.2f} dB PAPR), "
                    f"RF={self._peak_unit_b:.3f} "
                    f"({20*np.log10(max(self._peak_unit_b,1e-12)):+.2f} dB PAPR)"
                )

            self.set_powers(initial_p_rf_dbm, initial_p_lo_dbm)

        def check_locks(self) -> None:
            for label, dev, chans in [("TX", self.usrp_sink, (0, 1)),
                                      ("RX", self.usrp_source, (0,))]:
                for ch in chans:
                    try:
                        s = dev.get_sensor("lo_locked", ch)
                        locked = bool(s.to_bool())
                        if self._verbose:
                            print(f"[gr] {label} CH{ch} lo_locked={locked}")
                        if not locked:
                            warnings.warn(f"{label} CH{ch} LO not locked", RuntimeWarning)
                    except Exception:
                        pass

        def set_powers(self, p_rf_dbm: float, p_lo_dbm: float) -> dict:
            rf = power_to_gain_amp(p_rf_dbm, p_max_dbm=self._p_max_dbm)
            lo = power_to_gain_amp(p_lo_dbm, p_max_dbm=self._p_max_dbm)

            ceil_rf = AMP_MAX / max(self._peak_unit_b, 1e-12)
            ceil_lo = AMP_MAX / max(self._peak_unit_a, 1e-12)
            amp_rf = float(rf.amp); clip_rf = False
            amp_lo = float(lo.amp); clip_lo = False
            if amp_rf > ceil_rf:
                amp_rf = ceil_rf; clip_rf = True
            if amp_lo > ceil_lo:
                amp_lo = ceil_lo; clip_lo = True

            peak_rf = amp_rf * self._peak_unit_b
            peak_lo = amp_lo * self._peak_unit_a

            actual_rms_rf = amp_rf
            actual_rms_lo = amp_lo

            rf_predicted_actual = (
                self._p_max_dbm + (rf.gain_db - GAIN_MAX_DB) + 20.0 * np.log10(max(actual_rms_rf, 1e-30))
            )
            lo_predicted_actual = (
                self._p_max_dbm + (lo.gain_db - GAIN_MAX_DB) + 20.0 * np.log10(max(actual_rms_lo, 1e-30))
            )

            if clip_lo and self._verbose:
                warnings.warn(
                    f"LO peak-limited: amp_rms {lo.amp:.4f} -> {amp_lo:.4f} "
                    f"(N={self._N}, template peak={self._peak_unit_a:.2f}); "
                    f"predicted -> {lo_predicted_actual:+.2f} dBm (was {lo.predicted_dbm:+.2f}).",
                    RuntimeWarning,
                )
            if clip_rf and self._verbose:
                warnings.warn(
                    f"RF peak-limited: amp_rms {rf.amp:.4f} -> {amp_rf:.4f} "
                    f"(N={self._N}, template peak={self._peak_unit_b:.2f}); "
                    f"predicted -> {rf_predicted_actual:+.2f} dBm (was {rf.predicted_dbm:+.2f}).",
                    RuntimeWarning,
                )

            self.mul_rf.set_k(complex(amp_rf, 0.0))
            self.mul_lo.set_k(complex(amp_lo, 0.0))
            self.usrp_sink.set_gain(rf.gain_db, 0)
            self.usrp_sink.set_gain(lo.gain_db, 1)

            def _actual_gain(ch):
                try:
                    return float(self.usrp_sink.get_gain(ch))
                except Exception:
                    return float("nan")

            def _actual_freq(ch):
                try:
                    return float(self.usrp_sink.get_center_freq(ch))
                except Exception:
                    return float("nan")

            return {
                "rf": {
                    "target_dbm": float(p_rf_dbm),
                    "gain_db": float(rf.gain_db),
                    "amp_rms_target": float(rf.amp),
                    "amp_rms_actual": float(actual_rms_rf),
                    "peak_actual": float(peak_rf),
                    "peak_limited": bool(clip_rf),
                    "predicted_dbm": float(rf_predicted_actual),
                    "actual_gain_db": _actual_gain(0),
                    "actual_freq_hz": _actual_freq(0),
                },
                "lo": {
                    "target_dbm": float(p_lo_dbm),
                    "gain_db": float(lo.gain_db),
                    "amp_rms_target": float(lo.amp),
                    "amp_rms_actual": float(actual_rms_lo),
                    "peak_actual": float(peak_lo),
                    "peak_limited": bool(clip_lo),
                    "predicted_dbm": float(lo_predicted_actual),
                    "actual_gain_db": _actual_gain(1),
                    "actual_freq_hz": _actual_freq(1),
                },
            }

else:
    IQGrabber = None
    DualUSRPVectorFlowgraph = None


class USRPVectorReader:
    def __init__(
        self,
        grabber,
        *,
        fs_hz: float = USRP_SAMPLE_RATE,
        if_center_hz: float = IF_FREQ_HZ,
        n_capture: int = RX_N_CAPTURE_DEFAULT,
        uv_scale: float = RX_UV_SCALE_DEFAULT,
        clip_thresh: float = RX_CLIP_THRESH,
        if_atten_db: float = IF_ATTEN_DB_DEFAULT,
        verbose: bool = True,
    ):
        self._g = grabber
        self._fs = float(fs_hz)
        self._if_center_hz = float(if_center_hz)
        self._n = int(n_capture)
        self._uv_scale = float(uv_scale)
        self._clip_thresh = float(clip_thresh)
        self._if_atten_db = float(if_atten_db)
        self._verbose = verbose
        self._clip_warned = False

    def read_uv(self) -> float:
        iq = self._g.grab_fresh(self._n)
        if iq.size == 0:
            return float("nan")
        a_dc = np.mean(iq)
        if not self._clip_warned:
            clip_frac = float(np.mean(np.abs(iq) > self._clip_thresh))
            if clip_frac > 1e-4:
                self._clip_warned = True
                warnings.warn(
                    f"RX clipping detected (~{clip_frac*100:.2f}% samples > {self._clip_thresh}); "
                    f"lower --rx-gain to avoid front-end saturation.",
                    RuntimeWarning,
                )
        return float(np.abs(a_dc) * self._uv_scale)

    def read_dbm(self, *, at_mixer_if: bool = True) -> float:
        uv = self.read_uv()
        if not np.isfinite(uv) or uv <= 0:
            return float("-inf")
        v_rms = uv * 1e-6
        p_w = (v_rms ** 2) / PORT_R_OHM
        dbm_at_rx = 10.0 * np.log10(p_w / 1e-3)
        return float(dbm_at_rx + (self._if_atten_db if at_mixer_if else 0.0))

    def capture_spectrum(self, n_fft: Optional[int] = None) -> dict:
        n_fft = int(n_fft or self._n)
        iq = self._g.grab_fresh(n_fft)
        if iq.size < 8:
            return {"ok": False}
        n_fft = int(iq.size)
        win = np.blackman(n_fft)
        win_gain = np.sum(win)
        spec = np.fft.fftshift(np.fft.fft(iq * win)) / max(win_gain, 1e-30)
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / self._fs))
        mag = np.abs(spec)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
        dc_bin = int(np.argmin(np.abs(freqs)))
        a_dc = np.mean(iq)
        peak = float(np.max(np.abs(iq)))
        clip_frac = float(np.mean(np.abs(iq) > self._clip_thresh))
        return {
            "ok": True,
            "freqs_hz": freqs,
            "mag_db": mag_db,
            "dc_bin": dc_bin,
            "dc_mag_db": float(mag_db[dc_bin]),
            "dc_complex": complex(a_dc),
            "abs_dc": float(np.abs(a_dc)),
            "peak": peak,
            "clip_frac": clip_frac,
            "iq": iq,
        }

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def run_sweep(
    set_powers: Callable[[float, float], dict],
    sa_read_uv: Callable[[], float],
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    df_tone_hz: float,
    p_lo_dbm_grid: np.ndarray,
    p_rf_dbm_grid: np.ndarray,
    *,
    repeats_each: int = REPEATS_EACH_DEFAULT,
    settle_s: float = SETTLE_S_DEFAULT,
    p_max_dbm: float = P_MAX_DBM_DEFAULT,
) -> dict:
    n_lo = int(len(p_lo_dbm_grid))
    n_rf = int(len(p_rf_dbm_grid))
    R = int(repeats_each)

    if_amp_uv_all = np.full((n_lo, n_rf, R), np.nan, dtype=float)
    actual_p_lo_dbm = np.zeros((n_lo, n_rf), dtype=float)
    actual_p_rf_dbm = np.zeros((n_lo, n_rf), dtype=float)
    actual_gain_lo = np.zeros((n_lo, n_rf), dtype=float)
    actual_gain_rf = np.zeros((n_lo, n_rf), dtype=float)
    actual_amp_lo = np.zeros((n_lo, n_rf), dtype=float)
    actual_amp_rf = np.zeros((n_lo, n_rf), dtype=float)
    peak_limited_lo = np.zeros((n_lo, n_rf), dtype=bool)
    peak_limited_rf = np.zeros((n_lo, n_rf), dtype=bool)

    N = int(vec_a.shape[0])
    print(f"[sweep] N={N}, df_tone={df_tone_hz/1e6:.3f} MHz, "
          f"|<a,b>|^2 = {ip_magnitude_squared(vec_a, vec_b):.6e}")
    print(f"[sweep] LO grid: {p_lo_dbm_grid[0]:+.2f} ~ {p_lo_dbm_grid[-1]:+.2f} dBm ({n_lo} pts)")
    print(f"[sweep] RF grid: {p_rf_dbm_grid[0]:+.2f} ~ {p_rf_dbm_grid[-1]:+.2f} dBm ({n_rf} pts)")
    print(f"[sweep] total cells: {n_lo * n_rf}, repeats per cell: {R}, settle: {settle_s*1e3:.0f} ms")

    t0 = time.time()
    interrupted = False
    total = n_lo * n_rf
    cell = 0
    try:
        for i in range(n_lo):
            p_lo = float(p_lo_dbm_grid[i])
            for j in range(n_rf):
                cell += 1
                p_rf = float(p_rf_dbm_grid[j])

                info = set_powers(p_rf, p_lo)
                actual_p_lo_dbm[i, j] = info["lo"]["predicted_dbm"]
                actual_p_rf_dbm[i, j] = info["rf"]["predicted_dbm"]
                actual_gain_lo[i, j] = info["lo"]["gain_db"]
                actual_gain_rf[i, j] = info["rf"]["gain_db"]
                actual_amp_lo[i, j] = info["lo"]["amp_rms_actual"]
                actual_amp_rf[i, j] = info["rf"]["amp_rms_actual"]
                peak_limited_lo[i, j] = info["lo"]["peak_limited"]
                peak_limited_rf[i, j] = info["rf"]["peak_limited"]

                time.sleep(float(settle_s))

                for r in range(R):
                    try:
                        if_amp_uv_all[i, j, r] = float(sa_read_uv())
                    except Exception as exc:
                        print(f"[sweep] RX read failed at (i={i},j={j},r={r}): {exc!r}")

                with np.errstate(all="ignore"):
                    mu = float(np.nanmean(if_amp_uv_all[i, j, :]))
                    sd = float(np.nanstd(if_amp_uv_all[i, j, :]))
                elapsed = time.time() - t0
                eta = elapsed / cell * (total - cell)
                clip_tag = ""
                if peak_limited_lo[i, j] or peak_limited_rf[i, j]:
                    clip_tag = "  PEAK-LIM"
                print(
                    f"[{cell:4d}/{total}] "
                    f"P_LO={p_lo:+6.2f}->{actual_p_lo_dbm[i,j]:+6.2f} dBm  "
                    f"P_RF={p_rf:+6.2f}->{actual_p_rf_dbm[i,j]:+6.2f} dBm  "
                    f"IF uV: mean={mu:10.4f} std={sd:8.4f}  "
                    f"ETA={eta:6.1f}s{clip_tag}"
                )
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[sweep] interrupted at cell {cell}/{total}; saving partial ...")

    dt = time.time() - t0
    print(f"[sweep] {'partial ' if interrupted else ''}done in {dt:.1f} s")

    with np.errstate(all="ignore"):
        if_amp_uv_mean = np.nanmean(if_amp_uv_all, axis=2)
        if_amp_uv_std = np.nanstd(if_amp_uv_all, axis=2, ddof=0)

    return {
        "p_lo_dbm_grid": np.asarray(p_lo_dbm_grid, dtype=float),
        "p_rf_dbm_grid": np.asarray(p_rf_dbm_grid, dtype=float),
        "actual_p_lo_dbm": actual_p_lo_dbm,
        "actual_p_rf_dbm": actual_p_rf_dbm,
        "actual_gain_lo": actual_gain_lo,
        "actual_gain_rf": actual_gain_rf,
        "actual_amp_lo": actual_amp_lo,
        "actual_amp_rf": actual_amp_rf,
        "peak_limited_lo": peak_limited_lo,
        "peak_limited_rf": peak_limited_rf,
        "if_amp_uv_all": if_amp_uv_all,
        "if_amp_uv_mean": np.asarray(if_amp_uv_mean, dtype=float),
        "if_amp_uv_std": np.asarray(if_amp_uv_std, dtype=float),
        "vec_a": np.asarray(vec_a, dtype=np.complex128),
        "vec_b": np.asarray(vec_b, dtype=np.complex128),
        "vec_N": int(vec_a.shape[0]),
        "df_tone_hz": float(df_tone_hz),
        "ip_magnitude_squared": ip_magnitude_squared(vec_a, vec_b),
        "if_freq_hz": float(IF_FREQ_HZ),
        "freq_rf_hz": float(FREQ_RF_HZ),
        "freq_lo_hz": float(FREQ_LO_HZ),
        "p_max_dbm": float(p_max_dbm),
        "elapsed_s": float(dt),
        "interrupted": bool(interrupted),
        "reader_backend": "usrp_rx",
        "measured_units": "arb_usrp_fc32",
    }


def measured_power_w(if_amp_uv_mean: np.ndarray) -> np.ndarray:
    v_rms = np.asarray(if_amp_uv_mean, dtype=float) * 1e-6
    return (v_rms ** 2) / PORT_R_OHM


def ideal_inner_product_power_w(
    p_lo_dbm: np.ndarray,
    p_rf_dbm: np.ndarray,
    ip_mag_sq: float,
) -> np.ndarray:
    p_lo_w = 10.0 ** (np.asarray(p_lo_dbm, dtype=float) / 10.0) * 1e-3
    p_rf_w = 10.0 ** (np.asarray(p_rf_dbm, dtype=float) / 10.0) * 1e-3
    return (p_lo_w[:, None] * p_rf_w[None, :] * float(ip_mag_sq)) / 1e-3


def _make_norm(data: np.ndarray, kind: str = "log", gamma: float = 0.5):
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return Normalize()
    vmax = float(np.max(finite))
    if kind == "log":
        pos = finite[finite > 0]
        if pos.size == 0 or vmax <= 0:
            return Normalize()
        vmin = float(np.min(pos))
        if vmax <= vmin:
            vmax = vmin * 10.0
        return LogNorm(vmin=vmin, vmax=vmax)
    if kind == "gamma":
        vmin = max(float(np.min(finite)), 0.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        return PowerNorm(gamma=float(gamma), vmin=vmin, vmax=vmax)
    vmin = float(np.min(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def plot_heatmaps(
    data: dict,
    out_png,
    *,
    norm: str = "log",
    gamma: float = 0.5,
    show: bool = False,
) -> None:
    p_lo = np.asarray(data["p_lo_dbm_grid"], dtype=float)
    p_rf = np.asarray(data["p_rf_dbm_grid"], dtype=float)

    ip_mag_sq = float(data.get("ip_magnitude_squared", 1.0))
    N = int(data.get("vec_N", 1))
    df_tone_hz = float(data.get("df_tone_hz", 0.0))

    p_meas = measured_power_w(data["if_amp_uv_mean"])
    p_ideal = ideal_inner_product_power_w(p_lo, p_rf, ip_mag_sq)

    z_meas = p_meas.T
    z_ideal = p_ideal.T

    norm_meas = _make_norm(z_meas, norm, gamma=gamma)
    norm_ideal = _make_norm(z_ideal, norm, gamma=gamma)

    if norm == "log":
        norm_tag = "log colour"
    elif norm == "gamma":
        norm_tag = f"gamma={gamma:g} colour"
    else:
        norm_tag = "linear colour"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    im1 = axes[0].pcolormesh(
        p_lo, p_rf, z_ideal,
        shading="auto", cmap="viridis", norm=norm_ideal,
    )
    axes[0].set_xlabel("LO port power (dBm)")
    axes[0].set_ylabel("RF port power (dBm)")
    axes[0].set_title(
        r"Ideal:  $P_{center}[\mathrm{dBm}] = P_{LO} + P_{RF} + 10\log_{10}|\langle a,b\rangle|^2$"
        f"   ({norm_tag})"
    )
    cb1 = fig.colorbar(im1, ax=axes[0])
    cb1.set_label("Power (W) — perfect inner-product multiplier")

    im2 = axes[1].pcolormesh(
        p_lo, p_rf, z_meas,
        shading="auto", cmap="viridis", norm=norm_meas,
    )
    axes[1].set_xlabel("LO port power (dBm)")
    axes[1].set_ylabel("RF port power (dBm)")
    axes[1].set_title(
        f"Measured (USRP RX @ {data['if_freq_hz']/1e6:.0f} MHz)  ({norm_tag})"
    )
    cb2 = fig.colorbar(im2, ax=axes[1])
    cb2.set_label("|DC bin| (arb. units, USRP fc32 — UNCALIBRATED)")

    df_tag = f", df_tone={df_tone_hz/1e6:.2f} MHz" if N > 1 else ""
    fig.suptitle(
        f"USRP X310 + UBX-160  ->  ZEM-4300  ->  USRP RX (GNU Radio)    "
        f"f_RF={data['freq_rf_hz']/1e9:.2f} GHz, f_LO={data['freq_lo_hz']/1e9:.2f} GHz, "
        f"f_IF={data['if_freq_hz']/1e6:.0f} MHz   |   "
        f"vector inner product N={N}{df_tag}, "
        f"|<a,b>|^2={ip_mag_sq:.3e}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"[plot] saved figure: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def plot_rx_spectrum(spec: dict, out_png, *, if_center_hz: float = IF_FREQ_HZ,
                     show: bool = False) -> None:
    freqs = np.asarray(spec["freqs_hz"], dtype=float)
    mag_db = np.asarray(spec["mag_db"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot((freqs + if_center_hz) / 1e6, mag_db, lw=0.7)
    ax.axvline(if_center_hz / 1e6, color="r", ls="--", lw=1.0,
               label=f"IF center {if_center_hz/1e6:.0f} MHz (inner product)")
    ax.set_xlabel("RF frequency (MHz)  [RX baseband + center]")
    ax.set_ylabel("Magnitude (dB, arb.)")
    ax.set_title("USRP RX spectrum @ IF — sanity check")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"[plot] saved RX spectrum: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


def save_npz(data: dict, out_npz) -> Path:
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in data.items()
              if isinstance(v, (np.ndarray, list))}
    scalars = {k: v for k, v in data.items()
               if not isinstance(v, (np.ndarray, list, dict))}
    np.savez_compressed(out_npz, **arrays,
                        meta_json=json.dumps(scalars, ensure_ascii=False))
    print(f"[npz] saved: {out_npz}")
    return out_npz


def load_npz(path) -> dict:
    path = Path(path)
    z = np.load(str(path), allow_pickle=False)
    out: dict = {}
    for k in z.files:
        if k == "meta_json":
            out.update(json.loads(str(z[k])))
        else:
            out[k] = np.asarray(z[k])
    return out


def make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def strip_trailing_n_tag(stem: str) -> str:
    base, sep, suffix = stem.rpartition("_N")
    if sep and suffix.isdigit():
        return base
    return stem


def resolve_output_dir(out_hint, *, timestamp: str, n: Optional[int] = None) -> Path:
    raw_path = Path(out_hint)
    stem = strip_trailing_n_tag(raw_path.stem if raw_path.suffix else raw_path.name)
    parent = raw_path.parent
    base_dir = parent if raw_path.is_absolute() else DATA_DIR / parent
    n_tag = f"_N{int(n)}" if n is not None else ""
    candidate = base_dir / f"{stem}_{timestamp}{n_tag}"
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    for idx in range(1, 1000):
        numbered = base_dir / f"{stem}_{timestamp}_{idx:03d}{n_tag}"
        if not numbered.exists():
            numbered.mkdir(parents=True, exist_ok=False)
            return numbered
    raise RuntimeError(f"failed to create unique output directory for {candidate}")


def resolve_output_file(run_dir, out_path, *, default_suffix: str) -> Path:
    raw_path = Path(out_path)
    suffix = raw_path.suffix or default_suffix
    stem = raw_path.stem if raw_path.suffix else raw_path.name
    return Path(run_dir) / f"{stem}{suffix}"


def resolve_input_path(in_path) -> Path:
    path = Path(in_path)
    if path.is_absolute() or path.exists():
        return path
    data_path = DATA_DIR / path
    if data_path.exists():
        return data_path
    return path


def synthesize_demo_data(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    df_tone_hz: float,
    p_lo_dbm_grid: np.ndarray,
    p_rf_dbm_grid: np.ndarray,
    *,
    conv_loss_db: float = 7.0,
    lo_drive_threshold_dbm: float = 0.0,
    noise_uv: float = 1.0,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(int(seed))
    n_lo = len(p_lo_dbm_grid)
    n_rf = len(p_rf_dbm_grid)

    p_lo = np.asarray(p_lo_dbm_grid, dtype=float)
    p_rf = np.asarray(p_rf_dbm_grid, dtype=float)
    margin = p_lo - lo_drive_threshold_dbm
    convgain_db = -conv_loss_db - 5.0 * np.exp(-margin / 3.0)
    ip_mag_sq = ip_magnitude_squared(vec_a, vec_b)
    p_if_dbm = p_rf[None, :] + convgain_db[:, None] + 10.0 * np.log10(max(ip_mag_sq, 1e-30))
    p_if_w = 10.0 ** (p_if_dbm / 10.0) * 1e-3
    v_if_rms = np.sqrt(p_if_w * PORT_R_OHM)
    v_if_uv = v_if_rms * 1e6

    R = 3
    if_amp_uv_all = (v_if_uv[:, :, None]
                     + rng.normal(0.0, float(noise_uv), size=(n_lo, n_rf, R)))

    return {
        "p_lo_dbm_grid": p_lo,
        "p_rf_dbm_grid": p_rf,
        "actual_p_lo_dbm": np.broadcast_to(p_lo[:, None], (n_lo, n_rf)).copy(),
        "actual_p_rf_dbm": np.broadcast_to(p_rf[None, :], (n_lo, n_rf)).copy(),
        "actual_gain_lo": np.full((n_lo, n_rf), np.nan),
        "actual_gain_rf": np.full((n_lo, n_rf), np.nan),
        "actual_amp_lo": np.full((n_lo, n_rf), np.nan),
        "actual_amp_rf": np.full((n_lo, n_rf), np.nan),
        "peak_limited_lo": np.zeros((n_lo, n_rf), dtype=bool),
        "peak_limited_rf": np.zeros((n_lo, n_rf), dtype=bool),
        "if_amp_uv_all": if_amp_uv_all,
        "if_amp_uv_mean": np.mean(if_amp_uv_all, axis=2),
        "if_amp_uv_std": np.std(if_amp_uv_all, axis=2, ddof=0),
        "vec_a": np.asarray(vec_a, dtype=np.complex128),
        "vec_b": np.asarray(vec_b, dtype=np.complex128),
        "vec_N": int(vec_a.shape[0]),
        "df_tone_hz": float(df_tone_hz),
        "ip_magnitude_squared": ip_mag_sq,
        "if_freq_hz": float(IF_FREQ_HZ),
        "freq_rf_hz": float(FREQ_RF_HZ),
        "freq_lo_hz": float(FREQ_LO_HZ),
        "p_max_dbm": float(P_MAX_DBM_DEFAULT),
        "elapsed_s": 0.0,
        "interrupted": False,
        "_synthetic": True,
        "_conv_loss_db": float(conv_loss_db),
        "reader_backend": "synthetic",
        "measured_units": "arb_synthetic",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GNU Radio: USRP X310 + UBX-160 -> ZEM-4300 -> USRP RX, "
                    "vector inner product heatmap (replaces N9020A with second USRP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--N", type=int, default=N_DEFAULT,
                   help="Vector length N. N=1 degenerates to a single tone.")
    p.add_argument("--df-tone-hz", type=float, default=DF_TONE_HZ_DEFAULT,
                   help="Adjacent tone spacing (Hz). Should be << RX bandwidth, and ideally divide fs (10 MHz) evenly.")
    p.add_argument("--vector-seed", type=int, default=VECTOR_SEED_DEFAULT,
                   help="Random seed determining vec_a, vec_b. Kept fixed within one sweep.")

    p.add_argument("--papr-trials", type=int, default=512,
                   help="Number of SLM random-phase trials. Set 0 to disable PAPR reduction.")
    p.add_argument("--papr-oversample", type=int, default=8,
                   help="Oversampling factor used to estimate the continuous-time peak.")
    p.add_argument("--papr-seed", type=int, default=12345,
                   help="SLM random seed.")

    p.add_argument("--out-npz", type=str, default=f"{DEFAULT_OUTPUT_STEM}.npz",
                   help="Output NPZ filename/path. Each run creates a new timestamped directory.")
    p.add_argument("--out-png", type=str, default=f"{DEFAULT_OUTPUT_STEM}.png",
                   help="Output PNG filename. Saved inside this run's directory.")

    p.add_argument("--p-lo-dbm-min", type=float, default=P_LO_DBM_MIN_DEFAULT)
    p.add_argument("--p-lo-dbm-max", type=float, default=P_LO_DBM_MAX_DEFAULT)
    p.add_argument("--n-lo", type=int, default=N_LO_DEFAULT)
    p.add_argument("--p-rf-dbm-min", type=float, default=P_RF_DBM_MIN_DEFAULT)
    p.add_argument("--p-rf-dbm-max", type=float, default=P_RF_DBM_MAX_DEFAULT)
    p.add_argument("--n-rf", type=int, default=N_RF_DEFAULT)
    p.add_argument("--repeats-each", type=int, default=REPEATS_EACH_DEFAULT)
    p.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT)

    p.add_argument("--tx-args", type=str, default=TX_ARGS)
    p.add_argument("--p-max-dbm", type=float, default=P_MAX_DBM_DEFAULT)

    p.add_argument("--rx-args", type=str, default=RX_ARGS)
    p.add_argument("--rx-gain", type=float, default=RX_GAIN_DB_DEFAULT,
                   help="RX analog gain (dB). Decrease on clipping; increase if the noise floor is too high.")
    p.add_argument("--rx-n-capture", type=int, default=RX_N_CAPTURE_DEFAULT,
                   help="Number of fresh samples grabbed per reading to estimate DC. Larger means lower noise but slower.")
    p.add_argument("--rx-uv-scale", type=float, default=RX_UV_SCALE_DEFAULT,
                   help="Arbitrary scaling constant (uncalibrated); only keeps the uV values at a reasonable magnitude.")
    p.add_argument("--if-atten-db", type=float, default=IF_ATTEN_DB_DEFAULT,
                   help="External fixed attenuator (dB) between mixer IF output and RX. Only refers "
                        "dBm-like readings back to the mixer output and records it in the NPZ; does not change the heatmap shape.")
    p.add_argument("--no-ext-ref", action="store_true",
                   help="Do not lock RX to the external reference (control run; readings will be inaccurate because the center tone slowly rotates).")

    p.add_argument("--norm", choices=("log", "gamma", "linear"), default="log")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--show", action="store_true")

    p.add_argument("--dry-run", action="store_true",
                   help="No hardware; plot from synthetic data (gnuradio not required).")
    p.add_argument("--replot", type=str, default=None,
                   help="Replot from an existing NPZ.")
    p.add_argument("--test-rx", action="store_true",
                   help="Only start the flowgraph, capture an RX segment at fixed power, save a spectrum PNG and print DC readings.")
    p.add_argument("--test-rx-p-lo", type=float, default=0.0,
                   help="LO power (dBm) used with --test-rx.")
    p.add_argument("--test-rx-p-rf", type=float, default=0.0,
                   help="RF power (dBm) used with --test-rx.")

    return p


def _build_flowgraph(args, vec_a, vec_b) -> "DualUSRPVectorFlowgraph":
    if not _GR_OK:
        raise RuntimeError(
            f"GNU Radio (gnuradio.gr/uhd/blocks) unavailable; cannot run live mode. "
            f"import failed: {_GR_ERR!r}\n"
            f"Run on a machine with GNU Radio + UHD installed (not needed for dry-run/replot)."
        )
    clock_source = "internal" if args.no_ext_ref else RX_CLOCK_SOURCE_DEFAULT
    tb = DualUSRPVectorFlowgraph(
        vec_a=vec_a, vec_b=vec_b, df_tone_hz=args.df_tone_hz,
        tx_args=args.tx_args, rx_args=args.rx_args,
        sample_rate=USRP_SAMPLE_RATE,
        freq_rf=FREQ_RF_HZ, freq_lo=FREQ_LO_HZ, lo_offset=USRP_LO_OFFSET,
        tx_antenna=TX_ANTENNA, rx_antenna=RX_ANTENNA,
        rx_gain_db=float(args.rx_gain),
        rx_clock_source=clock_source,
        p_max_dbm=float(args.p_max_dbm),
        verbose=True,
    )
    return tb


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_timestamp = make_run_timestamp()

    if args.N < 1:
        raise ValueError(f"--N must be >= 1, got {args.N}")

    n_tag = f"_N{args.N}"
    if args.replot is None:
        args.out_npz = args.out_npz.replace(".npz", f"{n_tag}.npz") if ".npz" in args.out_npz else f"{args.out_npz}{n_tag}.npz"
        args.out_png = args.out_png.replace(".png", f"{n_tag}.png") if ".png" in args.out_png else f"{args.out_png}{n_tag}.png"

    if args.replot is not None:
        data = load_npz(resolve_input_path(args.replot))
        replot_n = int(np.asarray(data.get("vec_N", args.N)).item())
        out_dir = resolve_output_dir(args.out_png, timestamp=run_timestamp, n=replot_n)
        out_png = resolve_output_file(out_dir, args.out_png, default_suffix=".png")
        print(f"[main] output DIR: {out_dir}")
        print(f"[main] output PNG: {out_png}")
        data["run_timestamp"] = run_timestamp
        data["output_dir"] = str(out_dir)
        plot_heatmaps(data, out_png, norm=args.norm, gamma=args.gamma, show=args.show)
        return 0

    vec_a, vec_b = generate_unit_norm_vectors(args.N, args.vector_seed)
    ip2 = ip_magnitude_squared(vec_a, vec_b)
    print(f"[main] N={args.N}, df_tone={args.df_tone_hz/1e6:.3f} MHz, "
          f"vector_seed={args.vector_seed}")
    print(f"[main] vec_a (LO, raw) = {np.array2string(vec_a, precision=4, suppress_small=True)}")
    print(f"[main] vec_b (RF, raw) = {np.array2string(vec_b, precision=4, suppress_small=True)}")
    print(f"[main] |<a,b>|^2 = {ip2:.6e}  ({10*np.log10(max(ip2,1e-30)):+.2f} dB rel. perfect single tone)")

    vec_a_raw = vec_a.copy()
    vec_b_raw = vec_b.copy()
    vec_a, vec_b, papr_info = reduce_papr_by_phase_rotation(
        vec_a, vec_b,
        df_tone_hz=args.df_tone_hz,
        fs_hz=USRP_SAMPLE_RATE,
        n_trials=int(args.papr_trials),
        oversample=int(args.papr_oversample),
        seed=int(args.papr_seed),
        verbose=True,
    )
    if papr_info["applied"]:
        print(f"[main] vec_a (LO, after PAPR opt) = "
              f"{np.array2string(vec_a, precision=4, suppress_small=True)}")
        print(f"[main] vec_b (RF, after PAPR opt) = "
              f"{np.array2string(vec_b, precision=4, suppress_small=True)}")

    def _attach_meta(data: dict, out_dir: Path) -> None:
        data["run_timestamp"] = run_timestamp
        data["output_dir"] = str(out_dir)
        data["vector_seed"] = int(args.vector_seed)
        data["vec_a_raw"] = vec_a_raw
        data["vec_b_raw"] = vec_b_raw
        data["papr_phi"] = papr_info["phi"]
        data["papr_applied"] = bool(papr_info["applied"])
        data["papr_a_before_db"] = float(papr_info["papr_a_before_db"])
        data["papr_b_before_db"] = float(papr_info["papr_b_before_db"])
        data["papr_a_after_db"] = float(papr_info["papr_a_after_db"])
        data["papr_b_after_db"] = float(papr_info["papr_b_after_db"])
        data["papr_n_trials"] = int(papr_info["n_trials"])
        data["papr_seed"] = int(papr_info["seed"])
        data["if_atten_db"] = float(args.if_atten_db)
        data["rx_gain_db"] = float(args.rx_gain)
        data["rx_uv_scale"] = float(args.rx_uv_scale)

    if args.dry_run:
        p_lo = np.linspace(args.p_lo_dbm_min, args.p_lo_dbm_max, args.n_lo)
        p_rf = np.linspace(args.p_rf_dbm_min, args.p_rf_dbm_max, args.n_rf)
        out_dir = resolve_output_dir(args.out_npz, timestamp=run_timestamp, n=args.N)
        out_npz = resolve_output_file(out_dir, args.out_npz, default_suffix=".npz")
        out_png = resolve_output_file(out_dir, args.out_png, default_suffix=".png")
        print(f"[main] output DIR: {out_dir}")
        print(f"[main] output NPZ: {out_npz}")
        print(f"[main] output PNG: {out_png}")
        print("[main] DRY RUN: no hardware, synthetic data.")
        data = synthesize_demo_data(vec_a, vec_b, args.df_tone_hz, p_lo, p_rf)
        _attach_meta(data, out_dir)
        save_npz(data, out_npz)
        plot_heatmaps(data, out_png, norm=args.norm, gamma=args.gamma, show=args.show)
        return 0

    print("[main] starting GNU Radio dual-USRP flowgraph ...")
    try:
        tb = _build_flowgraph(args, vec_a, vec_b)
    except Exception:
        traceback.print_exc()
        return 1

    tb.start()
    time.sleep(0.5)
    try:
        tb.check_locks()
    except Exception:
        pass

    reader = USRPVectorReader(
        tb.grabber, fs_hz=USRP_SAMPLE_RATE, if_center_hz=IF_FREQ_HZ,
        n_capture=int(args.rx_n_capture), uv_scale=float(args.rx_uv_scale),
        if_atten_db=float(args.if_atten_db),
        verbose=True,
    )

    if args.test_rx:
        out_dir = resolve_output_dir(args.out_png, timestamp=run_timestamp, n=args.N)
        out_png = resolve_output_file(
            out_dir, args.out_png.replace(".png", "_rxtest.png"), default_suffix=".png")
        print(f"[main] TEST-RX: P_LO={args.test_rx_p_lo:+.1f} dBm, "
              f"P_RF={args.test_rx_p_rf:+.1f} dBm, IF atten={args.if_atten_db:.1f} dB")
        tb.set_powers(args.test_rx_p_rf, args.test_rx_p_lo)
        time.sleep(0.3)
        for k in range(3):
            try:
                uv = reader.read_uv()
                dbm_rx = reader.read_dbm(at_mixer_if=False)
                dbm_if = reader.read_dbm(at_mixer_if=True)
                print(f"[test-rx] read #{k}: IF DC = {uv:.4f} uV-like  |  "
                      f"{dbm_rx:+.2f} dBm-like @RX in  ->  {dbm_if:+.2f} dBm-like @mixer IF out "
                      f"(+{args.if_atten_db:.0f} dB pad, UNCAL)")
            except Exception as exc:
                print(f"[test-rx] read #{k} FAILED: {exc!r}")
        spec = reader.capture_spectrum()
        if spec.get("ok"):
            print(f"[test-rx] DC |.|={spec['abs_dc']:.5f}  peak={spec['peak']:.4f}  "
                  f"clip_frac={spec['clip_frac']*100:.3f}%")
            plot_rx_spectrum(spec, out_png, if_center_hz=IF_FREQ_HZ, show=args.show)
        tb.stop()
        tb.wait()
        return 0

    p_lo = np.linspace(args.p_lo_dbm_min, args.p_lo_dbm_max, args.n_lo)
    p_rf = np.linspace(args.p_rf_dbm_min, args.p_rf_dbm_max, args.n_rf)
    out_dir = resolve_output_dir(args.out_npz, timestamp=run_timestamp, n=args.N)
    out_npz = resolve_output_file(out_dir, args.out_npz, default_suffix=".npz")
    out_png = resolve_output_file(out_dir, args.out_png, default_suffix=".png")
    print(f"[main] output DIR: {out_dir}")
    print(f"[main] output NPZ: {out_npz}")
    print(f"[main] output PNG: {out_png}")

    try:
        data = run_sweep(
            set_powers=tb.set_powers,
            sa_read_uv=reader.read_uv,
            vec_a=vec_a, vec_b=vec_b, df_tone_hz=args.df_tone_hz,
            p_lo_dbm_grid=p_lo, p_rf_dbm_grid=p_rf,
            repeats_each=args.repeats_each, settle_s=args.settle_s,
            p_max_dbm=args.p_max_dbm,
        )
    except Exception:
        traceback.print_exc()
        tb.stop()
        tb.wait()
        return 1

    tb.stop()
    tb.wait()

    _attach_meta(data, out_dir)
    save_npz(data, out_npz)
    plot_heatmaps(data, out_png, norm=args.norm, gamma=args.gamma, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
