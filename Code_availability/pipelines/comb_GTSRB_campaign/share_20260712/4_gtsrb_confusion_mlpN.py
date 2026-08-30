#!/usr/bin/env python3

from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import argparse
import gc
import json
import math
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception as _exc:
    _MPL_OK = False
    _MPL_ERR = _exc

_GR_OK = True
_GR_ERR: Optional[BaseException] = None
try:
    from gnuradio import gr, blocks, uhd
    import pmt
except Exception as _exc:
    _GR_OK = False
    _GR_ERR = _exc


SCRIPT_DIR = _data_dir(__file__)
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_STEM = "gtsrb_confusion_mlpN"


TX_ARGS = "addr=192.168.30.2,second_addr=192.168.40.2,master_clock_rate=200e6"
USRP_SAMPLE_RATE = 10e6
USRP_LO_OFFSET = 5e6
TX_ANTENNA = "TX/RX"

FREQ_RF_HZ = 1.20e9
FREQ_LO_HZ = 0.90e9
IF_FREQ_HZ = abs(FREQ_RF_HZ - FREQ_LO_HZ)


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
RX_CLIP_THRESH = 0.98
RX_RING_CAPACITY = 1 << 24

RF_ATTEN_DB_DEFAULT = 30.0
IF_ATTEN_DB_DEFAULT = 0.0


FFT_LEN_DEFAULT = 16384
CP_LEN_DEFAULT = 512
K0_DEFAULT = 6
GAP_SAMPLES_DEFAULT = 8192
SYNC_BAND_FRAC = 8
SYNC_SEED_DEFAULT = 12345
BW_FRAC_DEFAULT = 0.8

IMAGES_PER_FRAME_DEFAULT = 300
N_TEST_DEFAULT = 300


P_LO_DBM_DEFAULT = -3.0
P_RF_DBM_DEFAULT = -35.0


SNR_THERMAL_DB = None

SETTLE_S_DEFAULT = 0.6
CAPTURE_RETRIES_DEFAULT = 6
DECODE_RETRIES_DEFAULT = 2
START_LEAD_S_DEFAULT = 0.2

GTSRB_MIRROR = ("https://codeload.github.com/parhamzm/"
                "German-Traffic-Signs-Dataset-GTSRB/tar.gz/refs/heads/master")
GTSRB_CACHE = "gtsrb_roi_32x32.npz"
PSTRETCH_LO, PSTRETCH_HI = 2.0, 98.0
NCLS = 43
IMG = 32
DIM = IMG * IMG * 3


@dataclass
class GainAmp:
    gain_db: float
    amp: float
    predicted_dbm: float


def power_to_gain_amp(target_dbm: float, p_max_dbm: float = P_MAX_DBM_DEFAULT,
                      amp_target: float = AMP_TARGET) -> GainAmp:
    gain = GAIN_MAX_DB + target_dbm - p_max_dbm - 20.0 * np.log10(amp_target)
    amp = amp_target
    if gain > GAIN_MAX_DB:
        gain = GAIN_MAX_DB
        amp = 10.0 ** ((target_dbm - p_max_dbm) / 20.0)
        if amp > AMP_MAX:
            warnings.warn(f"target {target_dbm:+.2f} dBm exceeds device max; "
                          f"clipping amp to {AMP_MAX}.", RuntimeWarning)
            amp = AMP_MAX
    elif gain < GAIN_MIN_DB:
        gain = GAIN_MIN_DB
        amp = 10.0 ** ((target_dbm - (p_max_dbm - GAIN_MAX_DB)) / 20.0)
        if amp < AMP_WARN:
            warnings.warn(f"target {target_dbm:+.2f} dBm requires amp={amp:.2e} "
                          f"(< {AMP_WARN:.0e}); may be below DAC SFDR.", RuntimeWarning)
    predicted = p_max_dbm + (gain - GAIN_MAX_DB) + 20.0 * np.log10(amp)
    return GainAmp(gain_db=gain, amp=float(amp), predicted_dbm=float(predicted))


def add_cp(sym: np.ndarray, cp: int) -> np.ndarray:
    return np.concatenate([sym[-cp:], sym])


def symbol_td_from_bins(vec: np.ndarray, bins: np.ndarray, L: int) -> np.ndarray:
    X = np.zeros(L, dtype=np.complex128)
    X[np.mod(bins, L)] = vec
    return np.fft.ifft(X)


def gen_sync_symbols(L: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:

    half = L // SYNC_BAND_FRAC
    bins = np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])
    rng_a = np.random.default_rng([int(seed), 777])
    rng_b = np.random.default_rng([int(seed), 778])
    qpsk = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * rng_a.integers(0, 4, bins.size)))
    sa = symbol_td_from_bins(qpsk, bins, L)
    qpsk = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * rng_b.integers(0, 4, bins.size)))
    sb = symbol_td_from_bins(qpsk, bins, L)
    return sa, sb


def zc_phase(D: int) -> np.ndarray:

    n = np.arange(D); cf = D % 2
    return np.exp(-1j * np.pi * n * (n + cf) / D)


@dataclass
class LayerPlan:

    D: int
    H: int
    K: int
    M: int
    R: int
    base: int
    k0: int
    kmax: int
    out_bins: np.ndarray
    guard_bins: np.ndarray


def _kmax_for(D: int, K: int, k0: int) -> Tuple[int, int, int]:

    M = K + 2
    base = -(M * (D - 1)) // 2
    b_lo, b_hi = base, M * (D - 1) + base
    a_lo = base - k0 - (K - 1)
    kmax = int(max(abs(b_lo), abs(b_hi), abs(a_lo)))
    return kmax, M, base


def plan_layer(D: int, H: int, L: int, k0: int, bw_frac: float,
               k_cap: Optional[int] = None) -> LayerPlan:

    limit = bw_frac * L / 2.0
    hi = int(min(H, k_cap)) if k_cap else int(H)
    K_max = 0
    for K in range(hi, 0, -1):
        kmax, _, _ = _kmax_for(D, K, k0)
        if kmax <= limit:
            K_max = K
            break
    if K_max < 1:
        raise ValueError(f"layer {D}->{H}: 带宽放不下 K=1 (kmax={_kmax_for(D,1,k0)[0]}"
                         f" > {limit:.0f}); 增大 --fft-len 或 --bw-frac")
    R = int(math.ceil(H / K_max))
    K = int(math.ceil(H / R))
    kmax, M, base = _kmax_for(D, K, k0)
    out_bins = k0 + np.arange(K, dtype=np.int64)
    guard_bins = np.array([k0 - 2, k0 - 1, k0 + K, k0 + K + 1], dtype=np.int64)
    return LayerPlan(D=D, H=H, K=K, M=M, R=R, base=base, k0=k0, kmax=kmax,
                     out_bins=out_bins, guard_bins=guard_bins)


@dataclass
class FrameSpecML:

    L: int
    cp: int
    plan: LayerPlan
    gap: int
    layer_tag: int
    n_vec: int
    n_sym: int
    labels: np.ndarray
    vec_index: np.ndarray
    y_digital: np.ndarray
    sA: np.ndarray
    sB: np.ndarray
    ref_pre: np.ndarray
    peak_a: float
    peak_b: float
    frame_len: int
    sync_payload_off: int

    @property
    def out_bins(self) -> np.ndarray:
        return self.plan.out_bins

    @property
    def guard_bins(self) -> np.ndarray:
        return self.plan.guard_bins


def build_frame_ml(W_t: np.ndarray, x_tilde: np.ndarray, labels: np.ndarray,
                   vec_index: np.ndarray, plan: LayerPlan, *, L: int, cp: int,
                   gap: int, sync_seed: int, layer_tag: int,
                   verbose: bool = True,
                   compute_digital: bool = True) -> FrameSpecML:

    H, D = W_t.shape
    assert (H, D) == (plan.H, plan.D)
    n_vec = x_tilde.shape[0]
    K, M, R, base, k0 = plan.K, plan.M, plan.R, plan.base, plan.k0
    n = np.arange(D, dtype=np.int64)
    binB = M * n + base


    wsyms_cp: List[np.ndarray] = []
    for r in range(R):
        Afreq = np.zeros(L, dtype=np.complex128)
        r0 = r * K
        for m in range(min(K, H - r0)):
            Afreq[np.mod(binB - k0 - m, L)] = np.conj(W_t[r0 + m, :])
        wsyms_cp.append(add_cp(np.fft.ifft(Afreq), cp))

    sa_pre, sb_pre = gen_sync_symbols(L, sync_seed)
    parts_a: List[np.ndarray] = [np.zeros(gap, np.complex128), add_cp(sa_pre, cp)]
    parts_b: List[np.ndarray] = [np.zeros(gap, np.complex128), add_cp(sb_pre, cp)]
    for q in range(n_vec):
        Bfreq = np.zeros(L, dtype=np.complex128)
        Bfreq[np.mod(binB, L)] = x_tilde[q]
        bsym_cp = add_cp(np.fft.ifft(Bfreq), cp)
        for r in range(R):
            parts_a.append(wsyms_cp[r])
            parts_b.append(bsym_cp)
    parts_a.append(np.zeros(gap, np.complex128))
    parts_b.append(np.zeros(gap, np.complex128))

    sA = np.concatenate(parts_a)
    sB = np.concatenate(parts_b)
    frame_len = sA.size
    act = slice(gap, frame_len - gap)

    rms_a = float(np.sqrt(np.mean(np.abs(sA[act]) ** 2)))
    rms_b = float(np.sqrt(np.mean(np.abs(sB[act]) ** 2)))
    sA /= max(rms_a, 1e-30)
    sB /= max(rms_b, 1e-30)
    peak_a = float(np.max(np.abs(sA)))
    peak_b = float(np.max(np.abs(sB)))

    ref = sb_pre * np.conj(sa_pre)
    ref = (ref / np.linalg.norm(ref)).astype(np.complex128)

    # Deployment/energy-honest path: with compute_digital=False the host never
    # performs W@x -- the mixer does the matrix operation and a FROZEN
    # per-column calibration (measured once, offline) replaces the per-image
    # digital reference. compute_digital=True is a lab/diagnostic convenience
    # only (per-image ground truth for equalization + drift metrics), and must
    # NOT be used when demonstrating the energy claim.
    if compute_digital:
        y_digital = (x_tilde @ W_t.T).astype(np.complex128)
    else:
        y_digital = np.zeros((0, W_t.shape[0]), np.complex128)

    fr = FrameSpecML(
        L=L, cp=cp, plan=plan, gap=gap, layer_tag=int(layer_tag),
        n_vec=n_vec, n_sym=n_vec * R,
        labels=labels.astype(np.int64), vec_index=vec_index.astype(np.int64),
        y_digital=y_digital,
        sA=sA.astype(np.complex64), sB=sB.astype(np.complex64),
        ref_pre=ref, peak_a=peak_a, peak_b=peak_b, frame_len=frame_len,
        sync_payload_off=gap + cp)
    if verbose:
        df = USRP_SAMPLE_RATE / L
        print(f"[frame L{layer_tag}] {D}->{H}: K={K}/符号, M={M}, R={R} 符号/样本, "
              f"base={base}; 本帧 {n_vec} 样本 = {fr.n_sym} data 符号")
        print(f"[frame L{layer_tag}] 输出 tone bins = {k0}..{k0+K-1} "
              f"({k0*df/1e3:.2f}..{(k0+K-1)*df/1e3:.2f} kHz), "
              f"保护 bins = {plan.guard_bins.tolist()}")
        print(f"[frame L{layer_tag}] 占用 |k|<={plan.kmax} "
              f"({100*plan.kmax/(L/2):.0f}% Nyquist); "
              f"帧长={frame_len} ({frame_len/USRP_SAMPLE_RATE:.3f} s)")
        print(f"[frame L{layer_tag}] unit-RMS 峰值: LO={peak_a:.3f} "
              f"({20*np.log10(peak_a):+.1f} dB PAPR), RF={peak_b:.3f} "
              f"({20*np.log10(peak_b):+.1f} dB PAPR)")
    return fr


def xcorr_os(y: np.ndarray, ref: np.ndarray, nfft: int = 1 << 18) -> np.ndarray:
    y = np.ascontiguousarray(y, dtype=np.complex128)
    Lr = int(ref.size)
    if nfft < 2 * Lr:
        nfft = 1 << int(math.ceil(math.log2(2 * Lr)))
    hop = nfft - Lr + 1
    H = np.conj(np.fft.fft(ref, nfft))
    n_out = y.size - Lr + 1
    if n_out <= 0:
        return np.zeros(0, np.complex128)
    out = np.empty(n_out, np.complex128)
    pos = 0
    while pos < n_out:
        blk = y[pos:pos + nfft]
        Z = np.fft.ifft(np.fft.fft(blk, nfft) * H)
        take = min(hop, n_out - pos, max(blk.size - Lr + 1, 0))
        if take <= 0:
            break
        out[pos:pos + take] = Z[:take]
        pos += take
    return out


@dataclass
class DecodeML:

    ok: bool
    reason: str = ""
    d0: int = -1
    peak_metric: float = 0.0
    frame_repeat_err: Optional[int] = None
    y_meas: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    vec_index: Optional[np.ndarray] = None
    G_hat: complex = 0j
    logit_rmse: float = float("nan")
    snr_bin_db: float = float("nan")
    sigma2_bin: float = float("nan")
    clip_frac: float = 0.0


def decode_ml(cap: np.ndarray, fr: FrameSpecML, fs: float,
              flip_output: bool = False, conj_output: bool = False,
              verbose: bool = True) -> DecodeML:

    L, cp = fr.L, fr.cp
    plan = fr.plan
    K, R, H, k0 = plan.K, plan.R, plan.H, plan.k0
    S = fr.n_sym
    need_after = L + S * (cp + L) + 64
    if cap.size < fr.frame_len // 2:
        return DecodeML(ok=False, reason=f"capture too short ({cap.size})")

    clip_frac = float(np.mean(np.abs(cap) > RX_CLIP_THRESH))


    corr = xcorr_os(cap, fr.ref_pre)
    mag = np.abs(corr)
    valid_last = cap.size - need_after
    if valid_last < 1:
        return DecodeML(ok=False, reason="capture shorter than one frame")
    d0 = int(np.argmax(mag[:valid_last]))
    pk = float(mag[d0])
    guard = np.copy(mag)
    lo_g = max(0, d0 - 3 * L); hi_g = min(mag.size, d0 + 3 * L)
    guard[lo_g:hi_g] = 0.0
    kk = 1
    while kk <= 5 and (d0 + kk * fr.frame_len < guard.size or d0 - kk * fr.frame_len >= 0):
        for rr in (d0 + kk * fr.frame_len, d0 - kk * fr.frame_len):
            if 0 <= rr < guard.size:
                guard[max(0, rr - 64): min(guard.size, rr + 64)] = 0.0
        kk += 1
    side = float(np.max(guard)) if guard.size else 1e-30
    peak_metric = pk / max(side, 1e-30)
    frame_repeat_err = None
    for rcand in (d0 + fr.frame_len, d0 - fr.frame_len):
        if 0 <= rcand < mag.size:
            w = mag[max(0, rcand - 64): min(mag.size, rcand + 65)]
            if w.size:
                frame_repeat_err = int(np.argmax(w) + max(0, rcand - 64) - rcand)
            break
    if verbose:
        print(f"[sync L{fr.layer_tag}] 峰位 d0={d0}, 峰/旁瓣={peak_metric:.1f}, "
              f"帧重复偏差={frame_repeat_err} 样本, clip={clip_frac*100:.3f}%")
    if peak_metric < 3.0:
        return DecodeML(ok=False, reason=f"sync peak weak ({peak_metric:.2f}); "
                        f"检查发射/功率/接线", d0=d0, peak_metric=peak_metric,
                        clip_frac=clip_frac)


    rd = (-fr.out_bins) if flip_output else fr.out_bins
    gd = (-fr.guard_bins) if flip_output else fr.guard_bins
    rd_idx = np.mod(rd, L); gd_idx = np.mod(gd, L)
    y_pad = np.zeros((fr.n_vec, R * K), np.complex128)
    y_noise = np.empty((S, fr.guard_bins.size), np.complex128)
    for s in range(S):
        st = d0 + L + s * (cp + L) + cp
        Y = np.fft.fft(cap[st:st + L].astype(np.complex128))
        q, r = divmod(s, R)
        y_pad[q, r * K:(r + 1) * K] = Y[rd_idx]
        y_noise[s] = Y[gd_idx]
    y_meas = y_pad[:, :H]
    if conj_output:
        y_meas = np.conj(y_meas)


    yd = fr.y_digital
    am = np.abs(y_meas)
    if yd.shape[0] == am.shape[0]:               # lab/diagnostic reference present
        ad = np.abs(yd)
        den = float(np.sum(ad ** 2))
        g = float(np.sum(am * ad) / den) if den > 0 else 0.0
        if g > 0:
            resid = am / g - ad
            logit_rmse = float(np.sqrt(np.mean(resid ** 2)) /
                               max(np.sqrt(np.mean(ad ** 2)), 1e-30))
        else:
            logit_rmse = float("nan")
    else:                                         # deployment: no digital W@x
        g = float("nan")
        logit_rmse = float("nan")
    G = complex(g)
    sigma2_bin = float(np.mean(np.abs(y_noise) ** 2))

    p_sig = float(np.mean(np.abs(y_meas) ** 2))
    snr_bin = max(p_sig - sigma2_bin, 1e-30) / max(sigma2_bin, 1e-30)

    if verbose:
        print(f"[decode L{fr.layer_tag}] {fr.n_vec} 样本 x {H} 输出: "
              f"幅度增益 g={abs(G):.3e}, 幅度 RMSE={logit_rmse:.4f}, "
              f"输出 bin SNR≈{10*np.log10(max(snr_bin,1e-30)):+.1f} dB")

    return DecodeML(
        ok=True, d0=d0, peak_metric=peak_metric, frame_repeat_err=frame_repeat_err,
        y_meas=y_meas, labels=fr.labels.copy(), vec_index=fr.vec_index.copy(),
        G_hat=G, logit_rmse=logit_rmse,
        snr_bin_db=float(10 * np.log10(max(snr_bin, 1e-30))),
        sigma2_bin=sigma2_bin, clip_frac=clip_frac)


if _GR_OK:

    class IQGrabber(gr.sync_block):


        def __init__(self, capacity: int = RX_RING_CAPACITY):
            gr.sync_block.__init__(self, name="IQGrabber",
                                   in_sig=[np.complex64], out_sig=None)
            self._cap = int(capacity)
            self._buf = np.zeros(self._cap, dtype=np.complex64)
            self._widx = 0
            self._total = 0
            self._lock = threading.Lock()
            self._rx_time_count = 0
            self._gap_count = 0
            self._last_gap_total = -1

        def work(self, input_items, output_items):
            x = input_items[0]
            n = len(x)
            if n:
                with self._lock:
                    try:
                        nread = self.nitems_read(0)
                        for tg in self.get_tags_in_window(0, 0, n):
                            if pmt.symbol_to_string(tg.key) == "rx_time":
                                self._rx_time_count += 1
                                if self._rx_time_count > 1:
                                    rel = max(0, int(tg.offset) - int(nread))
                                    self._last_gap_total = self._total + rel
                                    self._gap_count += 1
                    except Exception:
                        pass
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

        def gap_count(self) -> int:
            with self._lock:
                return int(self._gap_count)

        def reset(self) -> None:

            with self._lock:
                self._buf[:] = 0
                self._widx = 0
                self._total = 0
                self._rx_time_count = 0
                self._gap_count = 0
                self._last_gap_total = -1

        def grab_fresh_ex(self, n: int, timeout_s: float = 30.0):
            n = int(min(int(n), self._cap))
            start = self.total_samples()
            t0 = time.time()
            while self.total_samples() - start < n:
                if time.time() - t0 > timeout_s:
                    break
                time.sleep(0.005)
            with self._lock:
                total_end = self._total
                last_gap = self._last_gap_total
                idx = (self._widx - n) % self._cap
                if idx + n <= self._cap:
                    out = self._buf[idx:idx + n].copy()
                else:
                    first = self._cap - idx
                    out = np.concatenate([self._buf[idx:], self._buf[:n - first]])
            win_lo = total_end - n
            gap_in_window = (last_gap >= win_lo) and (last_gap < total_end)
            got = total_end - start
            info = {"clean": bool(got >= n and not gap_in_window),
                    "got": int(got), "need": int(n),
                    "gap_in_window": bool(gap_in_window)}
            return out, info

    class MVMFlowgraph(gr.top_block):


        def __init__(self, frame: FrameSpecML, lo_path: str, rf_path: str, *,
                     tx_args: str = TX_ARGS, rx_args: str = RX_ARGS,
                     sample_rate: float = USRP_SAMPLE_RATE,
                     freq_rf: float = FREQ_RF_HZ, freq_lo: float = FREQ_LO_HZ,
                     lo_offset: float = USRP_LO_OFFSET, tx_antenna: str = TX_ANTENNA,
                     rx_antenna: str = RX_ANTENNA, rx_gain_db: float = RX_GAIN_DB_DEFAULT,
                     rx_clock_source: str = RX_CLOCK_SOURCE_DEFAULT,
                     rx_time_source: str = RX_TIME_SOURCE_DEFAULT,
                     ring_capacity: int = RX_RING_CAPACITY,
                     p_max_dbm: float = P_MAX_DBM_DEFAULT, verbose: bool = True):
            gr.top_block.__init__(self, "MLPN MVM GTSRB")
            self._verbose = verbose
            self._sr = float(sample_rate)
            self._p_max_dbm = float(p_max_dbm)
            self._frame = frame

            if verbose:
                print(f"[gr-tx] connecting TX USRP: {tx_args}")
            self.usrp_sink = uhd.usrp_sink(
                ",".join((tx_args, "")),
                uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0, 1]))
            self.usrp_sink.set_samp_rate(self._sr)
            for ch, freq in [(0, freq_rf), (1, freq_lo)]:
                self.usrp_sink.set_center_freq(uhd.tune_request(freq, lo_offset), ch)
                self.usrp_sink.set_antenna(tx_antenna, ch)
                self.usrp_sink.set_gain(15.0, ch)
                if verbose:
                    print(f"[gr-tx] CH{ch}: freq~{freq/1e6:9.3f} MHz ant={tx_antenna}")

            self.src_rf = blocks.file_source(gr.sizeof_gr_complex, str(rf_path), True)
            self.src_lo = blocks.file_source(gr.sizeof_gr_complex, str(lo_path), True)
            self.mul_rf = blocks.multiply_const_cc(complex(0.0, 0.0))
            self.mul_lo = blocks.multiply_const_cc(complex(0.0, 0.0))
            self.connect((self.src_rf, 0), (self.mul_rf, 0), (self.usrp_sink, 0))
            self.connect((self.src_lo, 0), (self.mul_lo, 0), (self.usrp_sink, 1))

            if verbose:
                print(f"[gr-rx] connecting RX USRP: {rx_args}")
            self.usrp_source = uhd.usrp_source(
                ",".join((rx_args, "")),
                uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]))
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

        def check_locks(self) -> None:
            for label, dev, chans in [("TX", self.usrp_sink, (0, 1)),
                                      ("RX", self.usrp_source, (0,))]:
                for ch in chans:
                    try:
                        s = dev.get_sensor("lo_locked", ch)
                        if not bool(s.to_bool()):
                            warnings.warn(f"{label} CH{ch} LO not locked", RuntimeWarning)
                        elif self._verbose:
                            print(f"[gr] {label} CH{ch} lo_locked=True")
                    except Exception:
                        pass
            try:
                s = self.usrp_source.get_mboard_sensor("ref_locked", 0)
                print(f"[gr] RX ref_locked={bool(s.to_bool())} (10 MHz 外参)")
            except Exception:
                pass

        def set_powers(self, p_rf_dbm: float, p_lo_dbm: float) -> dict:
            rf = power_to_gain_amp(p_rf_dbm, p_max_dbm=self._p_max_dbm)
            lo = power_to_gain_amp(p_lo_dbm, p_max_dbm=self._p_max_dbm)
            ceil_rf = AMP_MAX / max(self._frame.peak_b, 1e-12)
            ceil_lo = AMP_MAX / max(self._frame.peak_a, 1e-12)

            def _resolve(want_dbm: float, ga: GainAmp, ceil: float, tag: str):


                gain, amp = float(ga.gain_db), float(ga.amp)
                if amp > ceil:
                    lost_db = 20.0 * np.log10(amp / max(ceil, 1e-30))
                    amp = ceil
                    gain = min(GAIN_MAX_DB, gain + lost_db)
                pred = self._p_max_dbm + (gain - GAIN_MAX_DB) + 20.0 * np.log10(max(amp, 1e-30))
                if pred < want_dbm - 0.3 and self._verbose:
                    head = GAIN_MAX_DB - gain
                    warnings.warn(
                        f"{tag} 仍达不到目标: 实际≈{pred:+.2f} dBm (目标 {want_dbm:+.2f}); "
                        f"增益余量仅 {head:.1f} dB, 波形 PAPR 太高. "
                        f"可降低该路目标功率或 PAPR.", RuntimeWarning)
                return gain, amp, pred

            g_rf, a_rf, pred_rf = _resolve(p_rf_dbm, rf, ceil_rf, "RF")
            g_lo, a_lo, pred_lo = _resolve(p_lo_dbm, lo, ceil_lo, "LO")
            self.mul_rf.set_k(complex(a_rf, 0.0))
            self.mul_lo.set_k(complex(a_lo, 0.0))
            self.usrp_sink.set_gain(g_rf, 0)
            self.usrp_sink.set_gain(g_lo, 1)
            return {"rf": {"gain_db": float(g_rf), "amp_rms": float(a_rf),
                           "predicted_dbm": float(pred_rf)},
                    "lo": {"gain_db": float(g_lo), "amp_rms": float(a_lo),
                           "predicted_dbm": float(pred_lo)}}

        def mute_tx(self) -> None:

            self.mul_rf.set_k(complex(0.0, 0.0))
            self.mul_lo.set_k(complex(0.0, 0.0))

        def start_timed(self, start_lead: float = START_LEAD_S_DEFAULT) -> None:

            try:
                self.usrp_sink.set_time_now(uhd.time_spec_t(0.0))
                self.usrp_sink.set_start_time(uhd.time_spec_t(float(start_lead)))
            except Exception as exc:
                warnings.warn(f"[tx] 定时启动设置失败({exc!r}); 退回即时启动 "
                              f"(可能仍有 U/L)", RuntimeWarning)
            self.start()

        def restart_with_files(self, lo_path: str, rf_path: str,
                               sA: Optional[np.ndarray] = None,
                               sB: Optional[np.ndarray] = None,
                               frame: Optional[FrameSpecML] = None,
                               start_lead: float = START_LEAD_S_DEFAULT) -> None:

            self.stop()
            self.wait()
            if frame is not None:
                self._frame = frame

            if sA is not None:
                np.asarray(sA, dtype=np.complex64).tofile(str(lo_path))
            if sB is not None:
                np.asarray(sB, dtype=np.complex64).tofile(str(rf_path))
            self.grabber.reset()
            self.src_lo.open(str(lo_path), True)
            self.src_rf.open(str(rf_path), True)
            self.start_timed(start_lead)

else:
    IQGrabber = None
    MVMFlowgraph = None


def sim_mix(sA: np.ndarray, sB: np.ndarray, model: str) -> np.ndarray:

    sA = sA.astype(np.complex128); sB = sB.astype(np.complex128)
    if model == "ideal":
        return sB * np.conj(sA)
    if model == "limiter":
        m = np.abs(sA); lo = np.zeros_like(sA); nz = m > 1e-9
        lo[nz] = sA[nz] / m[nz]
        return sB * np.conj(lo)
    raise ValueError(f"unknown mixer model {model!r}")


def synth_capture_ml(fr: FrameSpecML, snr_db: float, mixer: str, seed: int,
                     fs: float, salt: int = 0, verbose: bool = True) -> np.ndarray:

    rng = np.random.default_rng([int(seed), 4242, int(salt),
                                 int(round(snr_db * 10)) if np.isfinite(snr_db) else 999999])
    y_clean = sim_mix(fr.sA, fr.sB, mixer)
    if np.isfinite(snr_db):

        bidx = np.mod(fr.out_bins, fr.L)
        pw = []
        for s in range(fr.n_sym):
            st = fr.sync_payload_off + fr.L + s * (fr.cp + fr.L) + fr.cp
            Y = np.fft.fft(y_clean[st:st + fr.L])
            pw.append(np.abs(Y[bidx]) ** 2)
        p_sig = float(np.mean(pw))
        sigma2_bin = p_sig / (10.0 ** (snr_db / 10.0))
        sigma_t = math.sqrt(sigma2_bin / fr.L)
    else:
        sigma_t = 0.0
    n_cap = int(round(2.3 * fr.frame_len))
    start = int(rng.integers(0, fr.frame_len))
    reps = int(math.ceil((start + n_cap) / fr.frame_len)) + 1
    tiled = np.tile(y_clean, reps)
    cap = tiled[start:start + n_cap].copy()
    if sigma_t > 0:
        cap += (rng.standard_normal(n_cap) + 1j * rng.standard_normal(n_cap)) * (sigma_t / math.sqrt(2.0))
    cap *= 0.3 / max(float(np.max(np.abs(cap))), 1e-30)
    if verbose:
        s = f"{snr_db:.1f} dB" if np.isfinite(snr_db) else "∞ (无噪)"
        print(f"[sim L{fr.layer_tag}] mixer={mixer}, 注入平均输出 bin SNR={s}")
    return cap.astype(np.complex64)


def inject_target_snr(cap: np.ndarray, fr: FrameSpecML, fs: float,
                      target_db: float, seed: int, verbose: bool = True) -> np.ndarray:

    meas = decode_ml(cap, fr, fs, verbose=False)
    if (not meas.ok) or (meas.y_meas is None):
        warnings.warn(f"[target-snr] 无法测量当前 SNR ({meas.reason}); 不注入, 原样返回",
                      RuntimeWarning)
        return cap
    p_total = float(np.mean(np.abs(meas.y_meas) ** 2))
    sigma2_now = float(meas.sigma2_bin)
    p_signal = max(p_total - sigma2_now, 1e-30)
    snr_now_db = 10.0 * np.log10(max(p_signal / max(sigma2_now, 1e-30), 1e-30))
    sigma2_target = p_signal / (10.0 ** (target_db / 10.0))
    sigma2_add = sigma2_target - sigma2_now
    if sigma2_add <= 0:
        warnings.warn(
            f"[target-snr L{fr.layer_tag}] 目标 {target_db:g} dB >= 链路原始 SNR "
            f"{snr_now_db:.1f} dB; 加噪只能降不能升. 要往上: 用 --avg-per-layer 做积分增益 "
            f"(平均 N 次 +10log10(N) dB), 或提高 RF 驱动 (--p-rf-dbm/--paper-snr-db) "
            f"把原始链路抬上去. 本层本帧不注入 (保持原始 SNR).",
            RuntimeWarning)
        return cap
    sigma_t = math.sqrt(sigma2_add / fr.L)
    rng = np.random.default_rng([int(seed), 7777, int(round(target_db * 10))])
    noise = ((rng.standard_normal(cap.size) + 1j * rng.standard_normal(cap.size))
             * (sigma_t / math.sqrt(2.0))).astype(np.complex64)
    out = (cap.astype(np.complex64) + noise).astype(np.complex64)
    if verbose:
        print(f"[target-snr L{fr.layer_tag}] 链路原始 SNR={snr_now_db:.1f} dB -> "
              f"叠加标定 AWGN -> 目标 {target_db:g} dB "
              f"(加噪/原噪 σ²={sigma2_add/max(sigma2_now,1e-30):.2f}x)")
    return out


def inject_thermal_snr(cap: np.ndarray, fr: FrameSpecML, fs: float,
                    snr_db: float, seed: int, verbose: bool = True) -> np.ndarray:

    meas = decode_ml(cap, fr, fs, verbose=False)
    if (not meas.ok) or (meas.y_meas is None):
        warnings.warn(f"[snr-thermal] 无法测量信号功率 ({meas.reason}); 不注入, 原样返回",
                      RuntimeWarning)
        return cap
    p_total = float(np.mean(np.abs(meas.y_meas) ** 2))
    sigma2_now = float(meas.sigma2_bin)
    p_signal = max(p_total - sigma2_now, 1e-30)
    sigma2_add = p_signal / (10.0 ** (snr_db / 10.0))
    sigma_t = math.sqrt(sigma2_add / fr.L)
    rng = np.random.default_rng([int(seed), 20260609, int(round(snr_db * 10))])
    noise = ((rng.standard_normal(cap.size) + 1j * rng.standard_normal(cap.size))
             * (sigma_t / math.sqrt(2.0))).astype(np.complex64)
    out = (cap.astype(np.complex64) + noise).astype(np.complex64)
    if verbose:
        print(f"[snr-thermal L{fr.layer_tag}] 设 signal/thermal SNR = {snr_db:g} dB "
              f"(注 AWGN σ²=P_signal/SNR)")
    return out


def measure_band_power(cap: np.ndarray, fr: FrameSpecML, bins: np.ndarray,
                       max_win: int = 128) -> float:

    L = fr.L
    nwin = int(min(max(cap.size // L, 1), max_win))
    bidx = np.mod(bins, L)
    acc = 0.0
    cnt = 0
    for w in range(nwin):
        seg = cap[w * L:(w + 1) * L]
        if seg.size < L:
            break
        Y = np.fft.fft(seg.astype(np.complex128))
        acc += float(np.mean(np.abs(Y[bidx]) ** 2))
        cnt += 1
    return acc / max(cnt, 1)


def preprocess(u8: np.ndarray) -> np.ndarray:

    x = u8.reshape(u8.shape[0], -1).astype(np.float64) / 255.0
    lo = np.percentile(x, PSTRETCH_LO, axis=1, keepdims=True)
    hi = np.percentile(x, PSTRETCH_HI, axis=1, keepdims=True)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


def _gtsrb_build_cache(root: Path, out_npz: Path, verbose: bool) -> None:

    import csv
    from PIL import Image

    def build(csvname):
        rows = list(csv.DictReader(open(root / csvname)))
        X = np.empty((len(rows), IMG, IMG, 3), np.uint8)
        y = np.empty(len(rows), np.int64)
        for i, r in enumerate(rows):
            im = Image.open(root / r["Path"]).convert("RGB")
            im = im.crop((int(r["Roi.X1"]), int(r["Roi.Y1"]),
                          int(r["Roi.X2"]), int(r["Roi.Y2"])))
            X[i] = np.asarray(im.resize((IMG, IMG), Image.BILINEAR), np.uint8)
            y[i] = int(r["ClassId"])
        return X, y

    Xtr, ytr = build("Train.csv")
    Xte, yte = build("Test.csv")
    names = [r["SignName"] for r in csv.DictReader(open(root / "signnames.csv"))]
    np.savez_compressed(out_npz, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        class_names=np.array(names))
    if verbose:
        print(f"[data] GTSRB ROI 32x32 缓存 -> {out_npz}")


def load_gtsrb_test(verbose: bool = True):

    import io
    import tarfile
    import urllib.request
    for cand in (SCRIPT_DIR / GTSRB_CACHE, Path.home() / GTSRB_CACHE):
        if cand.exists():
            d = np.load(cand, allow_pickle=True)
            if verbose:
                print(f"[data] 载入缓存 {cand}")
            return preprocess(d["Xte"]), d["yte"].astype(np.int64)
    root = SCRIPT_DIR / "GTSRB"
    if not (root / "Test.csv").exists():
        if verbose:
            print(f"[data] 下载 GTSRB 镜像 (~300 MB): {GTSRB_MIRROR}")
        buf = urllib.request.urlopen(GTSRB_MIRROR, timeout=600).read()
        with tarfile.open(fileobj=io.BytesIO(buf), mode="r:gz") as tf:
            top = tf.getnames()[0].split("/")[0]
            tf.extractall(SCRIPT_DIR)
        (SCRIPT_DIR / top).rename(root)
        if verbose:
            print(f"[data] 解压 -> {root}")
    out = SCRIPT_DIR / GTSRB_CACHE
    _gtsrb_build_cache(root, out, verbose)
    d = np.load(out, allow_pickle=True)
    return preprocess(d["Xte"]), d["yte"].astype(np.int64)


def load_weights_ml(path: Path):

    d = np.load(path, allow_pickle=True)
    n_layers = int(d["n_layers"]) if "n_layers" in d else 3
    Ws = [d[f"W{l+1}"].astype(np.complex128) for l in range(n_layers)]
    dims = [Ws[0].shape[1]] + [W.shape[0] for W in Ws]
    act = str(d["act"]) if "act" in d else "abs"
    test_acc = float(d["test_acc"]) if "test_acc" in d else float("nan")
    qlev = int(d["quant_levels"]) if "quant_levels" in d else 0
    if act != "abs":
        raise ValueError(f"权重 act={act!r} != 'abs'; 本脚本层间激活为模值 |·|, "
                         f"权重必须由 gtsrb_train_digital_deep.py 以同一激活训练")
    for l in range(1, n_layers):
        if Ws[l].shape[1] != Ws[l - 1].shape[0]:
            raise ValueError(f"W{l+1} 形状 {Ws[l].shape} 与 W{l} 输出 "
                             f"{Ws[l-1].shape[0]} 不匹配")
    return Ws, dims, test_acc, qlev


def load_input_eq(path, D: int, cap: float = 3.0,
                  current_args=None) -> np.ndarray:
    """载入层 1 逐输入 tone 响应 h (make_input_eq 拟合; 键 'h', 复数, 长度 D).

    ⚠ 诊断-h 因果模型已被 run 6 硬件干预否证 (72.3% vs 预测 ~92.9%, 见
    eq_evidence/README.md) — 此路径保留为干预工具, 不是已验证的精度改进.

    返回 TX 预均衡因子 scale = 1/h, |scale| 截幅在 cap (保相位) — 防止
    近零 h 的 tone 把 PAPR/DAC 撑爆. 打印溯源 + 截幅统计; 有
    fingerprint_json 且给了 current_args 则逐项对比 (同 --calib-npz 约定,
    不一致只告警). 硬校验: cap >= 1, 1/h 不溢出, 截幅比例 <= 50%."""
    if not (cap >= 1.0):
        raise ValueError(f"--input-eq-cap 必须 >= 1 (got {cap!r}); cap 是 "
                         f"|1/h| 提升上限 (均值 |h|=1 规范下 cap<1 无意义)")
    z = np.load(path, allow_pickle=True)
    if "h" not in z.files:
        raise ValueError(f"{path}: 缺 'h' 键 (make_input_eq 生成的 npz)")
    h = np.asarray(z["h"], np.complex128)
    if h.shape != (D,):
        raise ValueError(f"{path}: h 形状 {h.shape} != ({D},)")
    if not np.all(np.isfinite(h)) or np.any(np.abs(h) == 0):
        raise ValueError(f"{path}: h 含非有限值或零")
    if "provenance" in z.files:
        print(f"[input-eq] provenance: {str(z['provenance'])}")
    if "fingerprint_json" in z.files and current_args is not None:
        try:
            fp = json.loads(str(z["fingerprint_json"]))
        except (ValueError, TypeError):
            fp = {}
        for k, v in fp.items():
            cur = getattr(current_args, k, None)
            if cur != v:
                print(f"[input-eq] ⚠ fingerprint mismatch: {k} eq={v!r} "
                      f"vs current={cur!r} (h 拟合源 run 的链路参数与当前"
                      f"不同, 预均衡可能不适用)")
    with np.errstate(over="ignore", invalid="ignore"):
        scale = 1.0 / h
        mag = np.abs(scale)
        n_clip = int(np.sum(mag > cap))
        scale = np.where(mag > cap, scale * (cap / mag), scale)
    if not np.all(np.isfinite(scale)):
        raise ValueError(f"{path}: 1/h 溢出 (min|h|={np.abs(h).min():.3e}); "
                         f"h 事实上不可逆, 拒绝上硬件")
    if n_clip / D > 0.5:
        raise ValueError(
            f"{path}: 截幅比例 {100*n_clip/D:.1f}% > 50% (run5a 参考 7.9%);"
            f" h 病态或 cap 错单位 — 重新拟合或有意识地调 --input-eq-cap")
    print(f"[input-eq] TX 预均衡: |1/h| 中位 {np.median(mag):.2f}, "
          f"截幅@{cap:g}: {n_clip}/{D} tones ({100*n_clip/D:.1f}%)")
    return scale


def load_calib_gains(path, dims: List[int],
                     current_args=None) -> List[np.ndarray]:
    """载入解码端逐神经元增益标定 (make_calibration.py 生成的 npz).

    校验: 每层键 g_l{t} 存在, 长度 = dims[t], 严格为正, 层内均值≈1
    (只削增益纹波, 不动全局标度). 返回 [g_l1, ..., g_lN].

    有 provenance 键 (source_npz/created) 则打印溯源; 有 fingerprint_json
    (标定源 run 的链路参数) 且给了 current_args 则逐项对比当前参数, 不一致
    只告警不报错 (标定可能不适用于当前链路配置)."""
    z = np.load(path, allow_pickle=True)
    prov = [f"{k}={str(z[k])}" for k in ("source_npz", "created")
            if k in z.files]
    if prov:
        print(f"[calib-npz] provenance: {', '.join(prov)}")
    if "fingerprint_json" in z.files and current_args is not None:
        try:
            fp = json.loads(str(z["fingerprint_json"]))
        except (ValueError, TypeError):
            fp = {}
        for k, v in fp.items():
            cur = getattr(current_args, k, None)
            if cur != v:
                print(f"[calib-npz] ⚠ fingerprint mismatch: {k} calib={v!r} "
                      f"vs current={cur!r} (标定源 run 的链路参数与当前不同, "
                      f"标定可能不适用)")
    n_layers = len(dims) - 1
    gains: List[np.ndarray] = []
    for t in range(1, n_layers + 1):
        key = f"g_l{t}"
        if key not in z.files:
            raise ValueError(f"{path}: 缺 {key} (标定层数与权重不符?)")
        g = np.asarray(z[key], np.float64)
        if g.shape != (dims[t],):
            raise ValueError(f"{path}: {key} 形状 {g.shape} != ({dims[t]},)")
        if np.any(g <= 0) or not np.isclose(g.mean(), 1.0, atol=1e-6):
            raise ValueError(f"{path}: {key} 必须严格为正且均值=1 "
                             f"(得 mean={g.mean():.6f})")
        gains.append(g)
    return gains


def apply_decode_calibration(arr: np.ndarray, g: np.ndarray,
                             power: bool = False) -> np.ndarray:
    """解码端逐神经元均衡: 幅度除以 g_j (power=True 时功率除以 g_j²)."""
    g = np.asarray(g, np.float64)
    return arr / (g ** 2 if power else g)[None, :]


def activation_abs(y_meas: np.ndarray, sigma2_bin: float, debias: bool) -> np.ndarray:

    if debias and np.isfinite(sigma2_bin):
        return np.sqrt(np.maximum(np.abs(y_meas) ** 2 - sigma2_bin, 0.0))
    return np.abs(y_meas)


def norm_rows(a: np.ndarray) -> np.ndarray:

    mx = np.max(a, axis=1, keepdims=True)
    return a / np.maximum(mx, 1e-12)


def quantize_act(a: np.ndarray, qlev: int) -> np.ndarray:

    if qlev is None or qlev <= 1:
        return a
    q = np.rint(np.asarray(a, np.float64) * (qlev - 1))
    return q / float(qlev - 1)


def quantize_int(a: np.ndarray, qlev: int) -> np.ndarray:

    return np.rint(np.asarray(a, np.float64) * (qlev - 1)).astype(np.int64)


def load_biases_ml(path: Path, n_layers: int):
    """Optional digital magnitude-domain biases b1..bN (ladder recipe)."""
    d = np.load(path, allow_pickle=True)
    if "b1" not in d.files:
        return None
    return [d[f"b{l+1}"].astype(np.float64) for l in range(n_layers)]


def digital_forward_ml(A0: np.ndarray, Ws: List[np.ndarray], qlev: int = 0,
                       normalize: bool = True, biases=None):

    A = A0.astype(np.float64)
    acts = []
    n_layers = len(Ws)
    y = None
    for l, W in enumerate(Ws):
        y = A.astype(np.complex128) @ W.T
        if biases is not None:
            y = np.maximum(np.abs(y) + biases[l][None, :], 0.0)
        if l < n_layers - 1:
            A = np.abs(y)
            if normalize:
                A = norm_rows(A)
            A = quantize_act(A, qlev)
            acts.append(A)
    return y, acts


def _draw_confusion_panel(ax, C_tp: np.ndarray, title: str, *, vmax: float = 100.0,
                          show_text=None):

    M = C_tp.T.astype(float)
    col = M.sum(axis=0, keepdims=True)
    Mn = 100.0 * M / np.maximum(col, 1.0)
    acc = float(np.trace(C_tp)) / max(float(C_tp.sum()), 1.0)
    im = ax.imshow(Mn, cmap="Greens", vmin=0.0, vmax=vmax)
    step = 1 if NCLS <= 12 else (2 if NCLS <= 30 else 5)
    ax.set_xticks(range(0, NCLS, step)); ax.set_yticks(range(0, NCLS, step))
    ax.tick_params(labelsize=(9 if NCLS <= 12 else 6.5))
    ax.set_xlabel("True class"); ax.set_ylabel("Predicted class")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(np.arange(-.5, NCLS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, NCLS, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=(0.6 if NCLS <= 12 else 0.25))
    ax.tick_params(which="minor", length=0)
    if show_text is None:
        show_text = NCLS <= 12
    if show_text:
        for i in range(NCLS):
            for j in range(NCLS):
                ax.text(j, i, f"{Mn[i, j]:.1f}", ha="center", va="center",
                        fontsize=6.3,
                        color="white" if Mn[i, j] > 55 else "#1b4332")
    return im, acc


def plot_confusion_pair(C_dig: np.ndarray, C_exp: np.ndarray, out_png: Path, *,
                        left_label: str, right_label: str,
                        suptitle: str = "", note: str = "") -> Tuple[float, float]:

    if not _MPL_OK:
        print(f"[warn] matplotlib 不可用, 跳过画图: {_MPL_ERR!r}")
        return float("nan"), float("nan")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))
    accd0 = 100 * np.trace(C_dig) / max(C_dig.sum(), 1)
    acce0 = 100 * np.trace(C_exp) / max(C_exp.sum(), 1)
    im0, accd = _draw_confusion_panel(axes[0], C_dig,
                                      f"Classification accuracy {accd0:.1f}%")
    im1, acce = _draw_confusion_panel(axes[1], C_exp,
                                      f"Classification accuracy {acce0:.1f}%")
    axes[0].text(0.5, -0.135, left_label, transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=9, linespacing=1.4)
    axes[1].text(0.5, -0.135, right_label, transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=9, linespacing=1.4)
    fig.subplots_adjust(left=0.055, right=0.875, top=0.85, bottom=0.205, wspace=0.20)
    cax = fig.add_axes([0.895, 0.24, 0.014, 0.52])
    cb = fig.colorbar(im1, cax=cax)
    cb.set_label("% of true-class column", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.965)
    if note:
        fig.text(0.465, 0.025, note, ha="center", va="bottom", fontsize=8.5)
    fig.savefig(out_png, dpi=175)
    plt.close(fig)
    print(f"[plot] 双混淆矩阵 (digital | 实测) -> {out_png}")
    return accd, acce


def plot_confusion(C: np.ndarray, acc: float, out_png: Path, title: str,
                   subtitle: str = "") -> None:

    if not _MPL_OK:
        print(f"[warn] matplotlib 不可用, 跳过画图: {_MPL_ERR!r}")
        return
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    im, _acc = _draw_confusion_panel(ax, C, f"Classification accuracy {acc*100:.1f}%")
    cap = title + (f"\n{subtitle}" if subtitle else "")
    ax.text(0.5, -0.15, cap, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("% of true-class column", fontsize=9)
    fig.subplots_adjust(bottom=0.2, top=0.92)
    fig.savefig(out_png, dpi=175)
    plt.close(fig)
    print(f"[plot] 混淆矩阵 -> {out_png}")


def confusion_from(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:

    C = np.zeros((NCLS, NCLS), np.int64)
    for t_, p_ in zip(labels, preds):
        C[int(t_), int(p_)] += 1
    return C


def plot_layer_fidelity(layer_diag: List[dict], out_png: Path, mode_txt: str) -> None:

    if not _MPL_OK:
        print(f"[warn] matplotlib 不可用, 跳过画图: {_MPL_ERR!r}")
        return
    nl = len(layer_diag)
    fig, axes = plt.subplots(1, nl, figsize=(4.6 * nl, 4.4))
    if nl == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    for l, (ax, dg) in enumerate(zip(axes, layer_diag)):
        yr = np.abs(dg["y_ref"]).ravel()
        ym = np.abs(dg["y_meas"] / (dg["G"] if abs(dg["G"]) > 0 else 1.0)).ravel()
        if yr.size > 4000:
            ii = rng.choice(yr.size, 4000, replace=False)
            yr, ym = yr[ii], ym[ii]
        lim = max(float(yr.max(initial=1e-9)), float(ym.max(initial=1e-9))) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
        ax.scatter(yr, ym, s=3.5, alpha=0.35, edgecolors="none", color="#1f77b4")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("|y| digital (this layer's own input)")
        if l == 0:
            ax.set_ylabel("|y| measured / g")
        navg = dg.get("n_avg", 1)
        atxt = f", {navg}\u00d7avg" if navg > 1 else ""
        ax.set_title(f"Layer {l+1}: {dg['D']}\u2192{dg['H']}\n"
                     f"g={abs(dg['G']):.2e}, RMSE={dg['rmse']:.3f}{atxt}",
                     fontsize=9.2)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Per-layer analog MVM fidelity ({mode_txt}); "
                 f"activation |\u00b7| + per-sample renorm in Python between passes",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=165)
    plt.close(fig)
    print(f"[plot] 逐层保真度散点 -> {out_png}")


def make_out_dir(stem: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = DATA_DIR / f"{Path(stem).stem}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _teardown_tb(tb) -> None:

    if tb is None:
        return
    try:
        tb.stop()
        tb.wait()
    except Exception:
        pass
    gc.collect()
    time.sleep(1.0)


def _capture_clean(tb, n_cap, settle_s, retries, grab_timeout,
                   strict: bool = False) -> np.ndarray:
    last = None
    for k in range(int(retries) + 1):
        time.sleep(float(settle_s))
        cap, cinfo = tb.grabber.grab_fresh_ex(n_cap, timeout_s=grab_timeout)
        if cap.size < n_cap or cinfo["got"] < n_cap:
            raise RuntimeError(f"抓取超时: 只拿到 {cinfo['got']}/{n_cap} 样本 (RX 流断了?)")
        last = cap
        if cinfo["clean"]:
            return cap
        if k < int(retries):
            warnings.warn(f"[rx] overflow/丢样, 丢弃重抓 {k+1}/{retries}", RuntimeWarning)
    if strict:
        # HIL training path: an unclean capture must be DROPPED upstream,
        # never trained on (the gain fit would launder it into plausible
        # magnitudes).
        raise RuntimeError(f"[rx] 重抓 {retries} 次仍 overflow (strict)")
    warnings.warn("[rx] 重抓多次仍 overflow; 用最后一次", RuntimeWarning)
    return last


class HWSession:


    def __init__(self, args, out_dir: Path):
        self.args = args
        self.lo_path = out_dir / "tx_lo_weights.cf32"
        self.rf_path = out_dir / "tx_rf_images.cf32"
        self.tb = None

    @property
    def started(self) -> bool:
        return self.tb is not None

    def ensure(self, fr: FrameSpecML, verbose_build: bool = False) -> dict:
        a = self.args
        clk = "internal" if a.no_ext_ref else RX_CLOCK_SOURCE_DEFAULT
        if self.tb is None:

            fr.sA.astype(np.complex64).tofile(self.lo_path)
            fr.sB.astype(np.complex64).tofile(self.rf_path)
            self.tb = MVMFlowgraph(fr, str(self.lo_path), str(self.rf_path),
                                   tx_args=a.tx_args, rx_args=a.rx_args,
                                   sample_rate=a.fs, rx_gain_db=float(a.rx_gain),
                                   rx_clock_source=clk,
                                   p_max_dbm=float(a.p_max_dbm),
                                   verbose=verbose_build)
            info = self.tb.set_powers(a.p_rf_dbm, a.p_lo_dbm)
            self.tb.start_timed(a.start_lead)
            time.sleep(2.0)
            self.tb.check_locks()
        else:
            self.tb.restart_with_files(str(self.lo_path), str(self.rf_path),
                                       sA=fr.sA, sB=fr.sB, frame=fr,
                                       start_lead=a.start_lead)
            info = self.tb.set_powers(a.p_rf_dbm, a.p_lo_dbm)
            time.sleep(2.0)
            if verbose_build:
                self.tb.check_locks()
        return info

    def capture(self, fr: FrameSpecML) -> np.ndarray:
        a = self.args
        n_cap = int(round(2.3 * fr.frame_len))
        gto = max(30.0, 4.0 * n_cap / a.fs)
        return _capture_clean(self.tb, n_cap, a.settle_s, a.capture_retries,
                              gto,
                              strict=bool(getattr(a, "capture_strict", False)))

    def close(self) -> None:
        _teardown_tb(self.tb)
        self.tb = None


def ring_cap_vectors(R: int, L: int, cp: int, gap: int) -> int:

    sym_budget = (RX_RING_CAPACITY / 2.3 - 2 * gap) / (L + cp) - 1.0
    return max(1, int(sym_budget // R))


def _fmt_pwr(tag: str, info: dict) -> str:
    return (f"[pwr {tag}] RF: gain={info['rf']['gain_db']:.1f} dB "
            f"amp={info['rf']['amp_rms']:.4f} -> {info['rf']['predicted_dbm']:+.2f} dBm | "
            f"LO: gain={info['lo']['gain_db']:.1f} dB "
            f"amp={info['lo']['amp_rms']:.4f} -> {info['lo']['predicted_dbm']:+.2f} dBm")


def _parse_avg_per_layer(spec: str, n_layers: int) -> List[int]:

    parts = [p.strip() for p in str(spec).split(",") if p.strip() != ""]
    if len(parts) == 1:
        vals = [max(1, int(parts[0]))] * n_layers
    elif len(parts) == n_layers:
        vals = [max(1, int(p)) for p in parts]
    else:
        raise ValueError(f"--avg-per-layer 需 1 个值或 {n_layers} 个逗号分隔值, "
                         f"收到 {len(parts)} 个: {spec!r}")
    return vals


def _parse_target_eff_snr(spec, n_layers: int) -> List[Optional[float]]:

    if spec is None:
        return [None] * n_layers
    parts = [p.strip() for p in str(spec).split(",")]
    if len(parts) == 1:
        v = parts[0]
        return [(float(v) if v != "" else None)] * n_layers
    if len(parts) != n_layers:
        raise ValueError(f"--target-eff-snr 需 1 个值或 {n_layers} 个逗号分隔值 "
                         f"(空=该层不动), 收到 {len(parts)} 个: {spec!r}")
    return [(float(p) if p != "" else None) for p in parts]


def run(args) -> int:

    if args.replot is not None:
        if args.calib_npz:
            print("[replot] --calib-npz 在 replot 模式下被忽略, 只重画已存结果 "
                  "(calib flag ignored in replot mode)")
        d = np.load(args.replot, allow_pickle=True)
        C_exp = d["confusion"]
        out_png = (Path(args.replot).with_suffix(".replot.png")
                   if args.out_png == f"{DEFAULT_OUTPUT_STEM}.png" else Path(args.out_png))
        if "confusion_digital_full" in d:
            paired = bool(d["digital_paired"]) if "digital_paired" in d else False
            C_left = (d["confusion_digital_paired"] if (paired and
                      "confusion_digital_paired" in d) else d["confusion_digital_full"])
            plot_confusion_pair(
                C_left, C_exp, out_png,
                left_label=str(d["left_label"]) if "left_label" in d else "Digital",
                right_label=str(d["right_label"]) if "right_label" in d else "Measured",
                suptitle=str(d["suptitle"]) if "suptitle" in d else "",
                note=str(d["note"]) if "note" in d else "")
        else:
            plot_confusion(C_exp, float(d["accuracy"]), out_png,
                           "N-layer MVM (replot)",
                           subtitle=str(d["subtitle"]) if "subtitle" in d else "")
        return 0


    wpath = Path(args.weights)
    if not wpath.is_absolute():
        wpath = SCRIPT_DIR / wpath
    if not wpath.exists():
        print(f"[error] 找不到权重 {wpath}; 先运行 gtsrb_train_digital_deep.py")
        return 1
    Ws, dims, digital_acc, qlev_w = load_weights_ml(wpath)
    n_layers = len(Ws)
    biases = load_biases_ml(wpath, n_layers)
    if biases is not None:
        print(f"[weights] 数字域逐神经元偏置 b1..b{n_layers} 已载入 (阶梯配方; "
              f"在层间数字步应用, 需 --rung3-json 的 |ref| 均衡定标)")
    dims_txt = "->".join(str(d) for d in dims)
    print(f"[weights] {wpath.name}: {n_layers} 层 {dims_txt} (逻辑权重, ZC 组帧时折入); "
          f"数字测试精度={digital_acc*100:.2f}%")

    qlev = int(qlev_w if args.quant_levels < 0 else args.quant_levels)
    if qlev > 1:
        print(f"[decide] 层间判定: Q={qlev} 电平 (判定值 = 0..{qlev-1} 的整数), "
              f"台阶={1.0/(qlev-1):.4f}")
        print(f"[decide] 判决要少翻转约需 有效幅度 SNR >~ "
              f"{20*np.log10(2.7*(qlev-1)):.0f} dB (3σ 不越半台阶, 估算); "
              f"不足时用 --avg-per-layer N 积分增益 +10log10(N) dB")
        print(f"[decide] ⚠ 实测结论 (见 README): 硬判决在本网络**不能**再生, 且各 SNR 下"
              f"都略差于不判定 —— 因为激活是**连续量**, 真值并不落在电平格点上, "
              f"判决无'正确电平'可回归, 只是引入量化失真; 且一次翻转 = 整整一个台阶的"
              f"误差, 会逐层级联 (翻转率随层号上升). 对照请跑 --quant-levels 0.")
    else:
        print(f"[decide] 层间判定: 关 (模拟值直传). 实测: 误差**不发散**, 逐层饱和 "
              f"(权重本身抑制各向同性噪声) -> 本模式是推荐设置")
    if args.quant_levels >= 0 and qlev != qlev_w:
        print(f"[decide] 注意: 命令行 Q={qlev} 覆盖了权重训练时的 Q={qlev_w}")

    calib_g: Optional[List[np.ndarray]] = None
    if args.calib_npz:
        cpath = Path(args.calib_npz)
        if not cpath.exists() and not cpath.is_absolute():
            cpath = SCRIPT_DIR / cpath
        if not cpath.exists():
            print(f"[error] 找不到标定文件 {cpath}; 先用 make_calibration.py "
                  f"从实测 run npz 生成")
            return 1
        calib_g = load_calib_gains(cpath, dims, current_args=args)
        print(f"[calib-npz] 解码端逐神经元增益均衡: {cpath.name} "
              f"({len(calib_g)} 层, 每层均值=1 只削纹波; 激活与末层判决都除以 g_j)")
        if args.test_rx:
            print("[calib-npz] 注意: --test-rx 诊断解码不应用增益均衡 "
                  "(calibration NOT applied in test-rx mode)")
        if args.dry_run:
            print("[calib-npz] 注意: dry-run 模拟器没有增益纹波, 均衡会注入 "
                  "1/g 的伪纹波 — 校准 dry-run 仅作管线冒烟测试 (calibrated "
                  "dry-run accuracy includes a spurious 1/g ripple; plumbing "
                  "smoke test only)")

    eq_scale: Optional[np.ndarray] = None
    if args.input_eq_npz:
        epath = Path(args.input_eq_npz)
        if not epath.exists() and not epath.is_absolute():
            epath = SCRIPT_DIR / epath
        if not epath.exists():
            print(f"[error] 找不到输入均衡文件 {epath}; 先用 make_input_eq 拟合")
            return 1
        eq_scale = load_input_eq(epath, dims[0], cap=float(args.input_eq_cap),
                                 current_args=args)
        print(f"[input-eq] 层 1 TX 侧逐输入 tone 预均衡启用: {epath.name} "
              f"(x'=x/h, 仅层 1; 保真度参考仍为真值 W·x)")
        if args.dry_run:
            print("[input-eq] 注意: dry-run 模拟器没有逐输入 tone 响应, 预均衡"
                  "会注入伪纹波 — 仅作管线冒烟测试")

    rung3 = None
    if args.rung3_json:
        if args.input_eq_npz:
            print("[error] --rung3-json 与 --input-eq-npz 不可同用 "
                  "(h 项与 TX 预均衡会重复校正同一响应)")
            return 1
        rpath = Path(args.rung3_json)
        if not rpath.exists() and not rpath.is_absolute():
            rpath = SCRIPT_DIR / rpath
        if not rpath.exists():
            print(f"[error] 找不到 rung-3 会话参数 {rpath}; 先用 rung3_refit.py "
                  f"从同会话 control run 生成")
            return 1
        from rung3_correction import Rung3Corrector
        rung3 = Rung3Corrector.from_session_params(rpath)
        print(f"[rung3] 读出端数字校正启用: {rpath.name} "
              f"(shadow={'on' if rung3.use_shadow else 'OFF(gated)'}, "
              f"h={'on' if rung3.use_h else 'OFF(gated)'}; "
              f"逐层 |ref| 均衡 + beta·delta 扣除; 原始 y_mag 同时存档)")
        if args.dry_run:
            print("[rung3] 注意: dry-run 模拟器只有 sim-mixer 失真; 校正增益不代表"
                  "硬件, 仅作管线冒烟测试")
    if biases is not None and rung3 is None:
        print("[error] 权重带数字偏置 (b1..bN) 但未启用 --rung3-json: 偏置以参考"
              "单位定义, 需要 rung-3 的逐列 |ref| 均衡定标后才能应用")
        return 1

    Xte, yte = load_gtsrb_test()
    off = int(args.offset)
    n_test = int(min(args.n_test, Xte.shape[0] - off))
    sel = np.arange(off, off + n_test)
    X = Xte[sel]; y = yte[sel]
    print(f"[data] 推理测试集第 {off}..{off+n_test-1} 张 (共 {n_test} 张)")


    y_last_dig, _ = digital_forward_ml(X, Ws, qlev, biases=biases)
    pred_dig_sel = np.argmax(np.abs(y_last_dig) ** 2, axis=1).astype(np.int64)
    y_full_dig, _ = digital_forward_ml(Xte, Ws, qlev, biases=biases)
    pred_full = np.argmax(np.abs(y_full_dig) ** 2, axis=1).astype(np.int64)
    C_dig_full = confusion_from(pred_full, yte)
    acc_dig_full = float((pred_full == yte).mean())


    plans = [plan_layer(dims[l], dims[l + 1], args.fft_len, args.k0,
                        args.bw_frac, args.k_per_symbol)
             for l in range(n_layers)]
    per_l: List[int] = []
    print(f"[plan] L={args.fft_len}, CP={args.cp_len}, K0={args.k0}, "
          f"带宽上限 {args.bw_frac:.2f}·Nyquist; 逐层规划:")
    for l, p in enumerate(plans):
        cap_v = ring_cap_vectors(p.R, args.fft_len, args.cp_len, args.gap)
        per = max(1, min(int(args.images_per_frame), cap_v))
        per_l.append(per)
        note = (f" (受 RX 环形缓冲收紧, ≤{cap_v})"
                if per < int(args.images_per_frame) else "")
        print(f"[plan]   L{l+1}: {p.D:>4d}->{p.H:<4d} K={p.K:>3d}/符号 M={p.M:>3d} "
              f"R={p.R} 符号/样本, |k|max={p.kmax} ({100*p.kmax/(args.fft_len/2):.0f}% Nyq); "
              f"每帧≤{per} 样本{note}")


    avg_per_l = _parse_avg_per_layer(args.avg_per_layer, n_layers)
    if any(a > 1 for a in avg_per_l):
        print(f"[avg] 逐层重复测量+功率平均: {avg_per_l} 次/层 "
              f"(弱层多测; 每层 +10log10(N) dB 有效幅度 SNR)")
        if not args.debias_act:
            print("[avg] 提示: 平均只压随机起伏; 近零处的 |·| 正偏置需 --debias-act "
                  "才能扣掉 (两者同开效果最好)")


    _MAG_SNR_GAIN_DB = 10.0 * math.log10(2.0)
    target_eff_l = _parse_target_eff_snr(args.target_eff_snr, n_layers)
    bin_target_l: List[Optional[float]] = [None] * n_layers
    if any(t is not None for t in target_eff_l):
        for l in range(n_layers):
            if target_eff_l[l] is not None:
                bin_target_l[l] = (target_eff_l[l] - _MAG_SNR_GAIN_DB
                                   - 10.0 * math.log10(max(avg_per_l[l], 1)))
        show = ", ".join(f"L{l+1}={target_eff_l[l]:g}dB(注到bin {bin_target_l[l]:.1f})"
                         for l in range(n_layers) if target_eff_l[l] is not None)
        print(f"[target-eff-snr] 逐层目标有效SNR: {show}")
        print(f"[target-eff-snr] 机制=叠加标定AWGN(只能往**下**调); bin目标已扣 3dB幅度增益"
              f"{' +逐层平均增益' if any(a>1 for a in avg_per_l) else ''}, "
              f"使报出的有效SNR命中目标. 某层目标>原始SNR会跳过并提示用 --avg-per-layer")
    elif args.target_snr_db is not None:
        bin_target_l = [float(args.target_snr_db)] * n_layers

    st_mode = args.snr_thermal_db is not None
    if st_mode:
        bin_target_l = [None] * n_layers
        print(f"[snr-thermal] 每层 signal/thermal SNR 设定 = {args.snr_thermal_db:g} dB "
              f"(注 AWGN, σ²=P_signal/SNR)")

    out_dir = make_out_dir(args.out_npz)
    print(f"[main] 输出目录: {out_dir}")


    hw: Optional[HWSession] = None
    pn_thermal: Optional[float] = None
    if not args.dry_run:
        if not _GR_OK:
            raise RuntimeError(f"GNU Radio 不可用: {_GR_ERR!r}\n"
                               f"请在装好 GNU Radio + UHD 的机器上运行 "
                               f"(dry-run/replot 不需要).")
        from check_rig_free import rig_free_or_die
        rig_free_or_die()          # standing rule: never transmit on a busy rig
        hw = HWSession(args, out_dir)

    if (hw is not None) and (args.calib_noise or args.paper_snr_db is not None):
        n0 = min(per_l[0], n_test)
        zc0 = zc_phase(dims[0])
        Wt0 = Ws[0] * np.conj(zc0)[None, :]
        fr0 = build_frame_ml(Wt0, (X[:n0] * zc0[None, :]).astype(np.complex128),
                             y[:n0], sel[:n0], plans[0], L=args.fft_len,
                             cp=args.cp_len, gap=args.gap, sync_seed=args.seed,
                             layer_tag=1, verbose=False)
        print("[calib] 起标定 flowgraph (量热噪底, 用第 1 层帧)...")
        hw.ensure(fr0, verbose_build=True)
        n_cap0 = int(round(2.3 * fr0.frame_len))
        gto = max(30.0, 4.0 * n_cap0 / args.fs)
        try:
            hw.tb.mute_tx(); time.sleep(0.6)
            ncap = _capture_clean(hw.tb, n_cap0, args.settle_s, args.capture_retries, gto)
            pn_thermal = measure_band_power(ncap, fr0, fr0.out_bins)
            noise_rms = float(np.sqrt(np.mean(np.abs(ncap) ** 2)))
            hw.tb.set_powers(args.p_rf_dbm, args.p_lo_dbm); time.sleep(0.6)
            scap = _capture_clean(hw.tb, n_cap0, args.settle_s, args.capture_retries, gto)
            res0 = decode_ml(scap, fr0, args.fs, flip_output=args.flip_output,
                             conj_output=args.conj_output, verbose=False)
            if (not res0.ok) or (res0.y_meas is None):
                raise RuntimeError(f"标定时解码失败: {res0.reason}; "
                                   f"先用 --test-rx 把链路调到能同步")
            p_sig0 = float(np.mean(np.abs(res0.y_meas) ** 2))
            snr_paper_now = 10.0 * np.log10(max(p_sig0 / max(pn_thermal, 1e-30), 1e-30))
            peak0 = float(np.max(np.abs(scap)))
            clip0 = float(np.mean(np.abs(scap) > RX_CLIP_THRESH))
            print(f"[calib] 热噪底 Pn(输出bin)={pn_thermal:.3e} (静默TX RMS={noise_rms:.2e})")
            print(f"[calib] --p-rf-dbm={args.p_rf_dbm:+.1f}: 信号 Py={p_sig0:.3e}, "
                  f"论文口径 SNR=Py/Pn={snr_paper_now:+.1f} dB "
                  f"(峰值|IQ|={peak0:.3f}, clip={clip0*100:.3f}%)")
            print(f"[calib] 对照 保护bin口径(信号/总噪声)={res0.snr_bin_db:+.1f} dB "
                  f"<- 这是会被损伤卡住的那个; 论文口径分母只有热噪所以高得多")
            if args.calib_noise:
                print("[calib] 命中各目标所需 --p-rf-dbm (信号∝发射功率, +1dB→+1dB):")
                for tgt in (15, 20, 25, 30, 35, 40):
                    need = args.p_rf_dbm + (tgt - snr_paper_now)
                    pk = peak0 * (10.0 ** ((tgt - snr_paper_now) / 20.0))
                    flag = "  ⚠RX会clip,需降--rx-gain" if pk > RX_CLIP_THRESH else ""
                    print(f"         SNR={tgt:>2d} dB -> --p-rf-dbm {need:+6.1f} "
                          f"(RX峰值≈{pk:.2f}){flag}")
            if args.paper_snr_db is not None:
                delta = float(args.paper_snr_db) - snr_paper_now
                new_p_rf = args.p_rf_dbm + delta
                new_peak = peak0 * (10.0 ** (delta / 20.0))
                print(f"[paper-snr] 目标 {args.paper_snr_db:g} dB(论文口径): "
                      f"--p-rf-dbm {args.p_rf_dbm:+.1f} -> {new_p_rf:+.1f} "
                      f"(Δ={delta:+.1f} dB); 三层每次发射 pass 都用这个工作点")
                if new_peak > RX_CLIP_THRESH:
                    warnings.warn(
                        f"[paper-snr] 目标功率下 RX 峰值预计≈{new_peak:.2f} "
                        f"(>{RX_CLIP_THRESH} 会 clip); 降 --rx-gain 后重跑.",
                        RuntimeWarning)
                args.p_rf_dbm = new_p_rf
        except Exception:
            hw.close()
            raise
        if args.calib_noise:
            hw.close()
            return 0
        info = hw.tb.set_powers(args.p_rf_dbm, args.p_lo_dbm)
        print(_fmt_pwr("calib", info))


    keep = np.ones(n_test, bool)
    A = X.astype(np.float64)
    Y_last: Optional[np.ndarray] = None
    P_last: Optional[np.ndarray] = None
    layer_diag: List[dict] = []
    snrs_paper: List[float] = []

    try:
        for l in range(n_layers):
            D, H = dims[l], dims[l + 1]
            plan = plans[l]
            zc = zc_phase(D)
            Wt = Ws[l] * np.conj(zc)[None, :]
            kidx = np.nonzero(keep)[0]
            if kidx.size == 0:
                raise RuntimeError("没有存活样本 (前面所有 chunk 都解码失败)")
            per = per_l[l]
            nch = int(math.ceil(kidx.size / per))
            print(f"\n===== 第 {l+1}/{n_layers} 层: 模拟域 MVM {D}->{H} "
                  f"({nch} 个 chunk, 每帧≤{per} 样本) =====")
            if l > 0:
                print(f"[layer {l+1}] 输入 = 第 {l} 层实测 |y| (Python 激活+归一), "
                      f"已重新编码成 RF 帧")

            n_avg = avg_per_l[l]
            Y_l = np.zeros((n_test, H), np.complex128)
            P_l = np.zeros((n_test, H), np.float64)
            Yref_l = np.zeros((n_test, H), np.complex128)
            got = np.zeros(n_test, bool)
            s2s: List[float] = []

            for c in range(nch):
                idx = kidx[c * per:(c + 1) * per]
                x_src = A[idx].astype(np.complex128)
                if eq_scale is not None and l == 0:
                    x_src = x_src * eq_scale[None, :]   # TX 预均衡 x'=x/h (截幅)
                x_t = (x_src * zc[None, :]).astype(np.complex128)
                fr = build_frame_ml(Wt, x_t, y[idx], sel[idx], plan,
                                    L=args.fft_len, cp=args.cp_len, gap=args.gap,
                                    sync_seed=args.seed, layer_tag=l + 1,
                                    verbose=(c == 0))
                if eq_scale is not None and l == 0:
                    # 保真度/激活参考 = 真值 W·x (预均衡只是补偿硬件响应,
                    # 目标输出不变)
                    fr.y_digital = ((A[idx] * zc[None, :]).astype(np.complex128)
                                    @ Wt.T).astype(np.complex128)


                if not args.dry_run:
                    first_build = not hw.started
                    info = hw.ensure(fr, verbose_build=first_build)
                    if c == 0:
                        print(_fmt_pwr(f"L{l+1}", info))

                if args.test_rx and l == 0 and c == 0:
                    cap0 = (synth_capture_ml(
                                fr, float(args.snr_db) if args.snr_db is not None
                                else float("inf"), args.sim_mixer, args.seed,
                                args.fs, salt=0, verbose=True)
                            if args.dry_run else hw.capture(fr))
                    _run_test_rx(args, fr, cap0, out_dir)
                    return 0


                pow_acc = np.zeros((idx.size, H), np.float64)
                sig2_acc = 0.0; n_ok = 0
                last_res: Optional[DecodeML] = None
                rep_att = 1 if args.dry_run else (1 + max(0, int(args.decode_retries)))
                for rep in range(n_avg):
                    res: Optional[DecodeML] = None
                    for att in range(rep_att):
                        if args.dry_run:
                            snr = args.snr_db if args.snr_db is not None else float("inf")
                            cap = synth_capture_ml(
                                fr, float(snr), args.sim_mixer, args.seed, args.fs,
                                salt=l * 1_000_000 + c * 10_000 + rep * 100 + att,
                                verbose=(c == 0 and rep == 0 and att == 0))
                        else:
                            if att > 0:
                                hw.ensure(fr)
                            cap = hw.capture(fr)
                        if args.snr_thermal_db is not None:
                            cap = inject_thermal_snr(
                                cap, fr, args.fs, float(args.snr_thermal_db),
                                seed=int(args.seed) + 2000 + l * 97 + c * 7 + rep,
                                verbose=(c == 0 and rep == 0))
                        elif bin_target_l[l] is not None:
                            cap = inject_target_snr(
                                cap, fr, args.fs, float(bin_target_l[l]),
                                seed=int(args.seed) + 1000 + l * 97 + c * 7 + rep,
                                verbose=(c == 0 and rep == 0))
                        res = decode_ml(cap, fr, args.fs, flip_output=args.flip_output,
                                        conj_output=args.conj_output,
                                        verbose=(rep == 0 or att > 0))
                        if res.ok:
                            break
                        if att < rep_att - 1:
                            warnings.warn(f"[L{l+1} chunk {c+1} rep {rep+1}] 解码失败 "
                                          f"({res.reason}); 重抓 {att+1}/{rep_att-1}",
                                          RuntimeWarning)
                    if (res is None) or (not res.ok):
                        if args.dry_run:
                            raise RuntimeError(f"dry-run 解码失败 (L{l+1} chunk {c+1} "
                                               f"rep {rep+1}): {res.reason if res else '??'}")
                        warnings.warn(f"[L{l+1} chunk {c+1}] rep {rep+1}/{n_avg} 仍失败; "
                                      f"用已得 {n_ok} 次平均", RuntimeWarning)
                        break
                    pow_acc += np.abs(res.y_meas) ** 2
                    sig2_acc += res.sigma2_bin
                    n_ok += 1; last_res = res

                if n_ok == 0:
                    if args.dry_run:
                        raise RuntimeError(f"dry-run 解码失败 (L{l+1} chunk {c+1})")
                    warnings.warn(f"[L{l+1} chunk {c+1}] 全部重抓失败; 丢弃 {idx.size} "
                                  f"样本 (整链剔除, 保持行对齐)", RuntimeWarning)
                    keep[idx] = False
                    continue

                Pbar = pow_acc / n_ok
                P_l[idx] = Pbar
                Y_l[idx] = last_res.y_meas
                Yref_l[idx] = fr.y_digital
                got[idx] = True
                s2s.append(sig2_acc / n_ok)
                if n_avg > 1 and c == 0:
                    print(f"[avg L{l+1}] chunk 1: {n_ok}/{n_avg} 次成功平均 "
                          f"(功率域; 估计起伏 -{10*np.log10(max(n_ok,1)):.1f} dB)")
                if pn_thermal is not None:
                    sp = 10.0 * np.log10(max(float(np.mean(Pbar)) /
                                             max(pn_thermal, 1e-30), 1e-30))
                    snrs_paper.append(sp)
                    print(f"[paper-snr L{l+1}] 本帧论文口径 SNR≈{sp:+.1f} dB (信号/热噪底)")
                if nch > 1:
                    print(f"[layer {l+1}] chunk {c+1}/{nch} 完成 ({idx.size} 样本)")

            keep &= got
            kidx = np.nonzero(keep)[0]
            sigma2_l = float(np.nanmean(s2s)) if s2s else float("nan")


            debias_off = sigma2_l if args.debias_act else 0.0
            am_all = np.sqrt(np.maximum(P_l[kidx] - debias_off, 0.0))
            if calib_g is not None:
                am_all = apply_decode_calibration(am_all, calib_g[l])
            ar_all = np.abs(Yref_l[kidx])
            r3_raw = r3_delta = None
            if rung3 is not None:
                r3_raw = am_all.copy()
                am_all, r3_delta, _r3_gj = rung3.correct(
                    l + 1, A[kidx], Ws[l],
                    (plans[l].D, plans[l].H, plans[l].K, plans[l].M,
                     plans[l].R), am_all, Yref_l[kidx],
                    W1=Ws[0],
                    plan_row1=(plans[0].D, plans[0].H, plans[0].K,
                               plans[0].M, plans[0].R))
                print(f"[rung3 L{l+1}] 校正已应用: ||beta·delta||/||ref|| = "
                      f"{float(np.linalg.norm(r3_delta) / max(np.linalg.norm(ar_all), 1e-30)):.4f}")
            den = float(np.sum(ar_all ** 2))
            g_l = float(np.sum(am_all * ar_all) / den) if den > 0 else 0.0
            if g_l > 0 and ar_all.size:
                rmse_l = float(np.sqrt(np.mean((am_all / g_l - ar_all) ** 2)) /
                               max(np.sqrt(np.mean(ar_all ** 2)), 1e-30))
            else:
                rmse_l = float("nan")
            snr_l = (10.0 * np.log10(max(float(np.mean(P_l[kidx])) - sigma2_l, 1e-30) /
                                     max(sigma2_l, 1e-30))
                     if (kidx.size and np.isfinite(sigma2_l)) else float("nan"))
            eff_l = (-20.0 * np.log10(max(rmse_l, 1e-6)) if np.isfinite(rmse_l)
                     else float("nan"))
            layer_diag.append(dict(
                D=D, H=H, K=plan.K, M=plan.M, R=plan.R, n_chunks=nch, n_avg=n_avg,
                y_meas=am_all, y_ref=ar_all, G=complex(g_l),
                y_meas_cplx=Y_l[kidx].copy(), y_ref_cplx=Yref_l[kidx].copy(),
                rmse=rmse_l, snr=snr_l, eff_snr=eff_l, sigma2=sigma2_l))
            if rung3 is not None:
                layer_diag[-1]["y_meas_raw"] = r3_raw
                layer_diag[-1]["rung3_delta"] = r3_delta
            snr_diag = "" if st_mode else (f", 链路 SNR≈{snr_l:+.1f} dB, "
                                           f"有效幅度 SNR≈{eff_l:+.1f} dB")
            print(f"[layer {l+1}] 汇总: 幅度增益 g={g_l:.3e}, RMSE={rmse_l:.4f}{snr_diag}"
                  f"{f' (x{n_avg} 平均)' if n_avg > 1 else ''}, 存活 {kidx.size}/{n_test}")

            if l < n_layers - 1:
                am_eff = (np.maximum(am_all + biases[l][None, :], 0.0)
                          if biases is not None else am_all)
                a = norm_rows(am_eff)
                if qlev > 1:

                    qi = quantize_int(a, qlev)
                    a = qi / float(qlev - 1)

                    step = 1.0 / (qlev - 1)
                    resid = np.abs(norm_rows(am_eff) - a) / step
                    hist = np.bincount(qi.ravel(), minlength=qlev)
                    frac_edge = float((resid > 0.4).mean())
                    print(f"[layer {l+1}] ★判定 Q={qlev}: 电平占比="
                          f"{np.round(hist/max(qi.size,1), 3).tolist()}")
                    print(f"[layer {l+1}]   判定值样例 (第1个样本前12个): "
                          f"{qi[0][:12].tolist()}")
                    print(f"[layer {l+1}]   距判决边界 >0.4 台阶的元素占比="
                          f"{frac_edge*100:.1f}% (越小越稳; 大则加 --avg-per-layer)")
                A = np.zeros((n_test, H), np.float64)
                A[kidx] = a
                layer_diag[-1]["act_next"] = a.astype(np.float32)
                print(f"[layer {l+1}] Python 侧激活: a=sqrt(平均|y|²"
                      f"{'-σ²' if args.debias_act else ''}) -> a/max(a)"
                      f"{f' -> 判定 Q={qlev}' if qlev > 1 else ''} -> "
                      f"第 {l+2} 层输入重新发射")
            else:
                Y_last = Y_l
                if rung3 is not None:
                    P_last = np.zeros_like(P_l)
                    lastm = (np.maximum(am_all + biases[l][None, :], 0.0)
                             if biases is not None else am_all)
                    P_last[kidx] = lastm ** 2   # 已均衡+rung-3 校正(+偏置)的幅度平方
                else:
                    P_last = (P_l if calib_g is None else
                              apply_decode_calibration(P_l, calib_g[l], power=True))
    finally:
        if hw is not None:
            hw.close()


    kidx = np.nonzero(keep)[0]
    if kidx.size == 0 or P_last is None:
        warnings.warn("没有成功样本; 不写结果. 检查发射/功率/接线.", RuntimeWarning)
        return 1
    n_drop = n_test - kidx.size


    preds = np.argmax(P_last[kidx], axis=1).astype(np.int64)
    labels = y[kidx]
    idxs = sel[kidx]
    acc = float((preds == labels).mean())
    C_exp = confusion_from(preds, labels)
    per_cls = C_exp.diagonal() / np.maximum(C_exp.sum(1), 1)

    preds_dig = pred_dig_sel[kidx]
    C_dig_paired = confusion_from(preds_dig, labels)
    acc_dig_paired = float((preds_dig == labels).mean())
    agree = float((preds == preds_dig).mean())

    if args.digital_paired:
        C_left = C_dig_paired; acc_left = acc_dig_paired; n_left = len(preds_dig)
        left_kind = "same images"
    else:
        C_left = C_dig_full; acc_left = acc_dig_full; n_left = int(C_dig_full.sum())
        left_kind = "full test set"

    print(f"\n========== N 层 MLP 模拟域推理结果 ({dims_txt}) ==========")
    mode = (("dry-run/" + args.sim_mixer +
             (f" @SNR{args.snr_db}dB" if (args.dry_run and args.snr_db is not None) else ""))
            if args.dry_run else "硬件 (USRP X310 x2 + ZEM-4300), 每层一遍模拟 MVM")
    print(f"  模式: {mode};  层间激活 |·| + 归一"
          f"{f' + 硬判定 Q={qlev}' if qlev > 1 else ' (无判定: 模拟值直传)'}"
          + (" + 噪声去偏" if args.debias_act else "")
          + (" + 逐神经元增益均衡 (--calib-npz)" if calib_g is not None else ""))
    for l, dg in enumerate(layer_diag):
        avg_txt = f" x{dg['n_avg']}avg" if dg.get("n_avg", 1) > 1 else ""
        snr_diag = "" if st_mode else (f" 链路SNR≈{dg['snr']:+.1f} "
                                       f"有效SNR≈{dg.get('eff_snr', float('nan')):+.1f} dB")
        print(f"  L{l+1} {dg['D']:>4d}->{dg['H']:<4d}: K={dg['K']:>3d} M={dg['M']:>3d} "
              f"R={dg['R']} x{dg['n_chunks']}ch{avg_txt} | g={abs(dg['G']):.3e} "
              f"RMSE={dg['rmse']:.4f}{snr_diag}")
    drop_txt = f" (丢弃 {n_drop} 张: 解码失败 chunk 整链剔除)" if n_drop else ""
    print(f"  实测推理 {len(preds)} 张{drop_txt}, 精度 = {acc*100:.2f}%")
    print(f"  数字端 (同一批 {len(preds_dig)} 张) = {acc_dig_paired*100:.2f}%;  "
          f"数字端 (全 {len(yte)} 张测试集) = {acc_dig_full*100:.2f}%")
    print(f"  实测 vs 数字 预测一致率 = {agree*100:.2f}%")
    if snrs_paper:
        print(f"  ★ 论文口径 SNR (信号/热噪底) ≈ {np.nanmean(snrs_paper):+.1f} dB "
              f"(逐层逐帧均值)")
    if st_mode:
        print(f"  ★ 本次测试 SNR (signal/thermal, 设定) = {args.snr_thermal_db:g} dB")
    if NCLS <= 12:
        print("  实测混淆矩阵 (行=真值, 列=预测):")
        print("        " + "".join(f"{j:>6d}" for j in range(NCLS)))
        for i in range(NCLS):
            print(f"    {i:>2d} |" + "".join(f"{C_exp[i,j]:>6d}" for j in range(NCLS)))
        print("  各类召回率: " + ", ".join(f"{d}:{per_cls[d]*100:.1f}%"
                                          for d in range(NCLS)))
    else:
        print(f"  实测各类召回率 ({NCLS} 类):")
        for r0 in range(0, NCLS, 8):
            print("    " + "  ".join(f"{d:>2d}:{per_cls[d]*100:5.1f}%"
                                     for d in range(r0, min(r0 + 8, NCLS))))
        worst = np.argsort(per_cls)[:6]
        print("  最差 6 类: " + ", ".join(f"{d}:{per_cls[d]*100:.1f}%" for d in worst))


    eff_layers_txt = "/".join(f"{dg.get('eff_snr', float('nan')):.0f}" for dg in layer_diag)
    snr_layers_txt = "/".join(f"{dg['snr']:.0f}" for dg in layer_diag)
    if args.snr_thermal_db is not None:
        snr_txt = f"SNR = {args.snr_thermal_db:g} dB (signal/thermal)"
    elif any(t is not None for t in target_eff_l):
        snr_txt = f"target eff-SNR set via AWGN \u2248 {eff_layers_txt} dB (L1/L2/L3)"
    elif snrs_paper:
        snr_txt = f"SNR (signal/thermal) \u2248 {np.nanmean(snrs_paper):.0f} dB"
    elif args.dry_run and args.snr_db is not None:
        snr_txt = f"injected SNR {args.snr_db:g} dB/layer"
    elif (not args.dry_run) and (args.target_snr_db is not None):
        snr_txt = f"hardware + calibrated AWGN to {args.target_snr_db:g} dB/layer"
    elif all(np.isfinite(dg["snr"]) for dg in layer_diag):
        snr_txt = f"measured tone SNR \u2248 {snr_layers_txt} dB (L1/L2/L3)"
    else:
        snr_txt = ""
    right_kind = ("dry-run, " + args.sim_mixer + " mixer") if args.dry_run        else "hardware: USRP X310 ×2 + ZEM-4300"
    h_txt = "\u2192".join(str(d) for d in dims)
    suptitle = (f"{n_layers}-layer complex MLP {h_txt} GTSRB — analog MVM per layer, "
                f"|\u00b7| activation in Python between RF passes; "
                f"decision = argmax |y\u2083|\u00b2")
    note = (args.note if args.note else
            f"{right_kind}" + (f"   |   {snr_txt}" if snr_txt else "")
            + f"   |   experiment\u2194digital agreement {agree*100:.1f}%")
    left_label = f"Digital (ideal multiplier)\nN={n_left}, {left_kind}"
    right_label = (f"{'Dry-run' if args.dry_run else 'Measured (hardware)'}\n"
                   f"N={len(preds)}, 3 analog passes")
    plot_confusion_pair(C_left, C_exp, out_dir / Path(args.out_png).name,
                        left_label=left_label, right_label=right_label,
                        suptitle=suptitle, note=note)
    plot_layer_fidelity(layer_diag, out_dir / "layer_fidelity.png",
                        "dry-run" if args.dry_run else "hardware")


    save = dict(
        confusion=C_exp, confusion_digital_paired=C_dig_paired,
        confusion_digital_full=C_dig_full,
        accuracy=acc, accuracy_digital_paired=acc_dig_paired,
        accuracy_digital_full=acc_dig_full, agreement=agree,
        per_class_recall=per_cls, preds=preds, preds_digital=preds_dig,
        labels=labels, img_index=idxs, keep_mask=keep,
        y_meas=Y_last[kidx].astype(np.complex64),
        mode=mode, suptitle=suptitle, note=note,
        left_label=left_label, right_label=right_label,
        digital_paired=bool(args.digital_paired),
        n_layers=n_layers, dims=np.asarray(dims, np.int64),
        snr_bin_db=np.array([dg["snr"] for dg in layer_diag]),
        logit_rmse=np.array([dg["rmse"] for dg in layer_diag]),
        eff_snr_db=np.array([dg.get("eff_snr", float("nan")) for dg in layer_diag]),
        avg_per_layer=np.asarray(avg_per_l, np.int64),
        snr_paper_db=np.array(snrs_paper) if snrs_paper else np.array([]),
        noise_floor_pn=(float(pn_thermal) if pn_thermal is not None else float("nan")),
        fft_len=args.fft_len, cp_len=args.cp_len, k0=args.k0, fs_hz=args.fs,
        bw_frac=args.bw_frac, debias_act=bool(args.debias_act),
        meta_json=json.dumps({k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                  else str(v)) for k, v in vars(args).items()},
                             ensure_ascii=False))
    for l, dg in enumerate(layer_diag):
        t = l + 1
        save[f"y_meas_l{t}"] = dg["y_meas_cplx"].astype(np.complex64)
        save[f"y_ref_l{t}"] = dg["y_ref_cplx"].astype(np.complex64)
        save[f"y_mag_l{t}"] = dg["y_meas"].astype(np.float32)
        save[f"g_mag_l{t}"] = float(abs(dg["G"]))
        save[f"G_hat_l{t}"] = np.complex128(dg["G"])
        save[f"logit_rmse_l{t}"] = float(dg["rmse"])
        save[f"snr_bin_db_l{t}"] = float(dg["snr"])
        save[f"eff_snr_db_l{t}"] = float(dg.get("eff_snr", float("nan")))
        save[f"n_avg_l{t}"] = int(dg.get("n_avg", 1))
        save[f"sigma2_bin_l{t}"] = float(dg["sigma2"])
        save[f"plan_l{t}"] = np.array([dg["D"], dg["H"], dg["K"], dg["M"], dg["R"]],
                                      np.int64)
        if "act_next" in dg:
            save[f"act_l{t}"] = dg["act_next"]
        if "y_meas_raw" in dg:
            save[f"y_mag_raw_l{t}"] = dg["y_meas_raw"].astype(np.float32)
            save[f"rung3_delta_l{t}"] = dg["rung3_delta"].astype(np.float32)
    np.savez_compressed(out_dir / Path(args.out_npz).name, **save)
    print(f"[npz] 结果 -> {out_dir / Path(args.out_npz).name}")
    return 0


def _run_test_rx(args, fr: FrameSpecML, cap: np.ndarray, out_dir: Path) -> None:

    plan = fr.plan
    print(f"[test-rx] 抓层 1 首 chunk 一帧 (D={plan.D}, K={plan.K}, M={plan.M}, "
          f"R={plan.R}), 看频谱 + 同步 + 解码诊断...")
    if _MPL_OK:
        n_fft = min(1 << 20, cap.size)
        seg = cap[:n_fft].astype(np.complex128)
        win = np.blackman(n_fft)
        spec = np.fft.fftshift(np.fft.fft(seg * win)) / max(np.sum(win), 1e-30)
        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / args.fs))
        df = args.fs / fr.L
        f_lo = fr.out_bins[0] * df / 1e6
        f_hi = fr.out_bins[-1] * df / 1e6
        comb_mhz = plan.kmax * df / 1e6
        figp = out_dir / "test_rx_spectrum.png"
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(freqs / 1e6, 20 * np.log10(np.maximum(np.abs(spec), 1e-12)), lw=0.5)
        ax.axvspan(f_lo, f_hi, color="r", alpha=0.25,
                   label=f"{plan.K} output tones (layer 1)")
        ax.set_xlabel("baseband freq (MHz), RX@300 MHz")
        ax.set_ylabel("dBFS-like")
        ax.set_title(f"RX spectrum (--test-rx, layer-1 chunk 1); "
                     f"{plan.K} tones near DC, combs within ±{comb_mhz:.1f} MHz")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(figp, dpi=160); plt.close(fig)
        print(f"[test-rx] 频谱 -> {figp}")
    peak = float(np.max(np.abs(cap)))
    print(f"[test-rx] 峰值 |IQ|max={peak:.3f} (>{RX_CLIP_THRESH} 为 clip), "
          f"RMS={float(np.sqrt(np.mean(np.abs(cap)**2))):.4f}")
    res = decode_ml(cap, fr, args.fs, flip_output=args.flip_output,
                    conj_output=args.conj_output, verbose=True)
    if res.ok:
        print(f"[test-rx] 解码 OK: 层 1 本 chunk {fr.n_vec} 张, "
              f"|G|={abs(res.G_hat):.3e}, 归一 RMSE={res.logit_rmse:.4f}, "
              f"输出 bin SNR≈{res.snr_bin_db:+.1f} dB.")
        if res.logit_rmse > 0.5:
            print("[test-rx] ⚠ RMSE 异常高: 若 dry-run 正常而硬件高, 试 --flip-output "
                  "(下变频可能镜像了频谱); 也检查 LO/RF 是否接反.")
        else:
            print("[test-rx] 链路就绪. 去掉 --test-rx 正式跑完整三层链 "
                  "(每层 发射->接收->|·| 激活->再发射).")
    else:
        print(f"[test-rx] 解码失败: {res.reason}")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="三层复数 MLP (784->H1->H2->10) 的 RF 模拟 MVM 推理: "
                    "每层一次 USRP+ZEM-4300 模拟矩阵乘, 层间在 Python 里做 |y| "
                    "模值激活并归一后再发射下一层. 权重由 "
                    "gtsrb_train_digital_deep.py 训练 (act='abs', 无偏置, 层间判定).")

    p.add_argument("--weights", type=str, default="gtsrb_mlp10_weights.npz",
                   help="N 层权重 npz (W1..WN, act='abs', 含 quant_levels)")
    p.add_argument("--quant-levels", type=int, default=-1,
                   help="层间判定电平数 Q; -1(默认)=沿用权重里训练时的 Q; "
                        "0=关判定(含噪模拟值直传, A/B 对照用)")
    p.add_argument("--n-test", type=int, default=N_TEST_DEFAULT, help="总推理图片数")
    p.add_argument("--offset", type=int, default=0, help="从测试集第几张开始")
    p.add_argument("--images-per-frame", type=int, default=IMAGES_PER_FRAME_DEFAULT,
                   help="每帧样本数上限 (各层再按符号重复数 R 与 RX 环形缓冲收紧)")

    p.add_argument("--fft-len", type=int, default=FFT_LEN_DEFAULT)
    p.add_argument("--cp-len", type=int, default=CP_LEN_DEFAULT)
    p.add_argument("--k0", type=int, default=K0_DEFAULT,
                   help="输出 bin 起点 (>=2, 需给下方留 2 个保护 bin)")
    p.add_argument("--gap", type=int, default=GAP_SAMPLES_DEFAULT)
    p.add_argument("--seed", type=int, default=SYNC_SEED_DEFAULT, help="SYNC 种子")
    p.add_argument("--fs", type=float, default=USRP_SAMPLE_RATE)
    p.add_argument("--bw-frac", type=float, default=BW_FRAC_DEFAULT,
                   help="梳齿占带比: |k|max <= bw_frac*L/2 (X310 DDC 平坦区)")
    p.add_argument("--k-per-symbol", type=int, default=None,
                   help="覆写每符号输出行数 K (默认按 --bw-frac 自动规划各层)")
    p.add_argument("--avg-per-layer", type=str, default="1",
                   help="逐层重复测量次数, 功率平均降噪. 单值广播全层 (如 '4'), 或逗号"
                        "列表按层 —— ★个数必须等于层数★ (10 层就要 10 个, 如 "
                        "'4,1,1,1,1,1,1,1,1,2'; 3 层的老写法 '1,3,15' 在 10 层会报错). "
                        "弱层多测, 每层 +10log10(N) dB 有效幅度 SNR. 建议配合 --debias-act")
    p.add_argument("--flip-output", action="store_true",
                   help="读 -k 处输出 (下变频镜像频谱时用)")
    p.add_argument("--conj-output", action="store_true",
                   help="解码后取共轭 (等效 I/Q 交换)")
    p.add_argument("--calib-npz", type=str, default=None,
                   help="解码端逐神经元增益标定 npz (make_calibration.py 从实测 "
                        "run npz 生成, 键 g_l1..g_lN, 每层均值=1). 设置后每层 "
                        "激活幅度与末层判决功率都先除以 g_j (功率除 g_j²) 再用")
    p.add_argument("--input-eq-npz", type=str, default=None,
                   help="层 1 逐输入 tone 响应 h 的 npz (键 'h', 复数 D 维). "
                        "设置后层 1 在 TX 侧预均衡 x'=x/h (|1/h| 截幅见 "
                        "--input-eq-cap); 保真度参考仍为真值 W·x. "
                        "⚠ 诊断-h 模型已被 run 6 干预否证 (72.3%% vs 预测 "
                        "~92.9%%, 见 eq_evidence/README.md) — 仅作干预工具, "
                        "不是已验证的精度改进")
    p.add_argument("--rung3-json", type=str, default=None,
                   help="rung-3 读出端数字校正参数 json (rung3_refit.py 从同会话 "
                        "control run 生成; 逐层 |ref| 均衡 + beta·delta 扣除, "
                        "原始 y_mag 保留存档; 与 --input-eq-npz 互斥)")
    p.add_argument("--input-eq-cap", type=float, default=3.0,
                   help="TX 预均衡的最大 |1/h| 提升倍数 (防 PAPR/DAC 撑爆)")
    p.add_argument("--debias-act", action="store_true",
                   help="激活用 sqrt(max(|y|^2-sigma2,0)) 去噪声偏置 (默认关: 直接 |y|)")

    p.add_argument("--p-lo-dbm", type=float, default=P_LO_DBM_DEFAULT)
    p.add_argument("--p-rf-dbm", type=float, default=P_RF_DBM_DEFAULT)
    p.add_argument("--rf-atten-db", type=float, default=RF_ATTEN_DB_DEFAULT,
                   help="RF 支路外接衰减 (仅记录进 meta, 不参与计算)")
    p.add_argument("--if-atten-db", type=float, default=IF_ATTEN_DB_DEFAULT,
                   help="IF 支路外接衰减 (仅记录进 meta)")
    p.add_argument("--rx-gain", type=float, default=RX_GAIN_DB_DEFAULT)
    p.add_argument("--p-max-dbm", type=float, default=P_MAX_DBM_DEFAULT)
    p.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT)
    p.add_argument("--capture-retries", type=int, default=CAPTURE_RETRIES_DEFAULT)
    p.add_argument("--decode-retries", type=int, default=DECODE_RETRIES_DEFAULT,
                   help="硬件解码失败时整帧重抓次数; 仍失败则丢弃该 chunk 样本")
    p.add_argument("--start-lead", type=float, default=START_LEAD_S_DEFAULT,
                   help="TX 定时启动提前量(秒), 消 U/L 下溢")
    p.add_argument("--no-ext-ref", action="store_true",
                   help="RX 不用外部 10 MHz 参考 (默认 external)")
    p.add_argument("--tx-args", type=str, default=TX_ARGS)
    p.add_argument("--rx-args", type=str, default=RX_ARGS)

    p.add_argument("--dry-run", action="store_true",
                   help="不碰硬件: 数字合成混频+捕获, 全解码/绘图链路照走")
    p.add_argument("--sim-mixer", choices=("ideal", "limiter"), default="ideal",
                   help="dry-run 混频模型: ideal=sB*conj(sA); limiter=LO 恒包络")
    p.add_argument("--snr-db", type=float, default=None,
                   help="dry-run 输出 bin SNR (缺省无噪)")
    p.add_argument("--target-snr-db", type=float, default=None,
                   help="硬件采集后数字注噪到指定输出 bin SNR (只能降不能升; 全层同值)")
    p.add_argument("--snr-thermal-db", type=float, default=SNR_THERMAL_DB,
                   help="把每层设到目标 signal-to-thermal SNR(dB) (信号对热噪底): "
                        "注 AWGN, σ²=P_signal/SNR 直接叠加. 全层同值, dry-run/硬件均可. "
                        f"缺省取脚本顶部常量 SNR_THERMAL_DB(={SNR_THERMAL_DB}). 例 --snr-thermal-db 15 / 25 / 40")
    p.add_argument("--target-eff-snr", type=str, default=None,
                   help="逐层目标有效SNR(dB): 用标定AWGN把该层降到目标. 单值广播全层, "
                        "或逗号列表按层(空=该层不动) —— ★个数必须等于层数★ (10 层要 10 个, "
                        "如 '15,,,,,,,,,' 只降 L1). 只能降; 要升用 --avg-per-layer(积分增益). "
                        "自动扣平均增益(bin目标=目标-10log10(N)), dry-run/硬件均可用")
    p.add_argument("--paper-snr-db", type=float, default=None,
                   help="论文口径 SNR (信号/热噪底): 标定后自动调 RF 功率逼近该值")
    p.add_argument("--calib-noise", action="store_true",
                   help="静默 TX 量热噪底并打印论文口径 SNR 标定表后退出")
    p.add_argument("--test-rx", action="store_true",
                   help="只抓层 1 首 chunk 一帧: 频谱+同步+解码诊断后退出")
    p.add_argument("--replot", type=str, default=None,
                   help="从已存 npz 重画混淆图后退出")

    p.add_argument("--digital-paired", action="store_true",
                   help="左图数字基线只用实测同批样本 (缺省: 全部测试集)")
    p.add_argument("--note", type=str, default="",
                   help="附加说明文字 (画进图 + 存进 npz)")
    p.add_argument("--out-npz", type=str, default=f"{DEFAULT_OUTPUT_STEM}.npz")
    p.add_argument("--out-png", type=str, default=f"{DEFAULT_OUTPUT_STEM}.png")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.k0 < 2:
        raise ValueError("--k0 必须 >= 2 (输出带下方要留 2 个保护 bin)")
    if not (0.0 < args.bw_frac <= 1.0):
        raise ValueError("--bw-frac 必须在 (0, 1] 内")
    if args.k_per_symbol is not None and args.k_per_symbol < 1:
        raise ValueError("--k-per-symbol 必须 >= 1")
    if args.n_test < 1 or args.images_per_frame < 1:
        raise ValueError("--n-test 与 --images-per-frame 必须 >= 1")
    if (args.paper_snr_db is not None) and (args.target_snr_db is not None):
        raise ValueError("--paper-snr-db (硬件设功率, 信号/热噪底) 与 --target-snr-db "
                         "(数字加噪, 信号/总噪声) 口径不同, 不能同时用; 二选一")
    if (args.target_eff_snr is not None) and (args.target_snr_db is not None
                                              or args.paper_snr_db is not None):
        raise ValueError("--target-eff-snr (逐层, 加噪降到目标有效SNR) 与 "
                         "--target-snr-db/--paper-snr-db 不能同用; 三选一")
    if (args.snr_thermal_db is not None) and (args.target_eff_snr is not None
                                           or args.target_snr_db is not None
                                           or args.paper_snr_db is not None):
        raise ValueError("--snr-thermal-db (signal/thermal, 注热噪等效底) 与 "
                         "--target-eff-snr/--target-snr-db/--paper-snr-db 口径不同, 不能同用")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
