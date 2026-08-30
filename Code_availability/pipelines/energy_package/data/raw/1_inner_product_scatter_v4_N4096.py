#!/usr/bin/env python3
# ===== 本文件为 N=4096 默认版 (手稿口径: N=4096, Q=200) =====
#   与 65536 版 (1_inner_product_scatter_v4.py) 唯一区别是默认常量:
#   N/L/CP/k0/gap = 4096/16384/512/6/8192 (原始频域几何, IP bin 仍在 +3.66 kHz);
#   其余逐行相同, 也可用 --N 65536 --fft-len 262144 --cp-len 8192 --k0 96
#   --gap 131072 跑回大尺寸.
# v4 变更 (相对 v3):
#   1. 指标从 RMSE 换成手稿定义的 ENOB = log2( std[y] / RMSE ):
#        y     = 数字真值内积 (Q=200 个 data slot, 归一化 c/sqrt(N));
#        RMSE  = sqrt( (1/Q) * sum_i |y_i - ŷ_i|^2 )   <- 手稿圈出的分母形式.
#      另给出 std[y-ŷ] 分母变体 (enob_errstd) 作参考, 两者仅差残余偏置
#      (导频复增益标定后 mean(y-ŷ)≈0, 数值几乎一致).
#   2. 新增 --repeats R (默认 3): 闭环收敛后功率固定不动, 独立重复抓取 R 次
#      得 ŷ^(1..R), 每次算一个 ENOB^(k); 汇总 mean ± 1 s.d. (ddof=1) 画误差棒,
#      --errorbar sem 可换成标准误 s.d./sqrt(n). 散点图并入全部重复点,
#      并额外输出 *_enob.png 误差棒图.
#   3. --replot 支持逗号分隔多个 NPZ: 把多次独立运行 (须相同 seed/N/q-data,
#      即同一组真值 y) 合并为重复实验重画误差棒; 旧 v3 NPZ 也能重画
#      (自动由存档的 RMSE 换算 ENOB).
#   4. --tune-metric 新增 enob (目标单位: bit).

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
import math
import os
import sys
import time
import threading
import warnings
from dataclasses import dataclass
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
DEFAULT_OUTPUT_STEM = "gr_fig3c_ip_scatter"

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
RX_CLIP_THRESH = 0.98

RX_RING_CAPACITY = 1 << 24

RF_ATTEN_DB_DEFAULT = 30.0
IF_ATTEN_DB_DEFAULT = 0.0

# N=4096 原始配套参数 (手稿口径; 65536 版即由这套 x16 得到):
#   梳最大占用 |k|max = 3N/2 + k0 - 2 = 6148 < L/2 = 8192 (占用 75% Nyquist);
#   CP = L/32, gap = L/2; k0=6 (≡0 mod 3), fs=10M 时 IP bin 在 k0*fs/L ≈ +3.66 kHz,
#   远离 DC 附近的 1/f 噪声与直流校正残留.
N_DEFAULT = 4096
FFT_LEN_DEFAULT = 16384
CP_LEN_DEFAULT = 512
COMB_M = 3
K0_DEFAULT = 6
Q_DATA_DEFAULT = 200
PILOT_EVERY_DEFAULT = 8
GAP_SAMPLES_DEFAULT = 8192
SYNC_BAND_FRAC = 8

VECTOR_SEED_DEFAULT = 0

P_LO_DBM_DEFAULT = -3.0
P_RF_DBM_INIT_DEFAULT = -35.0
TUNE_MAX_ITERS_DEFAULT = 4
TUNE_TOL_DB_DEFAULT = 0.4
SETTLE_S_DEFAULT = 0.6
CAPTURE_RETRIES_DEFAULT = 6
DECODE_RETRIES_DEFAULT = 2



@dataclass
class GainAmp:
    gain_db: float
    amp: float
    predicted_dbm: float


def power_to_gain_amp(
    target_dbm: float,
    p_max_dbm: float = P_MAX_DBM_DEFAULT,
    amp_target: float = AMP_TARGET,
    peak_factor: float = 1.0,
) -> GainAmp:
    """峰值感知的 (gain, amp) 联合求解.

    修复要点: 旧版先按 amp_target 定 gain, 再在 set_powers 里为满足数字峰值
    amp*peak <= AMP_MAX 单方面削 amp 而不补 gain, 导致高 PAPR 波形 (N=65536
    未削波时 ~16 dB) 的实际输出比目标低 10+ dB (LO 目标 -3 dBm 实发 -14.9 dBm,
    把 ZEM-4300 推进欠驱动扩张区, 造成 RMSE≈0.115 的确定性 floor).
    现在把峰值约束放进求解: amp 上限 = AMP_MAX/peak_factor, 差额用模拟 gain 补,
    只有 gain 顶满 31.5 dB 仍不够时才真正受限并告警.
    """
    amp_ceil = min(AMP_MAX, AMP_MAX / max(float(peak_factor), 1e-9))
    amp = min(float(amp_target), amp_ceil)
    gain = GAIN_MAX_DB + target_dbm - p_max_dbm - 20.0 * np.log10(max(amp, 1e-30))
    if gain > GAIN_MAX_DB:
        gain = GAIN_MAX_DB
        amp = 10.0 ** ((target_dbm - p_max_dbm) / 20.0)
        if amp > amp_ceil:
            warnings.warn(
                f"target {target_dbm:+.2f} dBm 超过该波形峰值约束下的最大平均功率 "
                f"≈{p_max_dbm + 20.0 * np.log10(amp_ceil):+.2f} dBm "
                f"(峰值系数 {peak_factor:.2f}, PAPR 限制); 按上限输出. "
                f"可用 --papr-clip-db 进一步降低 PAPR 换取平均功率.",
                RuntimeWarning,
            )
            amp = amp_ceil
    elif gain < GAIN_MIN_DB:
        gain = GAIN_MIN_DB
        amp = 10.0 ** ((target_dbm - (p_max_dbm - GAIN_MAX_DB)) / 20.0)
        if amp < AMP_WARN:
            warnings.warn(
                f"target {target_dbm:+.2f} dBm requires amp={amp:.2e} "
                f"(< {AMP_WARN:.0e}); may be below DAC SFDR.",
                RuntimeWarning,
            )
    predicted = p_max_dbm + (gain - GAIN_MAX_DB) + 20.0 * np.log10(max(amp, 1e-30))
    return GainAmp(gain_db=float(gain), amp=float(amp), predicted_dbm=float(predicted))



def comb_bins_lo(N: int) -> np.ndarray:
    n = np.arange(N, dtype=np.int64)
    return 3 * (n - N // 2) + 1


def gen_vec_pair(N: int, base_seed: int, slot_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([int(base_seed), int(slot_idx)])
    ra = rng.uniform(0.0, 1.0, N)
    pa = rng.uniform(0.0, 2.0 * np.pi, N)
    rb = rng.uniform(0.0, 1.0, N)
    pb = rng.uniform(0.0, 2.0 * np.pi, N)
    a = (ra * np.exp(1j * pa)).astype(np.complex128)
    b = (rb * np.exp(1j * pb)).astype(np.complex128)
    return a, b


def symbol_td_from_bins(vec: np.ndarray, bins: np.ndarray, L: int) -> np.ndarray:
    X = np.zeros(L, dtype=np.complex128)
    X[np.mod(bins, L)] = vec
    return np.fft.ifft(X)


def add_cp(sym: np.ndarray, cp: int) -> np.ndarray:
    return np.concatenate([sym[-cp:], sym])


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


@dataclass
class SlotInfo:
    idx: int
    kind: str
    seed_idx: int
    c_true: complex


@dataclass
class FrameSpec:
    N: int
    L: int
    cp: int
    k0: int
    gap0: int
    gap1: int
    slots: List[SlotInfo]
    sA: np.ndarray
    sB: np.ndarray
    ref_pre: np.ndarray
    peak_a: float
    peak_b: float
    frame_len: int
    sync_payload_off: int
    n_pilot: int
    n_data: int
    dev_lim: Optional[np.ndarray] = None
    dev_sq: Optional[np.ndarray] = None


def build_frame(
    N: int, L: int, cp: int, k0: int, q_data: int, pilot_every: int,
    gap0: int, gap1: int, seed: int, verbose: bool = True,
    papr_clip_db: float = 0.0, mixer_diag: bool = True,
) -> FrameSpec:
    bins_a = comb_bins_lo(N)
    bins_b = bins_a + int(k0)
    kmax = int(max(np.max(np.abs(bins_a)), np.max(np.abs(bins_b))))
    if kmax >= L // 2:
        raise ValueError(f"占用 bin |k|max={kmax} >= L/2={L//2}; 增大 --fft-len 或减小 --N")
    occ_frac = kmax / (L / 2.0)
    if occ_frac > 0.8:
        warnings.warn(f"占用带宽达 Nyquist 的 {occ_frac*100:.0f}% (>80%), "
                      f"可能进入 DDC 滤波器滚降区.", RuntimeWarning)

    kinds: List[str] = []
    nd = 0
    while nd < q_data:
        kinds.append("P")
        for _ in range(pilot_every):
            if nd < q_data:
                kinds.append("D")
                nd += 1
    kinds.append("P")

    sa_pre, sb_pre = gen_sync_symbols(L, seed)

    parts_a: List[np.ndarray] = [np.zeros(gap0, np.complex128), add_cp(sa_pre, cp)]
    parts_b: List[np.ndarray] = [np.zeros(gap0, np.complex128), add_cp(sb_pre, cp)]
    slots: List[SlotInfo] = []
    for s, kind in enumerate(kinds):
        seed_idx = s if kind == "D" else 100000 + s
        a, b = gen_vec_pair(N, seed, seed_idx)
        c_true = complex(np.vdot(a, b))
        slots.append(SlotInfo(idx=s, kind=kind, seed_idx=seed_idx, c_true=c_true))
        parts_a.append(add_cp(symbol_td_from_bins(a, bins_a, L), cp))
        parts_b.append(add_cp(symbol_td_from_bins(b, bins_b, L), cp))
    parts_a.append(np.zeros(gap1, np.complex128))
    parts_b.append(np.zeros(gap1, np.complex128))

    sA = np.concatenate(parts_a)
    sB = np.concatenate(parts_b)
    frame_len = sA.size
    act = slice(gap0, frame_len - gap1)

    rms_a = float(np.sqrt(np.mean(np.abs(sA[act]) ** 2)))
    rms_b = float(np.sqrt(np.mean(np.abs(sB[act]) ** 2)))
    sA /= max(rms_a, 1e-30)
    sB /= max(rms_b, 1e-30)

    if papr_clip_db and papr_clip_db > 0.0:
        # 数字削波: 未削波时 N=65536 的 OFDM PAPR ≈ 16 dB, 把可达平均功率压到
        # ≈+2 dBm; 削到 ~10 dB 可换回 ~6 dB 平均功率上限, 失真 (EVM) 实测很小.
        thr = 10.0 ** (papr_clip_db / 20.0)
        for tag in ("LO", "RF"):
            x = sA if tag == "LO" else sB
            mag = np.abs(x)
            papr_before = 20.0 * np.log10(max(float(np.max(mag)), 1e-30))
            frac = float(np.mean(mag > thr))
            y = np.where(mag > thr, x * (thr / np.maximum(mag, 1e-30)), x)
            evm = float(np.mean(np.abs(y - x) ** 2))
            y /= max(float(np.sqrt(np.mean(np.abs(y[act]) ** 2))), 1e-30)
            papr_after = 20.0 * np.log10(max(float(np.max(np.abs(y))), 1e-30))
            if tag == "LO":
                sA = y
            else:
                sB = y
            if verbose:
                print(f"[frame] {tag} 数字削波 @ {papr_clip_db:.1f} dB: PAPR "
                      f"{papr_before:+.1f} -> {papr_after:+.1f} dB, 削顶 "
                      f"{frac * 100:.3f}% 样本, 失真 EVM = "
                      f"{10.0 * np.log10(max(evm, 1e-30)):+.1f} dBc")

    peak_a = float(np.max(np.abs(sA)))
    peak_b = float(np.max(np.abs(sB)))

    sl_pre = slice(gap0 + cp, gap0 + cp + L)
    ref = (sB[sl_pre] * np.conj(sA[sl_pre]))
    ref = (ref / np.linalg.norm(ref)).astype(np.complex128)

    dev_lim = dev_sq = None
    if mixer_diag:
        # 预计算两个混频器包络非线性的逐 slot 偏差基底 (以最终发射波形计):
        #   limiter: g(A)=1 (过驱动/限幅);  square: g(A)=A^2 (欠驱动/扩张).
        # 解码时用残差与基底的相关符号判断 LO 驱动落在哪一侧, --mixer-comp 可扣除.
        c_all_b = np.array([s.c_true for s in slots], dtype=np.complex128)
        S_b = len(slots)
        Yl = np.empty(S_b, np.complex128)
        Ysq = np.empty(S_b, np.complex128)
        for s in range(S_b):
            off = gap0 + (cp + L) * (1 + s) + cp
            a_t = sA[off:off + L]
            b_t = sB[off:off + L]
            mm = np.abs(a_t)
            lo1 = np.where(mm > 1e-12, a_t / np.maximum(mm, 1e-30), 0.0)
            Yl[s] = np.fft.fft(b_t * np.conj(lo1))[k0 % L]
            Ysq[s] = np.fft.fft(b_t * np.conj(a_t * mm))[k0 % L]
        wc = max(float(np.sum(np.abs(c_all_b) ** 2)), 1e-30)
        sqN_b = math.sqrt(N)
        dev_lim = ((Yl / (np.sum(Yl * np.conj(c_all_b)) / wc)) - c_all_b) / sqN_b
        dev_sq = ((Ysq / (np.sum(Ysq * np.conj(c_all_b)) / wc)) - c_all_b) / sqN_b
        if verbose:
            print(f"[frame] 混频非线性诊断基底: |dev_limiter| rms="
                  f"{float(np.sqrt(np.mean(np.abs(dev_lim) ** 2))):.4f}, "
                  f"|dev_square| rms="
                  f"{float(np.sqrt(np.mean(np.abs(dev_sq) ** 2))):.4f}")

    n_pilot = sum(1 for s in slots if s.kind == "P")
    fr = FrameSpec(
        N=N, L=L, cp=cp, k0=k0, gap0=gap0, gap1=gap1, slots=slots,
        sA=sA.astype(np.complex64), sB=sB.astype(np.complex64),
        ref_pre=ref, peak_a=peak_a, peak_b=peak_b, frame_len=frame_len,
        sync_payload_off=gap0 + cp, n_pilot=n_pilot, n_data=q_data,
        dev_lim=dev_lim, dev_sq=dev_sq,
    )
    if verbose:
        df = None
        print(f"[frame] N={N}, L={L} (Δf=fs/L), CP={cp}, K0={k0}, "
              f"slots={len(slots)} (pilot={n_pilot}, data={q_data})")
        print(f"[frame] frame_len={frame_len} samples; 占用 |k|<= {kmax} "
              f"({occ_frac*100:.0f}% Nyquist)")
        print(f"[frame] unit-RMS 帧峰值: LO={peak_a:.3f} ({20*np.log10(peak_a):+.1f} dB PAPR), "
              f"RF={peak_b:.3f} ({20*np.log10(peak_b):+.1f} dB PAPR)")
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
class DecodeResult:
    ok: bool
    reason: str = ""
    d0: int = -1
    peak_metric: float = 0.0
    frame_repeat_err: Optional[int] = None
    c_true: Optional[np.ndarray] = None
    Y_data: Optional[np.ndarray] = None
    c_hat: Optional[np.ndarray] = None
    c_true_norm: Optional[np.ndarray] = None
    G_hat: complex = 0j
    drift_rad_per_slot: float = 0.0
    pilot_rmse: float = float("nan")
    rmse: float = float("nan")
    std_y: float = float("nan")
    enob: float = float("nan")
    enob_errstd: float = float("nan")
    snr3_db: float = float("nan")
    snr1_db: float = float("nan")
    snr_02mhz_db: float = float("nan")
    sigma2_bin: float = float("nan")
    p_ip_raw: float = float("nan")
    clip_frac: float = 0.0
    spec_win_db: Optional[np.ndarray] = None
    spec_win_bins: Optional[np.ndarray] = None
    mix_corr_lim: complex = 0j
    mix_corr_sq: complex = 0j
    mixer_comp: bool = False


def decode_capture(cap: np.ndarray, fr: FrameSpec, fs: float,
                   verbose: bool = True, mixer_comp: bool = False) -> DecodeResult:
    L, cp, k0 = fr.L, fr.cp, fr.k0
    S = len(fr.slots)
    need_after = L + S * (cp + L) + 64
    if cap.size < fr.frame_len + need_after // 4:
        return DecodeResult(ok=False, reason=f"capture too short ({cap.size})")

    clip_frac = float(np.mean(np.abs(cap) > RX_CLIP_THRESH))

    corr = xcorr_os(cap, fr.ref_pre)
    mag = np.abs(corr)
    del corr  # N=65536 时 corr 约 2.3 GB (complex128), 之后只用 mag, 及时释放
    valid_last = cap.size - need_after
    if valid_last < 1:
        return DecodeResult(ok=False, reason="capture shorter than one frame")
    mag_valid = mag[:valid_last]
    d0 = int(np.argmax(mag_valid))
    pk = float(mag_valid[d0])
    guard = np.copy(mag)
    lo_g = max(0, d0 - 3 * L); hi_g = min(mag.size, d0 + 3 * L)
    guard[lo_g:hi_g] = 0.0
    # 屏蔽所有 ±k*frame_len 处的帧重复峰: 2.3 帧窗内可能出现第 3 个前导
    # (概率 ~28%, 取决于抓取相位), 原实现只屏蔽 +1 帧处, 会把真峰当旁瓣
    # 导致 峰/旁瓣≈1.0 的假性同步失败 (硬件同样会随机踩中).
    n_rep = int(mag.size // max(fr.frame_len, 1)) + 2
    for k in range(1, n_rep):
        for rep_k in (d0 + k * fr.frame_len, d0 - k * fr.frame_len):
            if 0 <= rep_k < guard.size:
                guard[max(0, rep_k - 64): min(guard.size, rep_k + 65)] = 0.0
    side = float(np.max(guard)) if guard.size else 1e-30
    peak_metric = pk / max(side, 1e-30)

    rep = d0 + fr.frame_len
    frame_repeat_err = None
    if rep + 1 < mag.size:
        w = mag[max(0, rep - 64): min(mag.size, rep + 65)]
        frame_repeat_err = int(np.argmax(w) + max(0, rep - 64) - rep)

    if verbose:
        print(f"[sync] 峰位 d0={d0}, 峰/旁瓣={peak_metric:.1f}, "
              f"帧重复偏差={frame_repeat_err} 样本, clip={clip_frac*100:.3f}%")
    if peak_metric < 3.0:
        return DecodeResult(ok=False, reason=f"sync peak weak ({peak_metric:.2f}); "
                            f"检查发射是否在跑/功率/接线", d0=d0,
                            peak_metric=peak_metric, clip_frac=clip_frac)

    # 噪声 bin 用 ±4/±5: 仍避开互调栅格 (k0±3j), 且在 Δf=fs/L≈38 Hz 时
    # 避开 IP 主线 ±1/±2 bin 处的近载波相位噪声裙边, sigma2 更接近真实加性噪底.
    noise_bins = np.array([k0 - 5, k0 - 4, k0 + 4, k0 + 5], dtype=int)
    win_bins = np.arange(k0 - 12, k0 + 13, dtype=int)
    Yk = np.empty(S, np.complex128)
    Yn = np.empty((S, noise_bins.size), np.complex128)
    spec_acc = np.zeros(win_bins.size, np.float64)
    for s in range(S):
        st = d0 + L + s * (cp + L) + cp
        Y = np.fft.fft(cap[st:st + L].astype(np.complex128))
        Yk[s] = Y[k0 % L]
        Yn[s, :] = Y[np.mod(noise_bins, L)]
        spec_acc += np.abs(Y[np.mod(win_bins, L)]) ** 2
    spec_win_db = 10.0 * np.log10(np.maximum(spec_acc / S, 1e-30))

    kinds = np.array([s.kind for s in fr.slots])
    c_all = np.array([s.c_true for s in fr.slots], dtype=np.complex128)
    is_p = kinds == "P"
    is_d = kinds == "D"

    sigma2_bin = float(np.mean(np.abs(Yn) ** 2))
    p_ip_raw = float(np.mean(np.abs(Yk[is_d]) ** 2))
    p_sig = max(p_ip_raw - sigma2_bin, 1e-30)
    snr3 = p_sig / (3.0 * sigma2_bin)
    snr1 = p_sig / sigma2_bin
    df_hz = fs / L
    snr_02 = p_sig / (sigma2_bin * (0.2e6 / df_hz))
    
    sp = np.where(is_p)[0]
    gp = Yk[sp] * np.conj(c_all[sp])
    wp = np.abs(c_all[sp]) ** 2
    ph = np.unwrap(np.angle(gp))
    try:
        slope, _icpt = np.polyfit(sp.astype(float), ph, 1, w=np.sqrt(np.maximum(wp, 1e-30)))
    except Exception:
        slope = 0.0
    derot = np.exp(-1j * slope * np.arange(S, dtype=float))
    Yd = Yk * derot
    G = complex(np.sum(Yd[sp] * np.conj(c_all[sp])) / max(np.sum(wp), 1e-30))
    if abs(G) <= 0:
        return DecodeResult(ok=False, reason="G_hat = 0 (无信号?)")

    sqN = math.sqrt(fr.N)
    chn = (Yd / G) / sqN
    ctn = c_all / sqN
    mix_corr_lim = mix_corr_sq = 0j
    comp_applied = False
    if fr.dev_lim is not None and fr.dev_sq is not None:
        res_all = chn - ctn
        den_r = float(np.sqrt(np.sum(np.abs(res_all) ** 2)))

        def _corr(D):
            den = den_r * float(np.sqrt(np.sum(np.abs(D) ** 2)))
            return complex(np.sum(res_all * np.conj(D)) / max(den, 1e-30))

        mix_corr_lim = _corr(fr.dev_lim)
        mix_corr_sq = _corr(fr.dev_sq)
        if mixer_comp:
            # 仅用导频 slot 拟合 (α_lim, α_sq), 应用到全部 slot (含数据):
            # 不用数据自拟合, 属于校准式后处理.
            Xp = np.stack([fr.dev_lim[sp], fr.dev_sq[sp]], axis=1)
            try:
                alpha, *_ = np.linalg.lstsq(Xp, res_all[sp], rcond=None)
                chn = chn - (fr.dev_lim * alpha[0] + fr.dev_sq * alpha[1])
                g2 = np.sum(chn[sp] * np.conj(ctn[sp])) / max(
                    np.sum(np.abs(ctn[sp]) ** 2), 1e-30)
                if abs(g2) > 0:
                    chn = chn / g2
                comp_applied = True
            except Exception as exc:
                warnings.warn(f"--mixer-comp 拟合失败, 跳过: {exc!r}", RuntimeWarning)
    pilot_rmse = float(np.sqrt(np.mean(np.abs(chn[sp] - ctn[sp]) ** 2)))
    c_hat = chn[is_d]
    c_true_n = ctn[is_d]
    rmse = float(np.sqrt(np.mean(np.abs(c_hat - c_true_n) ** 2)))
    # --- ENOB (手稿定义): ENOB = log2( std[y] / sqrt((1/Q)Σ|y_i-ŷ_i|²) ) ---
    # 分母取手稿圈出的 RMSE 形式; 复数 std 按 numpy 约定 = sqrt(E|z-mean(z)|²),
    # 分子分母同一口径, 故与 "实部虚部拼成 2Q 个实数" 的算法给出完全相同的比值.
    err = c_hat - c_true_n
    std_y = float(np.std(c_true_n))
    err_std = float(np.std(err))
    enob = math.log2(max(std_y, 1e-300) / max(rmse, 1e-300))
    enob_errstd = math.log2(max(std_y, 1e-300) / max(err_std, 1e-300))

    if verbose:
        print(f"[decode] |G^|={abs(G):.4e}, drift={slope:+.2e} rad/slot, "
              f"pilot_RMSE={pilot_rmse:.4f}")
        print(f"[decode] SNR_3bin={10*np.log10(max(snr3,1e-30)):+.2f} dB "
              f"(1bin {10*np.log10(max(snr1,1e-30)):+.2f} dB, "
              f"0.2MHz 口径 {10*np.log10(max(snr_02,1e-30)):+.2f} dB); "
              f"RMSE={rmse:.4f}, std[y]={std_y:.4f} -> "
              f"ENOB=log2(std[y]/RMSE)={enob:.2f} bit")
    if verbose and fr.dev_lim is not None:
        rl, rs = float(np.real(mix_corr_lim)), float(np.real(mix_corr_sq))
        if rl < -0.5:
            hint = "=> LO 欠驱动 (扩张区): 建议升高 --p-lo-dbm / 检查 LO 实际功率"
        elif rl > 0.5:
            hint = "=> LO 过驱动 (限幅区): 建议降低 --p-lo-dbm"
        else:
            hint = "=> 混频接近理想乘法器"
        tail = ("" if (comp_applied or max(abs(rl), abs(rs)) < 0.5)
                else "; 可加 --mixer-comp 用导频拟合扣除该确定性误差")
        print(f"[decode] 混频非线性指纹: corr(limiter)={rl:+.2f}, "
              f"corr(square)={rs:+.2f} {hint}{tail}"
              + (" [已补偿]" if comp_applied else ""))

    return DecodeResult(
        ok=True, d0=d0, peak_metric=peak_metric, frame_repeat_err=frame_repeat_err,
        c_true=c_all[is_d], Y_data=Yk[is_d], c_hat=c_hat, c_true_norm=c_true_n,
        G_hat=G, drift_rad_per_slot=float(slope), pilot_rmse=pilot_rmse, rmse=rmse,
        std_y=std_y, enob=enob, enob_errstd=enob_errstd,
        snr3_db=float(10 * np.log10(max(snr3, 1e-30))),
        snr1_db=float(10 * np.log10(max(snr1, 1e-30))),
        snr_02mhz_db=float(10 * np.log10(max(snr_02, 1e-30))),
        sigma2_bin=sigma2_bin, p_ip_raw=p_ip_raw, clip_frac=clip_frac,
        spec_win_db=spec_win_db, spec_win_bins=win_bins,
        mix_corr_lim=mix_corr_lim, mix_corr_sq=mix_corr_sq,
        mixer_comp=comp_applied,
    )


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

        def grab_fresh(self, n: int, timeout_s: float = 30.0) -> np.ndarray:
            out, _info = self.grab_fresh_ex(n, timeout_s)
            return out

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
                    "gap_in_window": bool(gap_in_window),
                    "last_gap_total": int(last_gap)}
            return out, info

    class Fig3cFlowgraph(gr.top_block):

        def __init__(
            self,
            frame: FrameSpec,
            lo_path: str,
            rf_path: str,
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
            verbose: bool = True,
        ):
            gr.top_block.__init__(self, "WISE Fig3c IP scatter")
            self._verbose = verbose
            self._sr = float(sample_rate)
            self._p_max_dbm = float(p_max_dbm)
            self._frame = frame

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
            try:
                s = self.usrp_source.get_mboard_sensor("ref_locked", 0)
                print(f"[gr] RX ref_locked={bool(s.to_bool())} (10 MHz 外参)")
            except Exception:
                pass

        def set_powers(self, p_rf_dbm: float, p_lo_dbm: float) -> dict:
            # 峰值约束在 power_to_gain_amp 内部与 gain 联合求解:
            # 数字峰值 amp*peak <= AMP_MAX 恒成立, 缺口由模拟 gain 补齐,
            # 仅当 gain 顶满仍不够时 predicted < target (真正 PAPR 受限).
            rf = power_to_gain_amp(p_rf_dbm, p_max_dbm=self._p_max_dbm,
                                   peak_factor=self._frame.peak_b)
            lo = power_to_gain_amp(p_lo_dbm, p_max_dbm=self._p_max_dbm,
                                   peak_factor=self._frame.peak_a)
            clip_rf = bool(rf.predicted_dbm < p_rf_dbm - 0.05)
            clip_lo = bool(lo.predicted_dbm < p_lo_dbm - 0.05)
            self.mul_rf.set_k(complex(rf.amp, 0.0))
            self.mul_lo.set_k(complex(lo.amp, 0.0))
            self.usrp_sink.set_gain(rf.gain_db, 0)
            self.usrp_sink.set_gain(lo.gain_db, 1)
            return {
                "rf": {"target_dbm": float(p_rf_dbm), "gain_db": float(rf.gain_db),
                       "amp_rms": float(rf.amp), "peak_limited": clip_rf,
                       "predicted_dbm": float(rf.predicted_dbm)},
                "lo": {"target_dbm": float(p_lo_dbm), "gain_db": float(lo.gain_db),
                       "amp_rms": float(lo.amp), "peak_limited": clip_lo,
                       "predicted_dbm": float(lo.predicted_dbm)},
            }

else:
    IQGrabber = None
    Fig3cFlowgraph = None



def sim_mix(sA: np.ndarray, sB: np.ndarray, model: str) -> np.ndarray:
    sA = sA.astype(np.complex128)
    sB = sB.astype(np.complex128)
    if model == "ideal":
        return sB * np.conj(sA)
    if model == "limiter":
        m = np.abs(sA)
        lo = np.zeros_like(sA)
        nz = m > 1e-9
        lo[nz] = sA[nz] / m[nz]
        return sB * np.conj(lo)
    raise ValueError(f"unknown mixer model {model!r}")


def synth_capture(fr: FrameSpec, snr3_db: float, mixer: str, seed: int,
                  fs: float, verbose: bool = True, rep: int = 0) -> np.ndarray:
    # rep>0 时扩展种子, 使每次重复测量的噪声与截取相位相互独立;
    # rep=0 保持与 v3 完全相同的随机流 (bit 级可复现).
    seed_key = [int(seed), 4242, int(round(snr3_db * 10))]
    if int(rep) > 0:
        seed_key.append(int(rep))
    rng = np.random.default_rng(seed_key)
    y_clean_frame = sim_mix(fr.sA, fr.sB, mixer)

    S = len(fr.slots)
    is_d = np.array([s.kind == "D" for s in fr.slots])
    Yk = np.empty(S, np.complex128)
    for s in range(S):
        st = fr.sync_payload_off + fr.L + s * (fr.cp + fr.L) + fr.cp
        Yk[s] = np.fft.fft(y_clean_frame[st:st + fr.L])[fr.k0 % fr.L]
    p_ip_clean = float(np.mean(np.abs(Yk[is_d]) ** 2))

    snr_lin = 10.0 ** (snr3_db / 10.0)
    sigma2_bin = p_ip_clean / (3.0 * snr_lin)
    sigma_t = math.sqrt(sigma2_bin / fr.L)

    n_cap = int(round(2.3 * fr.frame_len))
    start = int(rng.integers(0, fr.frame_len))
    # 环形取样: 从无限循环播放的帧中任意相位处截取 2.3 帧.
    # (修复原实现 "拼 3 帧再切片" 在 start > 0.7*frame_len 时切片不足 n_cap
    #  导致与 noise 广播报错的潜在 bug, 且省去 ~3 帧的大数组内存.)
    cap = np.empty(n_cap, np.complex128)
    pos, idx = 0, start
    while pos < n_cap:
        take = min(fr.frame_len - idx, n_cap - pos)
        cap[pos:pos + take] = y_clean_frame[idx:idx + take]
        pos += take
        idx = 0
    noise = (rng.standard_normal(n_cap) + 1j * rng.standard_normal(n_cap)) * (sigma_t / math.sqrt(2.0))
    cap += noise
    cap *= 0.3 / max(float(np.max(np.abs(cap))), 1e-30)
    if verbose:
        print(f"[sim] mixer={mixer}, 目标 SNR_3bin={snr3_db:.1f} dB, "
              f"理论 AWGN RMSE={1.0/math.sqrt(27.0*snr_lin):.4f} (ideal 混频时)")
    return cap.astype(np.complex64)



def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in str(s).replace(";", ",").split(",") if x.strip()]


def tune_point(
    capture_fn: Callable[[], np.ndarray],
    set_rf_fn: Optional[Callable[[float], None]],
    fr: FrameSpec, fs: float,
    target: float, metric: str,
    p_rf_init: float, max_iters: int, tol_db: float,
    decode_retries: int = 0,
    verbose: bool = True,
    mixer_comp: bool = False,
) -> Tuple[DecodeResult, float, List[dict]]:
    p_rf = float(p_rf_init)
    history: List[dict] = []
    res = None
    for it in range(max_iters):
        res = None
        for attempt in range(int(decode_retries) + 1):
            cap = capture_fn()
            res = decode_capture(cap, fr, fs, verbose=verbose, mixer_comp=mixer_comp)
            if res.ok:
                break
            if attempt < int(decode_retries):
                warnings.warn(
                    f"[tune] iter{it} 解码失败 ({res.reason}); "
                    f"重抓 {attempt+1}/{int(decode_retries)}", RuntimeWarning)
        if res is None or not res.ok:
            reason = res.reason if res is not None else "no capture"
            raise RuntimeError(f"解码失败 (重试 {int(decode_retries)} 次后): {reason}")
        if metric == "snr":
            err_db = float(target) - res.snr3_db
        elif metric == "enob":
            err_db = 6.02 * (float(target) - res.enob)   # 1 bit ≈ 6.02 dB SNR
        else:
            err_db = 20.0 * math.log10(max(res.rmse, 1e-9) / float(target))
        history.append({"iter": it, "p_rf_dbm": p_rf, "snr3_db": res.snr3_db,
                        "rmse": res.rmse, "enob": res.enob, "err_db": err_db})
        if verbose:
            print(f"[tune] iter{it}: P_rf(TX口)={p_rf:+.2f} dBm -> "
                  f"SNR_3bin={res.snr3_db:+.2f} dB, RMSE={res.rmse:.4f}, "
                  f"ENOB={res.enob:.2f} bit (误差 {err_db:+.2f} dB)")
        if set_rf_fn is None or abs(err_db) <= tol_db or it == max_iters - 1:
            break
        p_rf += float(np.clip(err_db, -12.0, 12.0))
        set_rf_fn(p_rf)
    assert res is not None
    return res, p_rf, history


def _mean_std(vals) -> Tuple[float, float, int]:
    """有限值的 (均值, 样本标准差 ddof=1, n); n<2 时 s.d.=nan."""
    v = np.asarray([float(x) for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
    return m, s, int(v.size)


def _bar_value(mean: float, sd: float, n: int, mode: str) -> Tuple[float, float, str]:
    """按 --errorbar 模式把 (mean, s.d., n) 折成 (中心值, 误差棒半长, 标签)."""
    tag = "s.d." if mode == "std" else "s.e.m."
    if int(n) < 2 or not np.isfinite(sd):
        return float(mean), float("nan"), tag
    err = float(sd) if mode == "std" else float(sd) / math.sqrt(int(n))
    return float(mean), err, tag


def _pm(mean: float, err: float, fmt: str = ".2f") -> str:
    if np.isfinite(err):
        return f"{mean:{fmt}} ± {err:{fmt}}"
    return f"{mean:{fmt}}" if np.isfinite(mean) else "nan"


def capture_and_decode(capture_fn: Callable[[], np.ndarray], fr: FrameSpec, fs: float,
                       decode_retries: int, mixer_comp: bool,
                       verbose: bool = True) -> DecodeResult:
    """闭环收敛后单次独立重复测量: 抓取+解码, 失败按 --decode-retries 重试."""
    last = "no capture"
    for attempt in range(int(decode_retries) + 1):
        cap = capture_fn()
        res = decode_capture(cap, fr, fs, verbose=verbose, mixer_comp=mixer_comp)
        if res.ok:
            return res
        last = res.reason
        if attempt < int(decode_retries):
            warnings.warn(f"[repeat] 解码失败 ({res.reason}); 重抓 "
                          f"{attempt + 1}/{int(decode_retries)}", RuntimeWarning)
    raise RuntimeError(f"重复测量解码失败 (重试 {int(decode_retries)} 次后): {last}")


C_REAL = "#3FBFBF"
C_IMAG = "#F2A29E"


def plot_scatter(points: List[dict], out_png: Path, show: bool = False,
                 errorbar: str = "std") -> None:
    if not _MPL_OK:
        raise RuntimeError(f"matplotlib 不可用: {_MPL_ERR!r}")
    P = len(points)
    fig, axes = plt.subplots(1, P, figsize=(3.1 * P, 3.4), squeeze=False)
    for i, pt in enumerate(points):
        ax = axes[0][i]
        ct = np.asarray(pt["c_true_norm"])
        reps = [np.asarray(c) for c in (pt.get("c_hat_reps") or [pt["c_hat"]])]
        R = len(reps)
        ch = np.concatenate(reps)
        ctt = np.tile(ct, R)
        s_plus, s_dot = (30, 16) if R == 1 else (16, 9)
        ax.axhline(0.0, color="0.8", lw=0.8, ls=":", zorder=0)
        ax.axvline(0.0, color="0.8", lw=0.8, ls=":", zorder=0)
        ax.plot([-1, 1], [-1, 1], "k--", lw=1.0, zorder=1)
        ax.scatter(ctt.real, ch.real, marker="+", s=s_plus, linewidths=1.1,
                   color=C_REAL, label="real", zorder=3)
        ax.scatter(ctt.imag, ch.imag, marker="o", s=s_dot, facecolors=C_IMAG,
                   edgecolors="none", alpha=0.85, label="imag", zorder=2)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
        ax.set_aspect("equal")
        m, e, _tag = _bar_value(pt.get("enob_mean", pt.get("enob", float("nan"))),
                                pt.get("enob_sd", float("nan")),
                                int(pt.get("n_reps", 1)), errorbar)
        title = (f"ENOB: {m:.2f} ± {e:.2f} bit" if np.isfinite(e)
                 else f"ENOB: {m:.2f} bit")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("c", fontsize=12)
        if i == 0:
            ax.set_ylabel(r"$\hat{c}$", fontsize=12)
            ax.legend(loc="upper left", frameon=False, fontsize=9,
                      handletextpad=0.2, borderaxespad=0.2)
        ax.text(0.96, 0.05, pt["label"], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=12, fontweight="bold")
        snr = pt.get("snr3_mean", pt.get("snr3_db", float("nan")))
        ax.text(0.04, 0.05,
                f"SNR$_{{3bin}}$ {snr:.1f} dB",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="0.4")
    n_max = max(int(pt.get("n_reps", 1)) for pt in points)
    sup = ("USRP X310 x2 + ZEM-4300  |  N=%d complex inner products (WISE Fig. 3c)"
           % points[0]["N"])
    if n_max > 1:
        tag = "s.d." if errorbar == "std" else "s.e.m."
        sup += f"  |  ENOB: mean ± 1 {tag} (n={n_max})"
    fig.suptitle(sup, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=200)
    print(f"[plot] saved -> {out_png}")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def plot_enob_errorbar(points: List[dict], out_png: Path, show: bool = False,
                       errorbar: str = "std") -> None:
    """ENOB 误差棒图: x=各 target, y=ENOB 的 mean, 棒长=±1 s.d. (或 s.e.m.)."""
    if not _MPL_OK:
        raise RuntimeError(f"matplotlib 不可用: {_MPL_ERR!r}")
    P = len(points)
    xs = np.arange(P, dtype=float)
    ys, es, ns = [], [], []
    for pt in points:
        m, e, _tag = _bar_value(pt.get("enob_mean", pt.get("enob", float("nan"))),
                                pt.get("enob_sd", float("nan")),
                                int(pt.get("n_reps", 1)), errorbar)
        ys.append(m); es.append(e if np.isfinite(e) else 0.0)
        ns.append(int(pt.get("n_reps", 1)))
    fig, ax = plt.subplots(figsize=(max(3.4, 1.35 * P + 1.8), 3.4))
    ax.errorbar(xs, ys, yerr=es, fmt="o", ms=6, color=C_REAL, ecolor="0.25",
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=3)
    for x, y, e in zip(xs, ys, es):
        ax.annotate(f"{y:.2f}", (x, y + e), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([pt["label"] for pt in points])
    ax.set_xlim(-0.6, P - 0.4)
    ax.set_ylabel("ENOB (bit)")
    ax.grid(axis="y", alpha=0.3)
    tag = "s.d." if errorbar == "std" else "s.e.m."
    n_max = max(ns) if ns else 1
    ax.set_title(rf"ENOB = $\log_2(\mathrm{{std}}[y]\,/\,\mathrm{{RMSE}})$,"
                 f"  mean ± 1 {tag} (n={n_max})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[plot] saved -> {out_png}")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def save_npz(points: List[dict], fr: FrameSpec, args, out_npz: Path) -> None:
    P = len(points)
    Q = fr.n_data

    def stack(key, dtype):
        return np.stack([np.asarray(pt[key], dtype=dtype) for pt in points])

    meta = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in vars(args).items()}
    payload = dict(
        c_true_norm=stack("c_true_norm", np.complex128),
        c_hat=stack("c_hat", np.complex128),
        Y_data_raw=stack("Y_data", np.complex128),
        rmse=np.array([pt["rmse"] for pt in points]),
        pilot_rmse=np.array([pt["pilot_rmse"] for pt in points]),
        snr3_db=np.array([pt["snr3_db"] for pt in points]),
        snr1_db=np.array([pt["snr1_db"] for pt in points]),
        snr_02mhz_db=np.array([pt["snr_02mhz_db"] for pt in points]),
        sigma2_bin=np.array([pt["sigma2_bin"] for pt in points]),
        p_ip_raw=np.array([pt["p_ip_raw"] for pt in points]),
        G_hat=np.array([pt["G_hat"] for pt in points], np.complex128),
        drift_rad_per_slot=np.array([pt["drift"] for pt in points]),
        p_rf_dbm_tx=np.array([pt["p_rf_dbm"] for pt in points]),
        p_lo_dbm_tx=np.array([pt["p_lo_dbm"] for pt in points]),
        clip_frac=np.array([pt["clip_frac"] for pt in points]),
        peak_metric=np.array([pt["peak_metric"] for pt in points]),
        mix_corr_lim=np.array([pt["mix_corr_lim"] for pt in points], np.complex128),
        mix_corr_sq=np.array([pt["mix_corr_sq"] for pt in points], np.complex128),
        mixer_comp=np.array([pt["mixer_comp"] for pt in points], bool),
        labels=np.array([pt["label"] for pt in points]),
        targets=np.array([pt["target"] for pt in points]),
        spec_win_db=stack("spec_win_db", np.float64),
        spec_win_bins=np.asarray(points[0]["spec_win_bins"], np.int64),
        tune_history_json=json.dumps([pt["history"] for pt in points]),
        vec_N=fr.N, fft_len=fr.L, cp_len=fr.cp, k0=fr.k0,
        gap0=fr.gap0, gap1=fr.gap1, n_pilot=fr.n_pilot, n_data=fr.n_data,
        frame_len=fr.frame_len, seed=int(args.seed), fs_hz=float(args.fs),
        slot_kinds=np.array([s.kind for s in fr.slots]),
        slot_seed_idx=np.array([s.seed_idx for s in fr.slots], np.int64),
        slot_c_true=np.array([s.c_true for s in fr.slots], np.complex128),
        meta_json=json.dumps(meta, ensure_ascii=False),
    )

    # --- v4: 逐次重复测量数据 (P x R_max, 不足处 NaN 填充) 与 ENOB 统计 ---
    R_max = max(int(pt.get("n_reps", 1)) for pt in points)

    def stack_reps(key):
        A = np.full((P, R_max), np.nan)
        for i, pt in enumerate(points):
            v = np.asarray(pt[key], float)
            A[i, :v.size] = v
        return A

    ch_reps = np.full((P, R_max, Q), np.nan, np.complex128)
    for i, pt in enumerate(points):
        for r, c in enumerate(pt["c_hat_reps"]):
            ch_reps[i, r, :] = np.asarray(c, np.complex128)
    payload.update(
        n_repeats=np.array([int(pt["n_reps"]) for pt in points], np.int64),
        c_hat_reps=ch_reps,
        rmse_reps=stack_reps("rmse_vals"),
        enob_reps=stack_reps("enob_vals"),
        enob_errstd_reps=stack_reps("enob_errstd_vals"),
        snr3_db_reps=stack_reps("snr3_vals"),
        pilot_rmse_reps=stack_reps("pilot_rmse_vals"),
        std_y=np.array([pt["std_y"] for pt in points]),
        enob=np.array([pt["enob"] for pt in points]),
        enob_mean=np.array([pt["enob_mean"] for pt in points]),
        enob_sd=np.array([pt["enob_sd"] for pt in points]),
        rmse_mean=np.array([pt["rmse_mean"] for pt in points]),
        rmse_sd=np.array([pt["rmse_sd"] for pt in points]),
    )
    if fr.dev_lim is not None:
        payload["mixer_dev_lim"] = np.asarray(fr.dev_lim, np.complex128)
        payload["mixer_dev_sq"] = np.asarray(fr.dev_sq, np.complex128)
    if getattr(args, "save_vectors", False):
        va = np.empty((Q, fr.N), np.complex64)
        vb = np.empty((Q, fr.N), np.complex64)
        qi = 0
        for s in fr.slots:
            if s.kind == "D":
                a, b = gen_vec_pair(fr.N, args.seed, s.seed_idx)
                va[qi], vb[qi] = a, b
                qi += 1
        payload["vec_a_data"] = va
        payload["vec_b_data"] = vb
    np.savez_compressed(out_npz, **payload)
    print(f"[npz] saved -> {out_npz}")


def _points_from_npz(npz_path: Path) -> List[dict]:
    """把单个 NPZ (v3 旧档或 v4 新档) 读成统一的 point dict 列表."""
    d = np.load(npz_path, allow_pickle=False)
    P = d["rmse"].shape[0]
    is_v4 = "enob_reps" in d.files
    points = []
    for i in range(P):
        ct = np.asarray(d["c_true_norm"][i])
        if is_v4:
            R = int(d["n_repeats"][i])
            reps = [np.asarray(d["c_hat_reps"][i, r]) for r in range(R)]
            enob_vals = [float(x) for x in d["enob_reps"][i, :R]]
            rmse_vals = [float(x) for x in d["rmse_reps"][i, :R]]
            snr3_vals = [float(x) for x in d["snr3_db_reps"][i, :R]]
            std_y = float(d["std_y"][i])
        else:
            # v3 旧档: 只有单次测量, 由存档 RMSE + 真值 std 现算 ENOB.
            rmse = float(d["rmse"][i])
            std_y = float(np.std(ct))
            reps = [np.asarray(d["c_hat"][i])]
            enob_vals = [math.log2(max(std_y, 1e-300) / max(rmse, 1e-300))]
            rmse_vals = [rmse]
            snr3_vals = [float(d["snr3_db"][i])]
        points.append(dict(
            c_true_norm=ct, c_hat=reps[0], c_hat_reps=reps, n_reps=len(reps),
            enob_vals=enob_vals, rmse_vals=rmse_vals, snr3_vals=snr3_vals,
            std_y=std_y, label=str(d["labels"][i]), N=int(d["vec_N"]),
        ))
    return points


def _finalize_point_stats(pt: dict) -> None:
    em, es, n = _mean_std(pt["enob_vals"])
    rm, rs, _ = _mean_std(pt["rmse_vals"])
    sm, _, _ = _mean_std(pt["snr3_vals"])
    pt.update(enob_mean=em, enob_sd=es, n_reps=n,
              rmse_mean=rm, rmse_sd=rs, snr3_mean=sm,
              enob=float(pt["enob_vals"][0]) if pt["enob_vals"] else float("nan"))


def replot_from_npz(npz_paths: List[Path], out_png: Path, show: bool,
                    errorbar: str = "std") -> None:
    """重画; 传入多个 NPZ 时按'同一组真值 y 的重复实验'合并后画误差棒."""
    base = _points_from_npz(npz_paths[0])
    for path in npz_paths[1:]:
        pts = _points_from_npz(path)
        if len(pts) != len(base):
            raise ValueError(f"{path}: target 数 ({len(pts)}) 与首个文件 "
                             f"({len(base)}) 不一致, 无法合并")
        for i, (b, q) in enumerate(zip(base, pts)):
            if b["c_true_norm"].shape != q["c_true_norm"].shape or \
               not np.allclose(b["c_true_norm"], q["c_true_norm"]):
                raise ValueError(
                    f"{path} 第 {i} 个 panel 的真值 y 与首个文件不同 "
                    f"(合并为重复实验要求相同 --seed/--N/--q-data)")
            if b["label"] != q["label"]:
                warnings.warn(f"{path} 第 {i} 个 panel 标签 {q['label']!r} 与 "
                              f"{b['label']!r} 不同; 仍按位置合并", RuntimeWarning)
            b["c_hat_reps"] += q["c_hat_reps"]
            b["enob_vals"] += q["enob_vals"]
            b["rmse_vals"] += q["rmse_vals"]
            b["snr3_vals"] += q["snr3_vals"]
    for b in base:
        _finalize_point_stats(b)
    plot_scatter(base, out_png, show=show, errorbar=errorbar)
    plot_enob_errorbar(base, out_png.with_name(out_png.stem + "_enob" + out_png.suffix),
                       show=show, errorbar=errorbar)



def make_out_dir(stem: str, N: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = DATA_DIR / f"{Path(stem).stem}_{ts}_N{N}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="复现 WISE Fig. 3c: 复内积散点图 (双 USRP + ZEM-4300); "
                    "指标 ENOB=log2(std[y]/RMSE), 支持 --repeats 重复测量画误差棒",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--N", type=int, default=N_DEFAULT,
                   help="向量长度 (改动时须保证 3N/2 + k0 < fft_len/2)")
    p.add_argument("--fft-len", type=int, default=FFT_LEN_DEFAULT, help="OFDM FFT 长度 L")
    p.add_argument("--cp-len", type=int, default=CP_LEN_DEFAULT, help="循环前缀长度")
    p.add_argument("--k0", type=int, default=K0_DEFAULT,
                   help="IP 落点 bin (须 ≡0 mod 3, !=0)")
    p.add_argument("--q-data", type=int, default=Q_DATA_DEFAULT, help="data 符号数 = 散点数")
    p.add_argument("--pilot-every", type=int, default=PILOT_EVERY_DEFAULT,
                   help="每多少个 data 插 1 个 pilot")
    p.add_argument("--gap", type=int, default=GAP_SAMPLES_DEFAULT, help="帧首尾零 gap 样本数")
    p.add_argument("--seed", type=int, default=VECTOR_SEED_DEFAULT, help="向量种子")
    p.add_argument("--fs", type=float, default=USRP_SAMPLE_RATE, help="采样率 (Hz)")
    p.add_argument("--tune-metric", choices=("snr", "rmse", "enob"), default="snr",
                   help="闭环对准的量: snr=字面 SNR_3bin(dB); rmse=直接对 RMSE 数值 "
                        "(对标论文用 --tune-metric rmse --targets 0.093,0.056); "
                        "enob=直接对 ENOB=log2(std[y]/RMSE) (单位 bit; 本数据 "
                        "std[y]≈1/3, RMSE 0.093/0.056 ≈ ENOB 1.84/2.57 bit)")
    p.add_argument("--targets", type=str, default="15,25",
                   help="逗号分隔的目标列表 (snr 模式单位 dB; rmse 模式为 RMSE 数值; "
                        "enob 模式单位 bit)")
    p.add_argument("--repeats", type=int, default=3,
                   help="每个 target 的独立重复测量次数 R (手稿 ŷ^(1..R); >=2 才有"
                        "误差棒): 闭环收敛后功率固定, 重复抓取+解码, 每次得一个 "
                        "ENOB^(k), 汇总 mean ± s.d.")
    p.add_argument("--errorbar", choices=("std", "sem"), default="std",
                   help="误差棒口径: std=样本标准差 (ddof=1); sem=标准误 s.d./sqrt(n)")
    p.add_argument("--labels", type=str, default=None,
                   help='panel 角标, 逗号分隔, 如 "15 dB,25 dB"; 默认按 target 生成')
    p.add_argument("--p-lo-dbm", type=float, default=P_LO_DBM_DEFAULT,
                   help="TX CH1 (LO) 口功率, 全程固定; LO 线无 pad 时即 mixer LO 口功率")
    p.add_argument("--p-rf-dbm-init", type=float, default=P_RF_DBM_INIT_DEFAULT,
                   help="TX CH0 (RF) 口初始功率, 闭环会自动调")
    p.add_argument("--rf-atten-db", type=float, default=RF_ATTEN_DB_DEFAULT,
                   help="TX CH0 -> mixer RF 口之间外接 pad (dB, 仅记账/打印)")
    p.add_argument("--if-atten-db", type=float, default=IF_ATTEN_DB_DEFAULT,
                   help="mixer IF -> RX 之间外接 pad (dB, 仅记账; 建议拆掉=0)")
    p.add_argument("--max-tune-iters", type=int, default=TUNE_MAX_ITERS_DEFAULT)
    p.add_argument("--tune-tol-db", type=float, default=TUNE_TOL_DB_DEFAULT)
    p.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT)
    p.add_argument("--capture-retries", type=int, default=CAPTURE_RETRIES_DEFAULT,
                   help="单次抓包检测到 RX overflow/丢样时, 静默重抓的最大次数")
    p.add_argument("--decode-retries", type=int, default=DECODE_RETRIES_DEFAULT,
                   help="解码/同步失败时重抓+重解的最大次数 (再不行才放弃该点)")
    p.add_argument("--abort-on-fail", action="store_true",
                   help="某 target 重试仍失败时直接终止; 默认 False=跳过该点并保留已采数据")
    p.add_argument("--skip-tune", action="store_true",
                   help="不闭环, 直接在 --p-rf-dbm-init 测一次 (每个 target 同功率)")
    p.add_argument("--tx-args", type=str, default=TX_ARGS)
    p.add_argument("--rx-args", type=str, default=RX_ARGS)
    p.add_argument("--p-max-dbm", type=float, default=P_MAX_DBM_DEFAULT)
    p.add_argument("--rx-gain", type=float, default=RX_GAIN_DB_DEFAULT,
                   help="RX analog gain (dB); clip 就调小, 噪底淹没信号就调大")
    p.add_argument("--no-ext-ref", action="store_true",
                   help="RX 不锁外参 (强烈不建议: CFO/采样钟漂移会毁掉相位标定)")
    p.add_argument("--dry-run", action="store_true",
                   help="不接硬件, 合成 RX 数据走同一条解码路径 (不需要 gnuradio)")
    p.add_argument("--sim-mixer", choices=("ideal", "limiter"), default="ideal",
                   help="dry-run 的混频器模型")
    p.add_argument("--papr-clip-db", type=float, default=10.0,
                   help="TX 数字削波 PAPR (dB), 0=不削波. N=4096 不削波时 PAPR≈14 dB, "
                        "平均功率上限仅 ≈+2 dBm; 削到 10 dB 换回 ~6 dB 且 EVM 极小")
    p.add_argument("--mixer-comp", action="store_true",
                   help="解码端用导频拟合 limiter/square 两个混频非线性基底并从数据扣除 "
                        "(校准式后处理, 默认关; 指纹 |corr|>0.5 时建议开)")
    p.add_argument("--no-mixer-diag", action="store_true",
                   help="跳过混频非线性诊断基底预计算 (build 可省 ~20 s, 指纹/补偿不可用)")
    p.add_argument("--test-rx", action="store_true",
                   help="起 flowgraph 在 --p-rf-dbm-init/--p-lo-dbm 抓一段: "
                        "存频谱 PNG + 跑前导同步 + 完整解码诊断")
    p.add_argument("--replot", type=str, default=None,
                   help="从已有 NPZ 重画; 逗号分隔多个文件时按重复实验合并画误差棒 "
                        "(要求各次运行的 --seed/--N/--q-data 相同, 即同一组真值 y)")
    p.add_argument("--out-npz", type=str, default=f"{DEFAULT_OUTPUT_STEM}.npz")
    p.add_argument("--out-png", type=str, default=f"{DEFAULT_OUTPUT_STEM}.png")
    p.add_argument("--save-vectors", action="store_true",
                   help="把全部 data 向量存进 NPZ (N=4096 时 ~25 MB; 默认只存种子, 可完整重建)")
    p.add_argument("--show", action="store_true")
    return p


def default_labels(targets: List[float], metric: str) -> List[str]:
    if metric == "snr":
        return [f"{t:g} dB" for t in targets]
    if metric == "enob":
        return [f"ENOB→{t:g}b" for t in targets]
    return [f"RMSE→{t:g}" for t in targets]


def print_link_budget(args) -> None:
    print("[link] 功率记账 (mixer 口 = TX 口 - 外接 pad):")
    print(f"[link]   LO: TX CH1 {args.p_lo_dbm:+.1f} dBm -> mixer LO 口 "
          f"{args.p_lo_dbm:+.1f} dBm  (论文最优区 ≈ -3 dBm, 可 ±3 dB 微扫)")
    print(f"[link]   RF: TX CH0 {args.p_rf_dbm_init:+.1f} dBm - pad {args.rf_atten_db:.0f} dB "
          f"-> mixer RF 口 {args.p_rf_dbm_init - args.rf_atten_db:+.1f} dBm "
          f"(论文 15/25/35 dB SNR ≈ -63/-53/-43 dBm)")
    if args.if_atten_db > 0:
        print(f"[link]   IF 线上仍有 {args.if_atten_db:.0f} dB pad: 本实验 IF 电平很低, "
              f"建议拆掉或加大 --rx-gain.")


def run_dry(args, fr: FrameSpec, targets: List[float], labels: List[str],
            out_dir: Path) -> List[dict]:
    R = max(1, int(getattr(args, "repeats", 1)))
    # 真值 y 的 std (只取 data slot, c/sqrt(N)); 本随机模型下理论值 = 1/3.
    ctn_d = np.array([s.c_true for s in fr.slots if s.kind == "D"],
                     np.complex128) / math.sqrt(fr.N)
    std_y_d = float(np.std(ctn_d))
    points = []
    for t, lab in zip(targets, labels):
        if args.tune_metric == "snr":
            snr_inj = float(t)
        elif args.tune_metric == "enob":
            rmse_t = std_y_d / (2.0 ** float(t))
            snr_inj = 10.0 * math.log10(1.0 / (27.0 * rmse_t ** 2))
            print(f"[sim] ENOB 目标 {t:g} bit (std[y]={std_y_d:.4f} -> RMSE≈"
                  f"{rmse_t:.4f}) -> 注入字面 SNR_3bin={snr_inj:.2f} dB (AWGN 等效)")
        else:
            snr_inj = 10.0 * math.log10(1.0 / (27.0 * float(t) ** 2))
            print(f"[sim] RMSE 目标 {t:g} -> 注入字面 SNR_3bin={snr_inj:.2f} dB (AWGN 等效)")
        caps = [synth_capture(fr, snr_inj, args.sim_mixer, args.seed, args.fs,
                              verbose=(k == 0), rep=k) for k in range(R)]
        res, p_rf, hist = tune_point(
            capture_fn=lambda c=caps[0]: c, set_rf_fn=None, fr=fr, fs=args.fs,
            target=t, metric=args.tune_metric, p_rf_init=args.p_rf_dbm_init,
            max_iters=1, tol_db=args.tune_tol_db,
            mixer_comp=bool(args.mixer_comp))
        reps = [res]
        for k in range(1, R):
            print(f"[sim] {lab}: 独立重复 {k + 1}/{R} (同参数, 新噪声实现)")
            rk = decode_capture(caps[k], fr, args.fs, verbose=False,
                                mixer_comp=bool(args.mixer_comp))
            if rk.ok:
                print(f"[sim]   -> RMSE={rk.rmse:.4f}, ENOB={rk.enob:.2f} bit")
                reps.append(rk)
            else:
                warnings.warn(f"[{lab}] 重复 {k + 1}/{R} 解码失败, 跳过该次: "
                              f"{rk.reason}", RuntimeWarning)
        points.append(_point_dict(reps, fr, t, lab, p_rf, args.p_lo_dbm, hist))
    return points


def _point_dict(res_list, fr: FrameSpec, target, label, p_rf, p_lo, hist) -> dict:
    """把同一 target 下 R 次独立重复的 DecodeResult 聚合成一个 point.

    兼容性: 旧的标量字段 (rmse/snr3_db/c_hat/...) 一律取第 1 次重复 (= 闭环收敛
    时那次), 语义与 v3 的单次测量一致; 新增 *_vals 逐次数组与 ENOB 统计.
    """
    if isinstance(res_list, DecodeResult):
        res_list = [res_list]
    reps = [r for r in res_list if r is not None and r.ok]
    if not reps:
        raise ValueError("_point_dict: 没有任何成功的重复测量")
    r0 = reps[0]
    enob_vals = [float(r.enob) for r in reps]
    enob_errstd_vals = [float(r.enob_errstd) for r in reps]
    rmse_vals = [float(r.rmse) for r in reps]
    snr3_vals = [float(r.snr3_db) for r in reps]
    pilot_rmse_vals = [float(r.pilot_rmse) for r in reps]
    enob_mean, enob_sd, n = _mean_std(enob_vals)
    rmse_mean, rmse_sd, _ = _mean_std(rmse_vals)
    snr3_mean, _, _ = _mean_std(snr3_vals)
    return dict(
        target=float(target), label=str(label), N=fr.N,
        c_true_norm=r0.c_true_norm, c_hat=r0.c_hat, Y_data=r0.Y_data,
        rmse=r0.rmse, pilot_rmse=r0.pilot_rmse,
        std_y=float(r0.std_y), enob=float(r0.enob),
        snr3_db=r0.snr3_db, snr1_db=r0.snr1_db, snr_02mhz_db=r0.snr_02mhz_db,
        sigma2_bin=r0.sigma2_bin, p_ip_raw=r0.p_ip_raw, G_hat=r0.G_hat,
        drift=r0.drift_rad_per_slot, p_rf_dbm=float(p_rf), p_lo_dbm=float(p_lo),
        clip_frac=r0.clip_frac, peak_metric=r0.peak_metric,
        spec_win_db=r0.spec_win_db, spec_win_bins=r0.spec_win_bins,
        mix_corr_lim=complex(r0.mix_corr_lim), mix_corr_sq=complex(r0.mix_corr_sq),
        mixer_comp=bool(r0.mixer_comp),
        history=hist,
        # --- v4: 重复测量 ---
        n_reps=int(n),
        c_hat_reps=[np.asarray(r.c_hat) for r in reps],
        enob_vals=enob_vals, enob_errstd_vals=enob_errstd_vals,
        rmse_vals=rmse_vals, snr3_vals=snr3_vals,
        pilot_rmse_vals=pilot_rmse_vals,
        enob_mean=enob_mean, enob_sd=enob_sd,
        rmse_mean=rmse_mean, rmse_sd=rmse_sd, snr3_mean=snr3_mean,
    )


def run_hardware(args, fr: FrameSpec, targets: List[float], labels: List[str],
                 out_dir: Path, test_rx_only: bool = False) -> List[dict]:
    if not _GR_OK:
        raise RuntimeError(
            f"GNU Radio (gnuradio.gr/uhd/blocks) 不可用: {_GR_ERR!r}\n"
            f"请在装好 GNU Radio + UHD 的机器上运行 (dry-run/replot 不需要).")
    lo_path = out_dir / "tx_lo.cf32"
    rf_path = out_dir / "tx_rf.cf32"
    fr.sA.astype(np.complex64).tofile(lo_path)
    fr.sB.astype(np.complex64).tofile(rf_path)
    print(f"[tx] 波形文件: {lo_path.name}/{rf_path.name} "
          f"({fr.frame_len} 样本/帧, {fr.frame_len/args.fs:.3f} s)")

    clock_source = "internal" if args.no_ext_ref else RX_CLOCK_SOURCE_DEFAULT
    if args.no_ext_ref:
        warnings.warn("--no-ext-ref: 无 10 MHz 外参时 CFO/采样钟漂移会显著抬高 RMSE; "
                      "仅作对照.", RuntimeWarning)
    n_cap = int(round(2.3 * fr.frame_len))
    ring_capacity = max(RX_RING_CAPACITY, 1 << int(math.ceil(math.log2(n_cap + 1))))
    if ring_capacity > RX_RING_CAPACITY:
        print(f"[rx] 环形缓冲自适应扩容: 2^{int(round(math.log2(ring_capacity)))} = "
              f"{ring_capacity} 样本 (~{ring_capacity * 8 / 2**30:.2f} GiB), "
              f"以容纳单次抓取 2.3 帧 = {n_cap} 样本")
    tb = Fig3cFlowgraph(
        fr, str(lo_path), str(rf_path),
        tx_args=args.tx_args, rx_args=args.rx_args, sample_rate=args.fs,
        rx_gain_db=float(args.rx_gain), rx_clock_source=clock_source,
        ring_capacity=ring_capacity,
        p_max_dbm=float(args.p_max_dbm), verbose=True)
    info = tb.set_powers(args.p_rf_dbm_init, args.p_lo_dbm)
    print(f"[pwr] RF: gain={info['rf']['gain_db']:.1f} dB amp={info['rf']['amp_rms']:.4f} "
          f"-> {info['rf']['predicted_dbm']:+.2f} dBm | "
          f"LO: gain={info['lo']['gain_db']:.1f} dB amp={info['lo']['amp_rms']:.4f} "
          f"-> {info['lo']['predicted_dbm']:+.2f} dBm")
    tb.start()
    time.sleep(2.0)
    tb.check_locks()

    grab_timeout = max(30.0, 4.0 * n_cap / args.fs)
    cap_retries = int(getattr(args, "capture_retries", CAPTURE_RETRIES_DEFAULT))

    def capture_fn() -> np.ndarray:
        last = None
        for k in range(cap_retries + 1):
            time.sleep(float(args.settle_s))
            cap, cinfo = tb.grabber.grab_fresh_ex(n_cap, timeout_s=grab_timeout)
            if cap.size < n_cap or cinfo["got"] < n_cap:
                raise RuntimeError(
                    f"抓取超时: 只拿到 {cinfo['got']}/{n_cap} 样本 (RX 流断了?)")
            last = cap
            if cinfo["clean"]:
                return cap
            if k < cap_retries:
                warnings.warn(
                    f"[rx] 检测到 overflow/丢样 (帧不连续), 丢弃本次抓包并重抓 "
                    f"{k+1}/{cap_retries}; 若频繁出现请降采样率或检查 10GbE/主机负载.",
                    RuntimeWarning)
        warnings.warn(
            f"[rx] 重抓 {cap_retries} 次仍检测到 overflow; 用最后一次 (可能同步失败).",
            RuntimeWarning)
        return last

    points: List[dict] = []
    try:
        if test_rx_only:
            _run_test_rx(args, fr, tb, capture_fn, out_dir)
            return []
        cur_p_rf = float(args.p_rf_dbm_init)

        def set_rf(p):
            nonlocal cur_p_rf
            cur_p_rf = float(p)
            tb.set_powers(cur_p_rf, args.p_lo_dbm)

        for t, lab in zip(targets, labels):
            print(f"\n===== target: {lab} ({args.tune_metric}={t}) =====")
            try:
                res, p_rf, hist = tune_point(
                    capture_fn=capture_fn,
                    set_rf_fn=None if args.skip_tune else set_rf,
                    fr=fr, fs=args.fs, target=t, metric=args.tune_metric,
                    p_rf_init=cur_p_rf, max_iters=int(args.max_tune_iters),
                    tol_db=float(args.tune_tol_db),
                    decode_retries=int(args.decode_retries),
                    mixer_comp=bool(args.mixer_comp))
            except RuntimeError as exc:
                if args.abort_on_fail:
                    raise
                warnings.warn(
                    f"[{lab}] 该点失败, 跳过并保留已采数据: {exc}", RuntimeWarning)
                continue
            if res.clip_frac > 1e-4:
                warnings.warn(f"RX clip {res.clip_frac*100:.2f}% — 调小 --rx-gain 后重跑此点",
                              RuntimeWarning)
            # 手稿的 ŷ^(1..R): 第 1 次 = 闭环收敛那次; 之后功率固定不动,
            # 再独立抓取 R-1 次 (每次都是新的噪声/漂移实现).
            reps = [res]
            R = max(1, int(getattr(args, "repeats", 1)))
            for k in range(1, R):
                print(f"[repeat] {lab}: 独立重复测量 {k + 1}/{R} "
                      f"(P_rf 固定 {p_rf:+.2f} dBm)")
                try:
                    rk = capture_and_decode(capture_fn, fr, args.fs,
                                            int(args.decode_retries),
                                            bool(args.mixer_comp), verbose=True)
                except RuntimeError as exc:
                    warnings.warn(f"[{lab}] 重复 {k + 1}/{R} 失败, 跳过该次 "
                                  f"(误差棒 n 相应减少): {exc}", RuntimeWarning)
                    continue
                if rk.clip_frac > 1e-4:
                    warnings.warn(f"RX clip {rk.clip_frac*100:.2f}% (重复 {k + 1})",
                                  RuntimeWarning)
                reps.append(rk)
            points.append(_point_dict(reps, fr, t, lab, p_rf, args.p_lo_dbm, hist))
    finally:
        tb.stop()
        tb.wait()
    return points


def _run_test_rx(args, fr: FrameSpec, tb, capture_fn, out_dir: Path) -> None:
    print("[test-rx] 抓取 2.3 帧, 看频谱 + 前导同步 + 完整解码...")
    cap = capture_fn()
    n_fft = min(1 << 20, cap.size)
    seg = cap[:n_fft].astype(np.complex128)
    win = np.blackman(n_fft)
    spec = np.fft.fftshift(np.fft.fft(seg * win)) / max(np.sum(win), 1e-30)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / args.fs))
    if _MPL_OK:
        figp = out_dir / "test_rx_spectrum.png"
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(freqs / 1e6, 20 * np.log10(np.maximum(np.abs(spec), 1e-12)), lw=0.5)
        ax.set_xlabel("baseband freq (MHz), RX@300 MHz")
        ax.set_ylabel("dBFS-like")
        ax.set_title("RX 频谱 (--test-rx); IP 梳应集中在 ±3.75 MHz 内")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(figp, dpi=160); plt.close(fig)
        print(f"[test-rx] 频谱 -> {figp}")
    peak = float(np.max(np.abs(cap)))
    print(f"[test-rx] 捕获峰值 |IQ|max={peak:.3f} (>{RX_CLIP_THRESH} 为 clip), "
          f"RMS={float(np.sqrt(np.mean(np.abs(cap)**2))):.4f}")
    res = decode_capture(cap, fr, args.fs, verbose=True, mixer_comp=bool(args.mixer_comp))
    if res.ok:
        print(f"[test-rx] 解码 OK: RMSE={res.rmse:.4f} (ENOB={res.enob:.2f} bit), "
              f"SNR_3bin={res.snr3_db:+.2f} dB. 链路就绪, 可以正式跑.")
    else:
        print(f"[test-rx] 解码失败: {res.reason}")


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.replot is not None:
        npz_list = [Path(s.strip()) for s in str(args.replot).split(",") if s.strip()]
        out_png = npz_list[0].with_suffix(".replot.png") \
            if args.out_png == f"{DEFAULT_OUTPUT_STEM}.png" else Path(args.out_png)
        replot_from_npz(npz_list, out_png, show=args.show, errorbar=args.errorbar)
        return 0

    if args.k0 % COMB_M != 0 or args.k0 == 0:
        raise ValueError(f"--k0 必须是 {COMB_M} 的非零倍数 (得到 {args.k0}), "
                         f"否则 IP/互调会落进噪声 bin")
    targets = parse_float_list(args.targets)
    labels = (str(args.labels).split(",") if args.labels
              else default_labels(targets, args.tune_metric))
    labels = [s.strip() for s in labels]
    if len(labels) != len(targets):
        raise ValueError("--labels 数量必须与 --targets 一致")

    out_dir = make_out_dir(args.out_npz, args.N)
    out_npz = out_dir / Path(args.out_npz).name
    out_png = out_dir / Path(args.out_png).name
    print(f"[main] 输出目录: {out_dir}")
    print(f"[main] Δf = {args.fs/args.fft_len:.2f} Hz, 符号 {args.fft_len/args.fs*1e3:.3f} ms, "
          f"IP bin = +{args.k0} ({args.k0*args.fs/args.fft_len/1e3:+.2f} kHz)")
    print_link_budget(args)

    fr = build_frame(args.N, args.fft_len, args.cp_len, args.k0, args.q_data,
                     args.pilot_every, args.gap, args.gap, args.seed,
                     papr_clip_db=float(args.papr_clip_db),
                     mixer_diag=not bool(args.no_mixer_diag))

    if args.dry_run:
        points = run_dry(args, fr, targets, labels, out_dir)
    else:
        points = run_hardware(args, fr, targets, labels, out_dir,
                              test_rx_only=bool(args.test_rx))
        if args.test_rx:
            return 0

    if not points:
        warnings.warn("没有任何成功的数据点 (所有 target 均失败); 不写 NPZ/PNG. "
                      "检查发射/功率/接线, 或加大 --decode-retries/--capture-retries.",
                      RuntimeWarning)
        return 1

    print("\n========== 结果汇总 ==========")
    if len(points) < len(targets):
        print(f"  ⚠ 仅 {len(points)}/{len(targets)} 个 target 成功 (其余被跳过); "
              f"以下为已采点.")
    for pt in points:
        m, e, tag = _bar_value(pt["enob_mean"], pt["enob_sd"], pt["n_reps"],
                               args.errorbar)
        rep_str = ", ".join(f"{v:.2f}" for v in pt["enob_vals"])
        print(f"  [{pt['label']:>10s}] ENOB={_pm(m, e)} bit "
              f"(±1 {tag}, n={pt['n_reps']}: [{rep_str}])  "
              f"RMSE={_pm(pt['rmse_mean'], pt['rmse_sd'], '.4f')}  "
              f"SNR_3bin={pt['snr3_mean']:+.2f} dB")
        print(f"  {'':>12s} P_rf TX口={pt['p_rf_dbm']:+.2f}"
              f"/mixer口≈{pt['p_rf_dbm']-args.rf_atten_db:+.1f} dBm  "
              f"pilot_RMSE={pt['pilot_rmse']:.4f}  "
              f"mix_corr={np.real(pt['mix_corr_lim']):+.2f}  "
              f"std[y]={pt['std_y']:.4f}")
    sy = float(points[0]["std_y"])
    def _e(r):  # 论文 RMSE -> 本定义 ENOB (用实测 std[y])
        return math.log2(max(sy, 1e-300) / r)
    print(f"  论文参考 (RMSE, 括号内换算 ENOB=log2(std[y]/RMSE), std[y]={sy:.3f}): "
          f"Fig3c 无线 0.093({_e(0.093):.2f}b)@'15dB' / 0.056({_e(0.056):.2f}b)@'25dB'; "
          f"有线 basic (FigS7) 0.058({_e(0.058):.2f}b)@15dB / ~0.04({_e(0.04):.2f}b)@25dB "
          f"/ 0.031({_e(0.031):.2f}b)@35dB")

    save_npz(points, fr, args, out_npz)
    plot_scatter(points, out_png, show=args.show, errorbar=args.errorbar)
    plot_enob_errorbar(points,
                       out_png.with_name(out_png.stem + "_enob" + out_png.suffix),
                       show=args.show, errorbar=args.errorbar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
