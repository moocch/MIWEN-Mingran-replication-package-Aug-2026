#!/usr/bin/env python3
"""
2_heatmap_vector_inner_product.py
==================================

USRP X310 + 双 UBX-160 -> ZEM-4300 混频器 -> N9020A,
**长度-N 向量内积** 测试 (1_heatmap_single_scalar.py 的多 tone 推广).

任务背景
--------
1_heatmap_single_scalar.py 测的是 scalar 乘法 (N=1, single tone):
    P_IF_center ∝ |a_0 · b_0|^2  -> 与 P_LO·P_RF 成正比 (理想乘法器).

本脚本测 length-N 向量内积:
    c = <a, b> = Σ_{i=0}^{N-1} a_i* · b_i.

TX 端在两个 port 上分别放 N 个 baseband 复 tone:
    s_LO_BB(t) = Σ_i a_i · exp(j 2π f_i t)        -> 上变频到 0.9 GHz
    s_RF_BB(t) = Σ_i b_i · exp(j 2π f_i t)        -> 上变频到 1.2 GHz
其中 f_i = (i - (N-1)/2) · df_tone, baseband 居中.

ZEM-4300 混频后, IF 端口 spectrum 总共 2N-1 个 tone, 位置:
    f_IF + (j - i) · df_tone,   i, j = 0,...,N-1
其中 f_IF = 300 MHz.

**只有 j==i 那一组 (中心 tone) 是内积**, 其幅度为
    P_center ∝ |Σ_i a_i* · b_i|^2 = |<a, b>|^2.
其他 2N-2 个 off-center tone 是 cross-correlation sidelobe, **不要**.

这与 WISE 论文式 (S57) 是一致的: MVM 结果 y_m = S_y[NM-1-m] 嵌在 S_y 的
中间 M 个 subcarrier; 在 M=1 (standalone IP) 情形下, 中间只剩一个 subcarrier,
即式 (S89) 中的 S_y↓[1] = y_0. 我们这里测的就是 N9020A 上这一个中心点.

读数方式
--------
现有 N9020A 设置在 IF center = 300 MHz, marker 固定在 X = 300 MHz, 不做
peak search -> **read_uv() 读到的恰好就是中心 tone (内积)**, 不受 2N-2 个
sideband 的干扰. 所以扫描循环里调用 sa.read_uv() 即可, 完全沿用单 tone 脚本.

N=1 退化
--------
N=1 时 df_tone 无效, baseband = a_0 (单复数常量), 就是 1_heatmap_single_scalar.py
的复值推广 (a_0 = b_0 = 1 时完全相同). 用 N=1 跑一遍可以 sanity check 整个
pipeline 与单 tone 脚本结果一致.

硬件 (与 1_heatmap_single_scalar.py 完全一致)
---------------------------------------------
TX0 (RF, 1.2 GHz)  -> ZEM-4300 RF port
TX1 (LO, 0.9 GHz)  -> ZEM-4300 LO port
ZEM-4300 IF        -> N9020A (VISA), center = 300 MHz

用法
----
    # sanity check (N=1, 应与 1_heatmap 结果一致)
    python 2_heatmap_vector_inner_product.py --N 1

    # N=2 向量
    python 2_heatmap_vector_inner_product.py --N 2

    # N=4, tone 间距 2 MHz
    python 2_heatmap_vector_inner_product.py --N 4 --df-tone-hz 2e6

    # 重画
    python 2_heatmap_vector_inner_product.py --replot data/xxx.npz

    # 不接硬件 dry-run
    python 2_heatmap_vector_inner_product.py --N 3 --dry-run

每跑一次会保存:
    - 一个独立实验目录 data/<输出名>_YYYYMMDD_HHMMSS/
    - 目录内一个 NPZ (含 vec_a / vec_b / |<a,b>|^2 / 所有原 single-tone 字段)
    - 目录内一个 PNG (左 ideal 右 measured)
"""



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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm


# =============================================================================
# Defaults (与 1_heatmap_single_scalar.py 同步)
# =============================================================================

SCRIPT_DIR = _data_dir(__file__)
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_STEM = "usrp_mixer_vector_heatmap"

# ---- USRP ----
USRP_ARGS = "addr=192.168.30.2,second_addr=192.168.40.2,master_clock_rate=200e6"
USRP_SAMPLE_RATE = 10e6
USRP_LO_OFFSET = 5e6
USRP_ANTENNA = "TX/RX"

FREQ_RF_HZ = 1.20e9
FREQ_LO_HZ = 0.90e9
IF_FREQ_HZ = abs(FREQ_RF_HZ - FREQ_LO_HZ)        # 300 MHz
PORT_R_OHM = 50.0

# UBX 静态功率模型 (RMS-based, 与单 tone 脚本一致)
P_MAX_DBM_DEFAULT = 18.0
GAIN_MIN_DB = 0.0
GAIN_MAX_DB = 31.5
AMP_TARGET = 0.6
AMP_MAX = 0.95
AMP_WARN = 1e-3

# ---- 向量编码默认 ----
N_DEFAULT = 4096
DF_TONE_HZ_DEFAULT = 1.0e6      # 默认 tone 间距 1 MHz, 与 N9020A 100 kHz RBW 留 10x 余量
VECTOR_SEED_DEFAULT = 0         # 固定种子, 整个 sweep 用同一对 (a, b)
MIN_SPB_TARGET = 4000           # 单次 send 的最小样本数 (保证 streamer 稳定)

# ---- N9020A ----
def _try_load_visa_config() -> dict:
    try:
        this_dir = str(_data_dir(__file__))
        cal_dir = os.path.join(this_dir, "mixer_cal_scripts")
        if cal_dir not in sys.path:
            sys.path.insert(0, cal_dir)
        import config as _cfg
        return {
            "sa_resource": getattr(_cfg, "SA_RESOURCE", None),
            "backend": str(getattr(_cfg, "VISA_BACKEND", "")).strip(),
        }
    except Exception:
        return {"sa_resource": None, "backend": ""}


_CFG_VISA = _try_load_visa_config()
SA_VISA_ADDRESS = _CFG_VISA["sa_resource"] or "TCPIP::169.254.190.220::inst0::INSTR"
SA_VISA_BACKEND = (
    os.getenv("PYVISA_BACKEND", "").strip()
    or _CFG_VISA["backend"]
    or ""
)

SA_SPAN_HZ = 20e6
SA_RBW_HZ = 100e3
SA_WINDOW_HZ = 2e6
SA_REF_LEVEL_DBM = +20.0
SA_INPUT_ATTEN_DB = 10
SA_TIMEOUT_MS = 60000
SA_OPEN_TIMEOUT_MS = 3000
SA_OPEN_RETRIES = 4
SA_OPEN_RETRY_SLEEP_S = 0.6

# ---- 扫描默认 (与 1_heatmap 一致) ----
P_LO_DBM_MIN_DEFAULT = -70.0
P_LO_DBM_MAX_DEFAULT = +10.0
N_LO_DEFAULT = 33
P_RF_DBM_MIN_DEFAULT = -70.0
P_RF_DBM_MAX_DEFAULT = +10.0
N_RF_DEFAULT = 33

REPEATS_EACH_DEFAULT = 3
SETTLE_S_DEFAULT = 0.05


# =============================================================================
# 1. 功率反推 (与 1_heatmap_single_scalar.py 同公式, 仅复制以保持自包含)
#
#    注意: 这里的 `amp` 在多 tone 情形下被定义为 baseband 复数样本的 **RMS**,
#    与单 tone 情形下 amp == 常量样本值一致 (因为单 tone 时 RMS = peak = 常量).
#    所以原 power_to_gain_amp 不用改, 但生成 baseband 时要把 RMS 归到 amp.
# =============================================================================

@dataclass
class GainAmp:
    gain_db: float
    amp: float                  # 解释为 baseband RMS
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


# =============================================================================
# 2. 向量 -> baseband 波形 (新的核心代码)
# =============================================================================

def tone_offsets_hz(N: int, df_tone_hz: float) -> np.ndarray:
    """N 个 tone 相对 carrier 的 baseband 频偏: f_i = (i - (N-1)/2) * df_tone."""
    if N <= 1:
        return np.zeros(1, dtype=float)
    return (np.arange(N, dtype=float) - (N - 1) / 2.0) * float(df_tone_hz)


def period_samples_for_vector(N: int, df_tone_hz: float, fs_hz: float) -> int:
    """
    使 baseband 波形在 buffer 长度内严格周期的最小样本数.

    f_i = (i - (N-1)/2) * df_tone
    奇 N: f_i 是 df_tone 的整数倍 -> 周期 T = 1/df_tone
    偶 N: f_i 是 df_tone 的半整数倍 -> 周期 T = 2/df_tone
    """
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
            "df_tone 太大或 fs 太小, baseband 周期不到 1 个样本."
        )
    return period


def choose_spb(period_samples: int, max_spb: int, min_total: int = MIN_SPB_TARGET) -> int:
    """选 spb 为 period_samples 的整数倍, 既 <= max_spb 又 >= min_total."""
    if period_samples <= 1:
        return max(min_total, 1)
    # 至少 1 个周期
    k = max(1, max_spb // period_samples)
    spb = k * period_samples
    # 不够长就再加几个周期 (允许小幅超过 max_spb, streamer 内部会分包)
    while spb < min_total:
        k += 1
        spb = k * period_samples
    return spb


def make_vector_baseband(
    vec: np.ndarray,
    df_tone_hz: float,
    fs_hz: float,
    n_samples: int,
) -> np.ndarray:
    """
    复 baseband 样本:
        s_BB[n] = Σ_i vec[i] · exp(j 2π f_i n/fs)

    输入 vec 不要求归一化; 输出原样反映 vec 的能量分布.
    """
    vec = np.asarray(vec, dtype=np.complex128)
    N = int(vec.shape[0])
    if N == 1:
        # 单 tone 退化: baseband 是常量 vec[0], 与原脚本一致
        return np.full(n_samples, vec[0], dtype=np.complex64)
    f_offsets = tone_offsets_hz(N, df_tone_hz)
    n = np.arange(n_samples, dtype=np.float64)
    # 用相位累加器避免大 n 时浮点累积误差
    phase = (2.0 * np.pi / float(fs_hz)) * np.outer(f_offsets, n)   # [N, n_samples]
    expj = np.exp(1j * phase)                                       # [N, n_samples]
    s_bb = vec @ expj                                               # [n_samples]
    return s_bb.astype(np.complex64)


def rms_normalize_to_amp(
    s_bb: np.ndarray,
    amp_rms_target: float,
    amp_max: float = AMP_MAX,
) -> tuple:
    """
    把 baseband 样本的 RMS 归到 amp_rms_target.
    如果归一化后峰值超过 amp_max, 再 peak-limit 一次, 这时实际 RMS 会下降,
    返回真正的 (RMS, peak), 调用方据此可重算实际输出 dBm.

    返回 (s_scaled[complex64], actual_rms, actual_peak, peak_limited[bool]).
    """
    s = np.asarray(s_bb, dtype=np.complex128)
    cur_rms = float(np.sqrt(np.mean(np.abs(s) ** 2)))
    if cur_rms <= 0.0 or not np.isfinite(cur_rms):
        return np.zeros_like(s, dtype=np.complex64), 0.0, 0.0, False
    s_unit = s / cur_rms                            # ||s_unit||_rms = 1
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


# =============================================================================
# 2b. PAPR reduction (per-tone phase rotation, <a,b> 不变)
#
#  动机
#  ----
#  multi-tone baseband 的 PAPR 随 N 增大 (~ sqrt(2 ln N) RMS):
#      N= 1 : PAPR = 0 dB
#      N= 4 : PAPR ~ 5-7 dB (随 vec)
#      N= 8 : PAPR ~ 7-9 dB
#      N=16 : PAPR ~ 9-11 dB
#  把 baseband RMS 推到 amp_rms_target=0.6 时, 峰值 = 0.6 * 10^(PAPR/20).
#  PAPR=9 dB -> peak ~ 1.69 > AMP_MAX=0.95 -> 被 peak-limit, 实际 RMS 缩水
#  -> 实际输出 dBm 比 target 低, 而且削顶引入谐波/失真 -> MVM 被污染.
#
#  关键性质
#  --------
#  对 a, b 同时施加相同的对角酉变换 D = diag(e^{jφ_i}):
#      a' = a ⊙ e^{jφ},  b' = b ⊙ e^{jφ}
#      <a', b'> = Σ conj(a_i e^{jφ_i}) (b_i e^{jφ_i})
#              = Σ conj(a_i) b_i = <a, b>            ← 内积**精确**不变
#  所以 N9020A 上中心 IF tone 的功率 = |<a,b>|^2 · P_LO · P_RF 完全不受影响,
#  我们却获得 N 个自由相位 φ_i 用来塑形 baseband 波形.
#
#  算法
#  ----
#  SLM (Selected Mapping) + 结构化候选 (Schroeder / Newman):
#      1. 候选集 = {零相位, Schroeder φ_k = -π k(k-1)/N, Newman φ_k = π k^2/N,
#                  + n_trials 组 U(-π, π)^N 随机 φ}
#      2. 对每个 φ, 在 oversampled (default 8x) 时间网格上算 max(PAPR_a, PAPR_b)
#      3. 选 metric 最小者
#  开销: O((n_trials + 3) · N · period · oversample), N=8 K=512 时 ~50 ms.
# =============================================================================

def baseband_papr_db(
    vec: np.ndarray,
    df_tone_hz: float,
    fs_hz: float,
    oversample: int = 8,
) -> float:
    """
    精确估计单个向量的 baseband PAPR (dB), 用 oversample x 过采样
    以避免离散网格漏估真实连续时间峰值.
    PAPR(dB) = 20 log10(peak / RMS). N=1 (常量) -> 0 dB.
    """
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
    """
    寻找 per-tone 相位旋转 φ ∈ R^N, 使得:
        a' = a ⊙ exp(jφ),   b' = b ⊙ exp(jφ)
    两路 baseband 的 PAPR 都尽量小, 同时 <a', b'> = <a, b> 精确不变.

    N <= 2 时 phase rotation 改不了 PAPR (单 tone 是常量;
    双 tone 峰值 = |a_0|+|a_1| 与相位无关), 直接返回原向量.

    返回:
        (a_new, b_new, info: dict)
        info keys: applied, phi, papr_{a,b}_{before,after}_db, n_trials, ...
    """
    a = np.asarray(vec_a, dtype=np.complex128).reshape(-1)
    b = np.asarray(vec_b, dtype=np.complex128).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"vec_a/vec_b shape mismatch: {a.shape} vs {b.shape}")
    N = int(a.shape[0])

    papr_a_before = baseband_papr_db(a, df_tone_hz, fs_hz, oversample)
    papr_b_before = baseband_papr_db(b, df_tone_hz, fs_hz, oversample)

    if N <= 2 or n_trials <= 0:
        reason = ("N<=2 (PAPR 与相位无关)" if N <= 2 else "n_trials<=0")
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

    # 在 oversampled 时间网格上预计算 tone basis [N, n_eval]
    period = period_samples_for_vector(N, df_tone_hz, fs_hz)
    osf = max(int(oversample), 1)
    n_eval = max(int(period) * osf, 64)
    fs_eval = float(fs_hz) * osf
    f_offsets = tone_offsets_hz(N, df_tone_hz)
    n_idx = np.arange(n_eval, dtype=np.float64)
    phase_grid = (2.0 * np.pi / fs_eval) * np.outer(f_offsets, n_idx)
    tone_basis = np.exp(1j * phase_grid).astype(np.complex128)         # [N, n_eval]

    rng = np.random.default_rng(int(seed))

    # 候选 phi 集合: 结构化 + 随机
    candidates = [np.zeros(N, dtype=float)]                            # baseline
    k_arr = np.arange(N, dtype=float)
    # Schroeder phases: 等幅度多 tone 时近最优; 不等幅时仍是个好初值
    candidates.append(-np.pi * k_arr * (k_arr - 1.0) / float(N))
    # Newman phases: 同类的低 PAPR 解析候选
    candidates.append(np.pi * (k_arr ** 2) / float(N))
    # 主体: 随机 SLM
    K = int(n_trials)
    candidates.extend(list(rng.uniform(-np.pi, np.pi, size=(K, N))))

    # 用 batch 矩阵乘加速: stack 所有 candidates, 一次性算 PAPR
    phis = np.stack([np.asarray(p, dtype=np.float64) for p in candidates], axis=0)  # [Ktot, N]
    d_mat = np.exp(1j * phis)                                                       # [Ktot, N]
    # s_a[k, n] = sum_i d_mat[k, i] * a[i] * tone_basis[i, n]
    sa_mat = (d_mat * a[None, :]) @ tone_basis                                      # [Ktot, n_eval]
    sb_mat = (d_mat * b[None, :]) @ tone_basis
    rms_a_arr = np.sqrt(np.mean(np.abs(sa_mat) ** 2, axis=1))                       # [Ktot]
    rms_b_arr = np.sqrt(np.mean(np.abs(sb_mat) ** 2, axis=1))
    peak_a_arr = np.max(np.abs(sa_mat), axis=1)
    peak_b_arr = np.max(np.abs(sb_mat), axis=1)
    papr_a_arr = peak_a_arr / np.maximum(rms_a_arr, 1e-30)
    papr_b_arr = peak_b_arr / np.maximum(rms_b_arr, 1e-30)
    metric_arr = np.maximum(papr_a_arr, papr_b_arr)                                 # both must be low
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
        # Headroom 提升 = 多少 amp_rms 才会让 peak 触及 AMP_MAX
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


# =============================================================================
# 3. 向量生成 (固定 seed; 整个 sweep 用同一对 (a, b))
# =============================================================================

def generate_unit_norm_vectors(N: int, seed: int) -> tuple:
    """
    生成两个独立的随机复 unit-norm 向量 a (LO), b (RF), 长度 N.
    N=1 时退化为 a = b = [1+0j] (实数 1) -- 与原单 tone 脚本完全等价.
    """
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
    """|<a, b>|^2, 用 numpy 算法 (Σ conj(a_i) * b_i)."""
    return float(np.abs(np.vdot(a, b)) ** 2)


# =============================================================================
# 4. USRP 多 tone 双通道 source (取代 USRPDualToneSource)
# =============================================================================

class USRPVectorSource:
    """
    X310 + 双 UBX-160, **每个 channel 发 length-N 多 tone baseband**.

    与 USRPDualToneSource 不同:
        - 内部存的 baseband 波形是 length-spb 的复 array, 而不是常量
        - set_powers() 在 (gain, amp_rms) 计算后, 把整段 baseband (波形模板)
          的 RMS 归到 amp_rms, 同时 peak-limit
        - vec_a, vec_b 在整个生命周期内固定, 切换 power 时只重缩放, 不重生成

    一次 send 的样本数 spb 选成 baseband 周期的整数倍, 保证 buffer 接缝处
    相位连续, 防止 streamer 在拼接时产生 spur.
    """

    def __init__(
        self,
        vec_a: np.ndarray,      # LO channel (CH1), 长度 N
        vec_b: np.ndarray,      # RF channel (CH0), 长度 N
        df_tone_hz: float,
        usrp_args: str = USRP_ARGS,
        sample_rate: float = USRP_SAMPLE_RATE,
        freq_rf: float = FREQ_RF_HZ,
        freq_lo: float = FREQ_LO_HZ,
        lo_offset: float = USRP_LO_OFFSET,
        antenna: str = USRP_ANTENNA,
        p_max_dbm: float = P_MAX_DBM_DEFAULT,
        initial_p_rf_dbm: float = -30.0,
        initial_p_lo_dbm: float = -30.0,
        spb: Optional[int] = None,
        verbose: bool = True,
    ):
        import uhd

        self._verbose = verbose
        self._sr = float(sample_rate)
        self._freq_rf = float(freq_rf)
        self._freq_lo = float(freq_lo)
        self._p_max_dbm = float(p_max_dbm)
        self._df_tone_hz = float(df_tone_hz)

        vec_a = np.asarray(vec_a, dtype=np.complex128).reshape(-1)
        vec_b = np.asarray(vec_b, dtype=np.complex128).reshape(-1)
        if vec_a.shape[0] != vec_b.shape[0]:
            raise ValueError(
                f"vec_a (len={vec_a.shape[0]}) and vec_b (len={vec_b.shape[0]}) must be same length."
            )
        self._N = int(vec_a.shape[0])
        self._vec_a = vec_a     # LO
        self._vec_b = vec_b     # RF

        if verbose:
            print(f"[usrp] connecting: {usrp_args}")
        self._usrp = uhd.usrp.MultiUSRP(usrp_args)

        # CH0 -> RF, CH1 -> LO
        for ch, freq in [(0, freq_rf), (1, freq_lo)]:
            self._usrp.set_tx_rate(sample_rate, ch)
            tune_req = uhd.types.TuneRequest(freq, lo_offset)
            self._usrp.set_tx_freq(tune_req, ch)
            self._usrp.set_tx_antenna(antenna, ch)
            self._usrp.set_tx_gain(15.0, ch)
            if verbose:
                print(
                    f"[usrp] CH{ch}: freq={self._usrp.get_tx_freq(ch)/1e6:9.3f} MHz "
                    f"rate={self._usrp.get_tx_rate(ch)/1e6:6.3f} MS/s "
                    f"ant={self._usrp.get_tx_antenna(ch)}"
                )

        time.sleep(0.5)
        for ch in (0, 1):
            locked = self._usrp.get_tx_sensor("lo_locked", ch).to_bool()
            if not locked:
                raise RuntimeError(f"USRP TX LO not locked on CH{ch}")
            if verbose:
                print(f"[usrp] CH{ch} LO locked")

        # streamer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0, 1]
        self._streamer = self._usrp.get_tx_stream(st_args)
        max_spb = int(self._streamer.get_max_num_samps())

        # 选 spb: baseband 周期的整数倍 (保证接缝处相位连续)
        period = period_samples_for_vector(self._N, self._df_tone_hz, self._sr)
        self._spb = int(spb) if spb is not None else choose_spb(period, max_spb)
        if verbose:
            print(
                f"[usrp] N={self._N}, df_tone={self._df_tone_hz/1e6:.3f} MHz, "
                f"baseband period={period} samples ({period/self._sr*1e6:.3f} us), "
                f"spb={self._spb} = {self._spb//period} periods"
            )

        # 预生成单位 baseband 模板 (RMS=1 的复数样本), 后续 set_powers 只做缩放
        s_a_unit = make_vector_baseband(self._vec_a, self._df_tone_hz, self._sr, self._spb)
        s_b_unit = make_vector_baseband(self._vec_b, self._df_tone_hz, self._sr, self._spb)
        # 模板归一化到 RMS=1; set_powers 时再乘 amp_rms 目标
        s_a_unit, _, _, _ = rms_normalize_to_amp(s_a_unit, amp_rms_target=1.0, amp_max=np.inf)
        s_b_unit, _, _, _ = rms_normalize_to_amp(s_b_unit, amp_rms_target=1.0, amp_max=np.inf)
        # 记录模板的 peak (相对于 RMS=1), 用来事后预测峰值
        self._peak_unit_a = float(np.max(np.abs(s_a_unit)))
        self._peak_unit_b = float(np.max(np.abs(s_b_unit)))
        self._s_a_unit = s_a_unit
        self._s_b_unit = s_b_unit

        if verbose:
            print(
                f"[usrp] unit-RMS template peak: LO={self._peak_unit_a:.3f} "
                f"({20*np.log10(self._peak_unit_a):+.2f} dB PAPR), "
                f"RF={self._peak_unit_b:.3f} "
                f"({20*np.log10(self._peak_unit_b):+.2f} dB PAPR)"
            )

        # 后台 streamer 缓冲 (与单 tone 脚本同结构)
        self._buff = np.zeros((2, self._spb), dtype=np.complex64)
        self._buff_lock = threading.Lock()
        self._uhd = uhd

        # 初始功率
        self.set_powers(initial_p_rf_dbm, initial_p_lo_dbm)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def _stream_loop(self):
        md = self._uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = False
        md.has_time_spec = False
        try:
            while not self._stop_event.is_set():
                with self._buff_lock:
                    buff = self._buff.copy()
                self._streamer.send(buff, md)
                md.start_of_burst = False
        finally:
            md.end_of_burst = True
            try:
                self._streamer.send(np.zeros((2, 0), dtype=np.complex64), md)
            except Exception:
                pass

    def set_powers(self, p_rf_dbm: float, p_lo_dbm: float) -> dict:
        """
        把两个 channel 的目标 port 总功率 (dBm) 设进去 (整段 baseband RMS).
        逻辑:
            1. power_to_gain_amp 反推 (gain, amp_rms_target)
            2. 把 unit-RMS baseband 模板乘以 amp_rms_target
            3. 如果 peak 超 AMP_MAX, peak-limit, 更新 actual RMS / 预测 dBm
            4. set_tx_gain + 写 buffer
        """
        rf = power_to_gain_amp(p_rf_dbm, p_max_dbm=self._p_max_dbm)   # -> CH0
        lo = power_to_gain_amp(p_lo_dbm, p_max_dbm=self._p_max_dbm)   # -> CH1

        # 缩放 baseband 模板
        s_a_scaled, actual_rms_a, peak_a, clip_a = rms_normalize_to_amp(
            self._s_a_unit, amp_rms_target=lo.amp, amp_max=AMP_MAX,
        )
        s_b_scaled, actual_rms_b, peak_b, clip_b = rms_normalize_to_amp(
            self._s_b_unit, amp_rms_target=rf.amp, amp_max=AMP_MAX,
        )

        # 如果 peak-limit 触发, 实际 RMS < amp_rms_target, 实际输出 dBm 也会低
        # 用实际 RMS 重算 predicted_dbm (保持模型 P = p_max + (gain-31.5) + 20log10(amp_rms))
        lo_predicted_actual = (
            self._p_max_dbm + (lo.gain_db - GAIN_MAX_DB) + 20.0 * np.log10(max(actual_rms_a, 1e-30))
        )
        rf_predicted_actual = (
            self._p_max_dbm + (rf.gain_db - GAIN_MAX_DB) + 20.0 * np.log10(max(actual_rms_b, 1e-30))
        )

        if clip_a and self._verbose:
            warnings.warn(
                f"LO peak-limited at amp_rms_target={lo.amp:.4f} "
                f"(N={self._N}, PAPR template={self._peak_unit_a:.2f}); "
                f"actual RMS={actual_rms_a:.4e}, predicted -> {lo_predicted_actual:+.2f} dBm "
                f"(was {lo.predicted_dbm:+.2f} dBm).",
                RuntimeWarning,
            )
        if clip_b and self._verbose:
            warnings.warn(
                f"RF peak-limited at amp_rms_target={rf.amp:.4f} "
                f"(N={self._N}, PAPR template={self._peak_unit_b:.2f}); "
                f"actual RMS={actual_rms_b:.4e}, predicted -> {rf_predicted_actual:+.2f} dBm "
                f"(was {rf.predicted_dbm:+.2f} dBm).",
                RuntimeWarning,
            )

        # 写 USRP analog gain
        self._usrp.set_tx_gain(rf.gain_db, 0)
        self._usrp.set_tx_gain(lo.gain_db, 1)

        # 写 baseband buffer (CH0 = RF, CH1 = LO)
        with self._buff_lock:
            self._buff[0, :] = s_b_scaled
            self._buff[1, :] = s_a_scaled

        return {
            "rf": {
                "target_dbm": float(p_rf_dbm),
                "gain_db": float(rf.gain_db),
                "amp_rms_target": float(rf.amp),
                "amp_rms_actual": float(actual_rms_b),
                "peak_actual": float(peak_b),
                "peak_limited": bool(clip_b),
                "predicted_dbm": float(rf_predicted_actual),
                "actual_gain_db": float(self._usrp.get_tx_gain(0)),
                "actual_freq_hz": float(self._usrp.get_tx_freq(0)),
            },
            "lo": {
                "target_dbm": float(p_lo_dbm),
                "gain_db": float(lo.gain_db),
                "amp_rms_target": float(lo.amp),
                "amp_rms_actual": float(actual_rms_a),
                "peak_actual": float(peak_a),
                "peak_limited": bool(clip_a),
                "predicted_dbm": float(lo_predicted_actual),
                "actual_gain_db": float(self._usrp.get_tx_gain(1)),
                "actual_freq_hz": float(self._usrp.get_tx_freq(1)),
            },
        }

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# =============================================================================
# 5. N9020A reader  (沿用单 tone 脚本)
#
#    关键: marker 固定 X = IF center = 300 MHz, 读 Y. **不做 peak search**.
#    这样不管 N 多大, 读到的恰好就是中心 tone, 即内积 <a, b> 的功率.
# =============================================================================

class N9020AReader:
    def __init__(
        self,
        visa_address: str,
        if_center_hz: float = IF_FREQ_HZ,
        if_window_hz: float = SA_WINDOW_HZ,
        if_span_hz: float = SA_SPAN_HZ,
        if_rbw_hz: float = SA_RBW_HZ,
        ref_level_dbm: float = SA_REF_LEVEL_DBM,
        input_atten_db: float = SA_INPUT_ATTEN_DB,
        timeout_ms: int = SA_TIMEOUT_MS,
        backend: str = "",
        open_timeout_ms: int = SA_OPEN_TIMEOUT_MS,
        open_retries: int = SA_OPEN_RETRIES,
        verbose: bool = True,
    ):
        import pyvisa

        self._verbose = verbose
        self._if_center_hz = float(if_center_hz)
        self._if_window_hz = float(if_window_hz)
        self._rm = None
        self._inst = None

        last_err: Optional[Exception] = None
        for attempt in range(int(open_retries)):
            rm = pyvisa.ResourceManager(backend) if backend else pyvisa.ResourceManager()
            try:
                inst = rm.open_resource(visa_address, open_timeout=int(open_timeout_ms))
                inst.timeout = int(timeout_ms)
                inst.write_termination = "\n"
                inst.read_termination = "\n"
                self._rm = rm
                self._inst = inst
                break
            except Exception as exc:
                last_err = exc
                try:
                    rm.close()
                except Exception:
                    pass
                if attempt < int(open_retries) - 1:
                    time.sleep(SA_OPEN_RETRY_SLEEP_S)
        if self._inst is None:
            raise RuntimeError(
                f"failed to open VISA {visa_address!r} after {open_retries} attempts; last: {last_err!r}"
            )

        idn = self._inst.query("*IDN?").strip()
        if verbose:
            print(f"[sa] backend: {backend or '(pyvisa default)'}")
            print(f"[sa] connected to {visa_address}")
            print(f"[sa] IDN: {idn}")

        self._inst.write("*CLS")
        self._inst.write(":SENS:ROSC:SOUR INT")
        self._inst.write(":TRIG:SOUR IMM")
        self._inst.write(":CONF:SAN")
        self._inst.write(":INIT:CONT OFF")
        self._inst.write(f":FREQ:CENT {if_center_hz}")
        self._inst.write(f":FREQ:SPAN {if_span_hz}")
        self._inst.write(f":BAND:RES {if_rbw_hz}")
        self._inst.write(":BAND:VID:AUTO ON")
        self._inst.write(":DET RMS")
        self._inst.write(":AVER:STAT OFF")
        self._inst.write(f":INP:ATT {input_atten_db}")
        self._inst.write(f":DISP:WIND:TRAC:Y:RLEV {ref_level_dbm}")
        self._inst.write(":UNIT:POW DBM")
        self._inst.write(":CALC:MARK1:STAT ON")
        self._inst.write(":CALC:MARK1:MODE POS")
        self._inst.write(f":CALC:MARK1:X {if_center_hz}")
        if 0 < if_window_hz < if_span_hz:
            left = if_center_hz - if_window_hz / 2.0
            right = if_center_hz + if_window_hz / 2.0
            self._inst.write(f":CALC:MARK:PEAK:SEAR:LIM:LEFT {left}")
            self._inst.write(f":CALC:MARK:PEAK:SEAR:LIM:RIGH {right}")
            self._inst.write(":CALC:MARK:PEAK:SEAR:LIM:STAT ON")
        self._inst.query("*OPC?")
        try:
            err = self._inst.query(":SYST:ERR?").strip()
            if not err.startswith("+0") and not err.startswith("0,"):
                print(f"[sa] WARNING SCPI error queue: {err}")
        except Exception:
            pass

        if verbose:
            print(
                f"[sa] center={if_center_hz/1e6:.1f} MHz  span={if_span_hz/1e6:.1f} MHz  "
                f"rbw={if_rbw_hz/1e3:.0f} kHz"
            )

    def read_uv(self) -> float:
        """读 IF center (固定 X) 的 marker Y, 返回 uV RMS."""
        self._inst.write(":INIT:IMM")
        self._inst.query("*OPC?")
        self._inst.write(f":CALC:MARK1:X {self._if_center_hz}")
        if_dbm = float(self._inst.query(":CALC:MARK1:Y?"))
        if_w = (10.0 ** (if_dbm / 10.0)) * 1e-3
        if_v_rms = float(np.sqrt(if_w * PORT_R_OHM))
        return if_v_rms * 1e6

    def read_dbm(self) -> float:
        self._inst.write(":INIT:IMM")
        self._inst.query("*OPC?")
        self._inst.write(f":CALC:MARK1:X {self._if_center_hz}")
        return float(self._inst.query(":CALC:MARK1:Y?"))

    def close(self):
        try:
            if self._inst is not None:
                self._inst.close()
        except Exception:
            pass
        try:
            if self._rm is not None:
                self._rm.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =============================================================================
# 6. 扫描循环
# =============================================================================

def run_sweep(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    df_tone_hz: float,
    p_lo_dbm_grid: np.ndarray,
    p_rf_dbm_grid: np.ndarray,
    sa_read_uv: Callable[[], float],
    *,
    repeats_each: int = REPEATS_EACH_DEFAULT,
    settle_s: float = SETTLE_S_DEFAULT,
    p_max_dbm: float = P_MAX_DBM_DEFAULT,
    usrp_args: str = USRP_ARGS,
) -> dict:
    """
    在 (P_LO, P_RF) 二维网格上扫一遍.
    每个 cell 用固定的 (vec_a, vec_b) (整个 sweep 不变), 只缩放幅度.
    每个 cell 读 R 次 SA (R 次完全独立 sweep, 同一 vec 同一 power),
    用来估方差.
    """
    n_lo = int(len(p_lo_dbm_grid))
    n_rf = int(len(p_rf_dbm_grid))
    R = int(repeats_each)

    if_amp_uv_all = np.full((n_lo, n_rf, R), np.nan, dtype=float)
    actual_p_lo_dbm = np.zeros((n_lo, n_rf), dtype=float)
    actual_p_rf_dbm = np.zeros((n_lo, n_rf), dtype=float)
    actual_gain_lo = np.zeros((n_lo, n_rf), dtype=float)
    actual_gain_rf = np.zeros((n_lo, n_rf), dtype=float)
    actual_amp_lo = np.zeros((n_lo, n_rf), dtype=float)   # = amp_rms_actual
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

    with USRPVectorSource(
        vec_a=vec_a, vec_b=vec_b, df_tone_hz=df_tone_hz,
        usrp_args=usrp_args, freq_rf=FREQ_RF_HZ, freq_lo=FREQ_LO_HZ,
        p_max_dbm=p_max_dbm, verbose=False,
    ) as src:
        total = n_lo * n_rf
        cell = 0
        try:
            for i in range(n_lo):
                p_lo = float(p_lo_dbm_grid[i])
                for j in range(n_rf):
                    cell += 1
                    p_rf = float(p_rf_dbm_grid[j])

                    info = src.set_powers(p_rf_dbm=p_rf, p_lo_dbm=p_lo)
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
                            print(f"[sweep] SA read failed at (i={i},j={j},r={r}): {exc!r}")

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
                        f"IF µV: mean={mu:8.3f} std={sd:6.3f}  "
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
        "actual_amp_lo": actual_amp_lo,         # RMS actual (after peak-limit)
        "actual_amp_rf": actual_amp_rf,
        "peak_limited_lo": peak_limited_lo,
        "peak_limited_rf": peak_limited_rf,
        "if_amp_uv_all": if_amp_uv_all,         # 中心 tone only (内积)
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
    }


# =============================================================================
# 7. Heatmap 计算 + 画图
# =============================================================================

def measured_power_w(if_amp_uv_mean: np.ndarray) -> np.ndarray:
    v_rms = np.asarray(if_amp_uv_mean, dtype=float) * 1e-6
    return (v_rms ** 2) / PORT_R_OHM


def ideal_inner_product_power_w(
    p_lo_dbm: np.ndarray,
    p_rf_dbm: np.ndarray,
    ip_mag_sq: float,
) -> np.ndarray:
    """
    理想 (无损完美乘法器) 的内积中心 tone 功率:
        P_center[W] = P_LO[W] * P_RF[W] * |<a, b>|^2 / 1mW
        P_center[dBm] = P_LO[dBm] + P_RF[dBm] + 10*log10(|<a,b>|^2)

    推导:
        - baseband LO RMS = sqrt(P_LO_W * R) / scale, 但模型里
          (gain, amp_rms) -> P_LO 已经标定好了
        - 中心 tone 的复幅度 ∝ (amp_LO_rms) * (amp_RF_rms) * <a, b>
        - 中心 tone 功率 ∝ amp_LO_rms^2 * amp_RF_rms^2 * |<a, b>|^2
                       ∝ P_LO * P_RF * |<a, b>|^2

        归一化常数对应 (P_LO, P_RF) = (0, 0) dBm, |<a, b>| = 1 时 P_center = 1mW
        (与 single-tone ideal 公式一致), 所以
            P_center = (P_LO_W * P_RF_W * |<a,b>|^2) / 1 mW
    """
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
        f"Measured:  N9020A IF center @ {data['if_freq_hz']/1e6:.0f} MHz  ({norm_tag})"
    )
    cb2 = fig.colorbar(im2, ax=axes[1])
    cb2.set_label("Power (W) — inner-product center tone")

    df_tag = f", df_tone={df_tone_hz/1e6:.2f} MHz" if N > 1 else ""
    fig.suptitle(
        f"USRP X310 + UBX-160  ->  ZEM-4300  ->  N9020A    "
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


# =============================================================================
# 8. NPZ 持久化  (与 1_heatmap 同结构, 额外保存 vec_a, vec_b, N, df_tone)
# =============================================================================

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
    """
    Create one directory per experiment.

    Relative hints live under this script's data/ directory. The directory name
    is ordered as stem_timestamp_Nx, while files inside keep stable names.
    """
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


# =============================================================================
# 9. Dry-run 合成数据
# =============================================================================

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
    """合成数据 (与 1_heatmap 的 synthesize_demo_data 同模型, 额外乘 |<a,b>|^2)."""
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
    }


# =============================================================================
# 10. CLI / main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="USRP X310 + UBX-160 -> ZEM-4300 -> N9020A, vector inner product heatmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Vector encoding (新参数) ---
    p.add_argument("--N", type=int, default=N_DEFAULT,
                   help="向量长度 N. N=1 退化为单 tone (与 1_heatmap_single_scalar.py 等价).")
    p.add_argument("--df-tone-hz", type=float, default=DF_TONE_HZ_DEFAULT,
                   help="相邻 tone 间隔 (Hz). 应 >> SA RBW (默认 100 kHz).")
    p.add_argument("--vector-seed", type=int, default=VECTOR_SEED_DEFAULT,
                   help="随机 seed 决定 vec_a, vec_b. 同一 sweep 内固定不变.")

    # --- PAPR reduction (per-tone phase rotation, <a,b> 不变) ---
    p.add_argument("--papr-trials", type=int, default=512,
                   help="SLM 随机相位试探次数. 设 0 关闭 PAPR reduction. "
                        "另外固定加 3 个结构化候选 (zero, Schroeder, Newman).")
    p.add_argument("--papr-oversample", type=int, default=8,
                   help="估计连续时间峰值用的过采样倍数 (越大越接近真实 peak).")
    p.add_argument("--papr-seed", type=int, default=12345,
                   help="SLM 随机种子. 与 --vector-seed 解耦, 改 seed 可以换不同 SLM 解.")

    # --- I/O ---
    p.add_argument("--out-npz", type=str, default=f"{DEFAULT_OUTPUT_STEM}.npz",
                   help="输出 NPZ 文件名/路径. 每次实验会基于该名字新建一个带时间戳的目录.")
    p.add_argument("--out-png", type=str, default=f"{DEFAULT_OUTPUT_STEM}.png",
                   help="输出 PNG 文件名. 会保存到本次实验目录内.")

    # --- Sweep grid (与单 tone 脚本完全一致) ---
    p.add_argument("--p-lo-dbm-min", type=float, default=P_LO_DBM_MIN_DEFAULT)
    p.add_argument("--p-lo-dbm-max", type=float, default=P_LO_DBM_MAX_DEFAULT)
    p.add_argument("--n-lo", type=int, default=N_LO_DEFAULT)
    p.add_argument("--p-rf-dbm-min", type=float, default=P_RF_DBM_MIN_DEFAULT)
    p.add_argument("--p-rf-dbm-max", type=float, default=P_RF_DBM_MAX_DEFAULT)
    p.add_argument("--n-rf", type=int, default=N_RF_DEFAULT)
    p.add_argument("--repeats-each", type=int, default=REPEATS_EACH_DEFAULT)
    p.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT)

    # --- Hardware ---
    p.add_argument("--usrp-args", type=str, default=USRP_ARGS)
    p.add_argument("--p-max-dbm", type=float, default=P_MAX_DBM_DEFAULT)
    p.add_argument("--sa-visa-address", type=str, default=SA_VISA_ADDRESS)
    p.add_argument("--sa-backend", type=str, default=SA_VISA_BACKEND)
    p.add_argument("--sa-span-hz", type=float, default=SA_SPAN_HZ)
    p.add_argument("--sa-rbw-hz", type=float, default=SA_RBW_HZ)
    p.add_argument("--sa-window-hz", type=float, default=SA_WINDOW_HZ)
    p.add_argument("--sa-ref-level-dbm", type=float, default=SA_REF_LEVEL_DBM)
    p.add_argument("--sa-input-atten-db", type=float, default=SA_INPUT_ATTEN_DB)

    # --- Plot ---
    p.add_argument("--norm", choices=("log", "gamma", "linear"), default="log")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--show", action="store_true")

    # --- Modes ---
    p.add_argument("--dry-run", action="store_true",
                   help="不连硬件, 合成数据画图.")
    p.add_argument("--replot", type=str, default=None,
                   help="从已有 NPZ 重画.")
    p.add_argument("--test-sa", action="store_true",
                   help="只连 N9020A 测一次.")

    return p


def _open_sa_reader(args) -> N9020AReader:
    return N9020AReader(
        visa_address=args.sa_visa_address,
        if_center_hz=IF_FREQ_HZ,
        if_window_hz=float(args.sa_window_hz),
        if_span_hz=float(args.sa_span_hz),
        if_rbw_hz=float(args.sa_rbw_hz),
        ref_level_dbm=float(args.sa_ref_level_dbm),
        input_atten_db=float(args.sa_input_atten_db),
        timeout_ms=SA_TIMEOUT_MS,
        backend=str(args.sa_backend),
        verbose=True,
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_timestamp = make_run_timestamp()

    if args.N < 1:
        raise ValueError(f"--N must be >= 1, got {args.N}")

    # 给 NPZ 文件名带上 N tag, 防互覆盖
    n_tag = f"_N{args.N}"
    if args.replot is None:
        args.out_npz = args.out_npz.replace(".npz", f"{n_tag}.npz") if ".npz" in args.out_npz else f"{args.out_npz}{n_tag}.npz"
        args.out_png = args.out_png.replace(".png", f"{n_tag}.png") if ".png" in args.out_png else f"{args.out_png}{n_tag}.png"

    # ---- replot ----
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

    # ---- test-sa ----
    if args.test_sa:
        print("[main] TEST-SA: 只连 N9020A 跑一次单点测量")
        with _open_sa_reader(args) as sa:
            for k in range(3):
                try:
                    dbm = sa.read_dbm()
                    uv = sa.read_uv()
                    print(f"[test-sa] read #{k}: IF@{IF_FREQ_HZ/1e6:.0f}MHz "
                          f"= {dbm:+7.2f} dBm = {uv:.3f} uV RMS")
                except Exception as exc:
                    print(f"[test-sa] read #{k} FAILED: {exc!r}")
        return 0

    # ---- 生成向量 ----
    vec_a, vec_b = generate_unit_norm_vectors(args.N, args.vector_seed)
    ip2 = ip_magnitude_squared(vec_a, vec_b)
    print(f"[main] N={args.N}, df_tone={args.df_tone_hz/1e6:.3f} MHz, "
          f"vector_seed={args.vector_seed}")
    print(f"[main] vec_a (LO, raw) = {np.array2string(vec_a, precision=4, suppress_small=True)}")
    print(f"[main] vec_b (RF, raw) = {np.array2string(vec_b, precision=4, suppress_small=True)}")
    print(f"[main] |<a,b>|^2 = {ip2:.6e}  ({10*np.log10(max(ip2,1e-30)):+.2f} dB rel. perfect single tone)")

    # ---- PAPR reduction (per-tone phase rotation, <a,b> 精确不变) ----
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

    # ---- 准备 sweep grid ----
    p_lo = np.linspace(args.p_lo_dbm_min, args.p_lo_dbm_max, args.n_lo)
    p_rf = np.linspace(args.p_rf_dbm_min, args.p_rf_dbm_max, args.n_rf)
    out_dir = resolve_output_dir(args.out_npz, timestamp=run_timestamp, n=args.N)
    out_npz = resolve_output_file(out_dir, args.out_npz, default_suffix=".npz")
    out_png = resolve_output_file(out_dir, args.out_png, default_suffix=".png")
    print(f"[main] output DIR: {out_dir}")
    print(f"[main] output NPZ: {out_npz}")
    print(f"[main] output PNG: {out_png}")

    # ---- dry-run ----
    if args.dry_run:
        print("[main] DRY RUN: 不连硬件, 合成数据.")
        data = synthesize_demo_data(vec_a, vec_b, args.df_tone_hz, p_lo, p_rf)
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
        save_npz(data, out_npz)
        plot_heatmaps(data, out_png, norm=args.norm, gamma=args.gamma, show=args.show)
        return 0

    # ---- live sweep ----
    print("[main] connecting N9020A ...")
    sa = _open_sa_reader(args)
    try:
        data = run_sweep(
            vec_a=vec_a, vec_b=vec_b, df_tone_hz=args.df_tone_hz,
            p_lo_dbm_grid=p_lo, p_rf_dbm_grid=p_rf,
            sa_read_uv=sa.read_uv,
            repeats_each=args.repeats_each, settle_s=args.settle_s,
            p_max_dbm=args.p_max_dbm, usrp_args=args.usrp_args,
        )
    except Exception:
        traceback.print_exc()
        sa.close()
        return 1
    sa.close()

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
    save_npz(data, out_npz)
    plot_heatmaps(data, out_png, norm=args.norm, gamma=args.gamma, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
