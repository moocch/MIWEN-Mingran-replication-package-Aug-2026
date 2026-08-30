#!/usr/bin/env python3
"""
usrp_mixer_heatmap.py
=====================

USRP X310 + 双 UBX-160 驱动 ZEM-4300 混频器, N9020A 经 VISA 读 IF。
单脚本完成: 扫描 (P_LO, P_RF) 网格 -> 存 NPZ -> 出两张并排 heatmap:

    左:  Python ideal   P_ideal[dBm] = P_LO[dBm] + P_RF[dBm]
                        (理想乘法器, V_ref = sqrt(R · 1mW), 单位真 W)
    右:  Measured       N9020A IF peak (W)

硬件连接 (与 dual_tone.py 一致)
------------------------------
TX0 (RF A)  -> ZEM-4300 RF 口  @ 1.2 GHz
TX1 (RF B)  -> ZEM-4300 LO 口  @ 0.9 GHz
ZEM-4300 IF -> N9020A          (经 VISA 控制)

依赖
----
* uhd, numpy, matplotlib, pyvisa
* 注: uhd / pyvisa 是惰性 import, --dry-run / --replot / --test-sa 各自
  按需导入, 不全装也能跑部分功能。
* N9020A: 默认从同目录 mixer_cal_scripts/config.py 读 SA_RESOURCE +
  VISA_BACKEND (跟 quick_check_mxa_awg.py 一致); 也可以用命令行
  --sa-visa-address / --sa-backend 覆盖。

用法
----
    # 0. 先确认 N9020A 能连上 (读 IDN + 跑 3 次单点)
    python usrp_mixer_heatmap.py --test-sa

    # 1. 不接硬件, 造合成数据画图, 验证脚本逻辑
    python usrp_mixer_heatmap.py --dry-run

    # 2. 全流程: 扫描 + 存 NPZ + 出 PNG (默认 13x18 格点)
    python usrp_mixer_heatmap.py

    # 3. 自定义参数
    python usrp_mixer_heatmap.py \\
        --sa-visa-address "TCPIP0::192.168.x.x::inst0::INSTR" \\
        --p-lo-dbm-min -50 --p-lo-dbm-max +10 --n-lo 13 \\
        --p-rf-dbm-min -70 --p-rf-dbm-max +15 --n-rf 18 \\
        --repeats-each 3 --settle-s 0.05 \\
        --p-max-dbm 19.0 \\
        --out-npz today.npz --out-png today.png

    # 4. 已有 NPZ 重画
    python usrp_mixer_heatmap.py --replot today.npz --out-png today_v2.png

功率模型 (USRP 端, ~±2 dB 精度)
------------------------------
UBX-160 datasheet 在 < 3.5 GHz 频段 max-gain + 满刻度 ≥ +18 dBm。 因此

    P_out[dBm] ≈ P_MAX_DBM + (gain - 31.5) + 20·log10(amp)

策略: 优先把 amp 钉在 0.6, 用 gain 粗调; gain 触到 [0, 31.5] 边界时用
amp 细调。如果第一次跑完发现绝对值偏一致, 改 --p-max-dbm 即可。
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
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm


# =============================================================================
# Defaults
# =============================================================================

# ---- USRP ----
USRP_ARGS = "addr=192.168.30.2,second_addr=192.168.40.2,master_clock_rate=200e6"
USRP_SAMPLE_RATE = 10e6
USRP_LO_OFFSET = 5e6
USRP_ANTENNA = "TX/RX"

FREQ_RF_HZ = 1.20e9
FREQ_LO_HZ = 0.90e9
IF_FREQ_HZ = abs(FREQ_RF_HZ - FREQ_LO_HZ)   # 300 MHz
PORT_R_OHM = 50.0

# UBX 静态功率模型常量
P_MAX_DBM_DEFAULT = 18.0    # < 3.5 GHz 频段在 max-gain + 满刻度的输出 (datasheet)
GAIN_MIN_DB = 0.0
GAIN_MAX_DB = 31.5
AMP_TARGET = 0.6            # gain 有余量时 amp 钉在这里
AMP_MAX = 0.95              # 单点上限, 避免 DAC 饱和
AMP_WARN = 1e-3             # 低于此值给警告

# ---- N9020A VISA 配置 ----
def _try_load_visa_config() -> dict:
    """从 mixer_cal_scripts/config.py 读 SA_RESOURCE / VISA_BACKEND, 失败返回空。"""
    try:
        this_dir = str(_data_dir(__file__))
        cal_dir = os.path.join(this_dir, "mixer_cal_scripts")
        if cal_dir not in sys.path:
            sys.path.insert(0, cal_dir)
        import config as _cfg   # noqa: WPS433
        return {
            "sa_resource": getattr(_cfg, "SA_RESOURCE", None),
            "backend": str(getattr(_cfg, "VISA_BACKEND", "")).strip(),
        }
    except Exception:
        return {"sa_resource": None, "backend": ""}


_CFG_VISA = _try_load_visa_config()

# 优先级: 命令行 > config.py > 占位字符串
SA_VISA_ADDRESS = _CFG_VISA["sa_resource"] or "TCPIP::169.254.190.220::inst0::INSTR"

# 优先级: PYVISA_BACKEND env > config.py > pyvisa 默认 (空串)
SA_VISA_BACKEND = (
    os.getenv("PYVISA_BACKEND", "").strip()
    or _CFG_VISA["backend"]
    or ""
)

# N9020A SA mode 配置
SA_SPAN_HZ = 20e6
SA_RBW_HZ = 100e3
SA_WINDOW_HZ = 2e6           # ±1 MHz marker peak search window
SA_REF_LEVEL_DBM = +20.0     # 顶端 +15 dBm RF 经 ZEM 仍不会 overload
SA_INPUT_ATTEN_DB = 10       # 跟你 SA_TONE/SA_IP3 模板一致
SA_TIMEOUT_MS = 60000
SA_OPEN_TIMEOUT_MS = 3000
SA_OPEN_RETRIES = 4
SA_OPEN_RETRY_SLEEP_S = 0.6

# ---- 默认扫描参数 ----
P_LO_DBM_MIN_DEFAULT = -50.0
P_LO_DBM_MAX_DEFAULT = +10.0
N_LO_DEFAULT = 13            # 5 dB 步进
P_RF_DBM_MIN_DEFAULT = -70.0
P_RF_DBM_MAX_DEFAULT = +15.0
N_RF_DEFAULT = 18            # 5 dB 步进

REPEATS_EACH_DEFAULT = 3
SETTLE_S_DEFAULT = 0.05


# =============================================================================
# 1. USRP 功率反推 (dBm -> gain + digital amp)
# =============================================================================

@dataclass
class GainAmp:
    """(gain_dB, amp) 对, 对应一个目标 dBm。"""
    gain_db: float
    amp: float
    predicted_dbm: float    # 模型预测的实际输出 dBm (sanity check)


def power_to_gain_amp(
    target_dbm: float,
    p_max_dbm: float = P_MAX_DBM_DEFAULT,
    amp_target: float = AMP_TARGET,
) -> GainAmp:
    """
    把目标 dBm 反推成 (gain, amp)。

    模型: P_out = p_max + (gain - GAIN_MAX_DB) + 20*log10(amp)

    策略:
        1. 先令 amp = amp_target, 由模型反推 gain
        2. gain 触顶 -> 把 amp 往上推
        3. gain 触底 -> 把 amp 往下推
    """
    gain = GAIN_MAX_DB + target_dbm - p_max_dbm - 20.0 * np.log10(amp_target)
    amp = amp_target

    if gain > GAIN_MAX_DB:
        gain = GAIN_MAX_DB
        amp = 10.0 ** ((target_dbm - p_max_dbm) / 20.0)
        if amp > AMP_MAX:
            warnings.warn(
                f"target {target_dbm:+.2f} dBm exceeds device max "
                f"(~+{p_max_dbm:.1f} dBm at gain={GAIN_MAX_DB} dB, "
                f"amp={AMP_MAX:.2f}); clipping amp to {AMP_MAX}.",
                RuntimeWarning,
            )
            amp = AMP_MAX
    elif gain < GAIN_MIN_DB:
        gain = GAIN_MIN_DB
        amp = 10.0 ** ((target_dbm - (p_max_dbm - GAIN_MAX_DB)) / 20.0)
        if amp < AMP_WARN:
            warnings.warn(
                f"target {target_dbm:+.2f} dBm requires amp={amp:.2e} "
                f"(< {AMP_WARN:.0e}); signal may be below DAC SFDR / "
                f"LO leakage floor — verify with N9020A.",
                RuntimeWarning,
            )

    predicted = p_max_dbm + (gain - GAIN_MAX_DB) + 20.0 * np.log10(amp)
    return GainAmp(gain_db=gain, amp=float(amp), predicted_dbm=float(predicted))


# =============================================================================
# 2. USRP 双通道 source
# =============================================================================

class USRPDualToneSource:
    """
    封装 X310 + 双 UBX-160 的双通道 CW source。

    对外暴露 set_powers(p_rf_dbm, p_lo_dbm), 内部:
        - 用 power_to_gain_amp() 反推 (gain, amp)
        - set_tx_gain() 设模拟 gain
        - 后台 daemon 线程持续 streamer.send() 基带样本 (DC + LO offset)

    用法:
        with USRPDualToneSource() as src:
            src.set_powers(p_rf_dbm=+10, p_lo_dbm=0)
            time.sleep(0.05)         # 让模拟链路 settle
            ... read SA ...
    """

    def __init__(
        self,
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
        # uhd 是惰性 import, 让 --dry-run / --replot 不需要它
        import uhd   # noqa: WPS433

        self._verbose = verbose
        self._sr = sample_rate
        self._freq_rf = freq_rf
        self._freq_lo = freq_lo
        self._p_max_dbm = p_max_dbm

        # ---- 1. 连接 ----
        if verbose:
            print(f"[usrp] connecting: {usrp_args}")
        self._usrp = uhd.usrp.MultiUSRP(usrp_args)

        # ---- 2. 配置两个通道 ----
        # CH0 -> RF (1.2 GHz), CH1 -> LO (0.9 GHz)
        for ch, freq in [(0, freq_rf), (1, freq_lo)]:
            self._usrp.set_tx_rate(sample_rate, ch)
            tune_req = uhd.types.TuneRequest(freq, lo_offset)
            self._usrp.set_tx_freq(tune_req, ch)
            self._usrp.set_tx_antenna(antenna, ch)
            self._usrp.set_tx_gain(15.0, ch)   # 初始 gain 给个中间值
            if verbose:
                print(
                    f"[usrp] CH{ch}: freq={self._usrp.get_tx_freq(ch)/1e6:9.3f} MHz "
                    f"rate={self._usrp.get_tx_rate(ch)/1e6:6.3f} MS/s "
                    f"ant={self._usrp.get_tx_antenna(ch)}"
                )

        # 等 LO 锁定
        time.sleep(0.5)
        for ch in (0, 1):
            locked = self._usrp.get_tx_sensor("lo_locked", ch).to_bool()
            if not locked:
                raise RuntimeError(f"USRP TX LO not locked on CH{ch}")
            if verbose:
                print(f"[usrp] CH{ch} LO locked")

        # ---- 3. streamer ----
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0, 1]
        self._streamer = self._usrp.get_tx_stream(st_args)
        self._spb = int(spb) if spb is not None else int(self._streamer.get_max_num_samps())

        # ---- 4. 缓冲区 (后台线程读, 主线程写) ----
        # LO offset, 基带就是 DC: complex64 实部=amp, 虚部=0
        self._buff = np.zeros((2, self._spb), dtype=np.complex64)
        self._buff_lock = threading.Lock()
        # 把 uhd module 暂存,后台线程要构造 metadata 用
        self._uhd = uhd

        # ---- 5. 初始功率 ----
        self.set_powers(initial_p_rf_dbm, initial_p_lo_dbm)

        # ---- 6. 启动后台流送 ----
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
            # 收尾: 发个 end-of-burst
            md.end_of_burst = True
            try:
                self._streamer.send(
                    np.zeros((2, 0), dtype=np.complex64), md,
                )
            except Exception:
                pass

    def set_powers(self, p_rf_dbm: float, p_lo_dbm: float) -> dict:
        """
        把两个通道的目标功率(dBm)一次性设进去。立即返回,
        调用方应自行 sleep 等模拟链路 settle 后再读 SA。
        """
        rf = power_to_gain_amp(p_rf_dbm, p_max_dbm=self._p_max_dbm)
        lo = power_to_gain_amp(p_lo_dbm, p_max_dbm=self._p_max_dbm)

        self._usrp.set_tx_gain(rf.gain_db, 0)
        self._usrp.set_tx_gain(lo.gain_db, 1)

        with self._buff_lock:
            self._buff[0, :] = np.complex64(rf.amp + 0j)
            self._buff[1, :] = np.complex64(lo.amp + 0j)

        if self._verbose:
            print(
                f"[usrp] RF target={p_rf_dbm:+7.2f} dBm -> "
                f"gain={rf.gain_db:5.2f} dB amp={rf.amp:7.5f} "
                f"(predicted {rf.predicted_dbm:+6.2f} dBm) | "
                f"LO target={p_lo_dbm:+7.2f} dBm -> "
                f"gain={lo.gain_db:5.2f} dB amp={lo.amp:7.5f} "
                f"(predicted {lo.predicted_dbm:+6.2f} dBm)"
            )

        return {
            "rf": {
                "target_dbm": float(p_rf_dbm),
                "gain_db": rf.gain_db,
                "amp": rf.amp,
                "predicted_dbm": rf.predicted_dbm,
                "actual_gain_db": float(self._usrp.get_tx_gain(0)),
                "actual_freq_hz": float(self._usrp.get_tx_freq(0)),
            },
            "lo": {
                "target_dbm": float(p_lo_dbm),
                "gain_db": lo.gain_db,
                "amp": lo.amp,
                "predicted_dbm": lo.predicted_dbm,
                "actual_gain_db": float(self._usrp.get_tx_gain(1)),
                "actual_freq_hz": float(self._usrp.get_tx_freq(1)),
            },
        }

    def stop(self):
        """停止后台流送并清理。"""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# =============================================================================
# 3. N9020A reader (纯 pyvisa)
# =============================================================================

class N9020AReader:
    """
    Keysight N9020A (MXA) 的 IF 固定频点读取器,经 VISA 直连。

    内部约定:
        * SA mode, single sweep
        * 内参考 + 内触发 (跟 quick_check_mxa_awg.py 一致)
        * Center = if_center_hz, Span = if_span_hz, RBW = if_rbw_hz
        * Detector = RMS, Average OFF
        * Marker 钉在 if_center_hz; 每次 read 前重写 :CALC:MARK1:X 防止被移动
        * 读出来的 dBm 自动换成 µV RMS (50 ohm)

    注: 单 tone CW 场景下信号线宽 (USRP + UBX 相噪 + N9020A 本振相噪) 远小于
    if_rbw_hz (默认 100 kHz), 固定频点读数 ≈ peak 读数, 但不会被 IF 频带内
    的杂散 / 噪声 spurs 误导. 测调制信号请改用 channel power 模式.
    """

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
        # pyvisa 是惰性 import, 让 --dry-run / --replot 不需要它
        import pyvisa   # noqa: WPS433

        self._verbose = verbose
        self._if_center_hz = float(if_center_hz)
        self._if_window_hz = float(if_window_hz)
        self._rm = None
        self._inst = None

        # ---- 打开 VISA 资源 (带重试, 跟 quick_check_mxa_awg.py 一致) ----
        last_err: Optional[Exception] = None
        for attempt in range(int(open_retries)):
            rm = (pyvisa.ResourceManager(backend) if backend
                  else pyvisa.ResourceManager())
            try:
                inst = rm.open_resource(
                    visa_address, open_timeout=int(open_timeout_ms),
                )
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
                f"failed to open VISA resource {visa_address!r} "
                f"after {open_retries} attempts; last error: {last_err!r}"
            )

        idn = self._inst.query("*IDN?").strip()
        if verbose:
            print(f"[sa] backend: {backend or '(pyvisa default)'}")
            print(f"[sa] connected to {visa_address}")
            print(f"[sa] IDN: {idn}")

        # ---- Standalone mode: 内参考 + 内触发 ----
        self._inst.write("*CLS")
        self._inst.write(":SENS:ROSC:SOUR INT")
        ref_now = self._inst.query(":SENS:ROSC:SOUR?").strip()
        self._inst.write(":TRIG:SOUR IMM")
        trig_now = self._inst.query(":TRIG:SOUR?").strip()
        if verbose:
            print(f"[sa] reference source: {ref_now}")
            print(f"[sa] trigger source : {trig_now}")
        if not ref_now.upper().startswith("INT"):
            raise RuntimeError(f"SA reference is not internal: {ref_now}")
        if not trig_now.upper().startswith("IMM"):
            raise RuntimeError(f"SA trigger source is not IMM (internal): {trig_now}")

        # ---- 配置 SA mode + sweep ----
        # 用 :CONF:SAN (跟你 instruments.py 的 XSeriesSA.select_spectrum_mode 一致)
        self._inst.write(":CONF:SAN")
        self._inst.write(":INIT:CONT OFF")
        self._inst.write(f":FREQ:CENT {if_center_hz}")
        self._inst.write(f":FREQ:SPAN {if_span_hz}")
        self._inst.write(f":BAND:RES {if_rbw_hz}")
        self._inst.write(":BAND:VID:AUTO ON")
        self._inst.write(":DET RMS")                          # RMS detector
        self._inst.write(":AVER:STAT OFF")                    # 单 sweep, 我们在 Python 层做 R 次重复
        self._inst.write(f":INP:ATT {input_atten_db}")        # 显式设输入衰减
        self._inst.write(f":DISP:WIND:TRAC:Y:RLEV {ref_level_dbm}")
        self._inst.write(":UNIT:POW DBM")

        # Marker
        self._inst.write(":CALC:MARK1:STAT ON")
        self._inst.write(":CALC:MARK1:MODE POS")
        self._inst.write(f":CALC:MARK1:X {if_center_hz}")
        # 注: 下面的 peak search limit 在当前 "固定频点" 读数模式下不生效
        # (read_uv / read_dbm 不再调用 :CALC:MARK1:MAX). 保留这几行是为了:
        #   1) 用户在仪器面板上手动按 Peak Search 时仍只搜窗口内的 peak;
        #   2) 将来想切回 peak-search 模式时不用再加回来.
        if 0 < if_window_hz < if_span_hz:
            left = if_center_hz - if_window_hz / 2.0
            right = if_center_hz + if_window_hz / 2.0
            self._inst.write(f":CALC:MARK:PEAK:SEAR:LIM:LEFT {left}")
            self._inst.write(f":CALC:MARK:PEAK:SEAR:LIM:RIGH {right}")
            self._inst.write(":CALC:MARK:PEAK:SEAR:LIM:STAT ON")

        self._inst.query("*OPC?")

        # 检查 SCPI 错误队列 (只警告, 不抛错)
        try:
            err = self._inst.query(":SYST:ERR?").strip()
            if not err.startswith("+0") and not err.startswith("0,"):
                print(f"[sa] WARNING SCPI error queue: {err}")
        except Exception:
            pass

        if verbose:
            print(f"[sa] center={if_center_hz/1e6:.1f} MHz  "
                  f"span={if_span_hz/1e6:.1f} MHz  "
                  f"rbw={if_rbw_hz/1e3:.0f} kHz  "
                  f"refL={ref_level_dbm:+.1f} dBm  "
                  f"atten={input_atten_db:.0f} dB  "
                  f"window=±{if_window_hz/2/1e6:.2f} MHz")

    def read_uv(self) -> float:
        """触发一次扫描, 读 IF 中心频点 (固定 X) 的 amplitude, 返回 µV RMS。"""
        self._inst.write(":INIT:IMM")
        self._inst.query("*OPC?")
        # 强制把 marker X 钉在 IF 中心频率, 然后读 Y. 不调用 :CALC:MARK1:MAX
        # (那是 peak search), 这样 trace 上 IF 频带内有杂散 / spurs 时也不会
        # 被误读.
        self._inst.write(f":CALC:MARK1:X {self._if_center_hz}")
        if_dbm = float(self._inst.query(":CALC:MARK1:Y?"))
        if_w = (10.0 ** (if_dbm / 10.0)) * 1e-3
        if_v_rms = float(np.sqrt(if_w * PORT_R_OHM))
        return if_v_rms * 1e6

    def read_dbm(self) -> float:
        """同 read_uv() 但直接返回 dBm, debug/test-sa 用。"""
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
# 4. 扫描
# =============================================================================

def run_sweep(
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
    在 (P_LO, P_RF) 二维网格上扫一遍, 每格读 R 次 SA。
    返回 dict, 含所有数据数组, 给 save_npz / plot_heatmaps 用。
    """
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

    print(f"[sweep] LO grid: {p_lo_dbm_grid[0]:+.2f} ~ {p_lo_dbm_grid[-1]:+.2f} dBm "
          f"({n_lo} pts)")
    print(f"[sweep] RF grid: {p_rf_dbm_grid[0]:+.2f} ~ {p_rf_dbm_grid[-1]:+.2f} dBm "
          f"({n_rf} pts)")
    print(f"[sweep] total cells: {n_lo * n_rf}, repeats per cell: {R}")
    print(f"[sweep] settle: {settle_s*1e3:.0f} ms")

    t0 = time.time()
    interrupted = False

    with USRPDualToneSource(
        usrp_args=usrp_args,
        freq_rf=FREQ_RF_HZ,
        freq_lo=FREQ_LO_HZ,
        p_max_dbm=p_max_dbm,
        verbose=False,
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
                    actual_amp_lo[i, j] = info["lo"]["amp"]
                    actual_amp_rf[i, j] = info["rf"]["amp"]

                    time.sleep(float(settle_s))

                    for r in range(R):
                        try:
                            if_amp_uv_all[i, j, r] = float(sa_read_uv())
                        except Exception as exc:
                            print(f"[sweep] SA read failed at "
                                  f"(i={i},j={j},r={r}): {exc!r}")

                    with np.errstate(all="ignore"):
                        mu = float(np.nanmean(if_amp_uv_all[i, j, :]))
                        sd = float(np.nanstd(if_amp_uv_all[i, j, :]))
                    elapsed = time.time() - t0
                    eta = elapsed / cell * (total - cell)
                    print(f"[{cell:4d}/{total}] "
                          f"P_LO={p_lo:+6.2f} dBm  P_RF={p_rf:+6.2f} dBm  "
                          f"IF µV: mean={mu:8.3f} std={sd:6.3f}  "
                          f"ETA={eta:6.1f}s")
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[sweep] interrupted by user at cell {cell}/{total}; "
                  "saving partial data ...")

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
        "if_amp_uv_all": if_amp_uv_all,
        "if_amp_uv_mean": np.asarray(if_amp_uv_mean, dtype=float),
        "if_amp_uv_std": np.asarray(if_amp_uv_std, dtype=float),
        "if_freq_hz": float(IF_FREQ_HZ),
        "freq_rf_hz": float(FREQ_RF_HZ),
        "freq_lo_hz": float(FREQ_LO_HZ),
        "p_max_dbm": float(p_max_dbm),
        "elapsed_s": float(dt),
        "interrupted": bool(interrupted),
    }


# =============================================================================
# 5. Heatmap 计算 + 画图
# =============================================================================

def measured_power_w(if_amp_uv_mean: np.ndarray) -> np.ndarray:
    """SA µV (RMS) -> W into 50 ohm.   P[W] = (V_rms)^2 / 50"""
    v_rms = np.asarray(if_amp_uv_mean, dtype=float) * 1e-6
    return (v_rms ** 2) / PORT_R_OHM


def ideal_mixer_power_w(p_lo_dbm: np.ndarray, p_rf_dbm: np.ndarray) -> np.ndarray:
    """
    Python ideal: 完美乘法器 y = V_LO · V_RF / V_ref 的输出功率。
    取 V_ref = sqrt(R · 1 mW) ≈ 0.224 V (即 0 dBm 端口电压做归一)。

    这样:
        P_ideal[W]   = P_LO[W] * P_RF[W] / 1mW
        P_ideal[dBm] = P_LO[dBm] + P_RF[dBm]

    例:
        (P_LO, P_RF) = ( 0,  0) dBm  ->  P_ideal =   0 dBm = 1 mW
        (P_LO, P_RF) = (+15,+15) dBm  ->  P_ideal = +30 dBm = 1 W
        (P_LO, P_RF) = (-50,-70) dBm  ->  P_ideal = -120 dBm = 1e-15 W
    """
    p_lo_w = 10.0 ** (np.asarray(p_lo_dbm, dtype=float) / 10.0) * 1e-3
    p_rf_w = 10.0 ** (np.asarray(p_rf_dbm, dtype=float) / 10.0) * 1e-3
    return (p_lo_w[:, None] * p_rf_w[None, :]) / 1e-3   # W


def _make_norm(data: np.ndarray, kind: str = "log", gamma: float = 0.5):
    """Build a Normalize-like object suitable for the data."""
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
    out_png: str | Path,
    *,
    norm: str = "log",
    gamma: float = 0.5,
    show: bool = False,
) -> None:
    """两张并排 heatmap:左 ideal, 右 measured。独立 colorbar。"""
    p_lo = np.asarray(data["p_lo_dbm_grid"], dtype=float)
    p_rf = np.asarray(data["p_rf_dbm_grid"], dtype=float)

    p_meas = measured_power_w(data["if_amp_uv_mean"])     # [n_lo, n_rf]
    p_ideal = ideal_mixer_power_w(p_lo, p_rf)             # [n_lo, n_rf]

    # pcolormesh 期望 Z[row=y, col=x], 我们的轴是 X=LO, Y=RF, 所以转置
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

    # --- 左: Python ideal ---
    im1 = axes[0].pcolormesh(
        p_lo, p_rf, z_ideal,
        shading="auto", cmap="viridis", norm=norm_ideal,
    )
    axes[0].set_xlabel("LO port power (dBm)")
    axes[0].set_ylabel("RF port power (dBm)")
    axes[0].set_title(
        r"Python ideal:  $P_{ideal}[\mathrm{dBm}] = P_{LO} + P_{RF}$"
        f"   ({norm_tag})"
    )
    cb1 = fig.colorbar(im1, ax=axes[0])
    cb1.set_label("Power (W) — perfect multiplier")

    # --- 右: Measured ---
    im2 = axes[1].pcolormesh(
        p_lo, p_rf, z_meas,
        shading="auto", cmap="viridis", norm=norm_meas,
    )
    axes[1].set_xlabel("LO port power (dBm)")
    axes[1].set_ylabel("RF port power (dBm)")
    axes[1].set_title(
        f"Measured:  N9020A IF @ {data['if_freq_hz']/1e6:.0f} MHz  "
        f"({norm_tag})"
    )
    cb2 = fig.colorbar(im2, ax=axes[1])
    cb2.set_label("Power (W)")

    fig.suptitle(
        f"USRP X310 + UBX-160  →  ZEM-4300  →  N9020A    "
        f"f_RF={data['freq_rf_hz']/1e9:.2f} GHz, "
        f"f_LO={data['freq_lo_hz']/1e9:.2f} GHz, "
        f"f_IF={data['if_freq_hz']/1e6:.0f} MHz",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"[plot] saved figure: {out_png}")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# 6. NPZ 持久化
# =============================================================================

def save_npz(data: dict, out_npz: str | Path) -> Path:
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


def load_npz(path: str | Path) -> dict:
    path = Path(path)
    z = np.load(str(path), allow_pickle=False)
    out: dict = {}
    for k in z.files:
        if k == "meta_json":
            out.update(json.loads(str(z[k])))
        else:
            out[k] = np.asarray(z[k])
    return out


# =============================================================================
# 7. Dry-run synthetic data (无硬件验证脚本逻辑)
# =============================================================================

def synthesize_demo_data(
    p_lo_dbm_grid: np.ndarray,
    p_rf_dbm_grid: np.ndarray,
    *,
    conv_loss_db: float = 7.0,
    lo_drive_threshold_dbm: float = 0.0,
    noise_uv: float = 1.0,
    seed: int = 42,
) -> dict:
    """
    合成数据,与硬件版的 dict 同结构。
    模型: P_IF = P_RF * convgain(P_LO),  convgain 在 LO drive 不足时塌掉。
    """
    rng = np.random.default_rng(int(seed))
    n_lo = len(p_lo_dbm_grid)
    n_rf = len(p_rf_dbm_grid)

    p_lo = np.asarray(p_lo_dbm_grid, dtype=float)
    p_rf = np.asarray(p_rf_dbm_grid, dtype=float)
    margin = p_lo - lo_drive_threshold_dbm
    convgain_db = -conv_loss_db - 5.0 * np.exp(-margin / 3.0)
    p_if_dbm = p_rf[None, :] + convgain_db[:, None]              # [n_lo, n_rf]
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
        "if_amp_uv_all": if_amp_uv_all,
        "if_amp_uv_mean": np.mean(if_amp_uv_all, axis=2),
        "if_amp_uv_std": np.std(if_amp_uv_all, axis=2, ddof=0),
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
# 8. CLI / main
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="USRP X310 + UBX-160 -> ZEM-4300 -> N9020A mixer heatmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--out-npz", type=str, default="usrp_mixer_heatmap.npz")
    p.add_argument("--out-png", type=str, default="usrp_mixer_heatmap.png")

    # Sweep grid
    p.add_argument("--p-lo-dbm-min", type=float, default=P_LO_DBM_MIN_DEFAULT)
    p.add_argument("--p-lo-dbm-max", type=float, default=P_LO_DBM_MAX_DEFAULT)
    p.add_argument("--n-lo", type=int, default=N_LO_DEFAULT)
    p.add_argument("--p-rf-dbm-min", type=float, default=P_RF_DBM_MIN_DEFAULT)
    p.add_argument("--p-rf-dbm-max", type=float, default=P_RF_DBM_MAX_DEFAULT)
    p.add_argument("--n-rf", type=int, default=N_RF_DEFAULT)
    p.add_argument("--repeats-each", type=int, default=REPEATS_EACH_DEFAULT)
    p.add_argument("--settle-s", type=float, default=SETTLE_S_DEFAULT)

    # Hardware
    p.add_argument("--usrp-args", type=str, default=USRP_ARGS)
    p.add_argument("--p-max-dbm", type=float, default=P_MAX_DBM_DEFAULT,
                   help="UBX max output (dBm) at gain=31.5 / amp=1. "
                        "等同对 UBX 输出做 1-pt 校准.")
    p.add_argument("--sa-visa-address", type=str, default=SA_VISA_ADDRESS,
                   help="N9020A 的 VISA 地址. 默认从 mixer_cal_scripts/config.py 读 SA_RESOURCE.")
    p.add_argument("--sa-backend", type=str, default=SA_VISA_BACKEND,
                   help="pyvisa backend (空串=默认). 默认从 PYVISA_BACKEND 环境变量 "
                        "或 mixer_cal_scripts/config.py 的 VISA_BACKEND 读取.")
    p.add_argument("--sa-span-hz", type=float, default=SA_SPAN_HZ)
    p.add_argument("--sa-rbw-hz", type=float, default=SA_RBW_HZ)
    p.add_argument("--sa-window-hz", type=float, default=SA_WINDOW_HZ)
    p.add_argument("--sa-ref-level-dbm", type=float, default=SA_REF_LEVEL_DBM)
    p.add_argument("--sa-input-atten-db", type=float, default=SA_INPUT_ATTEN_DB)

    # Plot
    p.add_argument("--norm", choices=("log", "gamma", "linear"), default="log")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--show", action="store_true",
                   help="额外 plt.show() 弹窗.")

    # Modes
    p.add_argument("--dry-run", action="store_true",
                   help="不连硬件, 造合成数据画图, 验证脚本逻辑.")
    p.add_argument("--replot", type=str, default=None,
                   help="跳过扫描, 从已有 NPZ 重画 heatmap.")
    p.add_argument("--test-sa", action="store_true",
                   help="只连 N9020A 跑一次单点测量 + 打印 IDN, 用于联调.")

    return p


def _open_sa_reader(args) -> N9020AReader:
    """打开并配置 N9020A, 返回 reader。出错让异常往上抛。"""
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

    # ---- replot only ----
    if args.replot is not None:
        data = load_npz(args.replot)
        plot_heatmaps(data, args.out_png,
                      norm=args.norm, gamma=args.gamma, show=args.show)
        return 0

    # ---- test-sa: 只连 N9020A 测一次 ----
    if args.test_sa:
        print("[main] TEST-SA: 只连 N9020A 跑一次单点测量")
        with _open_sa_reader(args) as sa:
            for k in range(3):
                try:
                    dbm = sa.read_dbm()
                    uv = sa.read_uv()
                    print(f"[test-sa] read #{k}: IF@{IF_FREQ_HZ/1e6:.0f}MHz "
                          f"= {dbm:+7.2f} dBm = {uv:.3f} µV RMS")
                except Exception as exc:
                    print(f"[test-sa] read #{k} FAILED: {exc!r}")
        return 0

    # ---- 准备扫描 grid ----
    p_lo = np.linspace(args.p_lo_dbm_min, args.p_lo_dbm_max, args.n_lo)
    p_rf = np.linspace(args.p_rf_dbm_min, args.p_rf_dbm_max, args.n_rf)

    # ---- dry-run: synthetic data ----
    if args.dry_run:
        print("[main] DRY RUN: 不连硬件, 造合成数据.")
        data = synthesize_demo_data(p_lo, p_rf)
        save_npz(data, args.out_npz)
        plot_heatmaps(data, args.out_png,
                      norm=args.norm, gamma=args.gamma, show=args.show)
        return 0

    # ---- live sweep ----
    print("[main] connecting N9020A ...")
    sa = _open_sa_reader(args)

    try:
        data = run_sweep(
            p_lo_dbm_grid=p_lo,
            p_rf_dbm_grid=p_rf,
            sa_read_uv=sa.read_uv,
            repeats_each=args.repeats_each,
            settle_s=args.settle_s,
            p_max_dbm=args.p_max_dbm,
            usrp_args=args.usrp_args,
        )
    except Exception:
        traceback.print_exc()
        sa.close()
        return 1

    sa.close()

    save_npz(data, args.out_npz)
    plot_heatmaps(data, args.out_png,
                  norm=args.norm, gamma=args.gamma, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
