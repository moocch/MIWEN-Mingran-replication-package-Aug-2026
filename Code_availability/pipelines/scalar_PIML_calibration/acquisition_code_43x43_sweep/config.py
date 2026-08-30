"""
config.py

把所有“需要经常改”的东西集中在这里：
- VISA 资源字符串（你已经在 quick_check.txt 里验证能通）
- M8195A 通道映射
- 采样率、波形长度、校准目标功率列表
- 安全限制（Vpp 上限、步进限制）
"""

from __future__ import annotations

# ---------------------------
# VISA 资源（按你的 quick_check.txt）
# ---------------------------
VISA_BACKEND = ""  # Empty means use system default VISA backend only.

AWG_RESOURCE = "TCPIP0::192.168.10.2::5025::SOCKET"      # Keysight M8195A
SA_RESOURCE  = "TCPIP::169.254.190.220::inst0::INSTR"    # Agilent/Keysight N9020A MXA (VXI-11)

# ---------------------------
# M8195A 通道映射
# ---------------------------
# 重要：M8195A 在 DUAL 模式下“可用通道”为 1 和 4（2、3 不用于输出）。
# 你的 Measurement Plan 写的是 AWG_CH1 / AWG_CH2；这里默认映射为：
AWG_CH_LO = 1      # 对应 AWG_CH1（LO）
AWG_CH_RF = 4      # 对应 AWG_CH2（RF）

AWG_DAC_MODE = "DUAL"  # SING | DUAL | FOUR | ...（这里用 DUAL）

# ---------------------------
# 波形与采样率（建议固定）
# ---------------------------
FS_HZ = 64e9  # 64 GSa/s（可按你设备实际支持范围调整）

# 单音：为了在 900MHz/1187.5/1200/1212.5 上做到“整数周期拼接”，建议用 N=5120（频率分辨率 12.5MHz）
N_SAMP_TONE = 5120

# 两音：Δf=1MHz -> 每个 tone 偏移 0.5MHz；为做到整数周期拼接，建议 N=128000（频率分辨率 0.5MHz）
N_SAMP_2TONE = 128000

# 波形数值缩放（8bit DAC 的满刻度是 ±127）
# 单音可以用接近满刻度以提高 SNR；两音要留裕度避免叠加峰值溢出
WAVE_SCALE_TONE  = 0.95
WAVE_SCALE_2TONE = 0.45

# ---------------------------
# Station 0 - CAL 目标功率列表（端口参考面）
# ---------------------------
PLO_LIST_DBM = [4.0, 7.0, 10.0]
PRF_EDGE_FREQS_HZ = [1187.5e6, 1200.0e6, 1212.5e6]
PRF_SWEEP_DBM = [-40.0, -30.0, -20.0, -10.0, 0.0]

# CAL-RF-2TONE：文档只要求“至少 Δf=1MHz 的 2–3 档每音功率”
# 这里给 3 档默认值（你可改成文档里 Pin_tone_list 的任意子集）
PRF_2TONE_PERTONE_DBM = [-30.0, -24.0, -18.0]
RF_2TONE_F1_HZ = 1199.5e6
RF_2TONE_F2_HZ = 1200.5e6

# ---------------------------
# 校准算法参数
# ---------------------------
TOL_DB = 0.2            # ±0.2 dB
MAX_ITER = 12           # 迭代上限
MAX_STEP_DB = 6.0       # 单次幅度修正上限（避免发散/过冲）
SETTLE_S = 0.15         # 每次改幅度后等待（秒）

# ---------------------------
# M8195A 输出幅度（Vpp）安全限制
# 注意：不同配置/选件/外接放大器可能不同，必要时请按你的硬件能力调小上限。
# ---------------------------
VPP_MIN = 0.075         # 75 mVpp（M8195A 当前实机查询下限）
VPP_MAX = 1.0           # 1.0 Vpp（M8195A 当前实机查询上限）

# 初始猜测（只是起点，脚本会闭环调到目标 dBm）
VPP_START_LO = 0.5
VPP_START_RF = 0.2
VPP_START_2TONE = 0.2

# ---------------------------
# SA 模板参数（按 Measurement Plan 6.x）
# ---------------------------
SA_ISO = dict(span_hz=20e6, rbw_hz=1e6, vbw_hz=1e6, avg_count=20, input_atten_db=15, detector="RMS", ref_level_dbm=10.0)
SA_TONE = dict(span_hz=5e6,  rbw_hz=1e3, vbw_hz=1e3, avg_count=10, input_atten_db=10, detector="RMS")
SA_IP3  = dict(span_hz=10e6, rbw_hz=100, vbw_hz=100, avg_count=20, input_atten_db=10, detector="RMS")

# 输出目录
DATA_DIR = "data"
