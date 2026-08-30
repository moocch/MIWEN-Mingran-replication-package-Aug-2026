#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
piml_4096_digital_twin.py — N=4096 双 USRP 混频器内积实验的 PIML 数字孪生
=========================================================================

测量链路 (与 gr_heatmap_vector_inner_product_v2.py 一致):
    USRP X310 TX  CH0 = RF @ 1.2 GHz (4096-tone 向量 a)
                  CH1 = LO @ 0.9 GHz (4096-tone 向量 b)
    -> ZEM-4300 无源环形混频器 -> 30 dB 衰减 -> 第二台 USRP RX @ 300 MHz
    读数 = |mean(IQ)| * 1e6  (基带 DC 匹配滤波, 未校准任意单位 uV)

核心物理发现 1 — 别名折叠 (aliasing):
    fs = 10 MS/s, df_tone = 1 MHz, N = 4096
    => 基带模板周期仅 20 个样本, 4096 个 tone 折叠到 10 个可分辨频点
       (±0.5 … ±4.5 MHz)。实际发射的是 10-tone 信号, 其有效复振幅为
       A_k = Σ_{i: f_i ≡ f_k (mod fs)} a_i。
    => 决定中心 DC 拍频功率的是 **折叠后有效归一化内积**
       ip_eff = |⟨A,B⟩|² / (‖A‖²‖B‖²) = 1.398e-2  (-18.55 dB)
       而非标称 |⟨a,b⟩|² = 7.679e-4 (-31.15 dB), 相差 +12.60 dB。

核心物理发现 2 — 双泵浦对称软饱和:
    高 PAPR(≈5.3 dB) 多 tone 泵浦下, 二极管开关导通的包络平均使 LO
    turn-on 由 N=1 的 Weibull 硬锁变为 Hill 软尾 (实测斜率验证 w_hill→1);
    且 RF 口功率高时同样能驱动二极管开关 (实测热图近似 LO↔RF 对称),
    需要双泵浦项:  sig ∝ ip_eff·[α_L(P_LO)·C(P_RF) + κ·α_R(P_RF)·C(P_LO)]。

模型结构 (沿用 N=1 / N=2,4,8 PIML 风格):
    log10 P_arb = log10 P_phys(双泵浦物理核, 15 参数, sigmoid 界限)
                  + delta_dB(有界 tanh MLP) / 10

三阶段训练:
    Stage 1  物理核精修 (Adam, NN 关闭, 自 scipy 初值热启动)
    Stage 2  NN-only (物理冻结)
    Stage 3  联合微调 (小学习率 + 对 Stage-1 解的弱锚正则)

用法:
    python3 piml_4096_digital_twin.py \
        --npz  4096code_meas/meas_raw/gr_usrp_mixer_vector_heatmap_N4096.npz \
        --init /path/to/work/fit_dualpump.npy \
        --outdir /path/to/work/4096_out
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# 发射链常数 (gr_heatmap_vector_inner_product_v2.py)
# --------------------------------------------------------------------------
P_MAX_DBM = 18.0
GAIN_MIN_DB = 0.0
GAIN_MAX_DB = 31.5
AMP_TARGET = 0.6
AMP_MAX = 0.95
R_OHM = 50.0


def power_to_gain_amp_np(target_dbm: np.ndarray):
    """与采集脚本 power_to_gain_amp() 完全一致的向量化重建 (gain_db, amp)。

    网格 -70…+10 dBm 内只会触发两个分支:
      target >= -17.94 dBm : amp = 0.6 固定, gain 线性
      target <  -17.94 dBm : gain = 0,  amp 指数下降
    -17.94 dBm 处的 amp 台阶正是实测热图中 ~-17.5 dBm 台阶的来源
    (amp=0.6 时 DUC 插值重建峰值 >1 产生轻度削顶)。
    """
    t = np.asarray(target_dbm, dtype=np.float64)
    gain = GAIN_MAX_DB + t - P_MAX_DBM - 20.0 * np.log10(AMP_TARGET)
    amp = np.full_like(t, AMP_TARGET)
    hi = gain > GAIN_MAX_DB
    gain = np.where(hi, GAIN_MAX_DB, gain)
    amp = np.where(hi, np.minimum(10.0 ** ((t - P_MAX_DBM) / 20.0), AMP_MAX), amp)
    lo = gain < GAIN_MIN_DB
    gain = np.where(lo, GAIN_MIN_DB, gain)
    amp = np.where(lo, 10.0 ** ((t - (P_MAX_DBM - GAIN_MAX_DB)) / 20.0), amp)
    return gain, amp


# --------------------------------------------------------------------------
# 数据加载 + 别名折叠
# --------------------------------------------------------------------------
@dataclass
class N4096Dataset:
    p_lo_dbm: np.ndarray          # (33,)
    p_rf_dbm: np.ndarray          # (33,)
    y_db_arb: np.ndarray          # (33,33) 10*log10((uv*1e-6)^2/50) 任意 dB
    uv_mean: np.ndarray
    uv_std: np.ndarray
    ip_nominal: float             # |<a,b>|^2 (标称, 4096 维)
    ip_eff: float                 # 折叠后有效归一化内积
    papr_lo_db: float
    papr_rf_db: float
    fold_A: np.ndarray            # 折叠后 LO 10-tone 复振幅 (‖·‖ 未归一)
    fold_B: np.ndarray
    meta: dict


def fold_tones(vec: np.ndarray, n: int, df_hz: float, fs_hz: float):
    """把 n 个间隔 df 的 tone 复振幅按采样率 fs 别名折叠到可分辨频点。"""
    f = (np.arange(n) - (n - 1) / 2.0) * df_hz
    fa = ((f + fs_hz / 2.0) % fs_hz) - fs_hz / 2.0       # 折叠后频率
    key = np.round(fa / df_hz * 2.0) / 2.0               # 半整数 df 格点
    uniq = np.unique(key)
    folded = np.array([vec[key == u].sum() for u in uniq])
    return uniq * df_hz, folded


def load_n4096_npz(path: Path) -> N4096Dataset:
    d = np.load(path, allow_pickle=True)
    meta = d["meta"].item() if "meta" in d.files else {}
    p_lo = np.asarray(d["p_lo_dbm_grid"], dtype=np.float64)
    p_rf = np.asarray(d["p_rf_dbm_grid"], dtype=np.float64)
    uv_mean = np.asarray(d["if_amp_uv_mean"], dtype=np.float64)
    uv_std = np.asarray(d["if_amp_uv_std"], dtype=np.float64)
    meas_w = np.maximum((uv_mean * 1e-6) ** 2 / R_OHM, 1e-30)
    y = 10.0 * np.log10(meas_w)

    a = np.asarray(d["vec_a"])
    b = np.asarray(d["vec_b"])
    n = int(meta.get("N", a.size))
    df = float(meta.get("df_tone_hz", 1.0e6))
    fs = float(meta.get("samp_rate", 10.0e6))

    ip_nom = float(np.abs(np.vdot(a, b)) ** 2)           # 向量已 RMS 归一
    _, A = fold_tones(a, n, df, fs)
    _, B = fold_tones(b, n, df, fs)
    ip_eff = float(np.abs(np.vdot(A, B)) ** 2 /
                   (np.sum(np.abs(A) ** 2) * np.sum(np.abs(B) ** 2)))

    papr_lo = float(meta.get("papr_b_after_db", 5.357))  # CH1 = LO = b
    papr_rf = float(meta.get("papr_a_after_db", 5.311))  # CH0 = RF = a
    return N4096Dataset(p_lo, p_rf, y, uv_mean, uv_std,
                        ip_nom, ip_eff, papr_lo, papr_rf, A, B, meta)


# --------------------------------------------------------------------------
# 物理核: 双泵浦对称软饱和混频器 (15 参数, sigmoid 界限重参数化)
# --------------------------------------------------------------------------
PHYS_NAMES = ["G", "PsatL", "betaL", "PsatR", "betaR", "w_hill",
              "PcompRF", "PcompLO", "betaC", "kappa",
              "c_papr_lo", "c_papr_rf", "leak", "C_cal", "floor"]

PHYS_BOUNDS = {
    "G":        (-30.0, 10.0),   # 转换增益 dB (混频 IF 平面)
    "PsatL":    (-30.0, 20.0),   # LO 泵浦导通点 dBm
    "betaL":    (0.3, 6.0),
    "PsatR":    (-40.0, 20.0),   # RF 泵浦导通点 dBm
    "betaR":    (0.3, 8.0),
    "w_hill":   (0.0, 1.0),      # 0=Weibull 硬锁, 1=Hill 软尾 (高 PAPR 包络平均)
    "PcompRF":  (-15.0, 25.0),   # RF 作信号的压缩拐点 dBm
    "PcompLO":  (-15.0, 25.0),
    "betaC":    (0.3, 20.0),
    "kappa":    (-30.0, 10.0),   # RF-泵浦镜像路径相对权重 dB
    "c_papr_lo": (-1.5, 1.5),    # PAPR 有效功率平移系数
    "c_papr_rf": (-1.5, 1.5),
    "leak":     (-110.0, -30.0), # RF→IF 直通泄漏 dB
    "C_cal":    (-50.0, 200.0),  # mixer IF dBm ↔ USRP 任意单位 全局标定 dB
    "floor":    (-200.0, 100.0), # RX 链 DC 估计噪声底 (任意 dB, 以 dBm 形参)
}

LN10_10 = math.log(10.0) / 10.0


def _to_raw(value: float, lo: float, hi: float) -> float:
    frac = (value - lo) / (hi - lo)
    frac = min(max(frac, 1e-6), 1.0 - 1e-6)
    return math.log(frac / (1.0 - frac))


class DualPumpPhysicsMixer(nn.Module):
    """P_arb = 10^(C_cal/10)·( G·ip_eff·[α_L·C_R + κ·α_R·C_L] + leak·P_RF )
               + P_floor,    α_*: Weibull/Hill 混合导通, C_*: 软压缩。"""

    def __init__(self, init: dict[str, float]):
        super().__init__()
        for k in PHYS_NAMES:
            lo, hi = PHYS_BOUNDS[k]
            self.register_parameter(
                f"raw_{k}", nn.Parameter(torch.tensor(_to_raw(init[k], lo, hi),
                                                      dtype=torch.float64)))

    def value(self, k: str) -> torch.Tensor:
        lo, hi = PHYS_BOUNDS[k]
        raw = getattr(self, f"raw_{k}")
        return lo + (hi - lo) * torch.sigmoid(raw)

    def params_dict(self) -> dict[str, float]:
        with torch.no_grad():
            return {k: float(self.value(k).detach()) for k in PHYS_NAMES}

    @staticmethod
    def _alpha_mix(p_dbm, psat, beta, w):
        log10x = torch.clamp((p_dbm - psat) / 10.0, -30.0, 5.0)
        xw = torch.pow(10.0, torch.clamp(beta * log10x, -30.0, 10.0))
        weib = -torch.expm1(-xw)
        xh = torch.pow(10.0, torch.clamp(beta * log10x, -30.0, 30.0))
        hill = xh / (1.0 + xh)
        return (1.0 - w) * weib + w * hill

    @staticmethod
    def _comp_w(p_dbm, pc_dbm, beta_c):
        pw = 1e-3 * torch.pow(10.0, p_dbm / 10.0)
        pcw = 1e-3 * torch.pow(10.0, pc_dbm / 10.0)
        ratio = torch.clamp(pw / pcw, min=1e-30)
        return pw * torch.pow(1.0 + torch.pow(ratio, beta_c), -1.0 / beta_c)

    def forward(self, p_lo_dbm, p_rf_dbm, ip_eff, papr_lo_db, papr_rf_db):
        v = self.value
        ple = p_lo_dbm + v("c_papr_lo") * papr_lo_db
        pre = p_rf_dbm + v("c_papr_rf") * papr_rf_db
        a_l = self._alpha_mix(ple, v("PsatL"), v("betaL"), v("w_hill"))
        a_r = self._alpha_mix(pre, v("PsatR"), v("betaR"), v("w_hill"))
        c_r = self._comp_w(pre, v("PcompRF"), v("betaC"))
        c_l = self._comp_w(ple, v("PcompLO"), v("betaC"))
        sig = (torch.pow(10.0, v("G") / 10.0) * ip_eff *
               (a_l * c_r + torch.pow(10.0, v("kappa") / 10.0) * a_r * c_l))
        leak_w = torch.pow(10.0, v("leak") / 10.0) * 1e-3 * torch.pow(10.0, p_rf_dbm / 10.0)
        arb = (torch.pow(10.0, v("C_cal") / 10.0) * (sig + leak_w) +
               1e-3 * torch.pow(10.0, v("floor") / 10.0))
        return torch.log10(torch.clamp(arb, min=1e-32))


# --------------------------------------------------------------------------
# 残差 NN: 有界 tanh MLP
# --------------------------------------------------------------------------
class ResidualNN4096(nn.Module):
    """delta_dB = max_db · tanh(MLP(features))

    特征 = [P_LO/30, P_RF/30, amp_LO/0.6, amp_RF/0.6, gain_LO/31.5, gain_RF/31.5]
    amp/gain 由采集脚本的 power_to_gain_amp() 精确重建, 显式编码
    -17.94 dBm 处 amp 台阶 (DUC 削顶边界), 让 NN 不必从 P 的折线中自学拐点。
    """

    def __init__(self, hidden: int = 48, max_db: float = 5.0):
        super().__init__()
        self.max_db = max_db
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        ).double()
        with torch.no_grad():
            self.net[-1].weight.mul_(1e-2)
            self.net[-1].bias.zero_()

    def forward(self, feats):
        return self.max_db * torch.tanh(self.net(feats)).squeeze(-1)


class PIML4096Mixer(nn.Module):
    def __init__(self, physics: DualPumpPhysicsMixer, nn_res: ResidualNN4096):
        super().__init__()
        self.physics = physics
        self.nn_res = nn_res

    def predict_log10_arb(self, p_lo, p_rf, ip_eff, papr_lo, papr_rf,
                          feats, use_nn: bool = True):
        log10_phys = self.physics(p_lo, p_rf, ip_eff, papr_lo, papr_rf)
        if not use_nn:
            return log10_phys
        return log10_phys + self.nn_res(feats) / 10.0


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------
def build_tensors(ds: N4096Dataset, device):
    PLO, PRF = np.meshgrid(ds.p_lo_dbm, ds.p_rf_dbm, indexing="ij")
    g_lo, a_lo = power_to_gain_amp_np(PLO)
    g_rf, a_rf = power_to_gain_amp_np(PRF)
    feats = np.stack([PLO / 30.0, PRF / 30.0,
                      a_lo / AMP_TARGET, a_rf / AMP_TARGET,
                      g_lo / GAIN_MAX_DB, g_rf / GAIN_MAX_DB], axis=-1)
    t = lambda x: torch.tensor(x, dtype=torch.float64, device=device)
    return dict(
        p_lo=t(PLO), p_rf=t(PRF), y=t(ds.y_db_arb), feats=t(feats),
        ip_eff=t(ds.ip_eff), papr_lo=t(ds.papr_lo_db), papr_rf=t(ds.papr_rf_db),
    )


def rmse_db(model, T, use_nn: bool) -> float:
    with torch.no_grad():
        pred = 10.0 * model.predict_log10_arb(
            T["p_lo"], T["p_rf"], T["ip_eff"], T["papr_lo"], T["papr_rf"],
            T["feats"], use_nn=use_nn)
        return float(torch.sqrt(torch.mean((pred - T["y"]) ** 2)))


def train(model: PIML4096Mixer, T, args, anchor_raw: dict[str, torch.Tensor]):
    history = []

    def loss_fn(use_nn: bool, lam_anchor: float):
        pred = 10.0 * model.predict_log10_arb(
            T["p_lo"], T["p_rf"], T["ip_eff"], T["papr_lo"], T["papr_rf"],
            T["feats"], use_nn=use_nn)
        loss = torch.mean((pred - T["y"]) ** 2)
        if use_nn and args.lam_nn > 0:
            delta = model.nn_res(T["feats"])
            loss = loss + args.lam_nn * torch.mean(delta ** 2)
        if lam_anchor > 0:
            reg = sum((getattr(model.physics, f"raw_{k}") - anchor_raw[k]) ** 2
                      for k in PHYS_NAMES)
            loss = loss + lam_anchor * reg
        return loss

    def run_stage(label, params, n_iter, lr, use_nn, lam_anchor=0.0):
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(n_iter):
            opt.zero_grad()
            loss = loss_fn(use_nn, lam_anchor)
            loss.backward()
            opt.step()
            if (it + 1) % max(1, n_iter // 10) == 0 or it == 0:
                r = rmse_db(model, T, use_nn)
                history.append({"stage": label, "iter": it + 1, "rmse_db": r})
                print(f"  [{label}] iter {it+1:5d}/{n_iter}  RMSE = {r:6.3f} dB")
        return history

    phys_params = [getattr(model.physics, f"raw_{k}") for k in PHYS_NAMES]
    nn_params = list(model.nn_res.parameters())

    print("\n=== Stage 1: 物理核精修 (NN 关闭, scipy 热启动) ===")
    print(f"  初始物理 RMSE = {rmse_db(model, T, use_nn=False):.3f} dB")
    run_stage("phys", phys_params, args.n_iter_phys, args.lr_phys, use_nn=False)
    rmse_phys = rmse_db(model, T, use_nn=False)
    # Stage-1 解作为后续锚点
    anchor_raw.update({k: getattr(model.physics, f"raw_{k}").detach().clone()
                       for k in PHYS_NAMES})

    print("\n=== Stage 2: 残差 NN 训练 (物理冻结) ===")
    for p in phys_params:
        p.requires_grad_(False)
    run_stage("nn", nn_params, args.n_iter_nn, args.lr_nn, use_nn=True)

    print("\n=== Stage 3: 联合微调 (物理解冻 + 弱锚正则) ===")
    for p in phys_params:
        p.requires_grad_(True)
    opt_groups = [{"params": phys_params, "lr": args.lr_joint_phys},
                  {"params": nn_params, "lr": args.lr_joint_nn}]
    opt = torch.optim.Adam(opt_groups)
    n_iter = args.n_iter_joint
    for it in range(n_iter):
        opt.zero_grad()
        loss = loss_fn(True, args.lam_anchor)
        loss.backward()
        opt.step()
        if (it + 1) % max(1, n_iter // 10) == 0 or it == 0:
            r = rmse_db(model, T, use_nn=True)
            history.append({"stage": "joint", "iter": it + 1, "rmse_db": r})
            print(f"  [joint] iter {it+1:5d}/{n_iter}  RMSE = {r:6.3f} dB")

    return history, rmse_phys


# --------------------------------------------------------------------------
# 基线
# --------------------------------------------------------------------------
def shape_rmse(pred_db: np.ndarray, y_db: np.ndarray) -> tuple[float, float]:
    """未校准接收: 允许一个全局常数偏移后的形状 RMSE。返回 (rmse, offset)。"""
    off = float(np.mean(y_db - pred_db))
    return float(np.sqrt(np.mean((pred_db + off - y_db) ** 2))), off


def ideal_surface_dbm(ds: N4096Dataset, use_eff: bool) -> np.ndarray:
    PLO, PRF = np.meshgrid(ds.p_lo_dbm, ds.p_rf_dbm, indexing="ij")
    ip = ds.ip_eff if use_eff else ds.ip_nominal
    return PLO + PRF + 10.0 * np.log10(ip)


def naive_n1_surface_dbm(ds: N4096Dataset, n1_summary: Path) -> np.ndarray:
    """N=1 已训练物理核 (Weibull) 直接乘 ip_eff —— '天真迁移' 基线。"""
    p = json.loads(n1_summary.read_text())["physics_params"]
    PLO, PRF = np.meshgrid(ds.p_lo_dbm, ds.p_rf_dbm, indexing="ij")
    x = np.power(10.0, np.clip(p["beta_LO"] * (PLO - p["P_LO_sat_dBm"]) / 10.0,
                               -30, 10))
    alpha = -np.expm1(-x)                                  # Weibull turn-on
    prf_w = 1e-3 * 10.0 ** (PRF / 10.0)
    pc_w = 1e-3 * 10.0 ** (p["P_RF_comp_dBm"] / 10.0)
    prf_e = prf_w / (1.0 + np.maximum(prf_w / pc_w, 1e-30) ** p["beta_RF"]) ** (1.0 / p["beta_RF"])
    sig = 10.0 ** (p["G_max_dB"] / 10.0) * alpha * prf_e * ds.ip_eff
    leak = 10.0 ** (p["leak_dB"] / 10.0) * prf_w
    noise = 1e-3 * 10.0 ** (p["P_noise_dBm"] / 10.0)
    return 10.0 * np.log10(np.maximum(sig + leak + noise, 1e-32))


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True,
                    help="scipy 双泵浦拟合参数 .npy (15,)")
    ap.add_argument("--n1-summary", type=Path,
                    default=Path("/path/to/work/4096CH/N=1/PIML_Figures/model/summary.json"))
    ap.add_argument("--outdir", type=Path, default=Path("piml_4096_out"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--max-db", type=float, default=5.0)
    ap.add_argument("--n-iter-phys", type=int, default=1200)
    ap.add_argument("--n-iter-nn", type=int, default=6000)
    ap.add_argument("--n-iter-joint", type=int, default=4000)
    ap.add_argument("--lr-phys", type=float, default=3e-3)
    ap.add_argument("--lr-nn", type=float, default=3e-3)
    ap.add_argument("--lr-joint-phys", type=float, default=2e-4)
    ap.add_argument("--lr-joint-nn", type=float, default=5e-4)
    ap.add_argument("--lam-nn", type=float, default=1e-4)
    ap.add_argument("--lam-anchor", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "model").mkdir(exist_ok=True)

    ds = load_n4096_npz(args.npz)
    print(f"网格: {ds.p_lo_dbm.size}×{ds.p_rf_dbm.size}, "
          f"P∈[{ds.p_lo_dbm.min():+.0f},{ds.p_lo_dbm.max():+.0f}] dBm")
    print(f"ip_nominal = {ds.ip_nominal:.4e} ({10*np.log10(ds.ip_nominal):+.2f} dB)")
    print(f"ip_eff(折叠) = {ds.ip_eff:.4e} ({10*np.log10(ds.ip_eff):+.2f} dB)   "
          f"差 = {10*np.log10(ds.ip_eff/ds.ip_nominal):+.2f} dB")
    print(f"PAPR: LO {ds.papr_lo_db:.3f} dB, RF {ds.papr_rf_db:.3f} dB")

    init_vec = np.load(args.init)
    init = dict(zip(PHYS_NAMES, [float(v) for v in init_vec]))
    # 边界内夹紧 (w_hill=1.000 恰在上界)
    for k in PHYS_NAMES:
        lo, hi = PHYS_BOUNDS[k]
        span = hi - lo
        init[k] = min(max(init[k], lo + 1e-3 * span), hi - 1e-3 * span)

    physics = DualPumpPhysicsMixer(init).to(device)
    nn_res = ResidualNN4096(hidden=args.hidden, max_db=args.max_db).to(device)
    model = PIML4096Mixer(physics, nn_res).to(device)
    T = build_tensors(ds, device)

    anchor_raw = {k: getattr(physics, f"raw_{k}").detach().clone()
                  for k in PHYS_NAMES}
    history, rmse_phys = train(model, T, args, anchor_raw)
    rmse_full = rmse_db(model, T, use_nn=True)

    # ---------------- 基线与预测面 ----------------
    y = ds.y_db_arb
    ideal_nom = ideal_surface_dbm(ds, use_eff=False)
    ideal_eff = ideal_surface_dbm(ds, use_eff=True)
    naive_n1 = naive_n1_surface_dbm(ds, args.n1_summary)
    r_ideal, off_ideal = shape_rmse(ideal_nom, y)
    r_ideal_eff, off_ideal_eff = shape_rmse(ideal_eff, y)
    r_naive, off_naive = shape_rmse(naive_n1, y)

    with torch.no_grad():
        log10_phys = model.physics(T["p_lo"], T["p_rf"], T["ip_eff"],
                                   T["papr_lo"], T["papr_rf"])
        delta = model.nn_res(T["feats"])
        pred_phys = (10.0 * log10_phys).cpu().numpy()
        delta_db = delta.cpu().numpy()
        pred_full = pred_phys + delta_db

    res_phys = pred_phys - y
    res_full = pred_full - y
    p95 = float(np.percentile(np.abs(res_full), 95))
    frac2 = float((np.abs(res_full) <= 2.0).mean())

    print("\n================ 结果汇总 (RMSE, dB) ================")
    print(f"Ideal  (标称 ip², 形状)     : {r_ideal:7.2f}   (offset {off_ideal:+.2f} dB)")
    print(f"Ideal  (折叠 ip_eff, 形状)  : {r_ideal_eff:7.2f}   (offset {off_ideal_eff:+.2f} dB)")
    print(f"Naive  N=1 孪生迁移 (形状)  : {r_naive:7.2f}   (offset {off_naive:+.2f} dB)")
    print(f"Physics-only (双泵浦, 本文) : {rmse_phys:7.2f}")
    print(f"Full PIML (物理+NN)         : {rmse_full:7.2f}   "
          f"|err|≤2dB: {frac2*100:.1f}%  p95={p95:.2f} dB")

    # ---------------- 保存 ----------------
    torch.save({"physics_state": physics.state_dict(),
                "nn_state": nn_res.state_dict(),
                "phys_names": PHYS_NAMES,
                "phys_bounds": PHYS_BOUNDS,
                "hidden": args.hidden, "max_db": args.max_db},
               args.outdir / "model" / "piml_mixer_4096.pt")

    summary = {
        "input_npz": str(args.npz),
        "aliasing": {
            "fs_hz": float(ds.meta.get("samp_rate", 1e7)),
            "df_tone_hz": float(ds.meta.get("df_tone_hz", 1e6)),
            "n_tones_nominal": 4096,
            "n_tones_folded": int(ds.fold_A.size),
            "ip_nominal": ds.ip_nominal,
            "ip_eff_folded": ds.ip_eff,
            "ip_gap_db": 10 * math.log10(ds.ip_eff / ds.ip_nominal),
        },
        "physics_params": physics.params_dict(),
        "papr_db": {"lo": ds.papr_lo_db, "rf": ds.papr_rf_db},
        "metrics_db": {
            "rmse_ideal_shape": r_ideal,
            "rmse_ideal_eff_shape": r_ideal_eff,
            "rmse_naive_n1_shape": r_naive,
            "rmse_physics_only": rmse_phys,
            "rmse_physics_plus_nn": rmse_full,
            "mae_physics_plus_nn": float(np.mean(np.abs(res_full))),
            "p95_abs_err": p95, "frac_abs_err_le_2db": frac2,
        },
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "history_tail": history[-12:],
    }
    (args.outdir / "model" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    np.savez(args.outdir / "twin_predictions_N4096.npz",
             p_lo_dbm_grid=ds.p_lo_dbm, p_rf_dbm_grid=ds.p_rf_dbm,
             meas_db_arb=y, uv_mean=ds.uv_mean, uv_std=ds.uv_std,
             ideal_nominal_dbm=ideal_nom, ideal_eff_dbm=ideal_eff,
             naive_n1_dbm=naive_n1,
             offset_ideal_db=off_ideal, offset_ideal_eff_db=off_ideal_eff,
             offset_naive_db=off_naive,
             phys_db_arb=pred_phys, full_db_arb=pred_full, delta_nn_db=delta_db,
             ip_nominal=ds.ip_nominal, ip_eff=ds.ip_eff,
             fold_A=ds.fold_A, fold_B=ds.fold_B,
             papr_lo_db=ds.papr_lo_db, papr_rf_db=ds.papr_rf_db)

    print(f"\n已保存: {args.outdir}/model/piml_mixer_4096.pt, summary.json, "
          f"twin_predictions_N4096.npz")


if __name__ == "__main__":
    main()
