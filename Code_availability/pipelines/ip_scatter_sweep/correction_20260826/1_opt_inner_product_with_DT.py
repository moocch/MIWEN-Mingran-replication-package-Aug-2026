#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 数字孪生 (PIML) 内积后端优化 v3 — Mingran Jia; batch edition.
#
# 用法: 把本文件放在含 gr_fig3c_ip_scatter.npz 的目录树里, 直接
#           python3 1_opt_inner_product_with_DT.py
#       会自动发现全部数据集 (含子目录), 逐个优化, 按 N 命名输出:
#           inner_product_optimized_N{N}.npz / .png
#       亦可显式指定: --npz path1.npz path2.npz
#
# 要点:
#   * 波形来源: 优先直读 npz 同目录的 tx_lo.cf32/tx_rf.cf32; 找不到则从种子
#     流式重建 (精确复现 papr_clip_db 削波, O(L) 内存). 两种来源都做
#     "理想乘法混频应复现 c_true" 自检, 防止拿错波形/约定漂移.
#   * 失真基底: 数字孪生 comp_ratio 基底 + NPZ 内置精确 limiter/square 基底
#     + 复增益列, k 折交叉验证联合拟合 (导频式泛化, 无自拟合); 同时输出
#     "孪生单基底(原方法)" 对照.
#   * 输出兼容: optimized_ip_15dB / optimized_ip_25dB 键保留.

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import argparse, glob, json, math, os
from pathlib import Path

import numpy as np

PIML_TWIN_PARAMS = {
    "G": 2.7293514786, "PsatL": -2.8049513064, "betaL": 1.8507810921,
    "PsatR": -8.0228142384, "betaR": 2.9163471945, "w_hill": 0.9998751648,
    "PcompRF": -3.9442364607, "PcompLO": -5.1990871823, "betaC": 0.6101367971,
    "kappa": -1.4910963470, "c_papr_lo": 0.5383216349, "c_papr_rf": 0.0681181224,
    "leak": -97.7029503059, "C_cal": 13.2314239355, "floor": -97.5820283021,
}


class DigitalTwin:

    def __init__(self, params: dict, name: str = "PIML-N4096"):
        self.p = dict(params)
        self.name = name

    @classmethod
    def from_summary_json(cls, path: str):
        S = json.load(open(path))
        params = S["physics_params"] if "physics_params" in S else S
        return cls(params, name=Path(path).stem)

    def alpha(self, p_dbm, which="L"):
        psat = self.p["PsatL"] if which == "L" else self.p["PsatR"]
        beta = self.p["betaL"] if which == "L" else self.p["betaR"]
        w = self.p["w_hill"]
        x = 10.0 ** ((np.asarray(p_dbm, float) - psat) / 10.0)
        weib = -np.expm1(-(x ** beta))
        hill = (x ** beta) / (1.0 + x ** beta)
        return (1.0 - w) * weib + w * hill

    def comp_ratio(self, p_dbm, port="LO"):
        pc = self.p["PcompLO"] if port == "LO" else self.p["PcompRF"]
        bc = self.p["betaC"]
        pw = 1e-3 * 10.0 ** (np.asarray(p_dbm, float) / 10.0)
        pcw = 1e-3 * 10.0 ** (pc / 10.0)
        cw = pw * (1.0 + (pw / pcw) ** bc) ** (-1.0 / bc)
        return cw / np.maximum(pw, 1e-300)

    def operating_regime(self, p_lo_dbm, p_rf_dbm):
        return dict(
            alpha_L=float(self.alpha(p_lo_dbm, "L")),
            lo_comp_ratio=float(self.comp_ratio(p_lo_dbm, "LO")),
            rf_comp_ratio=float(self.comp_ratio(p_rf_dbm, "RF")),
        )


# ---------------- 帧几何 / 波形来源 ----------------

def _gen_vec_pair(N, seed, slot):
    r = np.random.default_rng([int(seed), int(slot)])
    a = r.uniform(0, 1, N) * np.exp(1j * r.uniform(0, 2 * np.pi, N))
    b = r.uniform(0, 1, N) * np.exp(1j * r.uniform(0, 2 * np.pi, N))
    return a.astype(np.complex128), b.astype(np.complex128)


def _comb_bins_lo(N):
    n = np.arange(N, dtype=np.int64)
    return 3 * (n - N // 2) + 1


def _sym_td(vec, bins, L):
    X = np.zeros(L, np.complex128)
    X[np.mod(bins, L)] = vec
    return np.fft.ifft(X)


def _sync(L, seed):
    h = L // 8
    b = np.concatenate([np.arange(-h, 0), np.arange(1, h + 1)])
    ra = np.random.default_rng([int(seed), 777])
    rb = np.random.default_rng([int(seed), 778])
    sa = _sym_td(np.exp(1j * (np.pi / 4 + np.pi / 2 * ra.integers(0, 4, b.size))), b, L)
    sb = _sym_td(np.exp(1j * (np.pi / 4 + np.pi / 2 * rb.integers(0, 4, b.size))), b, L)
    return sa, sb


class Geometry:
    def __init__(self, d):
        self.N = int(d["vec_N"]); self.L = int(d["fft_len"])
        self.CP = int(d["cp_len"]); self.K0 = int(d["k0"])
        self.g0 = int(d["gap0"]); self.g1 = int(d["gap1"])
        self.seed = int(d["seed"])
        self.kinds = np.asarray(d["slot_kinds"])
        self.sidx = np.asarray(d["slot_seed_idx"], np.int64)
        self.S = len(self.kinds)
        self.di = np.where(self.kinds == "D")[0]
        meta = json.loads(str(d["meta_json"])) if "meta_json" in d.files else {}
        self.meta = meta
        self.papr_clip_db = float(meta.get("papr_clip_db", 0.0) or 0.0)
        self.rf_pad = float(meta.get("rf_atten_db", 30.0))
        self.bins_a = _comb_bins_lo(self.N)
        self.bins_b = self.bins_a + self.K0

    def payload_off(self, s):
        return self.g0 + (self.CP + self.L) * (1 + int(s)) + self.CP


class FilePayloads:
    """直读采集脚本写出的最终发射波形 (已含削波与归一化)."""

    def __init__(self, geo: Geometry, path_lo: str, path_rf: str):
        self.geo = geo
        self.A = np.memmap(path_lo, dtype=np.complex64, mode="r")
        self.B = np.memmap(path_rf, dtype=np.complex64, mode="r")
        need = geo.payload_off(geo.S - 1) + geo.L
        if self.A.size < need or self.B.size < need:
            raise ValueError(f"tx 文件长度不足 ({self.A.size} < {need})")

    def __call__(self, s):
        off = self.geo.payload_off(s)
        return (np.asarray(self.A[off:off + self.geo.L], np.complex128),
                np.asarray(self.B[off:off + self.geo.L], np.complex128))


class RebuildPayloads:
    """从种子流式重建逐 slot 波形, 精确复现采集脚本的归一化与 PAPR 削波.
    O(L) 内存; 初始化做 1~2 遍全帧统计 (N=65536 单核约 1 分钟)."""

    def __init__(self, geo: Geometry, verbose=True):
        self.geo = geo
        g = geo
        self._pre = _sync(g.L, g.seed)
        if verbose:
            print(f"[rebuild] pass1: 统计全帧 RMS (共 {g.S} slot) ...")
        pa = pb = 0.0
        n = 0
        for sym_a, sym_b in self._iter_syms():
            pa += float(np.sum(np.abs(sym_a) ** 2) + np.sum(np.abs(sym_a[-g.CP:]) ** 2))
            pb += float(np.sum(np.abs(sym_b) ** 2) + np.sum(np.abs(sym_b[-g.CP:]) ** 2))
            n += g.L + g.CP
        self.rms_a = math.sqrt(pa / n)
        self.rms_b = math.sqrt(pb / n)
        self.thr = 10.0 ** (g.papr_clip_db / 20.0) if g.papr_clip_db > 0 else None
        self.rms2_a = self.rms2_b = 1.0
        if self.thr is not None:
            if verbose:
                print(f"[rebuild] pass2: 复现 {g.papr_clip_db:.1f} dB 削波后的归一化 ...")
            p2a = p2b = 0.0
            for sym_a, sym_b in self._iter_syms():
                ya = np.minimum(np.abs(sym_a) / self.rms_a, self.thr)
                yb = np.minimum(np.abs(sym_b) / self.rms_b, self.thr)
                p2a += float(np.sum(ya ** 2) + np.sum(ya[-g.CP:] ** 2))
                p2b += float(np.sum(yb ** 2) + np.sum(yb[-g.CP:] ** 2))
            self.rms2_a = math.sqrt(p2a / n)
            self.rms2_b = math.sqrt(p2b / n)

    def _iter_syms(self):
        g = self.geo
        yield self._pre
        for s in range(g.S):
            a, b = _gen_vec_pair(g.N, g.seed, int(g.sidx[s]))
            yield _sym_td(a, g.bins_a, g.L), _sym_td(b, g.bins_b, g.L)

    def _final(self, sym, rms, rms2):
        y = sym / rms
        if self.thr is not None:
            m = np.abs(y)
            y = np.where(m > self.thr, y * (self.thr / np.maximum(m, 1e-30)), y)
            y = y / rms2
        return y

    def __call__(self, s):
        g = self.geo
        a, b = _gen_vec_pair(g.N, g.seed, int(g.sidx[s]))
        return (self._final(_sym_td(a, g.bins_a, g.L), self.rms_a, self.rms2_a),
                self._final(_sym_td(b, g.bins_b, g.L), self.rms_b, self.rms2_b))


def _selfcheck_reader(geo: Geometry, reader, d, n_check=3, tol=1e-3, verbose=True):
    """理想乘法混频应精确复现 c_true —— 确认波形与 NPZ 同源/约定一致."""
    di = geo.di[:: max(1, len(geo.di) // n_check)][:n_check]
    Y = np.empty(len(di), complex)
    c = (np.asarray(d["c_true_norm"][0], np.complex128) * math.sqrt(geo.N))[
        np.searchsorted(geo.di, di)]
    for i, s in enumerate(di):
        a_t, b_t = reader(int(s))
        Y[i] = np.fft.fft(b_t * np.conj(a_t))[geo.K0 % geo.L]
    g = np.sum(Y * np.conj(c)) / np.sum(np.abs(c) ** 2)
    dev = float(np.sqrt(np.mean(np.abs(Y / g - c) ** 2)) / math.sqrt(geo.N))
    ok = dev < tol
    if verbose:
        print(f"[波形] 一致性自检: ideal 混频复现偏差 = {dev:.2e} "
              f"({'OK' if ok else '不匹配'})")
    return ok


def _locate_payloads(geo: Geometry, npz_path: str, args, verbose=True):
    cand = []
    if args.tx_lo and args.tx_rf:
        cand.append((args.tx_lo, args.tx_rf))
    for base in (Path(npz_path).resolve().parent,
                 _data_dir(__file__), Path.cwd()):
        cand.append((str(base / "tx_lo.cf32"), str(base / "tx_rf.cf32")))
    d = np.load(npz_path, allow_pickle=False)
    for pa, pb in cand:
        if os.path.isfile(pa) and os.path.isfile(pb):
            try:
                rd = FilePayloads(geo, pa, pb)
            except Exception as exc:
                if verbose:
                    print(f"[波形] 跳过 {pa}: {exc}")
                continue
            if _selfcheck_reader(geo, rd, d, verbose=verbose):
                if verbose:
                    print(f"[波形] 使用发射波形文件: {pa}")
                return rd
            if verbose:
                print(f"[波形] {pa} 与本 NPZ 不一致, 忽略.")
    if verbose:
        print("[波形] 未找到匹配 tx 文件, 采用流式重建 (含削波复现).")
    rd = RebuildPayloads(geo, verbose=verbose)
    assert _selfcheck_reader(geo, rd, d, verbose=verbose), \
        "重建波形未通过一致性自检: 采集脚本的波形构造约定可能已改变"
    return rd


# ---------------- 失真基底 / 交叉验证 ----------------

def compute_bases(d, geo: Geometry, reader, twin: DigitalTwin,
                  p_lo_mixer_dbm: float, verbose=True):
    di = geo.di
    Q = len(di)
    sqN = math.sqrt(geo.N)
    c_true = np.asarray(d["c_true_norm"][0], np.complex128)
    c_raw = c_true * sqN

    have_exact = ("mixer_dev_lim" in d.files) and ("mixer_dev_sq" in d.files)
    Y_tw = np.empty(Q, np.complex128)
    Y_lim = None if have_exact else np.empty(Q, np.complex128)
    Y_sq = None if have_exact else np.empty(Q, np.complex128)
    if verbose:
        src = "NPZ 内置" if have_exact else "现场计算"
        print(f"[基底] 逐 slot 计算孪生失真基底 (Q={Q}; limiter/square: {src}) ...")
    for q, s in enumerate(di):
        a_t, b_t = reader(int(s))
        m = np.abs(a_t)
        p_inst = p_lo_mixer_dbm + 20.0 * np.log10(np.maximum(m, 1e-12))
        ratio = np.sqrt(np.maximum(twin.comp_ratio(p_inst, port="LO"), 0.0))
        Y_tw[q] = np.fft.fft(b_t * np.conj(a_t * ratio))[geo.K0 % geo.L]
        if not have_exact:
            lo1 = np.where(m > 1e-12, a_t / np.maximum(m, 1e-30), 0.0)
            Y_lim[q] = np.fft.fft(b_t * np.conj(lo1))[geo.K0 % geo.L]
            Y_sq[q] = np.fft.fft(b_t * np.conj(a_t * m))[geo.K0 % geo.L]

    def _dev(Y):
        g = np.sum(Y * np.conj(c_raw)) / max(np.sum(np.abs(c_raw) ** 2), 1e-30)
        return (Y / g - c_raw) / sqN

    cols = [_dev(Y_tw)]
    names = ["twin"]
    if have_exact:
        cols += [np.asarray(d["mixer_dev_lim"], np.complex128)[di],
                 np.asarray(d["mixer_dev_sq"], np.complex128)[di]]
    else:
        cols += [_dev(Y_lim), _dev(Y_sq)]
    names += ["limiter", "square"]
    cols.append(c_true.copy())        # 复增益列: 吸收残余整体增益误差
    names.append("gain")
    return np.stack(cols, axis=1), names


def _cv_correct(r, B, n_folds, rng):
    Q = len(r)
    out = np.zeros(Q, np.complex128)
    coefs = []
    for f in np.array_split(rng.permutation(Q), n_folds):
        mk = np.ones(Q, bool)
        mk[f] = False
        beta, *_ = np.linalg.lstsq(B[mk], r[mk], rcond=None)
        out[f] = r[f] - B[f] @ beta
        coefs.append(beta)
    return out, np.mean(np.stack(coefs), axis=0)


def _stat(v, mode, n):
    v = np.asarray(v, float)
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
    if mode == "sem" and v.size > 1:
        sd = sd / math.sqrt(v.size)
    return m, sd


def _fmt(m, e):
    """GUM 规范: 不确定度保留 1-2 位有效数字 (首位为 1 时保留 2 位),
    均值四舍五入到与不确定度末位相同的十进制位; 处理进位换位 (0.096->0.10)."""
    m = float(m)
    if not np.isfinite(e) or e <= 0:
        return f"{m:.4g}"
    e = float(e)
    for _ in range(2):                       # 第二轮处理进位后位数变化
        exp = math.floor(math.log10(e))
        sf = 2 if int(e / 10 ** exp) == 1 else 1
        dec = (sf - 1) - exp
        er = round(e, dec) if dec > 0 else round(e / 10 ** (-dec)) * 10 ** (-dec)
        if er == e:
            break
        e = er
    if dec > 0:
        return f"{round(m, dec):.{dec}f} ± {e:.{dec}f}"
    q = 10 ** (-dec)
    return f"{round(m / q) * q:.0f} ± {round(e / q) * q:.0f}"


def optimize_one(twin: DigitalTwin, npz_path: str, args):
    d = np.load(npz_path, allow_pickle=False)
    geo = Geometry(d)
    labels = [str(x) for x in d["labels"]]
    c_true = np.asarray(d["c_true_norm"][0], np.complex128)
    if "c_hat_reps" in d.files:
        reps = np.asarray(d["c_hat_reps"], np.complex128)      # (P, R, Q)
    else:
        reps = np.asarray(d["c_hat"], np.complex128)[:, None, :]
    P, R, Q = reps.shape
    ebmode = str(geo.meta.get("errorbar", "std") or "std")
    std_y = float(np.std(c_true))
    print(f"\n############ 数据集: N={geo.N}  ({npz_path}) ############")
    print(f"[数据] {R} 次独立重复抓取; 误差棒 = 1 {'s.d.' if ebmode=='std' else 's.e.m.'} (ddof=1)")
    if "mixer_comp" in d.files and np.any(np.asarray(d["mixer_comp"])):
        print("[!] 该 NPZ 采集时已开 --mixer-comp, 基底系数会较小, 结果仍有效.")

    p_lo = float(np.mean(d["p_lo_dbm_tx"]))
    for p in range(P):
        reg = twin.operating_regime(p_lo, float(d["p_rf_dbm_tx"][p]) - geo.rf_pad)
        print(f"[twin] [{labels[p]}] α_L={reg['alpha_L']:.3f}, "
              f"LO 压缩比={reg['lo_comp_ratio']:.3f}, RF 压缩比={reg['rf_comp_ratio']:.3f}")

    reader = _locate_payloads(geo, npz_path, args)
    B, names = compute_bases(d, geo, reader, twin, p_lo)
    for p in range(P):
        r0 = reps[p, 0] - c_true
        cc = ", ".join(f"{nm}={np.real(np.sum(r0*np.conj(B[:,i]))/max(np.linalg.norm(r0)*np.linalg.norm(B[:,i]),1e-30)):+.2f}"
                       for i, nm in enumerate(names[:-1]))
        print(f"[诊断] [{labels[p]}][rep0] 残差-基底相关: {cc}")

    after = np.zeros_like(reps)
    twin1 = np.zeros_like(reps)
    coef_tab = []
    for p in range(P):
        for r in range(R):
            res_pr = reps[p, r] - c_true
            r2, beta = _cv_correct(res_pr, B, args.folds,
                                   np.random.default_rng([args.cv_seed, p, r]))
            after[p, r] = c_true + r2
            r1, _ = _cv_correct(res_pr, B[:, [0]], args.folds,
                                np.random.default_rng([args.cv_seed, p, r]))
            twin1[p, r] = c_true + r1
            if r == 0:
                coef_tab.append(beta)

    rmse_pr = lambda X: np.sqrt(np.mean(np.abs(X - c_true[None, None, :]) ** 2, axis=2))
    rb, rt, ra = rmse_pr(reps), rmse_pr(twin1), rmse_pr(after)      # (P, R)
    enob = lambda rr: np.log2(np.maximum(std_y, 1e-300) / np.maximum(rr, 1e-300))
    res = dict(
        vec_N=geo.N, labels=labels, c_true=c_true, basis_names=names,
        errorbar=ebmode, n_repeats=R, std_y=std_y,
        c_hat_before_reps=reps, c_hat_after_reps=after, c_hat_twin_reps=twin1,
        coeffs=np.stack(coef_tab),
        rmse_before_reps=rb, rmse_twin_reps=rt, rmse_after_reps=ra,
        enob_before_reps=enob(rb), enob_after_reps=enob(ra),
        rmse_awgn=np.array([1.0 / math.sqrt(27.0 * 10.0 ** (float(x) / 10.0))
                            for x in d["snr3_db"]]),
    )
    for key in ("rmse_before", "rmse_twin", "rmse_after", "enob_before", "enob_after"):
        M = res[key + "_reps"]
        stats = [_stat(M[p], ebmode, R) for p in range(P)]
        res[key + "_mean"] = np.array([x[0] for x in stats])
        res[key + "_err"] = np.array([x[1] for x in stats])

    print(f"  {'面板':>8s}  {'优化前 RMSE':>18s}  {'联合优化 RMSE':>18s}  "
          f"{'AWGN':>7s}  {'ENOB 前->后 (bit)':>24s}")
    for p, lab in enumerate(labels):
        print(f"  {lab:>8s}  {_fmt(res['rmse_before_mean'][p], res['rmse_before_err'][p]):>18s}  "
              f"{_fmt(res['rmse_after_mean'][p], res['rmse_after_err'][p]):>18s}  "
              f"{res['rmse_awgn'][p]:7.4f}  "
              f"{_fmt(res['enob_before_mean'][p], res['enob_before_err'][p])} -> "
              f"{_fmt(res['enob_after_mean'][p], res['enob_after_err'][p])}")
    return res


# ---------------- 出图 / 保存 ----------------

def plot_before_after(res, save_path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c_true = res["c_true"]
    labels = res["labels"]
    n = len(labels)
    rows = [("Before", res["c_hat_before_reps"],
             res["rmse_before_mean"], res["rmse_before_err"],
             res["enob_before_mean"], res["enob_before_err"]),
            ("After", res["c_hat_after_reps"],
             res["rmse_after_mean"], res["rmse_after_err"],
             res["enob_after_mean"], res["enob_after_err"])]
    fig, axes = plt.subplots(2, n, figsize=(3.35 * n + 1.35, 6.9))
    axes = np.asarray(axes).reshape(2, n)
    for r, (rname, chat, rm, re_, em, ee) in enumerate(rows):
        for p in range(n):
            ax = axes[r, p]
            ax.axhline(0, color="0.88", lw=0.8, ls=":", zorder=0)
            ax.axvline(0, color="0.88", lw=0.8, ls=":", zorder=0)
            ax.plot([-1, 1], [-1, 1], "k--", lw=1.0, zorder=1)
            ct = np.tile(c_true, chat.shape[1])          # 并入全部重复点
            ch = chat[p].reshape(-1)
            ax.scatter(ct.real, ch.real, marker="+", s=38,
                       color="#2fb6a8", linewidths=1.1, label="real", zorder=3)
            ax.scatter(ct.imag, ch.imag, marker="o", s=11,
                       color="salmon", alpha=0.8, edgecolors="none",
                       label="imag", zorder=2)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=9)
            ax.set_title(f"RMSE: {_fmt(rm[p], re_[p])}\n"
                         f"ENOB: {_fmt(em[p], ee[p])} bit", fontsize=9.5)
            ax.text(0.95, 0.05, labels[p], transform=ax.transAxes,
                    ha="right", va="bottom", fontweight="bold", fontsize=12)
            if r == 0 and p == 0:
                ax.legend(loc="upper left", frameon=False, fontsize=9,
                          handletextpad=0.2)
            if p == 0:
                ax.set_ylabel(r"$\hat{c} = \widehat{(\mathbf{a},\,\mathbf{b})}/\sqrt{N}$",
                              fontsize=10)
                ax.annotate(rname, xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-0.60, 0.5), textcoords="axes fraction",
                            fontsize=17, ha="center", va="center")
            if r == 1:
                ax.set_xlabel(r"$c = \langle\mathbf{a},\,\mathbf{b}\rangle/\sqrt{N}$",
                              fontsize=10)
    fig.subplots_adjust(left=0.215, right=0.985, top=0.925, bottom=0.085,
                        hspace=0.38, wspace=0.16)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[图] {save_path}")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def save_result(res, out_path):
    kw = dict(vec_N=res["vec_N"], c_true=res["c_true"],
              labels=np.array(res["labels"]),
              basis_names=np.array(res["basis_names"]), coeffs=res["coeffs"],
              errorbar=res["errorbar"], n_repeats=res["n_repeats"],
              std_y=res["std_y"], rmse_awgn=res["rmse_awgn"],
              c_hat_before_reps=res["c_hat_before_reps"],
              c_hat_after_reps=res["c_hat_after_reps"],
              c_hat_twin_reps=res["c_hat_twin_reps"])
    for key in ("rmse_before", "rmse_twin", "rmse_after",
                "enob_before", "enob_after"):
        kw[key + "_reps"] = res[key + "_reps"]
        kw[key + "_mean"] = res[key + "_mean"]
        kw[key + "_err"] = res[key + "_err"]
    kw["c_hat_before"] = res["c_hat_before_reps"][:, 0]
    kw["c_hat_after"] = res["c_hat_after_reps"][:, 0]
    if len(res["labels"]) >= 2:                    # 兼容键: 取 rep0 校正结果
        kw["optimized_ip_15dB"] = res["c_hat_after_reps"][0, 0]
        kw["optimized_ip_25dB"] = res["c_hat_after_reps"][1, 0]
    np.savez(out_path, **kw)
    print(f"[输出] {out_path}")


def _discover():
    here = _data_dir(__file__)
    hits = []
    for pat in ("gr_fig3c_ip_scatter.npz", "*/gr_fig3c_ip_scatter.npz",
                "data/*/gr_fig3c_ip_scatter.npz"):
        hits += glob.glob(str(here / pat))
    hits = sorted(set(hits), key=os.path.getmtime)
    return hits


def main():
    ap = argparse.ArgumentParser(description="数字孪生内积后端优化 (v3, 批处理)")
    ap.add_argument("--npz", nargs="*", default=None,
                    help="一个或多个实测 NPZ (缺省: 自动发现本目录及子目录)")
    ap.add_argument("--twin", default=None, help="孪生 summary.json (缺省用内置 PIML)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=0)
    ap.add_argument("--tx-lo", default=None)
    ap.add_argument("--tx-rf", default=None)
    ap.add_argument("--show", action="store_true", help="出图后弹窗显示")
    ap.add_argument("--replot", default=None,
                    help="仅重画: 给 inner_product_optimized_N*.npz 路径")
    args = ap.parse_args()

    if args.replot:
        d = np.load(args.replot, allow_pickle=False)
        res = {k: d[k] for k in ("c_true", "c_hat_before_reps", "c_hat_after_reps",
                                 "rmse_before_mean", "rmse_before_err",
                                 "rmse_after_mean", "rmse_after_err",
                                 "enob_before_mean", "enob_before_err",
                                 "enob_after_mean", "enob_after_err")}
        res["labels"] = [str(x) for x in d["labels"]]
        plot_before_after(res, Path(args.replot).with_suffix(".png"), show=args.show)
        return

    paths = args.npz or _discover()
    if not paths:
        raise SystemExit("未找到任何 gr_fig3c_ip_scatter.npz; 请用 --npz 指定")
    twin = (DigitalTwin.from_summary_json(args.twin) if args.twin
            else DigitalTwin(PIML_TWIN_PARAMS))

    for p in paths:
        res = optimize_one(twin, p, args)
        stem = f"inner_product_optimized_N{res['vec_N']}"
        save_result(res, stem + ".npz")
        plot_before_after(res, stem + ".png", show=args.show)


if __name__ == "__main__":
    main()
