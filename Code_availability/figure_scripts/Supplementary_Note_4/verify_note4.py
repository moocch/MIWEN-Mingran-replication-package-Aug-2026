# -*- coding: utf-8 -*-
"""Verify the numbers of Supplementary Note 4 (carrier coherence, inner-product
recovery, measured noise floor, correction controls) from archived bytes.

Inputs (nothing else):
  ../note3_client_energy/raw/<run>/gr_fig3c_ip_scatter.npz   raw campaign data
  ../note3_client_energy/raw/1_inner_product_scatter_v4*.py  acquisition code
  ./ip_optimized_N65536_20260826.npz, ./ip_optimized_N4096_20260826.npz
      per-capture corrected/uncorrected/twin-only decodes (c_hat_before_reps,
      c_hat_after_reps, c_hat_twin_reps, c_true) from the 2026-08-26
      out-of-sample correction run (fig3/d chain).

The S4.2 transmit-side facts (PAPR clip, comb geometry) are re-derived by
regenerating the transmitted frames from the stored seed with the exact
synthesis of the archived acquisition script; the regeneration is proven
byte-faithful by matching all 226 stored true inner products to ~1e-13.

Run:  python verify_note4.py     (numpy only; ~30 s: the N=65536 frame is
rebuilt from seed; exits nonzero on any mismatch)
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir, code_dir as _code_dir
# ---------------------------------------------------------------------------

import json
import math
import os
import re

import numpy as np

HERE = str(_data_dir(__file__))
RAW_DIR = os.path.join(os.path.dirname(HERE), "Supplementary_Note_3", "raw")
RAW = {
    4096: os.path.join(RAW_DIR, "gr_fig3c_ip_scatter_20260810_011915_N4096",
                       "gr_fig3c_ip_scatter.npz"),
    65536: os.path.join(RAW_DIR, "gr_fig3c_ip_scatter_20260810_002043_N65536",
                        "gr_fig3c_ip_scatter.npz"),
}
OPT = {
    4096: os.path.join(HERE, "ip_optimized_N4096_20260826.npz"),
    65536: os.path.join(HERE, "ip_optimized_N65536_20260826.npz"),
}
ACQ_SCRIPT = os.path.join(str(_code_dir(__file__)), "..",
                          "Supplementary_Note_3", "raw",
                          "1_inner_product_scatter_v4.py")

n_pass = 0
warnings_ = []


def check(name, ok, detail=""):
    global n_pass
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    assert ok, name
    n_pass += 1


def note(name, detail):
    warnings_.append((name, detail))
    print(f"[NOTE] {name}  ({detail})")


# ===================================================================== S4.2
# frame regeneration, exactly as in the archived acquisition script
# (comb_bins_lo / gen_vec_pair / symbol_td_from_bins / gen_sync_symbols /
#  build_frame of 1_inner_product_scatter_v4.py)

def comb_bins_lo(N):
    n = np.arange(N, dtype=np.int64)
    return 3 * (n - N // 2) + 1


def gen_vec_pair(N, base_seed, slot_idx):
    rng = np.random.default_rng([int(base_seed), int(slot_idx)])
    ra = rng.uniform(0.0, 1.0, N)
    pa = rng.uniform(0.0, 2.0 * np.pi, N)
    rb = rng.uniform(0.0, 1.0, N)
    pb = rng.uniform(0.0, 2.0 * np.pi, N)
    return ra * np.exp(1j * pa), rb * np.exp(1j * pb)


def symbol_td(vec, bins, L):
    X = np.zeros(L, np.complex128)
    X[np.mod(bins, L)] = vec
    return np.fft.ifft(X)


def gen_sync(L, seed, band_frac=8):
    half = L // band_frac
    bins = np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])
    out = []
    for salt in (777, 778):
        rng = np.random.default_rng([int(seed), salt])
        q = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * rng.integers(0, 4, bins.size)))
        out.append(symbol_td(q, bins, L))
    return out


def regen_frame_stats(N, L, cp, k0, gap0, gap1, seed, q_data=200, pilot_every=8):
    """Rebuild both transmitted combs; return per-comb clip stats + true IPs."""
    bins_a = comb_bins_lo(N)
    bins_b = bins_a + k0
    kinds, nd = [], 0
    while nd < q_data:
        kinds.append("P")
        for _ in range(pilot_every):
            if nd < q_data:
                kinds.append("D")
                nd += 1
    kinds.append("P")
    sa_pre, sb_pre = gen_sync(L, seed)
    syms = {"LO": [np.concatenate([sa_pre[-cp:], sa_pre])],
            "RF": [np.concatenate([sb_pre[-cp:], sb_pre])]}
    c_true = []
    for s, kind in enumerate(kinds):
        si = s if kind == "D" else 100000 + s
        a, b = gen_vec_pair(N, seed, si)
        c_true.append(np.vdot(a, b))
        ta, tb = symbol_td(a, bins_a, L), symbol_td(b, bins_b, L)
        syms["LO"].append(np.concatenate([ta[-cp:], ta]))
        syms["RF"].append(np.concatenate([tb[-cp:], tb]))
    frame_len = gap0 + sum(x.size for x in syms["LO"]) + gap1
    thr = 10.0 ** (10.0 / 20.0)          # papr_clip_db = 10 (meta_json)
    stats = {}
    for tag in ("LO", "RF"):
        act_pow = sum(float((np.abs(x) ** 2).sum()) for x in syms[tag])
        act_n = sum(x.size for x in syms[tag])
        rms = math.sqrt(act_pow / act_n)  # unit-RMS normalization (active part)
        nclip, eclip = 0, 0.0
        for x in syms[tag]:
            m = np.abs(x) / rms
            nclip += int((m > thr).sum())
            eclip += float(((m - thr)[m > thr] ** 2).sum())
        stats[tag] = dict(frac_samples=nclip / frame_len,
                          frac_energy=eclip / act_n,
                          kmax=int(max(np.max(np.abs(bins_a if tag == "LO" else bins_b)),
                                       0)))
    kmin = int(min(bins_a.min(), bins_b.min()))
    kmax = int(max(bins_a.max(), bins_b.max()))
    return stats, np.array(c_true), frame_len, (kmin, kmax)


acq_src = open(ACQ_SCRIPT, encoding="utf-8").read()

print("===== S4.1/S4.2: synchronisation quality and frame geometry =====")
geo_exp = {4096: (16384, 512, 6, 610.35), 65536: (262144, 8192, 96, 38.15)}
raw_d = {}
for N in (4096, 65536):
    d = np.load(RAW[N], allow_pickle=True)
    raw_d[N] = d
    L_exp, cp_exp, k0_exp, df_exp = geo_exp[N]
    L, cp, k0 = int(d["fft_len"]), int(d["cp_len"]), int(d["k0"])
    fs = float(d["fs_hz"])
    check(f"N={N}: {L_exp}-point symbols, {cp_exp}-sample CP",
          L == L_exp and cp == cp_exp)
    check(f"N={N}: subcarrier spacing {df_exp} Hz",
          abs(fs / L - df_exp) < 0.005, f"fs/L = {fs/L:.4f} Hz")
    check(f"N={N}: output bin k0={k0_exp} on the every-3rd-bin grid, +3.66 kHz",
          k0 == k0_exp and k0 % 3 == 0 and abs(k0 * fs / L - 3662.1) < 0.5,
          f"k0*fs/L = {k0*fs/L:.1f} Hz")
    check(f"N={N}: identical geometry ratios across sizes (CP=L/32, gap=L/2)",
          cp * 32 == L and int(d["gap0"]) * 2 == L and int(d["gap1"]) * 2 == L)
    check(f"N={N}: 26 pilot slots interleaved one-per-eight with 200 data slots",
          int(d["n_pilot"]) == 26 and int(d["n_data"]) == 200
          and json.loads(str(d["meta_json"]))["pilot_every"] == 8)

    pm = np.asarray(d["peak_metric"], float)
    check(f"N={N}: preamble peak-to-sidelobe ratios in 28.6-31.4",
          bool(np.all(pm >= 28.6) and np.all(pm <= 31.45)),
          "measured " + np.array2string(pm, precision=2))
    dr = np.abs(np.asarray(d["drift_rad_per_slot"], float))
    check(f"N={N}: fitted pilot drift slopes < 3.2e-3 rad/slot",
          bool(np.all(dr < 3.2e-3)), "|slope| " + np.array2string(dr, precision=6))
    check(f"N={N}: receiver ADC clipping fraction zero",
          bool(np.all(np.asarray(d["clip_frac"]) == 0.0)))

check("acceptance threshold 3 for the sync peak (archived acquisition code)",
      re.search(r"peak_metric\s*<\s*3\.0", acq_src) is not None
      and "sync peak weak" in acq_src)
check("noise estimated from 4 guard bins at k0+-4, k0+-5 (archived code)",
      re.search(r"k0\s*-\s*5,\s*k0\s*-\s*4,\s*k0\s*\+\s*4,\s*k0\s*\+\s*5", acq_src)
      is not None)
pm_all = np.concatenate([np.asarray(raw_d[N]["peak_metric"], float)
                         for N in (4096, 65536)])
check("PSR range endpoints quote as 28.6 and 31.4",
      round(float(pm_all.min()), 1) == 28.6 and round(float(pm_all.max()), 1) == 31.4,
      f"min {pm_all.min():.3f}, max {pm_all.max():.3f}")

print("\n===== S4.2: transmitted combs regenerated from the stored seed =====")
clip_all = {}
for N in (4096, 65536):
    d = raw_d[N]
    meta = json.loads(str(d["meta_json"]))
    check(f"N={N}: digital clip commanded at 10-dB PAPR (meta_json)",
          float(meta["papr_clip_db"]) == 10.0)
    stats, c_true, frame_len, (kmin, kmax) = regen_frame_stats(
        N, int(d["fft_len"]), int(d["cp_len"]), int(d["k0"]),
        int(d["gap0"]), int(d["gap1"]), int(d["seed"]))
    dev = float(np.max(np.abs(c_true - d["slot_c_true"])))
    check(f"N={N}: regenerated frame reproduces all 226 stored true inner "
          f"products", dev < 1e-9 and frame_len == int(d["frame_len"]),
          f"max |dev| = {dev:.2e}")
    span_mhz = (kmax - kmin) * float(d["fs_hz"]) / int(d["fft_len"]) / 1e6
    nyq_frac = kmax / (int(d["fft_len"]) / 2.0)
    check(f"N={N}: combs occupy ~7.5 MHz = 75% of Nyquist",
          7.4 < span_mhz < 7.6 and 0.74 < nyq_frac < 0.76,
          f"span {span_mhz:.3f} MHz, |k|max/(L/2) = {nyq_frac:.4f}")
    for tag in ("LO", "RF"):
        clip_all[(N, tag)] = stats[tag]
        check(f"N={N} {tag} comb: clipping touches a <0.03% sliver of samples",
              stats[tag]["frac_samples"] < 3e-4,
              f"samples {stats[tag]['frac_samples']*100:.4f}%, "
              f"energy {stats[tag]['frac_energy']*100:.4f}%")
fs_frac = [v["frac_samples"] for v in clip_all.values()]
fe_frac = [v["frac_energy"] for v in clip_all.values()]
note("S4.2 quotes the clip as 'touching 0.006% of samples'; byte-exact "
     "regeneration gives touched-SAMPLE fractions "
     + ", ".join(f"{x*100:.4f}%" for x in fs_frac)
     + " (0.019-0.021%) and clipped-ENERGY fractions "
     + ", ".join(f"{x*100:.4f}%" for x in fe_frac)
     + " (0.005%)",
     "the 0.006% figure matches neither exactly; nearest is the clipped-energy"
     " fraction - see README (likely erratum: energy fraction, not samples)")

print("\n===== S4.3: measured noise floor and sigma2_bin control =====")
for N in (4096, 65536):
    d = raw_d[N]
    s2 = np.asarray(d["sigma2_bin"], float)
    ddb = abs(10.0 * math.log10(s2[1] / s2[0]))
    pdiff = float(d["p_rf_dbm_tx"][1] - d["p_rf_dbm_tx"][0])
    check(f"N={N}: sigma2_bin agrees between operating points within 0.6 dB",
          ddb < 0.6, f"{ddb:.3f} dB apart")
    check(f"N={N}: while the data-comb power differs by ~10 dB",
          9.5 < pdiff < 10.6, f"{pdiff:.2f} dB")

snr = {N: np.asarray(raw_d[N]["snr3_db_reps"], float) for N in (4096, 65536)}
u15 = np.concatenate([snr[N][0] for N in (4096, 65536)])
u25 = np.concatenate([snr[N][1] for N in (4096, 65536)])
check("guard-bin SNRs 15.14-15.44 dB (union, three repeats each panel)",
      round(float(u15.min()), 2) == 15.14 and round(float(u15.max()), 2) == 15.44,
      f"min {u15.min():.4f}, max {u15.max():.4f}")
check("guard-bin SNRs 24.37-25.20 dB (union over both vector lengths)",
      round(float(u25.min()), 2) == 24.37 and round(float(u25.max()), 2) == 25.20,
      f"min {u25.min():.4f}, max {u25.max():.4f}")
check("N=65536 alone: 25.03-25.20 dB",
      round(float(snr[65536][1].min()), 2) == 25.03
      and round(float(snr[65536][1].max()), 2) == 25.20,
      f"min {snr[65536][1].min():.4f}, max {snr[65536][1].max():.4f}")

print("\n===== S4.3: bias-corrected repeat decomposition =====")
# r_iq = d_q + n_iq;  sigma^2 = <(1/(R-1)) sum_i |r_iq - rbar_q|^2>_q ;
# |d|^2 = <|rbar_q|^2>_q - sigma^2/R.  Floors: 1/sqrt(27*SNR) at the panel's
# mean guard-bin SNR (closed form of the normalized receiver noise floor).
EXP_S43 = {  # (sigma, floor, |d|, frac_after_%, frac_before_%)
    (65536, 0): (0.0324, 0.0331, 0.0156, 18.9, 74.6),
    (65536, 1): (0.0108, 0.0107, 0.0134, 60.6, 96.3),
    (4096, 0): (None, None, 0.0163, 22.0, 79.9),
    (4096, 1): (None, None, 0.0158, 71.5, 97.3),
}
ratios = {}
for N in (65536, 4096):
    o = np.load(OPT[N])
    g = raw_d[N]
    tru = o["c_true"]
    check(f"N={N}: optimized npz true values == raw npz c_true_norm "
          f"(archives agree)",
          bool(np.allclose(tru, g["c_true_norm"][0], atol=1e-12))
          and bool(np.allclose(tru, g["c_true_norm"][1], atol=1e-12)))
    check(f"N={N}: c_hat_before_reps == raw c_hat_reps (raw-side residuals "
          f"identical in both archives)",
          bool(np.allclose(o["c_hat_before_reps"], g["c_hat_reps"], atol=1e-12)))
    for p, lbl in ((0, "15 dB"), (1, "25 dB")):
        snr_db = float(np.mean(g["snr3_db_reps"][p]))
        floor = 1.0 / math.sqrt(27.0 * 10.0 ** (snr_db / 10.0))
        res = {}
        for stage, src in (("after", o["c_hat_after_reps"]),
                           ("before", g["c_hat_reps"])):
            r = src[p] - tru[None, :]
            R = r.shape[0]
            rbar = r.mean(axis=0)
            sig2 = ((np.abs(r - rbar[None, :]) ** 2).sum(axis=0) / (R - 1)).mean()
            d2 = max((np.abs(rbar) ** 2).mean() - sig2 / R, 0.0)
            res[stage] = (math.sqrt(sig2), math.sqrt(d2),
                          100.0 * d2 / float(np.mean(np.abs(r) ** 2)))
        sa, da, fa = res["after"]
        _, _, fb = res["before"]
        e_sig, e_floor, e_d, e_fa, e_fb = EXP_S43[(N, p)]
        if e_sig is not None:
            check(f"N={N} {lbl}: corrected sigma-hat = {e_sig}",
                  abs(sa - e_sig) < 5e-4, f"computed {sa:.4f}")
            check(f"N={N} {lbl}: closed-form floor = {e_floor}",
                  abs(floor - e_floor) < 5e-4, f"computed {floor:.4f}")
            ratios[p] = sa / floor
        check(f"N={N} {lbl}: reproducible |d-hat| = {e_d}",
              abs(da - e_d) < 5e-4, f"computed {da:.4f}")
        check(f"N={N} {lbl}: |d|^2 carries {e_fa}% of corrected residual power",
              abs(fa - e_fa) < 0.15, f"computed {fa:.1f}%")
        check(f"N={N} {lbl}: uncorrected deterministic fraction {e_fb}% "
              f"(from the raw npz repeats)",
              abs(fb - e_fb) < 0.15, f"computed {fb:.1f}%")
check("corrected random component sits at 0.98x / 1.01x the measured floor",
      round(ratios[0], 2) == 0.98 and round(ratios[1], 2) == 1.01,
      f"ratios {ratios[0]:.4f}, {ratios[1]:.4f}")

print("\n===== S4.4: controls on the out-of-sample correction =====")
# Template correlation. The scale-invariant correlation |<t,r>|/(|t||r|)
# between the twin-template component actually removed by the twin-only
# correction (c_hat_before - c_hat_twin, proportional to the replayed
# template; correlation magnitude is invariant to the fitted complex
# coefficient) and the raw residual (c_hat_before - c_true), per capture,
# averaged over the three captures of each of the four operating points.
rhos = {}
for N in (65536, 4096):
    o = np.load(OPT[N])
    tru = o["c_true"]
    for p, lbl in ((0, "15 dB"), (1, "25 dB")):
        t = o["c_hat_before_reps"][p] - o["c_hat_twin_reps"][p]
        r = o["c_hat_before_reps"][p] - tru[None, :]
        cc = [abs(np.vdot(t[i], r[i]))
              / math.sqrt(float((np.abs(t[i]) ** 2).sum())
                          * float((np.abs(r[i]) ** 2).sum()))
              for i in range(t.shape[0])]
        rhos[(N, lbl)] = float(np.mean(cc))
rv = list(rhos.values())
check("twin template vs measured residuals: rho = 0.79-0.91 across all four "
      "operating points",
      all(0.785 <= x <= 0.915 for x in rv)
      and round(min(rv), 2) == 0.79 and round(max(rv), 2) == 0.91,
      "; ".join(f"N={k[0]} {k[1]}: {v:.3f}" for k, v in rhos.items()))

# Calibrated twin compression ratio at the -3 dBm LO drive: pure algebra on
# the frozen twin constants (PIML_TWIN_PARAMS of the archived correction
# script, fig3/d chain): PcompLO = -5.1990871823 dBm, betaC = 0.6101367971,
# eta(P) = (1 + (P/Pcomp)^betaC)^(-1/betaC).
P_COMP_LO_DBM = -5.1990871823
BETA_C = 0.6101367971
pw = 1e-3 * 10.0 ** (-3.0 / 10.0)
pcw = 1e-3 * 10.0 ** (P_COMP_LO_DBM / 10.0)
eta = (1.0 + (pw / pcw) ** BETA_C) ** (-1.0 / BETA_C)
check("twin compression ratio at -3 dBm LO drive = 0.24",
      round(eta, 2) == 0.24, f"computed {eta:.4f}")

print(f"\nAll {n_pass} checks passed.")
if warnings_:
    print(f"{len(warnings_)} documented item(s) NOT asserted (see [NOTE] above "
          "and README.md).")
