#!/usr/bin/env python3
"""Serial per-product ENOB at (0,0), offline (2026-08-24): naive
(vs ideal multiplier) versus TWIN-INVERTED (deterministic curve
removed by per-pair inversion — legitimate pre-sum, at the pair level).
Data: the validated 256-pair diagnostic capture. The gap between the
two curves = the precision recoverable by training through the twin.

ENOB convention: per-|xw|-bin SINAD against the full used product
range, ENOB = (SINAD - 1.76)/6.02 (ADC convention). Single capture:
error includes noise + residual model error together (inseparable
here; Stage A adds repeats).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import importlib.util, sys
sys.path.insert(0, '.')
import sync_v2
spec = importlib.util.spec_from_file_location("core",
                                              "4_gtsrb_confusion_mlpN.py")
core = importlib.util.module_from_spec(spec); sys.modules["core"] = core
spec.loader.exec_module(core)

# ---- decode the diagnostic capture (validated convention) -------------
L, CP = 16384, 512; SYM = L + CP; fs = 1e7
NS, TS, GD = 256, 128, 8
rng = np.random.default_rng(2026)
x = rng.standard_normal(NS); w = rng.standard_normal(NS)
xw = x * w
ta, tb = sync_v2._chirps(core, L)
cap = np.load('serial_diag_cap.npy').astype(np.complex128)
c = np.abs(np.correlate(cap, np.conj(ta) * tb, mode='valid'))
lim = len(cap) - SYM - NS * TS
d0 = int(min(k for k in np.argsort(c)[::-1][:6] if k <= lim))
pay = cap[d0 + SYM:d0 + SYM + NS * TS]
t = np.arange(NS * TS) / fs
yh = (pay * np.exp(+2j * np.pi * 1.0e6 * t)).reshape(
    NS, TS)[:, GD:-GD].mean(axis=1)

# ---- twin surface ------------------------------------------------------
TM = json.load(open('serial_twin_model.json'))
thp = np.array(TM["thp"]); rg = np.array(TM["ridges"]).reshape(-1, 4)
DUTY = TM["duty_db"]

def pdb(pl, pr):
    a0, klo, krf = thp[0], thp[1], thp[2]
    xl = 10 ** ((pl - klo) / 10.0); xr = 10 ** ((pr - krf) / 10.0)
    y = a0 + 10 * np.log10((xl / (1 + xl)) * (xr / (1 + xr)) + 1e-30)
    for wi, ai, bi, ci in rg:
        y = y + wi * np.tanh(ai * pl / 40 + bi * pr / 40 + ci)
    return y

xrms = np.sqrt(np.mean(x ** 2)); wrms = np.sqrt(np.mean(w ** 2))
p_lo = 20 * np.log10(np.maximum(np.abs(w) / wrms, 1e-6)) + DUTY
p_rf = 20 * np.log10(np.maximum(np.abs(x) / xrms, 1e-6)) + DUTY
A_twin = 10 ** (pdb(p_lo, p_rf) / 20.0)
pred_twin = np.sign(xw) * A_twin

# ---- global scales -----------------------------------------------------
ym = np.real(yh * np.exp(-1j * np.angle(np.vdot(xw, yh))))  # align phase
g_naive = np.dot(ym, xw) / np.dot(xw, xw)
g_twin = np.dot(ym, pred_twin) / np.dot(pred_twin, pred_twin)

# ---- twin inversion per pair: solve f(x_hat; w) = y, product-referred --
xw_hat = np.zeros(NS)
grid_x = np.linspace(1e-4, np.abs(x).max() * 1.5, 4096)
for i in range(NS):
    pr_g = 20 * np.log10(np.maximum(grid_x / xrms, 1e-6)) + DUTY
    f_g = 10 ** (pdb(p_lo[i], pr_g) / 20.0) * g_twin
    yi = abs(ym[i])
    j = np.searchsorted(f_g, yi)
    if j <= 0:
        xh = grid_x[0]
    elif j >= len(grid_x):
        xh = grid_x[-1]
    else:
        f0, f1 = f_g[j - 1], f_g[j]
        xh = grid_x[j - 1] + (grid_x[j] - grid_x[j - 1]) \
            * (yi - f0) / max(f1 - f0, 1e-30)
    xw_hat[i] = np.sign(ym[i]) * xh * np.abs(w[i]) * np.sign(w[i]) \
        * np.sign(xw[i]) * np.sign(xw[i])  # sign from measurement
    xw_hat[i] = np.sign(ym[i]) * xh * np.abs(w[i])

err_naive = ym / g_naive - xw
err_twin = xw_hat - xw

# ---- binned ENOB -------------------------------------------------------
rng_p = xw.max() - xw.min()          # used product range (full scale)
def enob(err):
    sinad = 10 * np.log10((rng_p ** 2 / 12) / max(np.mean(err ** 2), 1e-30))
    return (sinad - 1.76) / 6.02 + np.log2(np.sqrt(12))  # consistent FS conv

def enob_fs(err):
    # effective bits of full scale: log2(range / (rms_err * sqrt(12)))
    return np.log2(rng_p / (np.sqrt(np.mean(err ** 2)) * np.sqrt(12)))

q = np.quantile(np.abs(xw), np.linspace(0, 1, 9))
bins, e_n, e_t, n_in = [], [], [], []
for a, b in zip(q[:-1], q[1:]):
    s = (np.abs(xw) >= a) & (np.abs(xw) < b)
    if s.sum() < 10:
        continue
    bins.append(0.5 * (a + b))
    e_n.append(enob_fs(err_naive[s]))
    e_t.append(enob_fs(err_twin[s]))
    n_in.append(int(s.sum()))
overall_n, overall_t = enob_fs(err_naive), enob_fs(err_twin)
print(f"overall ENOB (FS-referred): naive {overall_n:.2f} bits, "
      f"twin-inverted {overall_t:.2f} bits")

# ---- figure ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
o = np.argsort(xw)
ax.plot(xw[o], (ym / g_naive)[o], ".", ms=4, color="C3",
        label="measured / naive gain")
ax.plot(xw[o], xw[o], "k--", lw=1, label="ideal")
ax.plot(xw[o], (g_twin * pred_twin / g_naive)[o], ".", ms=3,
        color="C2", alpha=0.7, label="twin prediction (same units)")
ax.set_xlabel("ideal product x·w")
ax.set_ylabel("per-slot readout (linearized units)")
ax.set_title("A: per-product transfer at (0,0) — 256 pairs")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(bins, e_n, "o-", color="C3", label=f"naive (vs ideal): "
        f"overall {overall_n:.2f} b")
ax.plot(bins, e_t, "s-", color="C2", label=f"twin-inverted: "
        f"overall {overall_t:.2f} b")
ax.fill_between(bins, e_n, e_t, color="C2", alpha=0.12,
                label="recoverable by training")
ax.set_xscale("log")
ax.set_xlabel("|x·w| (bin center)")
ax.set_ylabel("ENOB [bits, full-scale referred]")
ax.set_title("B: per-product ENOB — deterministic curve removed vs not")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.suptitle("Serial per-product ENOB at (0,0) — offline, 256-pair "
             "diagnostic capture (single capture: noise+model residual "
             "combined); Stage A extends across drive with repeats",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("serial_enob_00_20260824.png", dpi=150)
print("saved serial_enob_00_20260824.png")
json.dump(dict(overall_naive_bits=float(overall_n),
               overall_twin_bits=float(overall_t),
               bins=bins, enob_naive=e_n, enob_twin=e_t, n=n_in),
          open("serial_enob_00_20260824.json", "w"), indent=1)
