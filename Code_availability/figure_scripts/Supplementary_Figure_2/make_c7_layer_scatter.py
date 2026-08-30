# -*- coding: utf-8 -*-
"""
make_c7_layer_scatter.py — per-layer multiplication-fidelity scatter of the
simulated fully analog GTSRB client (three passes), for the Reviewer-1
Comment-7 supplementary note.

One panel per analog pass. Each panel scatters the chain's actual layer
output (mixer envelope compression at the planned drive, the measured
per-element jitter, and the link noise of the archived budget) against the
ideal matrix--vector product of that same layer's own input — the same
per-layer convention as Supplementary Note 5. A single reported gain g per
layer is fitted for display only. Because the unpowered ladder decays
(−10 → −34.3 → −58.5 dBm; per-tone SNR 43.0 → 32.6 → 8.4 dB), the
multiplication gets progressively worse with depth, exactly as the link
budget predicts.

Physics is the same validated numpy port as make_c7_activation_figs.py
(plan_Z / ed_video / selfmix reproduced from the archived
comb_analog_sim.py; the plan is asserted against results/link_budget.json
before anything is drawn).

Outputs: figures/c7_analog_layer_scatter.pdf/.png and
         c7_layer_scatter_numbers.json
"""
from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- reproducibility-archive path handling (ONLY change vs. the recovered
# --- original, make_c7_layer_scatter_original_recovered.py.txt): inputs are
# --- read from the shared fully-analog simulation archive next to this
# --- folder, and all outputs go to reproduced\ so the recovered originals
# --- are never overwritten. Physics, constants, seed and the <0.05 dB
# --- link-budget gate are untouched.
HERE = _data_dir(__file__)
SIM = HERE.parent / "_shared_fully_analog_simulation"
FIGS = HERE / "reproduced"

# ---------------------------------------------------------------- constants
K_ED = 208.7
CL_PASS = 7.47
BOOST_DB = 18.5
R_V = 3500.0
IL_STEP = 1.0
R0 = 50.0
P_N = 10 ** (-73.6 / 10)          # mW, per-bin effective noise floor
NOISE_C = 27.0
P3 = 10 ** (-7.53 / 10)           # mW, PAPR-shifted compression point
S_ELEM = 0.06                     # measured per-element jitter (archive)
KB, T0 = 1.380649e-23, 290.0
SIG_TH = float(np.sqrt(4 * KB * T0 * R0 * 10e6))

mw = lambda d: 10.0 ** (np.asarray(d, float) / 10.0)
dbm = lambda p: 10.0 * np.log10(np.maximum(p, 1e-300))


def ed_video(v):
    return 0.5 * v * (1.0 - np.exp(-2.0 * K_ED * v))


def plan_Z(L, dims, p1_dbm=-10.0):
    rng = np.random.default_rng(0)
    powers, lifts = [float(mw(p1_dbm))], []
    P = float(mw(p1_dbm))
    for l in range(L - 1):
        P_mix = P * 10 ** (-CL_PASS / 10)
        P_det = P_mix * 10 ** (BOOST_DB / 10)
        lifts.append(P_det)
        K = dims[l + 1]
        a = rng.normal(size=4 * K) + 1j * rng.normal(size=4 * K)
        env = np.abs(np.fft.ifft(a)) * len(a)
        env *= np.sqrt(P_det * 1e-3 * 2 * R0 / np.mean(env ** 2))
        vid = ed_video(env)
        vid = vid - vid.mean()
        P = np.mean(vid ** 2) / (4 * R_V) * 1e3 * 10 ** (-IL_STEP / 10)
        powers.append(float(P))
    return powers, lifts


def comb_compress_batch(a, P_mw):
    """Mixer envelope compression at planned power (batch-mean scaling, as
    in the fielded ZH mode: one frozen scalar, not per-sample AGC)."""
    n = a.shape[-1]
    nt = 4 * n
    p = np.mean(np.sum(np.abs(a) ** 2, -1)) + 1e-30
    a_s = a * np.sqrt(P_mw / p)
    env = np.fft.ifft(a_s, n=nt, axis=-1) * nt
    g = (1.0 + np.abs(env) ** 2 / (0.5 * P3)) ** -0.5
    s_c = np.fft.fft(env * g, axis=-1)[..., :n] / nt
    return s_c * np.sqrt(p / P_mw)


def selfmix(y, P_mw, rng):
    K = y.shape[-1]
    nt = 4 * K
    p_now = np.sum(np.abs(y) ** 2, -1, keepdims=True) + 1e-30
    env = np.fft.ifft(y * np.sqrt(P_mw / p_now), n=nt, axis=-1) * nt
    vid = ed_video(np.abs(env) * np.sqrt(2.0 * R0 * 1e-3))
    vid = vid + SIG_TH * rng.standard_normal(vid.shape)
    r = np.abs(np.fft.fft(vid, axis=-1)[..., :K] / nt)
    return (r / (np.max(r, -1, keepdims=True) + 1e-12)).astype(np.complex128)


# ------------------------------------------------- validate plan vs archive
budget = json.loads((SIM / "results" / "link_budget.json").read_text())
dims = [3072, 128, 128, 43]
powers, lifts = plan_Z(3, dims)
rows = budget["GTSRB"]["rows"][:3]
for l, row in enumerate(rows):
    assert abs(dbm(powers[l]) - row["drive_dBm"]) < 0.05, "ladder mismatch"
snr_db = [row["tone_snr_dB"] for row in rows]
print("ladder:", [f"{dbm(p):+.2f} dBm" for p in powers], "SNR:", snr_db)

# --------------------------------------------------------- chain simulation
z = np.load(SIM / "weights" / "ckpt_ZH_L3.npz")
W = [np.asarray(z[f"p{i}"]).astype(np.complex128) for i in range(3)]
d = np.load(SIM / "data" / "gtsrb_roi_32x32.npz")


def prep(X):
    x = X.reshape(len(X), -1).astype(np.float64) / 255.0
    lo = np.percentile(x, 2.0, axis=1, keepdims=True)
    hi = np.percentile(x, 98.0, axis=1, keepdims=True)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


NIMG = 40
X = prep(d["Xte"][:NIMG]).astype(np.complex128)

rng = np.random.default_rng(0)
a = X
layers = []
for l in range(3):
    P_l, D_l = powers[l], dims[l]
    y_ideal = a @ W[l].T                          # exact MVM of this input
    a_c = comb_compress_batch(a, P_l)             # mixer compression
    y_det = a_c @ W[l].T                          # deterministic part only
    y = y_det * (1.0 + S_ELEM * rng.standard_normal(y_det.shape))  # jitter
    p_sig = np.mean(np.abs(y) ** 2, -1, keepdims=True)
    snr = NOISE_C * P_l / (D_l * P_N)
    sig = np.sqrt(p_sig / snr / 2.0)
    y = y + sig * (rng.standard_normal(y.shape) +
                   1j * rng.standard_normal(y.shape))       # link noise

    def rel_fit(m_meas, m_ref):
        g = float(np.sum(m_meas * m_ref) / np.sum(m_ref ** 2))
        return g, float(np.linalg.norm(m_meas / g - m_ref) /
                        np.linalg.norm(m_ref))

    m_chain, m_ideal = np.abs(y).ravel(), np.abs(y_ideal).ravel()
    m_det = np.abs(y_det).ravel()
    g, rel = rel_fit(m_chain, m_ideal)
    g_det, rel_det = rel_fit(m_det, m_ideal)
    layers.append(dict(l=l, ideal=m_ideal, chain=m_chain / g,
                       det=m_det / g_det, g=g, rel=rel, rel_det=rel_det,
                       drive=float(dbm(P_l)), snr=float(snr_db[l])))
    print(f"pass {l+1}: drive {dbm(P_l):+.1f} dBm, SNR {snr_db[l]:.1f} dB, "
          f"g={g:.3f}, relRMSE={rel:.3f} (deterministic {rel_det:.3f})")
    if l < 2:
        a = selfmix(y, lifts[l], rng)             # analog activation

# ------------------------------------------------------------------ figure
OK_BLUE, OK_ORANGE, INK, GRAY = "#0072B2", "#E69F00", "#1A1A1A", "#8A8A8A"
plt.rcParams.update({
    "font.size": 8.2, "axes.labelsize": 8.8, "axes.titlesize": 9.2,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "axes.linewidth": 0.7, "figure.dpi": 200, "savefig.dpi": 300,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.55))
fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.17,
                    wspace=0.28)
DIMTXT = [r"$3072 \to 128$", r"$128 \to 128$", r"$128 \to 43$"]
for ax, L in zip(axes, layers):
    lim = float(np.percentile(L["ideal"], 99.7)) * 1.25
    ax.plot([0, lim], [0, lim], "--", color=GRAY, lw=0.9, zorder=1)
    ax.plot(L["ideal"], L["chain"], ".", ms=1.8, alpha=0.30, color=OK_BLUE,
            rasterized=True, zorder=2)
    ax.plot(L["ideal"], L["det"], ".", ms=1.2, alpha=0.30, color=OK_ORANGE,
            rasterized=True, zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(f"pass {L['l']+1}  ({DIMTXT[L['l']]})", fontsize=9)
    ax.text(0.03, 0.97,
            f"{L['drive']:+.1f} dBm, tone SNR {L['snr']:.1f} dB\n"
            f"rel. RMSE {L['rel']:.2f} (det. {L['rel_det']:.2f})",
            transform=ax.transAxes, va="top", fontsize=7.2)
    ax.tick_params(direction="out", length=3, width=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("chain output  $|y|/g$")
axes[1].set_xlabel("ideal matrix--vector product  $|y_{\\mathrm{ideal}}|$")
h_full = plt.Line2D([], [], marker=".", ls="none", ms=7, color=OK_BLUE)
h_det = plt.Line2D([], [], marker=".", ls="none", ms=6, color=OK_ORANGE)
axes[2].legend([h_full, h_det],
               ["full chain (noise on)", "deterministic only (noise off)"],
               loc="lower right", frameon=False, fontsize=6.8,
               handletextpad=0.2, borderaxespad=0.2)

FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / "c7_analog_layer_scatter.pdf")
fig.savefig(FIGS / "c7_analog_layer_scatter.png")
print(f"saved -> {FIGS/'c7_analog_layer_scatter.pdf'}")

nums = dict(ladder_dbm=[float(dbm(p)) for p in powers], snr_db=snr_db,
            gains=[L["g"] for L in layers], rel_rmse=[L["rel"] for L in layers],
            rel_rmse_det=[L["rel_det"] for L in layers],
            n_images=NIMG)
(FIGS / "c7_layer_scatter_numbers.json").write_text(json.dumps(nums, indent=1))
print(json.dumps(nums, indent=1))
