"""
prep_fig4_assets.py
===================
Prepares the derived assets for fig4_v5 from the verified data package
(data/fig4_serial_results.npz, built by fig4_verify_package.py from
V2/hardware_aware_training_v2) and the rebuilt GTSRB test cache
(data/gtsrb_roi_32x32_test.npz, gate-checked against
battery_frozen_slim.npz labels 1200/1200).

1. Per-product transfer + ENOB at (0,0) dBm from the archived Stage-A
   drive-ladder payload (serial_stationA_20260823.npz -> ladder_yhat),
   applying the exact convention of
   V2/hardware_aware_training_v2/07_supporting_analysis_enob_and_driveladder/
   serial_enob.py (phase alignment, LS gains, per-pair twin inversion on a
   4096-point grid, FS-referred ENOB = log2(range/(rms*sqrt(12))),
   duty_db = -3.05) -- and cross-checks the overall bits against the
   archived diag-capture values 2.53 / 4.38.
2. Battery photo montage assets: the evaluated 600 images are
   battery_random1200_idx[0:600] into the official GTSRB test set.
3. Twin transfer slices for the panel-b "chip" glyph.

Outputs -> data/fig4_panel_assets.npz, photos_battery/*.png,
contact_sheet.png
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import json
import os

import numpy as np
from PIL import Image

HERE = str(_data_dir(__file__))
D = HERE
HA = os.path.join(HERE, "..", "raw", "serial_hardware_campaign")

R = np.load(os.path.join(D, "fig4_serial_results.npz"), allow_pickle=True)
TM = json.load(open(os.path.join(
    HA, "twin_fit", "serial_twin_model.json")))
thp = np.array(TM["thp"])
rg = np.array(TM["ridges"]).reshape(-1, 4)
DUTY = TM["duty_db"]


def pdb(pl, pr):
    a0, klo, krf = thp[0], thp[1], thp[2]
    xl = 10 ** ((pl - klo) / 10.0)
    xr = 10 ** ((pr - krf) / 10.0)
    y = a0 + 10 * np.log10((xl / (1 + xl)) * (xr / (1 + xr)) + 1e-30)
    for wi, ai, bi, ci in rg:
        y = y + wi * np.tanh(ai * pl / 40 + bi * pr / 40 + ci)
    return y


def enob_fs(err, rng_p):
    return np.log2(rng_p / (np.sqrt(np.mean(err ** 2)) * np.sqrt(12)))


# ---- 1. per-product transfer + ENOB at (0,0), archived stationA payload ----
P_IDX = 2                       # ladder_powers_dbm[2] = 0.0 dBm
xs = {0: R["ladder_x_d0"], 1: R["ladder_x_d1"]}
ws = {0: R["ladder_w_d0"], 1: R["ladder_w_d1"]}
res = dict(scatter=[], enob_naive=[], enob_twin=[])
bin_edges_ref = None
per_bin_n, per_bin_t = [], []
for draw in (0, 1):
    x, w = xs[draw], ws[draw]
    xw = x * w
    rng_p = xw.max() - xw.min()
    xrms = np.sqrt(np.mean(x ** 2))
    wrms = np.sqrt(np.mean(w ** 2))
    p_lo = 20 * np.log10(np.maximum(np.abs(w) / wrms, 1e-6)) + DUTY
    p_rf = 20 * np.log10(np.maximum(np.abs(x) / xrms, 1e-6)) + DUTY
    pred_twin = np.sign(xw) * 10 ** (pdb(p_lo, p_rf) / 20.0)
    grid_x = np.linspace(1e-4, np.abs(x).max() * 1.5, 4096)
    for rep in (0, 1):
        yh = R["ladder_yhat"][P_IDX, draw, rep]
        ym = np.real(yh * np.exp(-1j * np.angle(np.vdot(xw, yh))))
        g_naive = np.dot(ym, xw) / np.dot(xw, xw)
        g_twin = np.dot(ym, pred_twin) / np.dot(pred_twin, pred_twin)
        xw_hat = np.zeros(len(xw))
        for i in range(len(xw)):
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
            xw_hat[i] = np.sign(ym[i]) * xh * np.abs(w[i])
        err_naive = ym / g_naive - xw
        err_twin = xw_hat - xw
        res["enob_naive"].append(enob_fs(err_naive, rng_p))
        res["enob_twin"].append(enob_fs(err_twin, rng_p))
        res["scatter"].append(
            np.stack([xw, ym / g_naive, xw_hat,
                      g_twin * pred_twin / g_naive]))

en = np.array(res["enob_naive"])
et = np.array(res["enob_twin"])
print(f"stationA (0,0) FS-referred ENOB over 4 captures:")
print(f"  naive         {en.mean():.2f} +/- {en.std():.2f}  {np.round(en,3)}")
print(f"  twin-inverted {et.mean():.2f} +/- {et.std():.2f}  {np.round(et,3)}")
print(f"  archived diag-capture reference: naive 2.53, twin 4.38")

# pooled per-|xw| bins over the 4 captures (same quantile recipe)
sc = np.concatenate([s.T for s in res["scatter"]])   # (4*256, 4)
xw_all, naive_all, twin_all = sc[:, 0], sc[:, 1], sc[:, 2]
rng_pool = xw_all.max() - xw_all.min()
q = np.quantile(np.abs(xw_all), np.linspace(0, 1, 9))
bins, e_n, e_t, n_in = [], [], [], []
for a, b in zip(q[:-1], q[1:]):
    s = (np.abs(xw_all) >= a) & (np.abs(xw_all) < b)
    if s.sum() < 10:
        continue
    bins.append(0.5 * (a + b))
    e_n.append(enob_fs(naive_all[s] - xw_all[s], rng_pool))
    e_t.append(enob_fs(twin_all[s] - xw_all[s], rng_pool))
    n_in.append(int(s.sum()))
print("bins:", np.round(bins, 3))
print("enob_naive:", np.round(e_n, 2))
print("enob_twin :", np.round(e_t, 2))

# ---- 2. battery photo assets ----------------------------------------------
G = np.load(os.path.join(D, "gtsrb_roi_32x32_test.npz"))
Xte, yte = G["Xte"], G["yte"]
bidx = np.load(os.path.join(
    HA, "frozen_inputs", "battery_random1200_idx.npy"))[:600]
hw_true = R["hw_true_twin"]
assert np.array_equal(yte[bidx], hw_true), "battery labels mismatch"
pdir = os.path.join(HERE, "photos_battery")
os.makedirs(pdir, exist_ok=True)
# contact sheet of the first 120 evaluated photos (upscaled x3)
S, NC, NR = 32 * 3, 15, 8
sheet = Image.new("RGB", (NC * (S + 2), NR * (S + 2)), (255, 255, 255))
from PIL import ImageDraw
dr = ImageDraw.Draw(sheet)
for k in range(NC * NR):
    im = Image.fromarray(Xte[bidx[k]]).resize((S, S), Image.NEAREST)
    cx, cy = (k % NC) * (S + 2), (k // NC) * (S + 2)
    sheet.paste(im, (cx, cy))
    dr.text((cx + 2, cy + 1), f"{k}:c{yte[bidx[k]]}", fill=(255, 60, 60))
sheet.save(os.path.join(HERE, "contact_sheet.png"))
print("wrote contact_sheet.png (first 120 evaluated photos, k:class)")

np.savez_compressed(
    os.path.join(D, "fig4_panel_assets.npz"),
    scatter_captures=np.stack(res["scatter"]),      # (4, 4, 256)
    enob_naive_caps=en, enob_twin_caps=et,
    enob_bins=np.array(bins), enob_bin_naive=np.array(e_n),
    enob_bin_twin=np.array(e_t), enob_bin_n=np.array(n_in),
    battery_idx600=bidx,
)
print("wrote data/fig4_panel_assets.npz")
