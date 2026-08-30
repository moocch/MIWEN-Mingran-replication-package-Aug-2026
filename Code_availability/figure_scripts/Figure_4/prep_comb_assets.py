"""
prep_comb_assets.py
===================
Assets for fig4_v6 round 2, all recomputed from raw archives with gates.

1. Comb-battery confusion matrices (N = 1,200):
   - measured in-physics (battery_frozen_slim.npz preds; gate 98.92%)
   - clean-trained digital (forward_digital from the frozen single-file
     reference miwen_frozen_reference.py, clean weights
     r35_r3plus_s0_hw.npz on the rebuilt test cache; gate 99.50%)
   - also gates the ns25 same-weights digital 98.42% and bit-identity
     with the archived digital_preds.
2. Panel-a implementation assets from real data/weights:
   - the STOP photograph + a real 5x5x3 patch (location + pixels)
   - 4 real conv1 kernel tiles (|w|, per-kernel normalized RGB)
   - 3 real layer-1 feature maps (14x14, post |.|/BN/ReLU/pool/maxnorm)
     for that photograph, channels chosen by variance.

Writes data/comb_assets.npz
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import importlib.util
import os
import sys

import numpy as np

HERE = str(_data_dir(__file__))
SHARE = os.path.join(str(_data_dir(__file__)), "..", "raw",
                     "comb_GTSRB_campaign", "share_20260712")
D5 = HERE
os.makedirs(HERE, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "mfr", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "pipelines", "comb_GTSRB_campaign",
                        "share_20260712", "miwen_frozen_reference.py"))
mfr = importlib.util.module_from_spec(spec)
sys.modules["mfr"] = mfr
spec.loader.exec_module(mfr)

G = np.load(os.path.join(D5, "gtsrb_roi_32x32_test.npz"))
Xte_u8, yte = G["Xte"], G["yte"].astype(np.int64)
X_rows = mfr.preprocess(Xte_u8)
bat = np.load(os.path.join(SHARE, "battery_random1200_idx.npy")).astype(int)

def conf43(true, pred):
    M = np.zeros((43, 43), np.int64)
    for t, p in zip(true, pred):
        M[t, p] += 1
    return M

# ---- measured in-physics (frozen-calibration rerun) ------------------------
slim_f = np.load(os.path.join(SHARE, "battery_frozen_slim.npz"),
                 allow_pickle=True)
assert np.array_equal(slim_f["img_index"], bat)
hw_pred, hw_lab = slim_f["preds"], slim_f["labels"]
assert np.array_equal(hw_lab, yte[bat]), "cache labels mismatch"
acc_hw = 100 * (hw_pred == hw_lab).mean()
print(f"measured in-physics battery: {acc_hw:.4f}%  (gate 98.9167)")
assert abs(acc_hw - 98.91666666666667) < 1e-9
conf_meas = conf43(hw_lab, hw_pred)

# ---- clean-trained digital -------------------------------------------------
layers_clean = mfr.load_weights(os.path.join(SHARE, "r35_r3plus_s0_hw.npz"))
pd_clean = mfr.forward_digital(X_rows[bat], layers_clean)
acc_clean = 100 * (pd_clean == yte[bat]).mean()
print(f"clean-trained digital battery: {acc_clean:.4f}%  (gate 99.50)")
assert abs(acc_clean - 99.50) < 0.005
conf_clean_dig = conf43(yte[bat], pd_clean)

# ---- same-weights (ns25) digital: gate + bit-identity ----------------------
layers_ns25 = mfr.load_weights(os.path.join(SHARE,
                                            "r35_r3plus_ns25_s0_hw.npz"))
pd_ns25 = mfr.forward_digital(X_rows[bat], layers_ns25)
acc_ns25 = 100 * (pd_ns25 == yte[bat]).mean()
slim = np.load(os.path.join(SHARE, "battery_slim.npz"), allow_pickle=True)
bit = bool((pd_ns25 == slim["digital_preds"]).all())
print(f"same-weights digital battery: {acc_ns25:.4f}%  (gate 98.4167); "
      f"bit-identical to archive: {bit}")
assert abs(acc_ns25 - 98.41666666666667) < 1e-9 and bit

# ---- panel-a implementation assets ----------------------------------------
A5 = np.load(os.path.join(D5, "fig4_panel_assets.npz"))
bidx600 = A5["battery_idx600"]
K_MAIN = 34                                  # dark STOP photo
img_u8 = Xte_u8[bidx600[K_MAIN]]

# a real 5x5 patch on the sign edge (visually interesting): pick the 5x5
# window with maximal intensity variance in the central region
gray = img_u8.astype(float).mean(2)
best, bloc = -1, (0, 0)
for r in range(6, 22):
    for c in range(6, 22):
        v = gray[r:r + 5, c:c + 5].var()
        if v > best:
            best, bloc = v, (r, c)
patch = img_u8[bloc[0]:bloc[0] + 5, bloc[1]:bloc[1] + 5]

# real conv1 kernels of the clean checkpoint: |w| per kernel, RGB-normalized
z = np.load(os.path.join(SHARE, "r35_r3plus_s0_hw.npz"), allow_pickle=True)
Wk = np.abs(z["c0r"] + 1j * z["c0i"])        # (32, 3, 5, 5)
kt = []
for k in range(4):
    t = Wk[k].transpose(1, 2, 0)             # (5,5,3)
    t = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    kt.append(t)
kern_tiles = np.stack(kt)

# real layer-1 feature maps for the STOP photo (clean weights)
lay1 = layers_clean[0]
A_img = X_rows[bidx600[K_MAIN]][None].reshape(-1, 32, 32, 3).transpose(
    0, 3, 1, 2)
P, Ho, Wo = mfr.im2col(A_img, lay1["k"], lay1["stride"])
am = np.abs(P.reshape(-1, P.shape[-1]) @ lay1["W"].T)
A_next, S = mfr.interpass(am, lay1, 1, 1)    # (1, 32, 14, 14) max-normed
fm = A_next[0]
order = np.argsort(fm.reshape(32, -1).var(1))[::-1]
featmaps = fm[order[:3]]
print("feature-map channels:", order[:3].tolist())

np.savez_compressed(
    os.path.join(HERE, "data", "comb_assets.npz"),
    conf_measured=conf_meas, conf_clean_digital=conf_clean_dig,
    acc_measured=acc_hw, acc_clean_digital=acc_clean,
    acc_same_weights_digital=acc_ns25,
    photo_u8=img_u8, patch_u8=patch, patch_rc=np.array(bloc),
    kern_tiles=kern_tiles, featmaps=featmaps)
print("wrote data/comb_assets.npz")
