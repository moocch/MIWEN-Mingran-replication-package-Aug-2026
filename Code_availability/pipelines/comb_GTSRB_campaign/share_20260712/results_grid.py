"""Deployment-matrix accuracy grid for the battery frame (N=1,200, frozen
random frame, seed 20260805): training recipe x execution substrate.
Three cells are exact/measured; clean-on-hardware was never run (would cost
a rig session) and is a digital-twin estimate at the packed noise levels —
labeled as such on the figure."""
import sys
import numpy as np
sys.path.insert(0, ".")
from miwen_frozen_reference import (load_weights, forward_digital, preprocess,
                                    im2col, interpass, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(HERE / "gtsrb_roi_32x32.npz", allow_pickle=True)
Xte = preprocess(d["Xte"]); yte = d["yte"].astype(np.int64)
sel = np.load(HERE / "battery_random1200_idx.npy")
y = yte[sel]
slim = np.load(HERE / "battery_slim.npz", allow_pickle=True)
PACKED = [0.204, 0.234, 0.073, 0.164]

def twin(weights, seeds=10):
    layers = load_weights(HERE / weights)
    accs = []
    for sd in range(seeds):
        rng = np.random.RandomState(sd)
        A = Xte[sel].reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2); n = len(sel)
        for tag, lay in enumerate(layers, 1):
            if lay["kind"] == "conv":
                P, _, _ = im2col(A, lay["k"], lay["stride"])
                am = np.abs(P.reshape(-1, P.shape[-1]) @ lay["W"].T)
            else:
                am = np.abs(A.reshape(n, -1) @ lay["W"].T)
            am = am + PACKED[tag-1]*np.sqrt(np.mean(am**2))*rng.randn(*am.shape)
            am = np.maximum(am, 0.0)
            A, S = interpass(am, lay, tag, n)
        accs.append(float((np.argmax(S**2, 1) == y).mean()))
    return np.mean(accs)*100, np.std(accs)*100

acc_meas = (slim["preds"] == y).mean()*100
sig_meas = np.sqrt(acc_meas/100*(1-acc_meas/100)/len(y))*100
acc_ns25_dig = (slim["digital_preds"] == y).mean()*100
clean = load_weights(HERE / "r35_r3plus_s0_hw.npz")
acc_clean_dig = (forward_digital(Xte[sel], clean) == y).mean()*100
tw_m, tw_s = twin("r35_r3plus_s0_hw.npz")

fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.set_axis_off()
cells = [[(f"{acc_clean_dig:.2f}", "exact digital", "#2c7fb8"),
          (f"{tw_m:.1f} ± {tw_s:.1f}", "TWIN ESTIMATE — never fielded", "#bbb")],
         [(f"{acc_ns25_dig:.2f}", "exact digital, same images", "#2c7fb8"),
          (f"{acc_meas:.2f} ± {sig_meas:.2f}", "MEASURED, pre-registered", "#c0392b")]]
rows = ["clean training\n(digital-optimal)", "ns25 training\n(hardware-optimal, FROZEN)"]
cols = ["digital execution", "MIWEN hardware execution"]
for i in range(2):
    for j in range(2):
        v, note, color = cells[i][j]
        ax.add_patch(plt.Rectangle((j*0.38+0.22, 0.55-0.32*i), 0.36, 0.28,
                     fill=True, color=color, alpha=0.12, ec="#555"))
        ax.text(j*0.38+0.40, 0.72-0.32*i, v, ha="center", fontsize=17,
                fontweight="bold", color=color)
        ax.text(j*0.38+0.40, 0.63-0.32*i, note, ha="center", fontsize=7.5,
                color="#444")
for i, r in enumerate(rows):
    ax.text(0.20, 0.69-0.32*i, r, ha="right", fontsize=9.5)
for j, c in enumerate(cols):
    ax.text(j*0.38+0.40, 0.88, c, ha="center", fontsize=10.5,
            fontweight="bold")
ax.text(0.02, 0.14, "Battery frame: N=1,200 random official-test images "
        "(seed 20260805), identical for every cell. Paired, same images: "
        "measured−ns25digital = +0.42 (b=4,c=9);\nclean-digital−measured = "
        "+0.67 (b=9,c=1) — the price of analog at matched architecture. "
        "Repeat session (run 2): 98.67 measured; per-session flip rate 0.17%.",
        fontsize=7.6, color="#333")
ax.set_title("MIWEN GTSRB — deployment matrix, battery frame N=1,200",
             fontsize=12)
fig.tight_layout()
fig.savefig("../docs/notes/figs/battery_results_grid_20260805.png", dpi=200)
print(f"cells: clean-dig {acc_clean_dig:.2f} | clean-twin {tw_m:.1f}±{tw_s:.1f} | "
      f"ns25-dig {acc_ns25_dig:.2f} | measured {acc_meas:.2f}±{sig_meas:.2f}")
