"""Confusion matrix (true class x predicted class, 43x43) for the measured
battery run (frozen algorithm, N=1,200 random frame, 98.83%)."""
import sys
import numpy as np
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

z = np.load("battery_slim.npz", allow_pickle=True)
pm, y = z["preds"], z["labels"]
C = np.zeros((43, 43), int)
for t, p in zip(y, pm):
    C[t, p] += 1

fig, ax = plt.subplots(figsize=(9.5, 8.6))
masked = np.ma.masked_equal(C, 0)
im = ax.imshow(masked, norm=LogNorm(vmin=1, vmax=C.max()), cmap="viridis")
for t, p in zip(*np.nonzero(C * (1 - np.eye(43, dtype=int)))):
    ax.annotate(str(C[t, p]), (p, t), color="red", fontsize=8,
                ha="center", va="center", fontweight="bold")
    ax.add_patch(plt.Rectangle((p-0.5, t-0.5), 1, 1, fill=False,
                               ec="red", lw=1.0))
ax.set_xlabel("predicted class")
ax.set_ylabel("true class")
ax.set_xticks(range(0, 43, 2)); ax.set_yticks(range(0, 43, 2))
ax.tick_params(labelsize=7)
n_err = int((pm != y).sum())
ax.set_title(f"MIWEN hardware inference — GTSRB, N=1,200 random test "
             f"images\naccuracy 98.83% ± 0.31  |  {n_err} errors "
             f"(red cells, counts)  |  log color scale", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.75, label="images (log)")
fig.tight_layout()
fig.savefig("../docs/notes/figs/battery_confusion_20260805.png", dpi=200)
err = [(int(t), int(p), int(C[t, p]))
       for t, p in zip(*np.nonzero(C * (1 - np.eye(43, dtype=int))))]
print("off-diagonal (true, pred, n):", err)
