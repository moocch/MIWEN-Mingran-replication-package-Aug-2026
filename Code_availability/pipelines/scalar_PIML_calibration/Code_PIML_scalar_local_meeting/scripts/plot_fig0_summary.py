# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piml_data import load_data, metrics

OUT = _data_dir(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

d = load_data()

MODELS = [
    ("ideal",  "Before -- Perfect multiplier (no Twin)",   d["dbm_ideal"], "#dc6464"),
    ("phys",   "Physics-only Digital Twin",                 d["dbm_phys"],  "#e8a13a"),
    ("full",   "After -- Full Digital Twin (Physics+NN)",   d["dbm_full"],  "#3a73c8"),
]
M = {k: metrics(arr, d["dbm_meas"]) for k, _, arr, _ in MODELS}

fig = plt.figure(figsize=(14.5, 8))
gs  = fig.add_gridspec(2, 3, height_ratios=[2.4, 1.0], hspace=0.45, wspace=0.18)

fig.suptitle(
    r"Digital-Twin quality vs. measurement   "
    r"($f_{RF}=1.20$ GHz,  $f_{LO}=0.90$ GHz,  $f_{IF}=300$ MHz   --   1849 grid points)",
    fontsize=12.5, fontweight="bold", y=0.995,
)

y_meas = d["dbm_meas"].ravel()
xlim = ylim = (-125, 32)

for i, (key, title, pred, color) in enumerate(MODELS):
    ax = fig.add_subplot(gs[0, i])
    y_pred = pred.ravel()

    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "k-", lw=1.5, zorder=3)
    ax.fill_between(xlim, [xlim[0]-5, xlim[1]-5], [xlim[0]+5, xlim[1]+5],
                    color="gray", alpha=0.12, zorder=0)
    ax.fill_between(xlim, [xlim[0]-2, xlim[1]-2], [xlim[0]+2, xlim[1]+2],
                    color="gray", alpha=0.20, zorder=0)
    ax.scatter(y_meas, y_pred, s=8, alpha=0.5, color=color, edgecolor="none")

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
    ax.set_xlabel("Measured  $y$  [dBm]")
    if i == 0:
        ax.set_ylabel(r"Predicted  $\hat{y}$  [dBm]")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)

    txt = (f"RMSE = {M[key]['RMSE_dB']:5.2f} dB\n"
           f"$R^2$  = {M[key]['R2']:6.4f}\n"
           f"|err|$\\leq$2dB : {M[key]['within_2dB_pct']:4.1f}%")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            fontsize=9.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="0.6", alpha=0.95))

ax_b = fig.add_subplot(gs[1, :])
labels    = ["Perfect mult.\n(no Twin)", "Physics-only\nTwin", "Full Digital Twin\n(Physics+NN)"]
colors    = [m[3] for m in MODELS]
rmse_vals = [M[k]["RMSE_dB"] for k, _, _, _ in MODELS]
r2_vals   = [M[k]["R2"]      for k, _, _, _ in MODELS]
x = np.arange(len(MODELS)); w = 0.34

ax_b.bar(x - w/2, rmse_vals, w, color=colors, alpha=0.90,
         edgecolor="black", linewidth=0.5, label="RMSE [dB]")
ax_b.set_xticks(x); ax_b.set_xticklabels(labels, fontsize=10)
ax_b.set_ylabel("RMSE  [dB]")
ax_b.set_ylim(0, max(rmse_vals) * 1.20)
ax_b.set_title(r"Headline metrics -- RMSE (lower better)  &  $R^{2}$ (closer to 1 better)",
               fontsize=11.5, fontweight="bold")
ax_b.grid(True, axis="y", linestyle=":", alpha=0.4)
for xi, v in zip(x - w/2, rmse_vals):
    ax_b.text(xi, v + 0.4, f"{v:.2f} dB", ha="center", va="bottom",
              fontsize=10, fontweight="bold")

ax_r2 = ax_b.twinx()
ax_r2.bar(x + w/2, r2_vals, w, color="none",
          edgecolor="black", linewidth=1.2, hatch="//", label=r"$R^{2}$")
ax_r2.set_ylabel(r"$R^{2}$"); ax_r2.set_ylim(0, 1.05)
for xi, v in zip(x + w/2, r2_vals):
    ax_r2.text(xi, v + 0.02, f"{v:.4f}", ha="center", va="bottom",
               fontsize=10, fontweight="bold")

h1, l1 = ax_b.get_legend_handles_labels()
h2, l2 = ax_r2.get_legend_handles_labels()
ax_r2.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.95)

out_file = OUT / "fig0_summary.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out_file}")
