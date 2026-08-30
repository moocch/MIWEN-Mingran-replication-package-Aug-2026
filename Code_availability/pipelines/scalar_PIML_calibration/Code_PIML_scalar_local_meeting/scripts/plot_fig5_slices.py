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
from piml_data import load_data

OUT = _data_dir(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

d = load_data()

PANELS = [
    (-50.0, "Weak LO drive\n$P_{LO}=-50$  dBm"),
    (-20.0, "Moderate LO drive\n$P_{LO}=-20$  dBm"),
    ( +10.0, "Strong LO drive  (saturated)\n$P_{LO}=+10$  dBm"),
]

p_lo_axis = d["p_lo_axis"]
p_rf_axis = d["p_rf_axis"]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True, sharey=True)
fig.suptitle(
    r"1-D slices through the $(P_{LO}, P_{RF})$ surface  --  measured vs. three models",
    fontsize=12, fontweight="bold",
)

for ax, (lo, title) in zip(axes, PANELS):
    i_lo = int(np.argmin(np.abs(p_lo_axis - lo)))

    ax.plot(p_rf_axis, d["dbm_ideal"][i_lo, :], color="#dc6464", lw=2.2,
            label="Perfect multiplier")
    ax.plot(p_rf_axis, d["dbm_phys"][i_lo, :],  color="#e8a13a", lw=2.0, ls="--",
            label="Physics-only Twin")
    ax.plot(p_rf_axis, d["dbm_full"][i_lo, :],  color="#3a73c8", lw=2.2,
            label="Full Digital Twin")
    ax.plot(p_rf_axis, d["dbm_meas"][i_lo, :],  color="black",   lw=0.8,
            marker="o", markersize=4, label="Measured")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"RF port power  $P_{RF}$  [dBm]")
    ax.set_xlim(-72, 16); ax.set_ylim(-125, 28)
    ax.grid(True, linestyle=":", alpha=0.45)

axes[0].set_ylabel("IF power  [dBm]")
axes[1].legend(loc="lower right", fontsize=10, framealpha=0.95)

out_file = OUT / "fig5_slices.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out_file}")
