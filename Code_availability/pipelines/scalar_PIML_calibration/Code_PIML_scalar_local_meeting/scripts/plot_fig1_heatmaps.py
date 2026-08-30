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
    ("A. Measured (N9020A)",                     d["dbm_meas"]),
    ("B. Perfect multiplier (no Digital Twin)",  d["dbm_ideal"]),
    ("C. Physics-only Digital Twin",             d["dbm_phys"]),
    ("D. Full Digital Twin  (Physics + NN)",     d["dbm_full"]),
]

vmin = float(np.percentile(d["dbm_meas"], 0.5))
vmax = float(np.percentile(d["dbm_meas"], 99.5))

extent = [d["p_lo_axis"].min(), d["p_lo_axis"].max(),
          d["p_rf_axis"].min(), d["p_rf_axis"].max()]

fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), constrained_layout=True)
fig.suptitle(
    r"IF-power surfaces -- measurement vs. three models   "
    r"($f_{RF}=1.20$ GHz,  $f_{LO}=0.90$ GHz,  $f_{IF}=300$ MHz)",
    fontsize=12, fontweight="bold",
)

ims = []
for ax, (title, Z) in zip(axes.flat, PANELS):
    im = ax.imshow(Z.T, origin="lower", extent=extent, aspect="equal",
                   vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
    ims.append(im)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"LO port power  $P_{LO}$  [dBm]")
    ax.set_ylabel(r"RF port power  $P_{RF}$  [dBm]")

cbar = fig.colorbar(ims[0], ax=axes, shrink=0.78, pad=0.02)
cbar.set_label("IF power  [dBm]")

out_file = OUT / "fig1_heatmaps.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out_file}")
