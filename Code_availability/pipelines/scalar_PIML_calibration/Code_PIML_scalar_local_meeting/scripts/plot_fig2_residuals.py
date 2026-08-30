# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piml_data import load_data, metrics

OUT = _data_dir(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

d = load_data()

PANELS = [
    ("B $-$ A.  Perfect multiplier  $-$  measured", d["res_ideal"], d["dbm_ideal"]),
    ("C $-$ A.  Physics-only  $-$  measured",        d["res_phys"],  d["dbm_phys"]),
    ("D $-$ A.  Full Digital Twin  $-$  measured",   d["res_full"],  d["dbm_full"]),
]

VMIN, VMAX = -50, 50
extent = [d["p_lo_axis"].min(), d["p_lo_axis"].max(),
          d["p_rf_axis"].min(), d["p_rf_axis"].max()]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
fig.suptitle(r"Where each model fails -- pointwise residuals on the $(P_{LO}, P_{RF})$ grid",
             fontsize=12)

ims = []
for ax, (title, R, pred) in zip(axes, PANELS):
    rmse = metrics(pred, d["dbm_meas"])["RMSE_dB"]
    im = ax.imshow(R.T, origin="lower", extent=extent, aspect="equal",
                   vmin=VMIN, vmax=VMAX, cmap="RdBu_r", interpolation="nearest")
    ims.append(im)
    ax.set_title(f"{title}\nRMSE = {rmse:.2f} dB", fontsize=10.5, fontweight="bold")
    ax.set_xlabel(r"LO port power  $P_{LO}$  [dBm]")
    ax.set_ylabel(r"RF port power  $P_{RF}$  [dBm]")

cbar = fig.colorbar(ims[0], ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Prediction error  (predicted $-$ measured)  [dB]")
cbar.set_ticks([-50, -20, -5, -2, 0, 2, 5, 20, 50])

out_file = OUT / "fig2_residuals.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out_file}")
