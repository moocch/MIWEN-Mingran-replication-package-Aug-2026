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
    ("Perfect multiplier (no Twin)",        d["res_ideal"], d["dbm_ideal"], "#dc6464"),
    ("Physics-only Twin",                   d["res_phys"],  d["dbm_phys"],  "#e8a13a"),
    ("Full Digital Twin (Physics+NN)",      d["res_full"],  d["dbm_full"],  "#3a73c8"),
]

fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(14.5, 5.0), constrained_layout=True)

bins = np.arange(-30, 30.5, 1.0)
for label, R, _, color in MODELS:
    ax_h.hist(R.ravel(), bins=bins, density=True, alpha=0.55,
              color=color, edgecolor="none", label=label)
ax_h.axvline(0, color="black", lw=1.0)
for x in (-5, -2, 2, 5):
    ax_h.axvline(x, color="0.6", lw=0.8, linestyle=":")
ax_h.set_xlim(-30, 30)
ax_h.set_xlabel(r"Residual  $\hat{y}-y$  [dB]")
ax_h.set_ylabel("density")
ax_h.set_title("Residual densities  (peak at 0 = perfect prediction)",
               fontsize=11.5, fontweight="bold")
ax_h.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax_h.grid(True, linestyle=":", alpha=0.4)

t_grid = np.linspace(0, 25, 501)
for label, R, pred, color in MODELS:
    abs_err = np.sort(np.abs(R.ravel()))
    rmse = metrics(pred, d["dbm_meas"])["RMSE_dB"]
    cdf = 100.0 * np.searchsorted(abs_err, t_grid, side="right") / len(abs_err)
    ax_c.plot(t_grid, cdf, color=color, lw=2.4,
              label=f"{label}   (RMSE = {rmse:.2f} dB)")

abs_err_full = np.sort(np.abs(d["res_full"].ravel()))
for t in (1.0, 2.0, 5.0):
    pct = 100.0 * np.searchsorted(abs_err_full, t, side="right") / len(abs_err_full)
    ax_c.text(t + 0.4, pct - 4, f"{pct:.1f} %",
              color="#3a73c8", fontweight="bold", fontsize=10)

ax_c.set_xlim(0, 25); ax_c.set_ylim(0, 102)
ax_c.set_xlabel(r"error tolerance  $t$  [dB]")
ax_c.set_ylabel(r"% of grid points with  $|\hat{y}-y| \leq t$")
ax_c.set_title("Cumulative error distribution\n"
               r"$\mathrm{''How\ much\ of\ the\ surface\ is\ within\ \pm t\ dB?''}$",
               fontsize=11.5, fontweight="bold")
for hline in (50, 90, 100):
    ax_c.axhline(hline, color="0.7", lw=0.7, linestyle=":")
for vline in (1, 2, 5):
    ax_c.axvline(vline, color="0.7", lw=0.7, linestyle=":")
ax_c.legend(loc="lower right", fontsize=10, framealpha=0.95)

out_file = OUT / "fig4_cdf.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out_file}")
