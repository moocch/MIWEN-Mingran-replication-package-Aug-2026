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
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from piml_data import load_data, metrics

OUT = _data_dir(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

d = load_data()

y_meas   = d["dbm_meas"].ravel()
y_before = d["dbm_ideal"].ravel()
y_after  = d["dbm_full"].ravel()
err_before = y_before - y_meas
err_after  = y_after  - y_meas

M_before = metrics(d["dbm_ideal"], d["dbm_meas"])
M_after  = metrics(d["dbm_full"],  d["dbm_meas"])

COL_BEFORE = "#dc6464"
COL_AFTER  = "#3a73c8"

fig = plt.figure(figsize=(14.5, 9.5))
gs  = GridSpec(2, 3, figure=fig,
               height_ratios=[1.0, 1.0],
               width_ratios=[2.0, 1.0, 1.4],
               hspace=0.32, wspace=0.30)

ax_sc = fig.add_subplot(gs[0, 0:2])
xlim = ylim = (-128, 35)
ax_sc.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "k-", lw=1.5, zorder=4)
ax_sc.fill_between(xlim, [xlim[0]-5, xlim[1]-5], [xlim[0]+5, xlim[1]+5],
                   color="gray", alpha=0.10, zorder=0)
ax_sc.fill_between(xlim, [xlim[0]-2, xlim[1]-2], [xlim[0]+2, xlim[1]+2],
                   color="gray", alpha=0.18, zorder=0)
ax_sc.scatter(y_meas, y_before, s=8, alpha=0.55, color=COL_BEFORE, edgecolor="none",
              label="Before -- Perfect multiplier", zorder=2)
ax_sc.scatter(y_meas, y_after,  s=8, alpha=0.7,  color=COL_AFTER,  edgecolor="none",
              label="After -- Full Digital Twin", zorder=3)
ax_sc.set_xlim(xlim); ax_sc.set_ylim(ylim)
ax_sc.set_xlabel(r"Measured IF power  $y$  [dBm]")
ax_sc.set_ylabel(r"Predicted IF power  $\hat{y}$  [dBm]")
ax_sc.set_title("Predicted vs. Measured  --  before / after Digital-Twin optimisation",
                fontsize=12, fontweight="bold")
ax_sc.grid(True, linestyle=":", alpha=0.45)
ax_sc.text(0.97, 0.04, r"$\pm 2$ dB and $\pm 5$ dB tolerance",
           transform=ax_sc.transAxes, ha="right", va="bottom",
           fontsize=10, color="0.4")

ax_hi = fig.add_subplot(gs[0, 2])
bins = np.arange(-45, 45, 1.0)
ax_hi.hist(err_before, bins=bins, orientation="horizontal",
           color=COL_BEFORE, alpha=0.75, edgecolor="none")
ax_hi.hist(err_after,  bins=bins, orientation="horizontal",
           color=COL_AFTER,  alpha=0.85, edgecolor="none")
ax_hi.axhline(0, color="black", lw=1)
ax_hi.set_ylim(-45, 45)
ax_hi.set_xlabel("count")
ax_hi.set_ylabel(r"Residual  $\hat{y}-y$  [dB]")
ax_hi.set_title("Residual distribution", fontsize=12, fontweight="bold")
ax_hi.grid(True, axis="x", linestyle=":", alpha=0.45)

ax_re = fig.add_subplot(gs[1, 0:2])
ax_re.axhline(0, color="black", lw=1)
ax_re.scatter(y_meas, err_before, s=8, alpha=0.55, color=COL_BEFORE, edgecolor="none")
ax_re.scatter(y_meas, err_after,  s=8, alpha=0.75, color=COL_AFTER,  edgecolor="none")
ax_re.set_xlim(xlim); ax_re.set_ylim(-45, 45)
ax_re.set_xlabel(r"Measured IF power  $y$  [dBm]")
ax_re.set_ylabel("residual  [dB]")
ax_re.set_title("Residual versus measured power", fontsize=12, fontweight="bold")
ax_re.grid(True, linestyle=":", alpha=0.45)

ax_tb = fig.add_subplot(gs[1, 2]); ax_tb.axis("off")
table_rows = [
    ("Metric",         "Before",                              "After"),
    ("RMSE",           f"{M_before['RMSE_dB']:6.2f} dB",       f"{M_after['RMSE_dB']:6.2f} dB"),
    ("MAE",            f"{M_before['MAE_dB']:6.2f} dB",        f"{M_after['MAE_dB']:6.2f} dB"),
    ("Bias",           f"{M_before['Bias_dB']:6.2f} dB",       f"{M_after['Bias_dB']:6.3f} dB"),
    ("sigma",          f"{M_before['sigma_dB']:6.2f} dB",      f"{M_after['sigma_dB']:6.2f} dB"),
    ("max |error|",    f"{M_before['max_abs_err_dB']:6.2f} dB",f"{M_after['max_abs_err_dB']:6.2f} dB"),
    ("R^2",            f"{M_before['R2']:6.4f}",               f"{M_after['R2']:6.4f}"),
    ("Pearson rho",    f"{M_before['Pearson_rho']:6.4f}",      f"{M_after['Pearson_rho']:6.4f}"),
    ("|err| <= 1 dB",  f"{M_before['within_1dB_pct']:5.1f} %", f"{M_after['within_1dB_pct']:5.1f} %"),
    ("|err| <= 2 dB",  f"{M_before['within_2dB_pct']:5.1f} %", f"{M_after['within_2dB_pct']:5.1f} %"),
    ("|err| <= 5 dB",  f"{M_before['within_5dB_pct']:5.1f} %", f"{M_after['within_5dB_pct']:5.1f} %"),
]
tbl = ax_tb.table(cellText=table_rows, loc="center", cellLoc="left",
                  colWidths=[0.40, 0.30, 0.30])
tbl.auto_set_font_size(False); tbl.set_fontsize(9.8); tbl.scale(1.0, 1.4)
for j in range(3):
    tbl[(0, j)].set_text_props(weight="bold")
    tbl[(0, j)].set_facecolor("#e6ecf3")

fig.legend(
    handles=[
        plt.Line2D([0],[0], color="black", lw=1.5, label=r"Identity  $\hat{y}=y$"),
        plt.Line2D([0],[0], marker="o", color=COL_BEFORE, lw=0, markersize=7,
                   label="Before -- Perfect multiplier"),
        plt.Line2D([0],[0], marker="o", color=COL_AFTER,  lw=0, markersize=7,
                   label="After -- Full Digital Twin"),
    ],
    loc="lower center", ncol=3, frameon=False, fontsize=11,
    bbox_to_anchor=(0.5, -0.02),
)

out_file = OUT / "fig3_scatter.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out_file}")
