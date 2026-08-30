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
M_a = metrics(d["dbm_ideal"], d["dbm_meas"])
M_b = metrics(d["dbm_phys"],  d["dbm_meas"])
M_c = metrics(d["dbm_full"],  d["dbm_meas"])

fig = plt.figure(figsize=(11.5, 14.0))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ACC = "#1f4e8a"
GRAY = "0.30"

def title(y, text):
    ax.text(0.05, y, text, fontsize=14, fontweight="bold", color=ACC,
            transform=ax.transAxes)

def equation(y, text, fontsize=15):
    ax.text(0.10, y, text, fontsize=fontsize, color=ACC, transform=ax.transAxes)

def para(y, text, italic=False, fontsize=10.5):
    ax.text(0.05, y, text, fontsize=fontsize, color=GRAY,
            style="italic" if italic else "normal", transform=ax.transAxes)

def label(y, sym, descr, unit=""):
    ax.text(0.05, y, sym, fontsize=12, fontweight="bold", color=ACC,
            transform=ax.transAxes)
    suffix = f"    {unit}" if unit else ""
    ax.text(0.16, y, descr + suffix, fontsize=12, fontweight="bold",
            color="black", transform=ax.transAxes)


title(0.965, "Quantitative metrics  --  definitions")
para(0.935,
     r"Notation:    $y_i$ = measured IF power (dBm) at grid point $i$,"
     r"    $\hat{y}_i$ = corresponding model prediction,"
     r"    $\bar{y} = \frac{1}{N}\sum_i y_i$,    $N = 1849$  (43 x 43 grid).",
     fontsize=10)
para(0.905,
     "All metrics are evaluated in dB-space because the raw IF power spans  ~7 decades;\n"
     "using linear W would let two top-corner cells dominate every average.",
     italic=True, fontsize=10)

label(0.860, "RMSE", "--  Root-Mean-Square Error", "[dB]")
equation(0.815, r"$\mathrm{RMSE} = \sqrt{\,\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2\,}$",
         fontsize=18)
para(0.770, "A typical magnitude of the error.  Squaring penalises large outliers more than small ones.")

label(0.730, "MAE", "--  Mean Absolute Error", "[dB]")
equation(0.690, r"$\mathrm{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i - y_i|$", fontsize=18)
para(0.650, "Robust counterpart of RMSE; describes the typical |error| without squaring.")

label(0.612, "Bias", "--  mean signed error", "[dB]")
equation(0.572, r"$\mathrm{Bias} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)$", fontsize=18)
para(0.534, r"Positive $\Rightarrow$ systematic over-prediction;     "
            r"negative $\Rightarrow$ systematic under-prediction.")

label(0.498, r"$R^{\,2}$", "--  coefficient of determination", "(dimensionless)")
equation(0.453, r"$R^{2} = 1 - \frac{\sum_{i}(\hat{y}_i - y_i)^2}"
                r"{\sum_{i}(y_i - \bar{y})^2} = 1 - \frac{\mathrm{SS_{res}}}{\mathrm{SS_{tot}}}$",
         fontsize=17)
para(0.408, r"Fraction of the variance in $y$ explained by the model.   "
            r"$R^{2}=1$: perfect.   $R^{2}=0$: same as predicting $\bar{y}$.   "
            r"$R^{2}<0$: worse than the mean.")

label(0.370, r"Pearson  $\rho$", "--  linear correlation", r"$(-1 \leq \rho \leq 1)$")
equation(0.318, r"$\rho = \frac{\sum_{i}(\hat{y}_i - \bar{\hat{y}})(y_i - \bar{y})}"
                r"{\sqrt{\sum_{i}(\hat{y}_i - \bar{\hat{y}})^2}\,\sqrt{\sum_{i}(y_i - \bar{y})^2}}$",
         fontsize=17)
para(0.270, r"$\rho=1$  $\Leftrightarrow$  predictions are an affine function of truth (perfect ranking).")


title(0.225, "Final scoreboard  --  three models, evaluated on the full 43 x 43 = 1849-point sweep")

scoreboard = [
    ("Metric",                  "Perfect mult.\n(no Twin)",          "Physics-only\nTwin",                "Full Digital Twin\n(Physics + NN)"),
    ("RMSE  [dB]",              f"{M_a['RMSE_dB']:6.2f}",            f"{M_b['RMSE_dB']:6.2f}",            f"{M_c['RMSE_dB']:6.2f}"),
    ("MAE   [dB]",              f"{M_a['MAE_dB']:6.2f}",             f"{M_b['MAE_dB']:6.2f}",             f"{M_c['MAE_dB']:6.2f}"),
    ("Bias  (mean err) [dB]",   f"{M_a['Bias_dB']:6.2f}",            f"{M_b['Bias_dB']:6.2f}",            f"{M_c['Bias_dB']:6.3f}"),
    ("sigma  (std of err) [dB]",f"{M_a['sigma_dB']:6.2f}",           f"{M_b['sigma_dB']:6.2f}",           f"{M_c['sigma_dB']:6.2f}"),
    ("max |error| [dB]",        f"{M_a['max_abs_err_dB']:6.2f}",     f"{M_b['max_abs_err_dB']:6.2f}",     f"{M_c['max_abs_err_dB']:6.2f}"),
    ("95th-pctile |error| [dB]",f"{M_a['p95_abs_err_dB']:6.2f}",     f"{M_b['p95_abs_err_dB']:6.2f}",     f"{M_c['p95_abs_err_dB']:6.2f}"),
    ("$R^{2}$",                 f"{M_a['R2']:6.4f}",                 f"{M_b['R2']:6.4f}",                 f"{M_c['R2']:6.4f}"),
    (r"Pearson  $\rho$",        f"{M_a['Pearson_rho']:6.4f}",        f"{M_b['Pearson_rho']:6.4f}",        f"{M_c['Pearson_rho']:6.4f}"),
    ("|err| $\\leq$ 1 dB    [%]",f"{M_a['within_1dB_pct']:5.1f}",    f"{M_b['within_1dB_pct']:5.1f}",     f"{M_c['within_1dB_pct']:5.1f}"),
    ("|err| $\\leq$ 2 dB    [%]",f"{M_a['within_2dB_pct']:5.1f}",    f"{M_b['within_2dB_pct']:5.1f}",     f"{M_c['within_2dB_pct']:5.1f}"),
    ("|err| $\\leq$ 3 dB    [%]",f"{M_a['within_3dB_pct']:5.1f}",    f"{M_b['within_3dB_pct']:5.1f}",     f"{M_c['within_3dB_pct']:5.1f}"),
    ("|err| $\\leq$ 5 dB    [%]",f"{M_a['within_5dB_pct']:5.1f}",    f"{M_b['within_5dB_pct']:5.1f}",     f"{M_c['within_5dB_pct']:5.1f}"),
]

tax = fig.add_axes([0.05, 0.04, 0.90, 0.165]); tax.set_axis_off()
tbl = tax.table(cellText=scoreboard, loc="center", cellLoc="center",
                colWidths=[0.34, 0.22, 0.22, 0.22])
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1.0, 1.55)

for j in range(4):
    cell = tbl[(0, j)]
    cell.set_text_props(weight="bold", color="white")
    cell.set_facecolor("#222b40")
for i in range(1, len(scoreboard)):
    tbl[(i, 0)].set_text_props(weight="bold")
    tbl[(i, 0)].set_facecolor("#f3f5f8")
    tbl[(i, 1)].set_facecolor("#f7e6e6")
    tbl[(i, 3)].set_facecolor("#e6eef9")
    tbl[(i, 3)].set_text_props(weight="bold")

ax.text(0.5, 0.012,
        "Improvement   Full-Twin vs. Perfect-mult :   "
        "RMSE  {:.1f}x lower,    MAE  {:.1f}x lower,    "
        r"$R^{{2}}$  {:.3f} $\to$ {:.4f}".format(
            M_a["RMSE_dB"] / M_c["RMSE_dB"],
            M_a["MAE_dB"]  / M_c["MAE_dB"],
            M_a["R2"], M_c["R2"]),
        ha="center", fontsize=11, fontweight="bold", style="italic",
        color=GRAY, transform=ax.transAxes)

out_file = OUT / "fig6_formulas.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out_file}")
