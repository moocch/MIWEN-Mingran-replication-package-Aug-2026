# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

from pathlib import Path
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE   = _data_dir(__file__)
PROJ   = Path("/path/to/work/PIML_Vector_Figures")
sys.path.insert(0, str(PROJ))
from piml_vector_data import load_one_N, metrics

OUT = HERE
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         110,
})

MODEL_LABEL = {
    "ideal":  "B. Perfect multiplier (no Digital Twin)",
    "phys":   "C. Physics-only Digital Twin",
    "full":   "D. Full Digital Twin (Physics + NN)",
}
COLOUR = {
    "ideal": "#c0504d",
    "phys":  "#e0a93b",
    "full":  "#3973ac",
}


def _w_to_dbm(p):
    return 10.0 * np.log10(np.maximum(p, 1e-30)) + 30.0


def safe_log_range(*arrays_dbm, lo_pct=1, hi_pct=99.9):
    w = np.concatenate([10.0**((a.ravel()-30.0)/10.0) for a in arrays_dbm])
    return float(np.percentile(w, lo_pct)), float(np.percentile(w, hi_pct))


def annotation_box(ax, lines, loc="upper left"):
    txt = "\n".join(lines)
    ax.text(0.03 if "left" in loc else 0.97, 0.97, txt,
            transform=ax.transAxes, va="top",
            ha="left" if "left" in loc else "right",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="black",
                      boxstyle="round,pad=0.4", alpha=0.95))


def fig_redblack_heatmap(d, N):
    meas_w = 10.0**((d["dbm_meas"] - 30.0) / 10.0)
    full_w = 10.0**((d["dbm_full"] - 30.0) / 10.0)
    vmin, vmax = safe_log_range(d["dbm_meas"], d["dbm_full"])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    extent = [d["p_rf_axis"][0], d["p_rf_axis"][-1],
              d["p_lo_axis"][0], d["p_lo_axis"][-1]]

    titles = ["Measured IF power (W)",
              "Digital twin (physics + NN) IF power (W)"]
    arrays = [meas_w, full_w]
    for ax, arr, t in zip(axes, arrays, titles):
        im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto",
                       norm=LogNorm(vmin=vmin, vmax=vmax), cmap="afmhot")
        ax.set_title(t)
        ax.set_xlabel("RF port power (dBm)")
        ax.set_ylabel("LO port power (dBm)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"N = {N}   ·   $|\\langle a,b\\rangle|^2$ = {d['ip_sq']:.3f}   ·   "
                 f"PAPR$_{{LO}}$ = {d['papr_lo']:.2f} dB, "
                 f"PAPR$_{{RF}}$ = {d['papr_rf']:.2f} dB",
                 fontsize=12.5, y=1.04)
    fname = OUT / f"fig_N{N}_1_redblack_heatmap.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_ABCD_cascade(d, N):
    panels = [("A. Measured (N9020A)",                    d["dbm_meas"]),
              ("B. Perfect multiplier (no Digital Twin)", d["dbm_ideal"]),
              ("C. Physics-only Digital Twin",            d["dbm_phys"]),
              ("D. Full Digital Twin (Physics + NN)",     d["dbm_full"])]

    vmin = float(np.percentile(d["dbm_meas"], 1.0))
    vmax = float(np.percentile(d["dbm_meas"], 99.5))

    fig = plt.figure(figsize=(20.5, 7.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 5, height_ratios=[2.6, 1.4],
                          width_ratios=[1, 1, 1, 1, 0.05],
                          wspace=0.30, hspace=0.55,
                          left=0.05, right=0.97, top=0.91, bottom=0.10)
    extent = [d["p_lo_axis"][0], d["p_lo_axis"][-1],
              d["p_rf_axis"][0], d["p_rf_axis"][-1]]

    im = None
    for j, (title, arr) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(arr.T, origin="lower", extent=extent, aspect="auto",
                       vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"LO port power $P_{LO}$ [dBm]")
        ax.set_ylabel(r"RF port power $P_{RF}$ [dBm]")
        ax.grid(False)

    cax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("IF power [dBm]")

    m_B = metrics(d["dbm_ideal"], d["dbm_meas"])
    m_C = metrics(d["dbm_phys"],  d["dbm_meas"])
    m_D = metrics(d["dbm_full"],  d["dbm_meas"])

    metric_names = ["RMSE", "MAE", "P95 |err|", "Max |err|"]
    keys         = ["RMSE_dB", "MAE_dB", "p95_abs_err_dB", "max_abs_err_dB"]
    vals_B = [m_B[k] for k in keys]
    vals_C = [m_C[k] for k in keys]
    vals_D = [m_D[k] for k in keys]

    ax_bar = fig.add_subplot(gs[1, 1:4])
    x = np.arange(len(metric_names))
    w = 0.27
    bars_B = ax_bar.bar(x - w, vals_B, w, label="Perfect multiplier",
                        color="#4b3a8a", edgecolor="black")
    bars_C = ax_bar.bar(x,     vals_C, w, label="Physics-only DT",
                        color="#3e8e93", edgecolor="black")
    bars_D = ax_bar.bar(x + w, vals_D, w, label="Full PIML",
                        color="#c2d72b", edgecolor="black")

    for bars, vals in [(bars_B, vals_B), (bars_C, vals_C), (bars_D, vals_D)]:
        for b, v in zip(bars, vals):
            ax_bar.text(b.get_x() + b.get_width()/2,
                        v * 1.06, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax_bar.set_yscale("log")
    ax_bar.set_ylim(0.5, max(vals_B + vals_C + vals_D) * 4.0)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylabel("Error [dB]   (log scale)")
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
                  framealpha=0.95, ncol=3)
    ax_bar.grid(True, which="both", axis="y", alpha=0.3)

    fig.suptitle(f"N = {N}   ·   step-by-step model cascade: "
                 f"A. Measured  →  B. Perfect multiplier  →  "
                 f"C. Physics-only DT  →  D. Full Digital Twin (Physics + NN)",
                 fontsize=13.5, fontweight="bold")

    fname = OUT / f"fig_N{N}_2_ABCD_cascade.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_scatter_three_way(d, N):
    panels = [
        ("Before — Perfect multiplier (no Twin)", d["dbm_ideal"], "ideal"),
        ("Physics-only Digital Twin",             d["dbm_phys"],  "phys"),
        ("After — Full Digital Twin (Physics+NN)", d["dbm_full"],  "full"),
    ]
    y = d["dbm_meas"].ravel()

    y_all = np.concatenate([y] + [p[1].ravel() for p in panels])
    lo = float(np.floor(np.percentile(y_all, 0.5) / 5) * 5)
    hi = float(np.ceil( np.percentile(y_all, 99.5)/ 5) * 5)
    lo_p = min(lo, -120)
    hi_p = max(hi, 30)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), constrained_layout=True,
                             sharey=True)
    for ax, (title, pred, key) in zip(axes, panels):
        yp = pred.ravel()
        m = metrics(pred, d["dbm_meas"])
        xs = np.linspace(lo - 5, hi + 5, 50)
        ax.fill_between(xs, xs - 2, xs + 2, color="gray", alpha=0.15, zorder=1)
        ax.plot(xs, xs, "k-", lw=1.5, zorder=2)
        ax.scatter(y, yp, s=10, alpha=0.55,
                   color=COLOUR[key], edgecolor="none", zorder=3)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel(r"Measured  $y$  [dBm]")
        if key == "ideal":
            ax.set_ylabel(r"Predicted  $\hat{y}$  [dBm]")
        ax.set_xlim(lo - 2, hi + 2)
        ax.set_ylim(lo_p, hi_p)
        ax.grid(True, alpha=0.3)
        annotation_box(ax, [
            f"RMSE = {m['RMSE_dB']:.2f} dB",
            f"$R^2$  = {m['R2']:.4f}",
            f"|err|≤2dB : {m['within_2dB_pct']:.1f} %",
        ])

    fig.suptitle(f"N = {N}   ·   Predicted vs Measured "
                 f"(Before → Physics-only → Full Digital Twin)",
                 fontsize=13.5, y=1.04, fontweight="bold")
    fname = OUT / f"fig_N{N}_3_scatter_3way.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_headline_bars(d, N):
    m_B = metrics(d["dbm_ideal"], d["dbm_meas"])
    m_C = metrics(d["dbm_phys"],  d["dbm_meas"])
    m_D = metrics(d["dbm_full"],  d["dbm_meas"])
    rmse = [m_B["RMSE_dB"], m_C["RMSE_dB"], m_D["RMSE_dB"]]
    r2   = [m_B["R2"],      m_C["R2"],      m_D["R2"]]
    labels = ["Perfect mult.\n(no Twin)",
              "Physics-only\nTwin",
              "Full Digital Twin\n(Physics + NN)"]
    cols   = [COLOUR["ideal"], COLOUR["phys"], COLOUR["full"]]

    fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    x = np.arange(3)
    w = 0.38
    bars_rmse = ax.bar(x - w/2, rmse, w, color=cols, edgecolor="black",
                       label="RMSE [dB]")
    ax2 = ax.twinx()
    bars_r2 = ax2.bar(x + w/2, r2, w, color="white", hatch="///",
                      edgecolor="black", label=r"$R^2$")

    for b, v in zip(bars_rmse, rmse):
        ax.text(b.get_x() + b.get_width()/2, v + 0.4,
                f"{v:.2f} dB", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    for b, v in zip(bars_r2, r2):
        if v >= 0:
            ax2.text(b.get_x() + b.get_width()/2, v + 0.03,
                     f"{v:.4f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")
        else:
            ax2.text(b.get_x() + b.get_width()/2, v - 0.05,
                     f"{v:.4f}", ha="center", va="top",
                     fontsize=11, fontweight="bold", color="darkred")

    ax.set_ylabel("RMSE [dB]")
    ax2.set_ylabel(r"$R^2$")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, max(rmse) * 1.30)
    r2_low = min(0.0, min(r2) - 0.15)
    ax2.set_ylim(r2_low, 1.15)
    ax2.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.grid(False); ax2.grid(False)
    ax.set_title(f"N = {N}   ·   Headline metrics — RMSE (lower better) & $R^2$ (closer to 1 better)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(handles=[bars_rmse[0], bars_r2[0]],
              labels=["RMSE [dB]", r"$R^2$"],
              loc="upper left", framealpha=0.95)

    fname = OUT / f"fig_N{N}_4_headline_bars.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_before_after(d, N):
    y       = d["dbm_meas"].ravel()
    y_pre   = d["dbm_ideal"].ravel()
    y_post  = d["dbm_full"].ravel()
    res_pre  = y_pre  - y
    res_post = y_post - y

    m_pre  = metrics(d["dbm_ideal"], d["dbm_meas"])
    m_post = metrics(d["dbm_full"],  d["dbm_meas"])

    fig = plt.figure(figsize=(15.5, 8.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.3],
                          width_ratios=[1.6, 1.0],
                          left=0.06, right=0.97, top=0.92, bottom=0.09,
                          hspace=0.30, wspace=0.20)

    ax = fig.add_subplot(gs[0, 0])
    y_all = np.concatenate([y, y_pre, y_post])
    lo_y = float(np.floor(np.percentile(y_all, 0.5) / 5) * 5)
    hi_y = float(np.ceil( np.percentile(y_all, 99.5)/ 5) * 5)
    xs = np.linspace(lo_y - 5, hi_y + 5, 50)
    ax.fill_between(xs, xs - 5, xs + 5, color="gray", alpha=0.10, zorder=1)
    ax.fill_between(xs, xs - 2, xs + 2, color="gray", alpha=0.20, zorder=1)
    ax.plot(xs, xs, "k-", lw=1.5, zorder=2)
    ax.scatter(y, y_pre,  s=8, alpha=0.55, color=COLOUR["ideal"],
               edgecolor="none", zorder=3, label="Before — Perfect multiplier")
    ax.scatter(y, y_post, s=8, alpha=0.80, color=COLOUR["full"],
               edgecolor="none", zorder=4, label="After — Full Digital Twin")
    ax.set_title("Predicted vs. Measured", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Measured IF power  $y$  [dBm]")
    ax.set_ylabel(r"Predicted IF power  $\hat{y}$  [dBm]")
    ax.set_xlim(lo_y - 2, hi_y + 2)
    ax.set_ylim(min(lo_y, -120), max(hi_y, 30))
    ax.text(0.97, 0.04, "±2 dB and ±5 dB tolerance",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=9.5, style="italic", color="dimgray")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.axhline(0, color="black", lw=1.0)
    ax_r.scatter(y, res_pre,  s=6, alpha=0.45, color=COLOUR["ideal"], edgecolor="none")
    ax_r.scatter(y, res_post, s=6, alpha=0.85, color=COLOUR["full"], edgecolor="none")
    ax_r.set_title("Residual versus measured power", fontsize=11.5, fontweight="bold")
    ax_r.set_xlabel("Measured IF power  $y$  [dBm]")
    ax_r.set_ylabel("residual [dB]")
    ax_r.set_xlim(lo_y - 2, hi_y + 2)
    yr = max(abs(np.percentile(res_pre, 1)), abs(np.percentile(res_pre, 99)),
             5.0) * 1.1
    ax_r.set_ylim(-yr, yr)

    ax_t = fig.add_subplot(gs[:, 1])
    ax_t.axis("off")
    rows = [
        ("RMSE",             f"{m_pre['RMSE_dB']:.2f} dB",
                              f"{m_post['RMSE_dB']:.2f} dB"),
        ("MAE",              f"{m_pre['MAE_dB']:.2f} dB",
                              f"{m_post['MAE_dB']:.2f} dB"),
        ("Bias (mean err.)", f"{m_pre['Bias_dB']:+.3f} dB",
                              f"{m_post['Bias_dB']:+.3f} dB"),
        ("sigma (std err.)", f"{m_pre['sigma_dB']:.2f} dB",
                              f"{m_post['sigma_dB']:.2f} dB"),
        ("max |error|",      f"{m_pre['max_abs_err_dB']:.2f} dB",
                              f"{m_post['max_abs_err_dB']:.2f} dB"),
        ("R^2",              f"{m_pre['R2']:.4f}",
                              f"{m_post['R2']:.4f}"),
        ("Pearson rho",      f"{m_pre['Pearson_rho']:.4f}",
                              f"{m_post['Pearson_rho']:.4f}"),
        ("|err| <= 1 dB",    f"{m_pre['within_1dB_pct']:.1f} %",
                              f"{m_post['within_1dB_pct']:.1f} %"),
        ("|err| <= 2 dB",    f"{m_pre['within_2dB_pct']:.1f} %",
                              f"{m_post['within_2dB_pct']:.1f} %"),
        ("|err| <= 5 dB",    f"{m_pre['within_5dB_pct']:.1f} %",
                              f"{m_post['within_5dB_pct']:.1f} %"),
    ]
    table_text = "{:<22s}{:>14s}{:>14s}\n".format("Metric", "Before", "After")
    table_text += "-" * 50 + "\n"
    for label, b, a in rows:
        table_text += "{:<22s}{:>14s}{:>14s}\n".format(label, b, a)

    ax_t.text(0.04, 0.99, table_text, family="monospace",
              fontsize=11.5, va="top", ha="left",
              bbox=dict(facecolor="white", edgecolor="black",
                        boxstyle="round,pad=0.7", alpha=0.97))

    fig.suptitle(f"N = {N}   ·   Before / After Digital-Twin optimisation  —  "
                 f"scatter, residuals, and headline metrics",
                 fontsize=13.5, fontweight="bold")
    fname = OUT / f"fig_N{N}_5_before_after.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_cross_N_summary(data_by_N):
    Ns = sorted(data_by_N.keys())
    metric_per_N = {N: {
        "ideal": metrics(data_by_N[N]["dbm_ideal"], data_by_N[N]["dbm_meas"]),
        "phys":  metrics(data_by_N[N]["dbm_phys"],  data_by_N[N]["dbm_meas"]),
        "full":  metrics(data_by_N[N]["dbm_full"],  data_by_N[N]["dbm_meas"]),
    } for N in Ns}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), constrained_layout=True)
    x = np.arange(len(Ns))
    w = 0.27

    ax = axes[0]
    rmse_B = [metric_per_N[N]["ideal"]["RMSE_dB"] for N in Ns]
    rmse_C = [metric_per_N[N]["phys"]["RMSE_dB"]  for N in Ns]
    rmse_D = [metric_per_N[N]["full"]["RMSE_dB"]  for N in Ns]
    bB = ax.bar(x - w, rmse_B, w, color=COLOUR["ideal"], edgecolor="black",
                label="Perfect multiplier")
    bC = ax.bar(x,     rmse_C, w, color=COLOUR["phys"],  edgecolor="black",
                label="Physics-only DT")
    bD = ax.bar(x + w, rmse_D, w, color=COLOUR["full"],  edgecolor="black",
                label="Full Digital Twin")
    for bars, vals in [(bB, rmse_B), (bC, rmse_C), (bD, rmse_D)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.4,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"N = {N}" for N in Ns])
    ax.set_ylabel("RMSE [dB]")
    ax.set_title("RMSE per model, per vector length N",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(rmse_B) * 1.25)
    ax.legend(loc="upper right", framealpha=0.95, ncol=3, fontsize=9.5)

    ax = axes[1]
    h2_B = [metric_per_N[N]["ideal"]["within_2dB_pct"] for N in Ns]
    h2_C = [metric_per_N[N]["phys"]["within_2dB_pct"]  for N in Ns]
    h2_D = [metric_per_N[N]["full"]["within_2dB_pct"]  for N in Ns]
    bB = ax.bar(x - w, h2_B, w, color=COLOUR["ideal"], edgecolor="black",
                label="Perfect multiplier")
    bC = ax.bar(x,     h2_C, w, color=COLOUR["phys"],  edgecolor="black",
                label="Physics-only DT")
    bD = ax.bar(x + w, h2_D, w, color=COLOUR["full"],  edgecolor="black",
                label="Full Digital Twin")
    for bars, vals in [(bB, h2_B), (bC, h2_C), (bD, h2_D)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"N = {N}" for N in Ns])
    ax.set_ylabel("|err| <= 2 dB  hit-rate [%]")
    ax.set_title("Within-2-dB hit-rate per model, per N",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.legend(loc="upper left", framealpha=0.95, ncol=3, fontsize=9.5)

    fig.suptitle("Vector-extension PIML digital twin  —  cross-N summary",
                 fontsize=13.5, fontweight="bold")
    fname = OUT / "fig_summary_across_N.png"
    fig.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    data_by_N = {N: load_one_N(N, run_dir=PROJ / "run") for N in (2, 4, 8)}

    print(f"{'N':>3} {'model':>10} {'RMSE[dB]':>10} {'R2':>8} {'|err|<=2dB %':>14}")
    for N, d in data_by_N.items():
        for key, label in [("ideal", "Perfect"),
                           ("phys",  "Phys-only"),
                           ("full",  "Full PIML")]:
            m = metrics(d[f"dbm_{key}"], d["dbm_meas"])
            print(f"{N:>3d} {label:>10s} {m['RMSE_dB']:>10.3f} "
                  f"{m['R2']:>8.4f} {m['within_2dB_pct']:>14.2f}")

    produced = []
    for N, d in data_by_N.items():
        produced.append(fig_redblack_heatmap(d, N))
        produced.append(fig_ABCD_cascade(d, N))
        produced.append(fig_scatter_three_way(d, N))
        produced.append(fig_headline_bars(d, N))
        produced.append(fig_before_after(d, N))
    produced.append(fig_cross_N_summary(data_by_N))

    print("\nFigures written:")
    for p in produced:
        print(f"  {p}")


if __name__ == "__main__":
    main()
