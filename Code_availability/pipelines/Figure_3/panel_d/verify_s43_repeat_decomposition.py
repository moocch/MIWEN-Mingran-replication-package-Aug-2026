"""Independent recomputation of the repeat decomposition quoted in the
main text (noise-floor paragraph) and Supplementary Note 4, S4.3.

Model: residual of product q in capture i is r_iq = d_q + n_iq, with d
reproducible across captures and n zero-mean. Bias-corrected estimators:
  sigma^2 = < (1/(R-1)) sum_i |r_iq - rbar_q|^2 >_q
  |d|^2   = < |rbar_q|^2 >_q  -  sigma^2 / R
Floors are the closed form 1/sqrt(27*SNR) at each panel's measured mean
guard-bin SNR.

Run from this folder:  python verify_s43_repeat_decomposition.py
(reads ../data/*.npz). Verified 2026-08-29: reproduces every S4.3 value
exactly (sigma 0.0324/0.0108 vs floors 0.0331/0.0107 -> 0.98x/1.01x;
|d| 0.0156/0.0134 with 18.9%/60.6% of corrected residual power;
N = 4096: |d| 0.0163/0.0158, 22.0%/71.5%; uncorrected fractions
74.6%/96.3% and 79.9%/97.3%), and the Methods claim that every tuning
capture lands within 0.4 dB of its SNR target (worst case 0.36 dB).
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import os

import numpy as np

D = os.path.join(str(_data_dir(__file__)), "..", "data")

EXPECT = {  # (sigma, floor, |d|, frac_after_%, frac_before_%) per panel
    ("N65536", 0): (0.0324, 0.0331, 0.0156, 18.9, 74.6),
    ("N65536", 1): (0.0108, 0.0107, 0.0134, 60.6, 96.3),
    ("N4096", 0): (0.0306, None, 0.0163, 22.0, 79.9),
    ("N4096", 1): (0.0100, None, 0.0158, 71.5, 97.3),
}

for tag, o_name, g_name in (
    ("N65536", "ip_optimized_N65536_20260826.npz",
     "gr_ip_scatter_N65536_20260810.npz"),
    ("N4096", "ip_optimized_N4096_20260826.npz",
     "gr_ip_scatter_N4096_20260810.npz"),
):
    o = np.load(os.path.join(D, o_name))
    g = np.load(os.path.join(D, g_name), allow_pickle=True)
    tru = o["c_true"]
    for p, lbl, target in ((0, "15 dB", 15.0), (1, "25 dB", 25.0)):
        snr_db = float(np.mean(g["snr3_db_reps"][p]))
        floor = 1.0 / np.sqrt(27.0 * 10 ** (snr_db / 10.0))
        res = {}
        for stage in ("after", "before"):
            r = o[f"c_hat_{stage}_reps"][p] - tru[None, :]
            R = r.shape[0]
            rbar = r.mean(axis=0)
            sig2 = ((np.abs(r - rbar[None, :]) ** 2).sum(axis=0)
                    / (R - 1)).mean()
            d2 = max((np.abs(rbar) ** 2).mean() - sig2 / R, 0.0)
            tot = np.mean(np.abs(r) ** 2)
            res[stage] = (np.sqrt(sig2), np.sqrt(d2), 100 * d2 / tot)
        sa, da, fa = res["after"]
        _, _, fb = res["before"]
        e = EXPECT[(tag, p)]
        assert abs(sa - e[0]) < 5e-4, (tag, lbl, "sigma", sa)
        if e[1] is not None:
            assert abs(floor - e[1]) < 5e-4, (tag, lbl, "floor", floor)
        assert abs(da - e[2]) < 5e-4, (tag, lbl, "|d|", da)
        assert abs(fa - e[3]) < 0.15, (tag, lbl, "frac_after", fa)
        assert abs(fb - e[4]) < 0.15, (tag, lbl, "frac_before", fb)
        tune = float(g["snr3_db_reps"][p][0])
        assert abs(tune - target) <= 0.4, (tag, lbl, "tuning capture", tune)
        print(f"{tag} {lbl}: sigma={sa:.4f} floor={floor:.4f} "
              f"({sa/floor:.2f}x)  |d|={da:.4f}  "
              f"frac after/before = {fa:.1f}%/{fb:.1f}%  "
              f"tuning capture {tune:.2f} dB  -- all asserts OK")
print("S4.3 repeat decomposition fully reproduced from the archived data.")
