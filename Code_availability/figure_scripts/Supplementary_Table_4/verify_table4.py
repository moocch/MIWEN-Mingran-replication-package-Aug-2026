#!/usr/bin/env python3
"""Verify every cell of Supplementary Table 4 ("Measured accuracy--energy
comparison of the two adaptation routes", also reproduced in the response
letter, R3 Comment 2) against the archived artifacts in THIS folder.

Inputs (all archive-local, copied verbatim from the pipeline archives —
see README.md for source paths and MD5s):
  fig5_energy_package/data/fig5_plot_data.npz   measured client-energy
      accounting of the inner-product primitive (25 dB panel; N=4096, 65536)
  ip_optimized_N65536_20260826.npz              out-of-sample twin-correction
      results at N=65,536 (rows: 15 dB, 25 dB)
  provenance_comb_accuracies.md                 provenance of the comb-route
      classification numbers 98.92 / 99.50 (N=1,200 GTSRB battery)
  README_hardware_aware_training_v2.md          provenance of the serial-route
      classification numbers 98.50 / 5.67 (N=600, same frozen battery[0:600])
  verify_accuracy.py                            campaign recomputation script
      for the serial-route numbers (raw prediction npz live in the campaign
      folder, not here; see below)

The ONLY arithmetic not read from an artifact is the client-side cost of
DEPLOYING the twin correction: applying its four complex coefficients per
answer = 16 real MACs = 16 pJ at the benchmark constant e_dig = 1 pJ/MAC
(read from meta_json of fig5_plot_data.npz). Feature templates depend only
on the transmitted waveforms and are precomputed off-client; the four
coefficients are fitted offline. Amortized per real MAC of the N-point
complex inner product (4N real MACs): fee = 16 pJ / (4N).

Usage:  python verify_table4.py          (numpy only)
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = _data_dir(__file__)
H100_FJ = 70.0  # the 70-fJ/MAC GPU (H100) reference of the table header

RESULTS = []


def check(cell, quoted, archive_value, source):
    ok = quoted == archive_value
    RESULTS.append((cell, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {cell}: quoted {quoted} | "
          f"archive {archive_value}  <- {source}")
    return ok


def check_close(cell, a, b, source, rtol=1e-9):
    ok = bool(np.isclose(a, b, rtol=rtol))
    RESULTS.append((cell, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {cell}: {a} vs {b}  <- {source}")
    return ok


def note(cell, ok, text):
    RESULTS.append((cell, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {cell}: {text}")


# ---------------------------------------------------------------- load data
d = np.load(HERE / "fig5_energy_package" / "data" / "fig5_plot_data.npz",
            allow_pickle=True)
c = np.load(HERE / "ip_optimized_N65536_20260826.npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))

iN = {int(n): i for i, n in enumerate(d["points_N"])}          # {4096:0, 65536:1}
iS = {str(s): i for i, s in enumerate(c["labels"])}            # {'15 dB':0, '25 dB':1}
e_dig_J = meta["e_dig_J"]                                       # 1 pJ / real MAC
h100_J = float(d["h100_line"])

e_ip_fJ = {n: float(d["points_e_ip"][i]) * 1e15 for n, i in iN.items()}

print("=" * 78)
print("Supplementary Table 4 — cell-by-cell verification against archived data")
print("=" * 78)

# ------------------------------------------------- benchmark constants
print("\n--- benchmark constants ---")
check("GPU reference = 70 fJ/MAC", H100_FJ, h100_J * 1e15,
      "fig5_plot_data.npz h100_line (meta: h100_J_per_mac)")
check("e_dig = 1 pJ per real MAC", 1.0, e_dig_J * 1e12,
      "fig5_plot_data.npz meta_json e_dig_J")
check("energy panel is the 25 dB panel", "25 dB", meta["panel"],
      "fig5_plot_data.npz meta_json panel")
check("correction file is N=65,536", 65536, int(c["vec_N"]),
      "ip_optimized_N65536_20260826.npz vec_N")

# ------------------------------------------------- deployment-fee convention
print("\n--- twin-correction deployment fee (the only new arithmetic) ---")
n_coeff = int(c["coeffs"].shape[1])
check("correction = 4 complex coefficients per answer", 4, n_coeff,
      "ip_optimized_N65536_20260826.npz coeffs.shape / basis_names "
      + str(list(map(str, c["basis_names"]))))
real_macs = 4 * n_coeff
check("=> 16 real MACs = 16 pJ per answer", 16, real_macs,
      "4 real MACs per complex MAC x 4 coefficients, at e_dig = 1 pJ/MAC")
fee_fJ = {n: real_macs * e_dig_J * 1e15 / (4 * n) for n in iN}
print(f"       fee per real MAC: N=65536 -> {fee_fJ[65536]:.4f} fJ, "
      f"N=4096 -> {fee_fJ[4096]:.4f} fJ  (16 pJ / 4N)")

# ------------------------------------------------- TOP BLOCK, N=65,536, 25 dB
print("\n--- top block: inner-product primitive, N=65,536, 25 dB ---")
i = iN[65536]
j = iS["25 dB"]
check("raw decode RMSE = 0.056", 0.056, round(float(d["points_rmse_mean"][i]), 3),
      f"fig5_plot_data.npz points_rmse_mean[N=65536] = {d['points_rmse_mean'][i]:.6f}")
check("raw ENOB = 2.58", 2.58, round(float(d["points_enob_mean"][i]), 2),
      f"fig5_plot_data.npz points_enob_mean[N=65536] = {d['points_enob_mean'][i]:.5f}")
check("raw energy = 0.588 fJ per real MAC", 0.588, round(e_ip_fJ[65536], 3),
      f"fig5_plot_data.npz points_e_ip[N=65536] = {e_ip_fJ[65536]:.6f} fJ")
check("raw advantage = 119x vs 70-fJ GPU", 119,
      round(float(d["points_speedup_vs_h100"][i])),
      f"fig5_plot_data.npz points_speedup_vs_h100[N=65536] = "
      f"{d['points_speedup_vs_h100'][i]:.4f}")
check_close("stored speedup == h100_line / points_e_ip (N=65536)",
            float(d["points_speedup_vs_h100"][i]),
            h100_J / float(d["points_e_ip"][i]), "internal consistency")
check("+ twin correction RMSE = 0.017", 0.017,
      round(float(c["rmse_after_mean"][j]), 3),
      f"ip_optimized npz rmse_after_mean[25 dB] = {c['rmse_after_mean'][j]:.6f}")
check("+ twin correction ENOB = 4.28", 4.28,
      round(float(c["enob_after_mean"][j]), 2),
      f"ip_optimized npz enob_after_mean[25 dB] = {c['enob_after_mean'][j]:.5f}")
tot65536 = e_ip_fJ[65536] + fee_fJ[65536]
check("+ correction energy = 0.649 fJ per real MAC", 0.649, round(tot65536, 3),
      f"0.587794 fJ + 16 pJ/(4*65536) = {tot65536:.6f} fJ")
check("+ correction advantage = 108x", 108, round(H100_FJ / tot65536),
      f"70 fJ / {tot65536:.6f} fJ = {H100_FJ / tot65536:.4f}")

print("\n--- cross-checks: the two npz files describe the same measurement ---")
check_close("points_rmse_mean[N=65536] == rmse_before_mean[25 dB]",
            float(d["points_rmse_mean"][i]), float(c["rmse_before_mean"][j]),
            "raw 25-dB RMSE identical in both archives")
check_close("points_enob_mean[N=65536] == enob_before_mean[25 dB]",
            float(d["points_enob_mean"][i]), float(c["enob_before_mean"][j]),
            "raw 25-dB ENOB identical in both archives")

# ------------------------------------------------- S3.5 companions
print("\n--- S3.5 companion numbers ---")
k = iN[4096]
check("N=4096 raw energy = 1.467 fJ", 1.467, round(e_ip_fJ[4096], 3),
      f"fig5_plot_data.npz points_e_ip[N=4096] = {e_ip_fJ[4096]:.6f} fJ")
tot4096 = e_ip_fJ[4096] + fee_fJ[4096]
check("N=4096 corrected energy = 2.444 fJ", 2.444, round(tot4096, 3),
      f"1.467273 fJ + 16 pJ/(4*4096) = {tot4096:.6f} fJ")
check("N=4096 raw advantage = 47.7x", 47.7,
      round(float(d["points_speedup_vs_h100"][k]), 1),
      f"fig5_plot_data.npz points_speedup_vs_h100[N=4096] = "
      f"{d['points_speedup_vs_h100'][k]:.4f}")
check("N=4096 corrected advantage = 28.6x", 28.6, round(H100_FJ / tot4096, 1),
      f"70 fJ / {tot4096:.6f} fJ = {H100_FJ / tot4096:.4f}")
m = iS["15 dB"]
check("15-dB raw RMSE = 0.064 (N=65,536)", 0.064,
      round(float(c["rmse_before_mean"][m]), 3),
      f"ip_optimized npz rmse_before_mean[15 dB] = {c['rmse_before_mean'][m]:.6f}")
check("15-dB corrected RMSE = 0.036 (N=65,536)", 0.036,
      round(float(c["rmse_after_mean"][m]), 3),
      f"ip_optimized npz rmse_after_mean[15 dB] = {c['rmse_after_mean'][m]:.6f}")

# ------------------------------------------------- BOTTOM BLOCK
print("\n--- bottom block: 43-class GTSRB classification ---")
comb = (HERE / "provenance_comb_accuracies.md").read_text(encoding="utf-8")
serial = (HERE / "README_hardware_aware_training_v2.md").read_text(encoding="utf-8")

# comb route: noise-injection training, measured 98.92 vs clean digital 99.50
m1 = re.search(r"\*\*98\.92 \u00b1 0\.30\*\*.*?(\d+)/1200", comb)
if m1:
    n_ok = int(m1.group(1))
    check("comb measured (noise-injection training) = 98.92%", 98.92,
          round(100 * n_ok / 1200, 2),
          f"provenance_comb_accuracies.md: '98.92 +/- 0.30 ... {n_ok}/1200', "
          "battery_frozen_slim.npz, energy-honest rerun 2026-08-07")
else:
    note("comb measured (noise-injection training) = 98.92%", False,
         "pattern '**98.92 \u00b1 0.30** ... n/1200' not found in provenance doc")
m2 = re.search(r"98\.9167% \((\d+)/1200\)", comb)
if m2:
    check("  curator recomputation agrees (1187/1200)", 98.92,
          round(100 * int(m2.group(1)) / 1200, 2),
          "provenance_comb_accuracies.md 'Verification performed while curating'")
else:
    note("  curator recomputation agrees (1187/1200)", False,
         "recomputation line not found in provenance doc")
note("comb digital baseline (clean-trained) = 99.50%",
     bool(re.search(r"\| 3 \| \*\*99\.50\*\* \| Clean-trained", comb)),
     "provenance_comb_accuracies.md row #3: clean-trained digital model, "
     "same architecture, same 1,200 images (r35_r3plus_s0_hw.npz)")

# serial route: twin-based training 98.50 vs clean weights 5.67
pairs = {p: int(n) for p, n in
         re.findall(r"\*\*(\d+\.\d+)\s*%\*\*\s*\((\d+)/600\)", serial)}
if "98.50" in pairs:
    check("serial measured (twin-based training) = 98.50%", 98.50,
          round(100 * pairs["98.50"] / 600, 2),
          f"README_hardware_aware_training_v2.md main table: "
          f"{pairs['98.50']}/600, hardware, (0,0) dBm, N=600")
else:
    note("serial measured (twin-based training) = 98.50%", False,
         "'**98.50 %** (n/600)' not found in serial README")
if "5.67" in pairs:
    check("serial measured (clean weights) = 5.67%", 5.67,
          round(100 * pairs["5.67"] / 600, 2),
          f"README_hardware_aware_training_v2.md main table: "
          f"{pairs['5.67']}/600 (collapsed: 590/600 predict class 12)")
else:
    note("serial measured (clean weights) = 5.67%", False,
         "'**5.67 %** (n/600)' not found in serial README")
note("serial clean-digital reference = 99.50% (ideal)",
     bool(re.search(r"99\.50\s?%", serial)),
     "README_hardware_aware_training_v2.md main table, clean row, digital column")

# added inference energy = 0 for both routes (convention, backed by the docs)
note("comb route: added inference energy = 0",
     "host computes NO W@x" in comb,
     "by construction — noise injection changes training only; the 98.92 run "
     "is the energy-honest rerun with frozen offline calibration "
     "('host computes NO W@x', provenance_comb_accuracies.md)")
note("serial route: added inference energy = 0",
     "\u63a8\u7406\u8def\u5f84\u91cc\u4e0d\u542b\u4efb\u4f55 twin \u6210\u5206"
     in serial,
     "by construction — the twin acts in the training loop only; the serial "
     "README states the inference path contains no twin component")

# executable proof of the serial numbers
raw_dirs = ["05_hardware_result_0dBm_N600", "06_digital_comparators",
            "08_frozen_inputs_and_labels"]
if all((HERE / r).is_dir() for r in raw_dirs):
    print("\n  raw serial prediction files found locally -> run: "
          "python verify_accuracy.py")
else:
    print("\n  NOTE: verify_accuracy.py (copied here for reference) needs the "
          "raw prediction npz\n  folders "
          + ", ".join(raw_dirs) + ",\n  which are NOT copied into this archive. "
          "The executable proof lives in the campaign folder\n  "
          r'"E:\archive\Manuscript\Overleaf'
          r"\202607_MIWEN_Manuscript\V2\hardware_aware_training_v2" + '"\n'
          "  (python verify_accuracy.py there prints clean 34/600 = 5.67% and "
          "twin 591/600 = 98.50%);\n  here the cells are asserted against the "
          "archived provenance documents above.")

# ---------------------------------------------------------------- summary
n_fail = sum(1 for _, ok in RESULTS if not ok)
print("\n" + "=" * 78)
print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed"
      + ("" if n_fail == 0 else f"  ({n_fail} FAILED)"))
print("=" * 78)
sys.exit(1 if n_fail else 0)
