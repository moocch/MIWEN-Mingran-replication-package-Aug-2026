# -*- coding: ascii -*-
"""verify_table6.py -- assert every cell of Supplementary Table 6
(Supplementary Note 8, S8.2: "Training ablation on the simulated chain",
\\label{tab:analogablation}) against the released simulation archive.

Primary source : <archive>/_shared_fully_analog_simulation/results/
                 hardware_aware_ablation.json
Fallback       : ./data/hardware_aware_ablation.json (byte-identical copy from
                 the released miwen_fully_analog_archive.zip, code/results/;
                 the shared results/ folder is missing this file even though
                 its MANIFEST lists it)
Cross-checks   : _shared .../results/results_summary.json  (MZ_L3 / Z_L3 /
                 MZD_L3 / ZD_L3 tags), _shared .../weights/train_summary.json
                 (digital baselines of the _nohw / _ideal checkpoints),
                 _shared .../reproduce.py (protocol: repeats=5, chain mode Z)

Table 6 (values in %; chain = mean +- 1 s.d.; drop in points):

  Task   Training loss         Digital   Chain            Drop
  MNIST  twin + link noise     96.07     95.62 +- 0.09    0.45
  MNIST  link noise, no twin   96.03     94.20 +- 0.15    1.83
  MNIST  ideal square law      95.77     31.97 +- 0.23    63.8
  GTSRB  twin + link noise     88.29     86.26 +- 0.13    2.03
  GTSRB  link noise, no twin   89.19     65.70 +- 0.17    23.5
  GTSRB  ideal square law      85.28      8.16 +- 0.35    77.1
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir, code_dir as _code_dir
# ---------------------------------------------------------------------------

import json
import re
import sys
from pathlib import Path

HERE = _data_dir(__file__)
SHARED = HERE.parent / "raw" / "fully_analog_simulation"

TOL = 0.005
n_pass = n_fail = 0


def check(label, got, want, tol=TOL):
    global n_pass, n_fail
    ok = abs(got - want) <= tol
    n_pass += ok
    n_fail += (not ok)
    print("  %-58s got %8.2f  want %8.2f  %s"
          % (label, got, want, "PASS" if ok else "FAIL"))
    return ok


def check_true(label, cond):
    global n_pass, n_fail
    n_pass += bool(cond)
    n_fail += (not cond)
    print("  %-58s %s" % (label, "PASS" if cond else "FAIL"))
    return cond


def load_json(name):
    for p in (SHARED / "results" / name, HERE / "data" / name):
        if p.exists():
            print("[source] %s -> %s" % (name, p))
            return json.load(open(p))
    raise FileNotFoundError(name)


print("=" * 78)
print("Supplementary Table 6 (tab:analogablation) -- training ablation,")
print("simulated chain -- verification against hardware_aware_ablation.json")
print("=" * 78)

abl = load_json("hardware_aware_ablation.json")
summ = load_json("results_summary.json")
train = json.load(open(SHARED / "weights" / "train_summary.json"))
print()

# SI row -> JSON key of hardware_aware_ablation.json
# (JSON labels differ from the SI wording -- see README)
ROWS = [
    # (SI row,                    JSON key,       digital, chain,  std,  SI drop, JSON drop)
    ("MNIST twin + link noise",   "MZ_L3",        96.07,  95.62, 0.09,  0.45,  0.45),
    ("MNIST link noise, no twin", "MZD_L3_nohw",  96.03,  94.20, 0.15,  1.83,  1.83),
    ("MNIST ideal square law",    "MZI_L3_ideal", 95.77,  31.97, 0.23, 63.8,  63.80),
    ("GTSRB twin + link noise",   "Z_L3",         88.29,  86.26, 0.13,  2.03,  2.03),
    ("GTSRB link noise, no twin", "ZD_L3_nohw",   89.19,  65.70, 0.17, 23.5,  23.49),
    ("GTSRB ideal square law",    "ZI_L3_ideal",  85.28,   8.16, 0.35, 77.1,  77.12),
]

for si_row, key, dig, ana, std, si_drop, js_drop in ROWS:
    r = abl[key]
    print("%s  (JSON key '%s', label '%s'):" % (si_row, key, r["label"]))
    check("digital (digital_pct)", r["digital_pct"], dig)
    check("chain (analog_pct)",    r["analog_pct"], ana)
    check("chain s.d. (analog_std)", r["analog_std"], std)
    check("drop (drop_pts)",       r["drop_pts"], js_drop)
    # SI prints the drop to 1 decimal for the two large collapses
    check("drop as printed in the SI table", round(r["drop_pts"], 1)
          if si_drop in (63.8, 23.5, 77.1) else r["drop_pts"], si_drop,
          tol=0.051)
    check_true("drop == digital - chain (2-dec rounding)",
               abs(r["drop_pts"] - round(r["digital_pct"]
                                         - r["analog_pct"], 2)) <= 0.011)

# ------------------------------------------------------- cross-checks ------
print("Cross-checks against results_summary.json (same release, repeats=5):")
check("tag MZ_L3 chain acc",  summ["MZ_L3"]["accuracy_pct"], 95.62)
check("tag MZ_L3 chain std",  summ["MZ_L3"]["std_pct"], 0.09)
check("tag MZD_L3 (weights MZ_L3, mode ZD) = MNIST digital",
      summ["MZD_L3"]["accuracy_pct"], 96.07)
check("tag Z_L3 chain acc",   summ["Z_L3"]["accuracy_pct"], 86.26)
check("tag Z_L3 chain std",   summ["Z_L3"]["std_pct"], 0.13)
check("tag ZD_L3 (weights Z_L3, mode ZD) = GTSRB digital",
      summ["ZD_L3"]["accuracy_pct"], 88.29)
check_true("MZ_L3 / Z_L3 evaluated with repeats == 5",
           summ["MZ_L3"]["repeats"] == 5 and summ["Z_L3"]["repeats"] == 5)

print("Cross-checks against weights/train_summary.json (digital baselines")
print("of the ablation checkpoints = their training-time accuracy):")
check("MZD_L3_nohw (mode ZD)",  100 * train["MZD_L3_nohw"]["acc"], 96.03)
check("MZI_L3_ideal (mode ZI)", 100 * train["MZI_L3_ideal"]["acc"], 95.77)
check("ZD_L3_nohw (mode ZD)",   100 * train["ZD_L3_nohw"]["acc"], 89.19)
check("ZI_L3_ideal (mode ZI)",  100 * train["ZI_L3_ideal"]["acc"], 85.28)

# ------------------------------------------------------- protocol ----------
print("Protocol ('five noise realizations', chain evaluation):")
src = (_code_dir(__file__).parent.parent / "pipelines"
       / "fully_analog_simulation" / "reproduce.py").read_text()
check_true("reproduce.py: def ablation(repeats=5)",
           re.search(r"def ablation\(repeats=5\)", src) is not None)
check_true("reproduce.py: chain column evaluated under mode 'Z', "
           "digital under ZD/ZI (1 draw)",
           'dig, ana = run(dmode, 1), run("Z", repeats)' in src)

print()
print("RESULT: %d passed, %d failed" % (n_pass, n_fail))
sys.exit(0 if n_fail == 0 else 1)
