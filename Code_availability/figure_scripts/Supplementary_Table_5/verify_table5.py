# -*- coding: ascii -*-
"""verify_table5.py -- assert every cell of Supplementary Table 5
(Supplementary Note 8, S8.1: "Hardware-aware 2x2 on the simulated fully
analog chain", \\label{tab:analog2x2}) against the released simulation
archive.

Primary source : <archive>/_shared_fully_analog_simulation/results/twin_2x2.json
Fallback       : ./data/twin_2x2.json  (byte-identical copy extracted from the
                 released miwen_fully_analog_archive.zip, code/results/;
                 the shared results/ folder in this reproducibility archive
                 is missing twin_2x2.json even though its MANIFEST lists it)
Cross-checks   : _shared .../weights/train_summary.json  (digital columns)
                 _shared .../results/results_summary.json (repeats == 5)
                 ./data/twin_2x2.py (evaluation protocol: 5 noise draws, ZH)

Table 5 (values in %; chain = mean +- 1 s.d. over five noise realizations):

  Task   Weights        Digital   Under twin (forecast)   Chain
  MNIST  clean          95.77     18.25                   15.77 +- 0.30
  MNIST  twin-trained   94.60     ---                     92.62 +- 0.08
  GTSRB  clean          85.28     8.65                     6.06 +- 0.08
  GTSRB  twin-trained   87.08     ---                     75.13 +- 0.18
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

HERE = _data_dir(__file__)
SHARED = HERE.parent / "raw" / "fully_analog_simulation"

TOL = 0.005          # all SI values are quoted to 2 decimals
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
    """Shared archive results/ first, local data/ copy second."""
    for p in (SHARED / "results" / name, HERE / "data" / name):
        if p.exists():
            print("[source] %s -> %s" % (name, p))
            return json.load(open(p))
    raise FileNotFoundError(name)


print("=" * 78)
print("Supplementary Table 5 (tab:analog2x2) -- hardware-aware 2x2,")
print("simulated fully analog chain -- verification against twin_2x2.json")
print("=" * 78)

twin = load_json("twin_2x2.json")
summ = load_json("results_summary.json")
train = json.load(open(SHARED / "weights" / "train_summary.json"))
print()

# --------------------------------------------------------------- MNIST ----
# JSON keys: twin_2x2.json[<dataset>][<cell>] with cell in
#   clean_digital / clean_under_twin / clean_hardware /
#   twin_digital  / twin_hardware      each {acc, std}
print("MNIST row 'clean' (weights MZC_L3; eval modes ZC / ZT / ZH):")
m = twin["mnist"]
check("digital (clean_digital.acc)",       m["clean_digital"]["acc"], 95.77)
check("digital s.d. (clean_digital.std)",  m["clean_digital"]["std"], 0.00)
check("under-twin forecast (clean_under_twin.acc)",
      m["clean_under_twin"]["acc"], 18.25)
check("chain (clean_hardware.acc)",        m["clean_hardware"]["acc"], 15.77)
check("chain s.d. (clean_hardware.std)",   m["clean_hardware"]["std"], 0.30)

print("MNIST row 'twin-trained' (weights MZT_L3; eval modes ZT / ZH):")
check("digital (twin_digital.acc)",        m["twin_digital"]["acc"], 94.60)
check("chain (twin_hardware.acc)",         m["twin_hardware"]["acc"], 92.62)
check("chain s.d. (twin_hardware.std)",    m["twin_hardware"]["std"], 0.08)

# --------------------------------------------------------------- GTSRB ----
print("GTSRB row 'clean' (weights ZC_L3; eval modes ZC / ZT / ZH):")
g = twin["gtsrb"]
check("digital (clean_digital.acc)",       g["clean_digital"]["acc"], 85.28)
check("digital s.d. (clean_digital.std)",  g["clean_digital"]["std"], 0.00)
check("under-twin forecast (clean_under_twin.acc)",
      g["clean_under_twin"]["acc"], 8.65)
check("chain (clean_hardware.acc)",        g["clean_hardware"]["acc"], 6.06)
check("chain s.d. (clean_hardware.std)",   g["clean_hardware"]["std"], 0.08)

print("GTSRB row 'twin-trained' (weights ZT_L3; eval modes ZT / ZH):")
check("digital (twin_digital.acc)",        g["twin_digital"]["acc"], 87.08)
check("chain (twin_hardware.acc)",         g["twin_hardware"]["acc"], 75.13)
check("chain s.d. (twin_hardware.std)",    g["twin_hardware"]["std"], 0.18)

# ------------------------------------------- digital-column cross-checks ---
print("Cross-check: digital column == training-time accuracy of the same")
print("checkpoint (weights/train_summary.json, acc in [0,1] x 100):")
check("MZC_L3 (MNIST clean, mode ZC)",  100 * train["MZC_L3"]["acc"], 95.77)
check("MZT_L3 (MNIST twin,  mode ZT)",  100 * train["MZT_L3"]["acc"], 94.60)
check("ZC_L3  (GTSRB clean, mode ZC)",  100 * train["ZC_L3"]["acc"], 85.28)
check("ZT_L3  (GTSRB twin,  mode ZT)",  100 * train["ZT_L3"]["acc"], 87.08)

# -------------------------------------------- five noise realizations ------
print("Caption claim 'mean +- 1 s.d. over five noise realizations':")
reps = sorted({v.get("repeats") for v in summ.values()})
check_true("every entry of results_summary.json has repeats == 5",
           reps == [5])
src = (HERE / "data" / "twin_2x2.py").read_text() \
    if (HERE / "data" / "twin_2x2.py").exists() else ""
check_true("twin_2x2.py main() default repeats=5 (chain cells)",
           re.search(r"def main\(.*repeats=5\)", src) is not None)
check_true("twin_2x2.py chain cells use mode ZH with repeats=repeats",
           'acc("ZH", ds, clean_w, repeats=repeats)' in src
           and 'acc("ZH", ds, twin_w, repeats=repeats)' in src)
check_true("twin_2x2.py digital/forecast cells deterministic (repeats=1)",
           re.search(r"def acc\(mode, ds, wtag, L=3, repeats=1\)", src)
           is not None)

print()
print("RESULT: %d passed, %d failed" % (n_pass, n_fail))
sys.exit(0 if n_fail == 0 else 1)
