# -*- coding: ascii -*-
"""verify_note8.py -- assert the Supplementary Note 8 text numbers that are
not Table 5/6 cells (S8.1 recovery sentence, S8.3 depth controls and
per-pass budget), and resolve the two flagged tensions:

  (a) the recovery endpoints 93.90 / 87.53 match no Table 5/6 row;
  (b) L = 3 gaps quoted as 0.40 / 1.30 (S8.3, "main text") vs. Table 6's
      drops 0.45 / 2.03.

Primary sources: <archive>/_shared_fully_analog_simulation/results/
                 results_summary.json, link_budget.json
Fallback (files missing from the shared results/ folder, byte-identical
copies from the released miwen_fully_analog_archive.zip): ./data/
                 twin_2x2.json, hardware_aware_ablation.json,
                 paper_numbers.json
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import json
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
    print("  %-62s got %9.2f  want %9.2f  %s"
          % (label, got, want, "PASS" if ok else "FAIL"))
    return ok


def check3(label, got, want, tol=0.0005):
    global n_pass, n_fail
    ok = abs(got - want) <= tol
    n_pass += ok
    n_fail += (not ok)
    print("  %-62s got %9.3f  want %9.3f  %s"
          % (label, got, want, "PASS" if ok else "FAIL"))
    return ok


def check_true(label, cond):
    global n_pass, n_fail
    n_pass += bool(cond)
    n_fail += (not cond)
    print("  %-62s %s" % (label, "PASS" if cond else "FAIL"))
    return cond


def load_json(name):
    for p in (SHARED / "results" / name, HERE / "data" / name):
        if p.exists():
            print("[source] %s -> %s" % (name, p))
            return json.load(open(p))
    raise FileNotFoundError(name)


print("=" * 78)
print("Supplementary Note 8 text numbers (S8.1 recovery, S8.3 depth controls")
print("and per-pass budget) + resolution of the two flagged tensions")
print("=" * 78)

summ = load_json("results_summary.json")
budget = load_json("link_budget.json")
twin = load_json("twin_2x2.json")
abl = load_json("hardware_aware_ablation.json")
paper = load_json("paper_numbers.json")
print()


def acc(tag):
    return summ[tag]["accuracy_pct"], summ[tag]["std_pct"]


# ------------------------------------------------ S8.1 recovery sentence ---
print("S8.1: link-noise recovery 92.62 -> 93.90 (MNIST), 75.13 -> 87.53")
print("(from = twin arm of Table 5, ZT-trained weights on the ZH chain;")
print(" to = ZH-trained weights on the ZH chain, tags MZH_L3 / ZH_L3):")
check("MNIST from: twin_2x2 mnist.twin_hardware.acc",
      twin["mnist"]["twin_hardware"]["acc"], 92.62)
a, s = acc("MZH_L3")
check("MNIST to: results_summary MZH_L3 acc (mode ZH)", a, 93.90)
check("MNIST to: MZH_L3 std", s, 0.10)
check("GTSRB from: twin_2x2 gtsrb.twin_hardware.acc",
      twin["gtsrb"]["twin_hardware"]["acc"], 75.13)
a, s = acc("ZH_L3")
check("GTSRB to: results_summary ZH_L3 acc (mode ZH)", a, 87.53)
check("GTSRB to: ZH_L3 std", s, 0.14)
check_true("both endpoints share the ZH-chain protocol (repeats=5)",
           summ["MZH_L3"]["repeats"] == 5 and summ["ZH_L3"]["repeats"] == 5)

# ------------------------------------------------- S8.3 depth controls -----
print("S8.3: L=2 gaps 0.22 (94.04 vs 93.82 +- 0.03), 0.87 (87.68 vs")
print("86.81 +- 0.05) -- tags MZT_L2d/MZH_L2 and ZT_L2d/ZH_L2:")
d, _ = acc("MZT_L2d")
a, s = acc("MZH_L2")
check("MNIST L=2 digital (MZT_L2d, mode ZT)", d, 94.04)
check("MNIST L=2 chain (MZH_L2, mode ZH)", a, 93.82)
check("MNIST L=2 chain std", s, 0.03)
check("MNIST L=2 gap", round(d - a, 2), 0.22)
d, _ = acc("ZT_L2d")
a, s = acc("ZH_L2")
check("GTSRB L=2 digital (ZT_L2d, mode ZT)", d, 87.68)
check("GTSRB L=2 chain (ZH_L2, mode ZH)", a, 86.81)
check("GTSRB L=2 chain std", s, 0.05)
check("GTSRB L=2 gap", round(d - a, 2), 0.87)

print("S8.3: L=3 gaps 0.40 / 1.30 'main text' -- tags MZT_L3d/MZH_L3,")
print("ZT_L3d/ZH_L3 (the Fig. 5d pairs 94.30/93.90 and 88.83/87.53):")
d, _ = acc("MZT_L3d")
check("MNIST L=3 digital (MZT_L3d, mode ZT)", d, 94.30)
check("MNIST L=3 gap", round(d - acc("MZH_L3")[0], 2), 0.40)
d, _ = acc("ZT_L3d")
check("GTSRB L=3 digital (ZT_L3d, mode ZT)", d, 88.83)
check("GTSRB L=3 gap", round(d - acc("ZH_L3")[0], 2), 1.30)

print("S8.3: L=4 collapse 85.02 vs 15.39 +- 0.23 -- tags MZT_L4d/MZH_L4:")
d, _ = acc("MZT_L4d")
a, s = acc("MZH_L4")
check("MNIST L=4 digital (MZT_L4d)", d, 85.02)
check("MNIST L=4 chain (MZH_L4)", a, 15.39)
check("MNIST L=4 chain std", s, 0.23)

# ------------------------------------------------- S8.3 link budget --------
print("S8.3: per-pass power-return efficiencies (link_budget.json")
print("<plan>.block_efficiency):")
for i, want in enumerate([2.255, 2.181, 1.267]):
    check3("MNIST pass %d" % (i + 1),
           budget["MNIST"]["block_efficiency"][i], want)
for i, want in enumerate([2.091, 2.115, 1.202]):
    check3("GTSRB pass %d" % (i + 1),
           budget["GTSRB"]["block_efficiency"][i], want)

print("S8.3: GTSRB link ladder (link_budget.json GTSRB.rows[i].drive_dBm /")
print(".tone_snr_dB; SI quotes SNRs and pass 4 to 1 decimal):")
rows = budget["GTSRB"]["rows"]
for i, (dr, sn) in enumerate([(-10.0, 43.0), (-34.27, 32.6), (-58.48, 8.4)]):
    check("pass %d drive (dBm)" % (i + 1), rows[i]["drive_dBm"], dr)
    check("pass %d tone SNR (dB, 1-dec)" % (i + 1),
          rows[i]["tone_snr_dB"], sn, tol=0.051)
check("pass 4 drive (dBm, 1-dec; JSON -85.16)",
      rows[3]["drive_dBm"], -85.2, tol=0.051)
check("pass 4 tone SNR (dB, 1-dec; JSON -18.31)",
      rows[3]["tone_snr_dB"], -18.3, tol=0.051)
mn = budget["MNIST"]["rows"]
check_true("'within 0.7 dB of the MNIST plan': max pass-1..3 drive gap",
           max(abs(rows[i]["drive_dBm"] - mn[i]["drive_dBm"])
               for i in range(3)) <= 0.7)

# ---------------------------------------------- flag (a) resolution --------
print()
print("FLAG (a): 93.90 / 87.53 match no Table 5/6 row -- provenance:")
pn = {(r["location"], r["quantity"]): r for r in paper}
r = pn[("Fig. 4 MNIST analog", "accuracy")]
check_true("paper_numbers: Fig.5d MNIST analog '93.9 +- 0.1 %' = weights "
           "MZH_L3, model ZH",
           r["value"].startswith("93.9 ") and r["weights"] == "MZH_L3"
           and r["model"] == "ZH")
r = pn[("Fig. 4 MNIST digital", "accuracy")]
check_true("paper_numbers: Fig.5d MNIST digital '94.3' = weights MZH_L3, "
           "model ZT",
           r["value"].startswith("94.3 ") and r["weights"] == "MZH_L3"
           and r["model"] == "ZT")
r = pn[("Fig. 4 GTSRB analog", "accuracy")]
check_true("paper_numbers: Fig.5d GTSRB analog '87.53 +- 0.14 %' = weights "
           "ZH_L3, model ZH",
           r["value"].startswith("87.53 ") and r["weights"] == "ZH_L3"
           and r["model"] == "ZH")
r = pn[("Fig. 4 GTSRB digital", "accuracy")]
check_true("paper_numbers: Fig.5d GTSRB digital '88.83' = weights ZH_L3, "
           "model ZT",
           r["value"].startswith("88.83 ") and r["weights"] == "ZH_L3"
           and r["model"] == "ZT")
check_true("results_summary: MZT_L3d/ZT_L3d reuse the SAME ZH checkpoint "
           "(weights field)",
           summ["MZT_L3d"]["weights"] == "MZH_L3"
           and summ["ZT_L3d"]["weights"] == "ZH_L3")
check_true("Table 6 'twin+link noise' row is a DIFFERENT checkpoint family "
           "(MZ_L3/Z_L3, mode Z)",
           summ["MZ_L3"]["mode"] == "Z" and summ["Z_L3"]["mode"] == "Z"
           and abs(abl["MZ_L3"]["analog_pct"] - 95.62) < TOL
           and abs(abl["Z_L3"]["analog_pct"] - 86.26) < TOL)
check_true("Table 6 chain == mode-Z evaluation in results_summary "
           "(MZ_L3 95.62, Z_L3 86.26)",
           abs(abl["MZ_L3"]["analog_pct"]
               - summ["MZ_L3"]["accuracy_pct"]) < TOL
           and abs(abl["Z_L3"]["analog_pct"]
                   - summ["Z_L3"]["accuracy_pct"]) < TOL)
check_true("Table 6 digital baselines use mode ZD on MZ_L3/Z_L3 weights "
           "(96.07 / 88.29), not mode ZT",
           summ["MZD_L3"]["mode"] == "ZD"
           and summ["MZD_L3"]["weights"] == "MZ_L3"
           and abs(summ["MZD_L3"]["accuracy_pct"]
                   - abl["MZ_L3"]["digital_pct"]) < TOL
           and summ["ZD_L3"]["mode"] == "ZD"
           and summ["ZD_L3"]["weights"] == "Z_L3"
           and abs(summ["ZD_L3"]["accuracy_pct"]
                   - abl["Z_L3"]["digital_pct"]) < TOL)

# ---------------------------------------------- flag (b) resolution --------
print("FLAG (b): L=3 gap 0.40/1.30 (S8.3, main text) vs 0.45/2.03 (Table 6):")
check("ZH family gap: MZT_L3d - MZH_L3",
      round(acc("MZT_L3d")[0] - acc("MZH_L3")[0], 2), 0.40)
check("Z family drop: MZD_L3 - MZ_L3 (Table 6)",
      round(summ["MZD_L3"]["accuracy_pct"]
            - summ["MZ_L3"]["accuracy_pct"], 2), 0.45)
check("ZH family gap: ZT_L3d - ZH_L3",
      round(acc("ZT_L3d")[0] - acc("ZH_L3")[0], 2), 1.30)
check("Z family drop: ZD_L3 - Z_L3 (Table 6)",
      round(summ["ZD_L3"]["accuracy_pct"]
            - summ["Z_L3"]["accuracy_pct"], 2), 2.03)
def wtag(t):
    # early GTSRB rows of results_summary.json omit the 'weights' field;
    # for those the tag itself is the checkpoint tag
    return summ[t].get("weights") or t


check_true("the two pairs of gaps come from disjoint checkpoint sets",
           {wtag("MZH_L3"), wtag("ZH_L3")}
           .isdisjoint({wtag("MZ_L3"), wtag("Z_L3")}))

print()
print("RESULT: %d passed, %d failed" % (n_pass, n_fail))
sys.exit(0 if n_fail == 0 else 1)
