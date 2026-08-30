# -*- coding: utf-8 -*-
"""
recompute_no_booster.py -- derives the no-booster levels of Supplementary
Note 5, S5.3: "with it removed, the second pass of the main-text plan would
arrive near -52 dBm and the third near -95 dBm --- some 20 dB below the
-73.6-dBm receiver floor --- so the client would support at most two passes."

Those two numbers appear in NO archived JSON (checked: the shared archive's
results/*.json, every JSON in the original miwen_fully_analog_archive.zip,
and Response_Letter_Reproductivity/Reviewer1_Comment7/b_activation_block_note/
c7_activation_numbers.json).  This script recomputes them from the archived
physics, three ways:

  (L) the note's own bookkeeping: the booster is "worth ~18.5 dB of
      delivered video power per recirculation", so removing it costs the
      archived ladder 18.5 dB at pass 2 and 2 x 18.5 dB at pass 3.
  (A) the full archived link recursion (a gated pure-numpy port of plan_Z
      from _shared_fully_analog_simulation/comb_analog_sim.py) with
      g_boost = 0 and everything else untouched (R_v kept at 3.5 kOhm).
  (B) same recursion with g_boost = 0 AND the video source impedance
      returned to 50 Ohm (R_v = 50 x 10^(0/10)), since the archive defines
      R_v as "the boosted node impedance 50 x 10^(BOOST/10)".

Before any variant runs, the port is GATED: it must reproduce the archived
link_budget.json drive ladder (both datasets, all four passes) to < 0.05 dB
and the block efficiencies to < 0.005 %, exactly as the manuscript's own
figure script is gated.

Run:  python recompute_no_booster.py     (writes recompute_no_booster.json)
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = _data_dir(__file__)
ARCHIVE = HERE.parents[1] / "_shared_fully_analog_simulation"

# ---- archived physics constants (results/link_budget.json "settings") ----
S = json.load(open(ARCHIVE / "results" / "link_budget.json"))["settings"]
CL_PASS = S["cl_pass_dB"]           # 7.47 = 6.65 (ZEM-4300+) + 0.82 (LFCN-490+)
BOOST_DB = S["boost_dB"]            # 18.5, Wang et al. ESSCIRC 2017
K_ED = S["k_ED_per_V"]              # 208.7 /V, Wang et al. SSC-L 2018
R_V = S["R_v_ohm"]                  # 3500 = boosted node impedance
IL_STEP = S["step_down_loss_dB"]    # 1.0 dB video step-down
P1_DBM = S["first_pass_dBm"]        # -10.0
N_STAGE = 1                         # no stacking credit (S5.7)
P_FLOOR = -73.6                     # per-bin receiver floor [dBm]
NOISE_C = 27.0                      # archived tone-SNR law constant

mw = lambda d: 10.0 ** (d / 10.0)
dbm = lambda p: 10.0 * math.log10(max(p, 1e-30))


def ed_video(v_pk):
    """Eq. S16 / comb_analog_sim.ed_video: measured pseudo-balun transfer."""
    return 0.5 * v_pk * (1.0 - np.exp(-2.0 * K_ED * v_pk))


def plan_Z(L, dims, p1_dbm=P1_DBM, g_boost=BOOST_DB, r_v=R_V):
    """Pure-numpy port of comb_analog_sim.plan_Z (same rng seed, same comb
    construction, same available-power video interface)."""
    rng = np.random.default_rng(0)
    P = mw(p1_dbm)
    powers = [P]
    for l in range(L - 1):
        P_mix = P * 10 ** (-CL_PASS / 10)
        P_det = P_mix * 10 ** (g_boost / 10)     # 50-ohm-equivalent node power
        K = dims[l + 1]
        a = rng.normal(size=4 * K) + 1j * rng.normal(size=4 * K)
        env = np.abs(np.fft.ifft(a)) * len(a)
        env *= np.sqrt(P_det * 1e-3 * 2 * 50.0 / np.mean(env ** 2))
        vid = ed_video(env)
        vid = vid - vid.mean()                   # video (AC) comb only
        p_av_w = np.mean(vid ** 2) / (4 * r_v) * N_STAGE
        P = p_av_w * 1e3 * 10 ** (-IL_STEP / 10)
        powers.append(float(P))
    return powers


def snr_tone_db(p_mw, D):
    return 10 * math.log10(NOISE_C * p_mw / (D * mw(P_FLOOR)))


DIMS = {"MNIST": [784, 100, 64, 64, 10],
        "GTSRB": [3072, 128, 128, 128, 43]}

lb = json.load(open(ARCHIVE / "results" / "link_budget.json"))
out = {"constants": dict(S, N_stage=N_STAGE, floor_dBm=P_FLOOR)}

# ------------------------------------------------------------------ gate --
print("=" * 72)
print("GATE: numpy port must reproduce the archived link budget")
print("=" * 72)
gate_ok = True
for ds, dims in DIMS.items():
    got = [dbm(p) for p in plan_Z(4, dims)]
    ref = [r["drive_dBm"] for r in lb[ds]["rows"]]
    eff_ref = lb[ds]["block_efficiency"]
    eff_got = [100 * 10 ** ((got[i + 1] - got[i] + CL_PASS) / 10)
               for i in range(3)]
    for i in range(4):
        d = abs(got[i] - ref[i])
        ok = d < 0.05
        gate_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {ds} pass {i+1}: "
              f"port {got[i]:8.2f} dBm  archived {ref[i]:8.2f} dBm  "
              f"|diff| = {d:.4f} dB")
    for i in range(3):
        d = abs(eff_got[i] - eff_ref[i])
        ok = d < 0.005
        gate_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {ds} block eff {i+1}: "
              f"port {eff_got[i]:.3f} %  archived {eff_ref[i]:.3f} %")
if not gate_ok:
    print("GATE FAILED -- port does not reproduce the archive; aborting.")
    sys.exit(1)
print("GATE PASSED: archived ladder reproduced to < 0.05 dB.\n")

# ------------------------------------------------------------- variants --
TEX_P2, TEX_P3 = -52.0, -95.0        # the note's "near -52 / near -95 dBm"


def report(name, levels, note=""):
    print(f"--- {name} ---")
    if note:
        print("    " + note)
    row = {}
    for ds, lv in levels.items():
        dims = DIMS[ds]
        n_above = sum(1 for x in lv if x > P_FLOOR)
        snrs = [snr_tone_db(mw(x), D) for x, D in zip(lv, dims)]
        print(f"  {ds}: " + " -> ".join(f"{x:7.2f}" for x in lv) + " dBm")
        print(f"        tone SNR " + ", ".join(f"{s:6.1f}" for s in snrs)
              + f" dB;  passes above the {P_FLOOR} dBm floor: {n_above}")
        row[ds] = dict(drive_dBm=[round(x, 2) for x in lv],
                       tone_snr_dB=[round(s, 2) for s in snrs],
                       passes_above_floor=n_above)
        d2, d3 = abs(lv[1] - TEX_P2), abs(lv[2] - TEX_P3)
        print(f"        vs tex: pass2 {lv[1]:7.2f} (|d|={d2:5.2f} dB of -52),"
              f"  pass3 {lv[2]:7.2f} (|d|={d3:5.2f} dB of -95)")
        row[ds]["abs_diff_from_tex_dB"] = [round(d2, 2), round(d3, 2)]
    out[name] = row
    print()
    return row


print("=" * 72)
print("NO-BOOSTER RECOMPUTATION  (tex claim: near -52 and -95 dBm, "
      "<= 2 passes)")
print("=" * 72)

# (L) the note's own linear bookkeeping on the archived ladder
levels_L = {}
for ds in DIMS:
    ref = [r["drive_dBm"] for r in lb[ds]["rows"]]
    levels_L[ds] = [ref[0], ref[1] - BOOST_DB, ref[2] - 2 * BOOST_DB]
rL = report("variant_L_bookkeeping", levels_L,
            "archived ladder minus 18.5 dB per completed recirculation "
            "(the 'worth ~18.5 dB of delivered video power' statement)")

# (A) full recursion, g_boost = 0, R_v untouched (3.5 kOhm)
rA = report("variant_A_recursion_Rv3500",
            {ds: [dbm(p) for p in plan_Z(3, dims, g_boost=0.0)]
             for ds, dims in DIMS.items()},
            "archived plan_Z recursion with g_boost = 0, R_v = 3500 Ohm")

# (B) full recursion, g_boost = 0 and R_v = 50 * 10^(0/10) = 50 Ohm
rB = report("variant_B_recursion_Rv50",
            {ds: [dbm(p) for p in plan_Z(3, dims, g_boost=0.0, r_v=50.0)]
             for ds, dims in DIMS.items()},
            "g_boost = 0 and video source impedance back to 50 Ohm "
            "(R_v = 50 x 10^(boost/10) with boost = 0)")

# ------------------------------------------------------------- verdict --
print("=" * 72)
print("VERDICT")
print("=" * 72)
verdict = []
for name, r in (("L", rL), ("A", rA), ("B", rB)):
    hit = all(r[ds]["abs_diff_from_tex_dB"][0] <= 1.5 and
              r[ds]["abs_diff_from_tex_dB"][1] <= 1.5 for ds in DIMS)
    two = all(r[ds]["passes_above_floor"] <= 2 for ds in DIMS)
    verdict.append((name, hit, two))
    print(f"variant {name}: reproduces -52/-95 within 1.5 dB: {hit};  "
          f"supports <= 2 passes: {two}")
p3L = rL["MNIST"]["drive_dBm"][2]
print(f"\n'~20 dB below the floor': variant L pass 3 = {p3L} dBm is "
      f"{abs(p3L - P_FLOOR):.1f} dB below {P_FLOOR} dBm "
      f"(GTSRB {rL['GTSRB']['drive_dBm'][2]} dBm: "
      f"{abs(rL['GTSRB']['drive_dBm'][2] - P_FLOOR):.1f} dB below).")

out["verdict"] = {f"variant_{n}": dict(reproduces_52_95=h,
                                       at_most_two_passes=t)
                  for n, h, t in verdict}
json.dump(out, open(HERE / "recompute_no_booster.json", "w"), indent=1)
print("\nwrote", HERE / "recompute_no_booster.json")
