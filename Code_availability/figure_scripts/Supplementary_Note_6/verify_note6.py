#!/usr/bin/env python3
"""Verify every checkable number of Supplementary Note 6 (comb-encoded
campaign) from the bytes archived in THIS folder.

TIER 1 (always runs, archived bytes only)
  [1.1] frozen-calibration rerun accuracy 98.92 +/- 0.30 % (1187/1200)
  [1.2] evaluation frame identity: img_index == battery_random1200_idx.npy
  [1.3] paired stats (i), measured 98.92 vs same-weights digital 98.42:
        10 vs 4 discordant, exact McNemar p ~= 0.18  (recomputed from
        battery_digital_preds_extract.npz with the archived
        mcnemar_paired.py itself)
  [1.4] paired stats (ii), measured vs clean-trained digital 99.50:
        8 vs 1 discordant -> p ~= 0.04 (p and accuracy arithmetic from
        archived bytes; the per-image clean-digital forward is re-run in
        Tier 2 when the frozen fig4 archive is reachable)
  [1.5] as-run battery 98.83 %, 3 of 1,200 prediction flips vs the
        frozen rerun
  [1.6] ns25 noise vector [0.177, 0.32, 0.102, 0.221] x 0.25, parsed
        from the archived training wrapper r35_ns25_train.py
  [1.7] frozen offline calibration structure: one gain per output
        column (32/64/128/43), fitted on a disjoint 450-image frame,
        no host W@x
  [1.8] clean r3plus training log ends at test 99.05 (the weights
        behind the 99.50 battery number)

TIER 2 (optional; runs when the read-only frozen archives are present)
  [2.1] extraction integrity vs the 55-MB battery_slim.npz source
  [2.2] full clean-digital forward on the 1,200-image battery
        (fig4 archive GTSRB test cache + clean weights) -> 99.50 %,
        and the exact 8-vs-1 discordant recount of [1.4]

DOCUMENTED, NOT ASSERTED (platform constants; no measurement bytes in
this folder can re-derive them -- see README.md for source pointers).

Run:  python verify_note6.py        (numpy only for Tier 1)
"""
from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir, code_dir as _code_dir
# ---------------------------------------------------------------------------


import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

HERE = _data_dir(__file__)
FAIL = 0


def check(tag: str, cond: bool, detail: str) -> None:
    global FAIL
    mark = "PASS" if cond else "FAIL"
    if not cond:
        FAIL += 1
    print(f"[{tag}] {mark}  {detail}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mc = load_module("mcnemar_paired", _code_dir(__file__) / "mcnemar_paired.py")

print("=" * 72)
print("Supplementary Note 6 -- comb-encoded campaign: verification")
print("=" * 72)

# ---------------------------------------------------------------- TIER 1
print("\nTIER 1 -- archived bytes in this folder\n")

f = np.load(HERE / "battery_frozen_slim.npz", allow_pickle=True)
idx = np.load(HERE / "battery_random1200_idx.npy")
ext = np.load(HERE / "battery_digital_preds_extract.npz", allow_pickle=True)

# [1.1] frozen rerun accuracy
ok = f["preds"] == f["labels"]
acc = ok.mean() * 100
se = np.sqrt(acc / 100 * (1 - acc / 100) / len(ok)) * 100
check("1.1a", int(ok.sum()) == 1187 and len(ok) == 1200,
      f"frozen rerun: {int(ok.sum())}/1200 correct")
check("1.1b", round(acc, 2) == 98.92,
      f"accuracy {acc:.4f}% -> quoted 98.92")
check("1.1c", abs(float(f["accuracy"]) - 0.9892) < 5e-5,
      f"stored accuracy scalar {float(f['accuracy'])}")
check("1.1d", round(se, 2) == 0.30,
      f"binomial s.e. {se:.4f} pts -> quoted +/-0.30")
check("1.1e", str(f["calibration"]) == "frozen (no host W@x)",
      f"calibration tag = '{f['calibration']}'")

# [1.2] frame identity
check("1.2", np.array_equal(f["img_index"], idx),
      "img_index identical to battery_random1200_idx.npy "
      f"(n={len(idx)}, GTSRB test indices {idx.min()}..{idx.max()})")

# [1.3] McNemar (i): frozen measured vs same-weights (ns25) digital MAC
b, c, n, acc_m, acc_d = mc.paired_discordants(
    f["preds"], ext["digital_preds"], f["labels"],
    labels_b=ext["labels"], img_index_a=f["img_index"],
    img_index_b=ext["img_index"])
p_i = mc.mcnemar_exact(b, c)
check("1.3a", (b, c) == (10, 4),
      f"discordants measured-only-correct={b}, digital-only-correct={c} "
      "-> quoted '10 versus 4'")
check("1.3b", round(acc_d * 100, 2) == 98.42,
      f"same-weights digital accuracy {acc_d*100:.4f}% -> quoted 98.42")
check("1.3c", round((acc_m - acc_d) * 100, 2) == 0.50,
      f"delta {100*(acc_m-acc_d):+.4f} pts -> quoted +0.50")
check("1.3d", abs(p_i - 0.18) < 0.005 and not p_i < 0.05,
      f"exact McNemar p = {p_i:.5f} -> quoted ~0.18, not significant")

# [1.4] McNemar (ii): measured vs clean-trained digital (99.50)
p_ii = mc.mcnemar_exact(8, 1)
check("1.4a", abs(p_ii - 0.04) < 0.002 and p_ii < 0.05,
      f"exact McNemar p(8,1) = {p_ii:.5f} -> quoted ~0.04, significant")
clean_correct = 1187 + 8 - 1     # measured correct - (measured-only) + (clean-only)
check("1.4b", clean_correct == 1194 and round(clean_correct / 12, 2) == 99.50,
      f"8-vs-1 discordants imply clean-digital {clean_correct}/1200 "
      f"= {clean_correct/12:.2f}% -> quoted 99.50")
check("1.4c", round((1187 - clean_correct) / 12, 2) == -0.58,
      f"implied delta {(1187-clean_correct)/12:+.4f} pts -> quoted -0.58")

# [1.5] as-run battery and flips
okr = ext["preds"] == ext["labels"]
flips = int((ext["preds"] != f["preds"]).sum())
check("1.5a", round(okr.mean() * 100, 2) == 98.83,
      f"as-run battery {okr.mean()*100:.4f}% ({int(okr.sum())}/1200)")
check("1.5b", flips == 3,
      f"{flips} prediction flips as-run vs frozen rerun -> quoted 3 of 1,200")

# [1.6] ns25 noise vector from the archived training wrapper
src = (_code_dir(__file__) / "r35_ns25_train.py").read_text()
mnum = re.search(r"SIG\s*=\s*\[([^\]]+)\]", src)
sig = [eval(t.strip()) for t in mnum.group(1).split(",")]  # noqa: S307 (literal arithmetic)
base = [round(s / 0.25, 3) for s in sig]
check("1.6", base == [0.177, 0.32, 0.102, 0.221],
      f"wrapper SIG/0.25 = {base} -> quoted [0.177, 0.32, 0.102, 0.221] x 0.25")

# [1.7] frozen calibration structure
cal = np.load(HERE / "frozen_calibration.npz", allow_pickle=True)
shapes = tuple(cal[k].shape[0] for k in ("g_l1", "g_l2", "g_l3", "g_l4"))
check("1.7a", shapes == (32, 64, 128, 43),
      f"per-column gain vectors {shapes} = conv32/conv64/dense128/dense43")
check("1.7b", "450" in str(cal["source"]),
      f"source tag = '{cal['source']}' (450 disjoint calibration images)")
check("1.7c", "without any digital W@x" in str(cal["note"]),
      f"note = '{cal['note']}'")

# [1.8] clean r3plus training record
log = (HERE / "r35_train_log.txt").read_text()
check("1.8", "=== r3plus: test 99.05; exported r35_r3plus_s0_hw.npz" in log,
      "clean r3plus log: SELECTED ckpt test full 99.05 -> the weights "
      "behind the 99.50 battery number")

# ---------------------------------------------------------------- TIER 2
print("\nTIER 2 -- optional cross-checks against the read-only frozen "
      "archives\n")

GI = Path(r"E:\archive\Manuscript"
          r"\Overleaf\202607_MIWEN_Manuscript\V2\GTSRB_inference"
          r"\share_20260712")
FIG4 = Path(r"E:\archive\Manuscript"
            r"\Overleaf\202607_MIWEN_v2\V3_Manu\Manuscript_Reproductivity"
            r"\fig4")

slim_path = GI / "battery_slim.npz"
if slim_path.exists():
    d = np.load(slim_path, allow_pickle=True)
    same = all(np.array_equal(ext[k], d[k])
               for k in ("preds", "digital_preds", "labels", "img_index"))
    check("2.1", same,
          "battery_digital_preds_extract.npz arrays identical to the "
          "55-MB battery_slim.npz source")
else:
    print("[2.1] SKIP  battery_slim.npz source not reachable "
          "(extract verified at creation; see extract_digital_preds.py)")

cache_path = FIG4 / "data" / "gtsrb_roi_32x32_test.npz"
w_clean = FIG4 / "a" / "r35_r3plus_s0_hw.npz"
ref_py = FIG4 / "a" / "miwen_frozen_reference.py"
if not (cache_path.exists() and w_clean.exists() and ref_py.exists()):
    # fall back to the provenance package for weights + reference code
    w_clean = GI / "r35_r3plus_s0_hw.npz"
    ref_py = GI / "miwen_frozen_reference.py"
if cache_path.exists() and w_clean.exists() and ref_py.exists():
    ref = load_module("miwen_frozen_reference", ref_py)
    cache = np.load(cache_path, allow_pickle=True)
    X = ref.preprocess(cache["Xte"][idx])
    y = cache["yte"].astype(np.int64)[idx]
    check("2.2a", np.array_equal(y, f["labels"]),
          "GTSRB test-cache labels at the battery indices match the "
          "archived battery labels")
    pd_clean = ref.forward_digital(X, ref.load_weights(w_clean))
    okc = pd_clean == y
    check("2.2b", round(okc.mean() * 100, 2) == 99.50,
          f"clean-trained digital forward: {okc.mean()*100:.4f}% "
          f"({int(okc.sum())}/1200) -> quoted 99.50")
    b2 = int((ok & ~okc).sum())
    c2 = int((~ok & okc).sum())
    p2 = mc.mcnemar_exact(b2, c2)
    check("2.2c", (c2, b2) == (8, 1),
          f"discordants clean-only-correct={c2}, measured-only-correct={b2} "
          "-> quoted '8 versus 1'")
    check("2.2d", abs(p2 - 0.04) < 0.002,
          f"exact McNemar p = {p2:.5f} -> quoted ~0.04")
else:
    print("[2.2] SKIP  fig4 frozen archive (GTSRB test cache + clean "
          "weights) not reachable; 8-vs-1 verified arithmetically in 1.4")

# ------------------------------------------------- DOCUMENTED, NOT ASSERTED
print("""
DOCUMENTED (platform constants of S6.1/S6.4; source pointers, no asserts)
  - TX power law P_out[dBm] = 18.0 + (g - 31.5) + 20*log10(a_RMS):
    P_MAX_DBM_DEFAULT = 18.0 and GAIN_MAX_DB = 31.5 in the frozen core
    4_gtsrb_confusion_mlpN.py (share_20260712, lines 54/56), with
    PAPR-aware peak limiting.
  - 10 MS/s baseband, 5-MHz digital LO offset: USRP_SAMPLE_RATE = 10e6,
    USRP_LO_OFFSET = 5e6 (same file, lines 45/46).
  - Matched-filter peak-to-sidelobe acceptance 3.0; inner-product
    acquisitions measured 28.6-31.4; pilot residual phase-drift slopes
    < 3.2e-3 rad/slot: session logs / Supplementary Note 4 campaign,
    not re-derivable from the arrays archived here.
  - Cascade frames: 16,384-point FFT (610.35 Hz spacing = 1e7/16384),
    512-sample cyclic prefix, seeded-QPSK sync symbol; comb RX 30 dB
    gain + 30 dB attenuator; operating point LO -3 dBm / RF -35 dBm
    (README_provenance.md; P_LO_DBM_DEFAULT / P_RF_DBM_DEFAULT in the
    frozen core).
  - Comb construction / Zadoff-Chu precoding equations (S6.4): code in
    4_gtsrb_confusion_mlpN.py (comb synthesis) at the source; equations
    are definitional.
  - FLAG (see README.md): the S6.3(iii) figure 'a conventional
    signed-activation digital CNN reaches ~99.7-99.8% in the
    literature' carries NO citation in the SI; it traces only to the
    repo note docs/notes/2026-08-06_digital_vs_miwen_table.md, which is
    equally uncited.""")

print("=" * 72)
print(f"RESULT: {'ALL ASSERTIONS PASSED' if FAIL == 0 else f'{FAIL} FAILURE(S)'}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
