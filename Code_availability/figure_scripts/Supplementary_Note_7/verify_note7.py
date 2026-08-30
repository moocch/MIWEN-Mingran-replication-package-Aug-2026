#!/usr/bin/env python3
"""Verify every checkable number of Supplementary Note 7 (time-serial
campaign) from the bytes archived in THIS folder.

  [1] Ridge-count sweep (S7.2): held-out RMSE 6.90/4.03/3.50/3.09/2.44/
      0.46/0.29/0.19/0.15/0.06 dB at K = 0/1/2/3/5/7/10/15/20/30,
      repeatability floor 0.015 dB   (twin_ksweep_20260825.json)
  [2] Fielded K=20 surface held-out 0.12 dB (S7.2): recomputed as a pure
      function evaluation of the archived serial_twin_model.json
      parameters on the archived CW map with the frozen 20% holdout
      split -- plus the resolution of the 0.15-vs-0.12 flag (see FLAG)
  [3] Drive-ladder quintet (S7.3): per-capture relative RMS residual of
      the best complex linear fit, mean of 4 captures per drive ->
      0.21/0.41/0.50/0.58/0.66 at (-9,-9)/(-3,-3)/(0,0)/(+3,+3)/(+7,+7)
      dBm   (recomputed from serial_stationA_20260823.npz, the
      convention of serial_stationA.py / serial_enob.py)
  [4] Amplitude-resolved ENOB (S7.3): overall 2.59+/-0.06 -> 4.32+/-0.06
      bits (mean +/- 1 s.d. over the four (0,0) captures) and pooled
      8-bin endpoints 6.5/2.1 -> 9.7/3.5 bits   (recomputed: twin
      inversion per product on a 4,096-point grid, serial_enob.py
      convention, full-scale-referred log2[range/(RMSE*sqrt(12))])
  [5] Hardware 2x2 headline: clean 5.67% (34/600) vs twin 98.50%
      (591/600), recounted from the raw chunk predictions; plus the
      pinned digital comparators; plus verify_accuracy.py run end-to-end
  [6] Session-gate / capture-format facts stored in archived metadata

DOCUMENTED, NOT ASSERTED at the end: session-log-only claims.

Run:  python verify_note7.py     (numpy only; [5] also runs
      verify_accuracy.py as a subprocess)
"""
from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir, code_dir as _code_dir
# ---------------------------------------------------------------------------


import json
import subprocess
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


print("=" * 72)
print("Supplementary Note 7 -- time-serial campaign: verification")
print("=" * 72)

# ---------------------------------------------------------------- [1] sweep
print("\n[1] Ridge-count sweep of the serial twin (S7.2)\n")
ks = json.load(open(HERE / "01_digital_twin_model" / "twin_ksweep_20260825.json"))
QUOTED = {0: 6.90, 1: 4.03, 2: 3.50, 3: 3.09, 5: 2.44,
          7: 0.46, 10: 0.29, 15: 0.19, 20: 0.15, 30: 0.06}
check("1a", ks["K"] == list(QUOTED.keys()), f"K grid {ks['K']}")
for K, r in zip(ks["K"], ks["rmse"]):
    check(f"1b K={K}", abs(r - QUOTED[K]) < 0.005,
          f"held-out {r:.4f} dB -> quoted {QUOTED[K]:.2f}")
check("1c", abs(ks["floor"] - 0.015) < 0.0005,
      f"three-repeat repeatability floor {ks['floor']:.5f} dB -> quoted 0.015")

# ------------------------------------------- [2] fielded K=20 held-out 0.12
print("\n[2] Fielded K=20 surface: held-out RMSE recompute (S7.2)\n")
TM = json.load(open(HERE / "01_digital_twin_model" / "serial_twin_model.json"))
thp = np.array(TM["thp"])
ridges = np.array(TM["ridges"])
m = np.load(HERE / "01_digital_twin_model" / "heatmap_unpadded_20260814.npz",
            allow_pickle=True)
LO, RF = np.meshgrid(m["p_lo_dbm_grid"].astype(float),
                     m["p_rf_dbm_grid"].astype(float), indexing="ij")
A_uv = m["if_amp_uv_mean"].astype(float)
clean = (m["clean_mask"].astype(bool) & (A_uv > 0)
         & (m["peak_fs"].astype(float) < 0.95) & (LO >= -35) & (RF >= -35))
lo, rf, y_db = LO[clean], RF[clean], 20 * np.log10(A_uv[clean])
n = len(y_db)


def s_pole(p, k):
    x = 10 ** ((p - k) / 10.0)
    return x / (1.0 + x)


def product_db(th, rg, lo_, rf_):
    yv = th[0] + 10 * np.log10(np.maximum(
        s_pole(lo_, th[1]) * s_pole(rf_, th[2]), 1e-300))
    if rg is not None:
        for wi, ai, bi, ci in rg.reshape(-1, 4):
            yv = yv + wi * np.tanh(ai * lo_ / 40 + bi * rf_ / 40 + ci)
    return yv


def model_db(th, rg, lo_, rf_):
    y_mw = (10 ** (product_db(th, rg, lo_, rf_) / 10.0)
            + 10 ** ((th[3] + lo_) / 10.0) + 10 ** ((th[4] + rf_) / 10.0))
    return 10 * np.log10(y_mw)


# the frozen 20% holdout split -- IDENTICAL construction in
# serial_twin_fit.py and twin_ksweep.py (numpy default_rng(9) over the
# same 520 product-region cells)
hold = np.random.default_rng(9).random(n) < 0.2
check("2a", (n, int(hold.sum())) == (520, 102),
      f"product-dominated region: {n} clean cells, {int(hold.sum())} held out "
      "(deterministic rng(9) split shared by fit and sweep)")
rmse_ho = float(np.sqrt(np.mean(
    (model_db(thp, ridges, lo[hold], rf[hold]) - y_db[hold]) ** 2)))
check("2b", abs(rmse_ho - TM["rmse_holdout"]) < 1e-9,
      f"recomputed held-out {rmse_ho:.5f} dB == stored rmse_holdout "
      f"{TM['rmse_holdout']:.5f} (exact function evaluation)")
check("2c", round(rmse_ho, 2) == 0.12,
      f"-> quoted '(held-out 0.12 dB)' for the fielded K={TM['K']} surface")
check("2d", TM["K"] == 20 and all(TM["gates"].values()),
      f"fielded twin: K={TM['K']}, gates {TM['gates']}")
rmse_prod = float(np.sqrt(np.mean(
    (product_db(thp, ridges, lo[hold], rf[hold]) - y_db[hold]) ** 2)))
check("2e", abs(rmse_prod - rmse_ho) < 1e-4,
      f"product-term-only evaluation {rmse_prod:.5f} dB: the fielded "
      "model's per-port feedthrough terms are negligible in-region")

print("""
FLAG RESOLUTION -- sweep says 0.15 dB at K=20, SI says the fielded K=20
surface has held-out 0.12 dB.  Both numbers are 20%-held-out RMSE on the
IDENTICAL data and IDENTICAL 102-cell holdout split (checks 2a/2b): not
a different split, and not the feedthrough terms (check 2e).  They are
two different FITS of the same K=20 model family: twin_ksweep.py re-fits
from scratch at every K with a leaner staged budget (3-parameter physics
stage, max_nfev 800/2500/1500) while the fielded serial_twin_fit.py used
a 4-start 5-parameter physics stage and larger budgets (2000/3000/2000).
An 83-parameter tanh-ridge least-squares has many near-degenerate local
minima; the sweep's K=20 refit landed at 0.152 dB, the fielded fit at
0.124 dB.  Re-running both staged fits during archiving (scipy 1.x on
the archive machine) landed at 0.115 and 0.138 dB respectively --
optimizer-trajectory scatter of ~0.03 dB at fixed capacity, an order of
magnitude above the 0.015-dB repeat floor and immaterial to the sweep's
conclusion (capacity saturates at K ~ 20-30).  The SI's parenthetical
quotes the fielded fit's own number, which check 2b reproduces exactly
from archived bytes.""")

# ----------------------------------------------------- [3] drive ladder
print("\n[3] Drive-ladder quintet (S7.3)\n")
d = np.load(HERE / "07_supporting_analysis_enob_and_driveladder"
            / "serial_stationA_20260823.npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
TAGS = {"-9_-9": ("(-9,-9)", 0.21), "-3_-3": ("(-3,-3)", 0.41),
        "+0_+0": ("(0,0)", 0.50), "+3_+3": ("(+3,+3)", 0.58),
        "+7_+7": ("(+7,+7)", 0.66)}
check("3a", meta["nslot"] == 256 and len(
    [k for k in d.files if k.startswith("yhat_")]) == 20,
    "256 pairs per capture; 4 captures (2 draws x 2 repeats) x 5 drives "
    "archived")
for tag, (name, quoted) in TAGS.items():
    rels = []
    for dr in range(2):
        xw = d[f"x_d{dr}"] * d[f"w_d{dr}"]
        for rp in range(2):
            yh = d[f"yhat_{tag}_d{dr}_r{rp}"]
            g = np.vdot(xw, yh) / np.vdot(xw, xw)     # best complex LS gain
            res = yh - g * xw
            rels.append(np.sqrt(np.mean(np.abs(res) ** 2)
                                / np.mean(np.abs(g * xw) ** 2)))
    mean = float(np.mean(rels))
    check(f"3b {name}", abs(mean - quoted) < 0.005,
          f"mean rel. linear-fit residual {mean:.4f} -> quoted {quoted:.2f} "
          f"(per capture: {', '.join(f'{r:.3f}' for r in rels)})")

# ------------------------------------------------------------- [4] ENOB
print("\n[4] Amplitude-resolved ENOB at (0,0) (S7.3)\n")
rg4 = ridges.reshape(-1, 4)
DUTY = TM["duty_db"]


def pdb(pl, pr):
    yv = thp[0] + 10 * np.log10(
        s_pole(pl, thp[1]) * s_pole(pr, thp[2]) + 1e-30)
    for wi, ai, bi, ci in rg4:
        yv = yv + wi * np.tanh(ai * pl / 40 + bi * pr / 40 + ci)
    return yv


naive_bits, twin_bits = [], []
pool_en, pool_et, pool_xw = [], [], []
for dr in range(2):
    x, w = d[f"x_d{dr}"], d[f"w_d{dr}"]
    xw = x * w
    xrms, wrms = np.sqrt(np.mean(x ** 2)), np.sqrt(np.mean(w ** 2))
    p_lo = 20 * np.log10(np.maximum(np.abs(w) / wrms, 1e-6)) + DUTY
    p_rf = 20 * np.log10(np.maximum(np.abs(x) / xrms, 1e-6)) + DUTY
    pred_twin = np.sign(xw) * 10 ** (pdb(p_lo, p_rf) / 20.0)
    grid_x = np.linspace(1e-4, np.abs(x).max() * 1.5, 4096)
    for rp in range(2):
        yh = d[f"yhat_+0_+0_d{dr}_r{rp}"]
        ym = np.real(yh * np.exp(-1j * np.angle(np.vdot(xw, yh))))
        g_naive = np.dot(ym, xw) / np.dot(xw, xw)
        g_twin = np.dot(ym, pred_twin) / np.dot(pred_twin, pred_twin)
        xw_hat = np.zeros(len(xw))
        for i in range(len(xw)):        # invert f(x_hat; w) = y per product
            pr_g = 20 * np.log10(np.maximum(grid_x / xrms, 1e-6)) + DUTY
            f_g = 10 ** (pdb(p_lo[i], pr_g) / 20.0) * g_twin
            yi = abs(ym[i])
            j = int(np.searchsorted(f_g, yi))
            if j <= 0:
                xh = grid_x[0]
            elif j >= len(grid_x):
                xh = grid_x[-1]
            else:
                f0, f1 = f_g[j - 1], f_g[j]
                xh = grid_x[j - 1] + (grid_x[j] - grid_x[j - 1]) \
                    * (yi - f0) / max(f1 - f0, 1e-30)
            xw_hat[i] = np.sign(ym[i]) * xh * np.abs(w[i])
        err_n, err_t = ym / g_naive - xw, xw_hat - xw
        rng_p = xw.max() - xw.min()
        ef = lambda e: float(np.log2(rng_p / (np.sqrt(np.mean(e ** 2))
                                              * np.sqrt(12))))
        naive_bits.append(ef(err_n))
        twin_bits.append(ef(err_t))
        pool_en.append(err_n)
        pool_et.append(err_t)
        pool_xw.append(xw)

mn, sn = float(np.mean(naive_bits)), float(np.std(naive_bits))   # ddof=0
mt, st = float(np.mean(twin_bits)), float(np.std(twin_bits))
print(f"    per-capture naive bits: {[round(v, 3) for v in naive_bits]}")
print(f"    per-capture twin  bits: {[round(v, 3) for v in twin_bits]}")
check("4a", abs(mn - 2.59) < 0.005 and abs(sn - 0.06) < 0.005,
      f"overall raw ENOB {mn:.3f} +/- {sn:.3f} bits -> quoted 2.59 +/- 0.06")
check("4b", abs(mt - 4.32) < 0.005 and abs(st - 0.06) < 0.005,
      f"overall twin-inverted ENOB {mt:.3f} +/- {st:.3f} bits "
      "-> quoted 4.32 +/- 0.06")

xw_all = np.concatenate(pool_xw)
en, et = np.concatenate(pool_en), np.concatenate(pool_et)
rng_p = xw_all.max() - xw_all.min()
ef = lambda e: float(np.log2(rng_p / (np.sqrt(np.mean(e ** 2)) * np.sqrt(12))))
q = np.quantile(np.abs(xw_all), np.linspace(0, 1, 9))
# serial_enob.py bin convention: half-open [a, b) -- the maximal
# product(s) fall outside the top bin
sel_lo = (np.abs(xw_all) >= q[0]) & (np.abs(xw_all) < q[1])
sel_hi = (np.abs(xw_all) >= q[7]) & (np.abs(xw_all) < q[8])
b_lo_n, b_hi_n = ef(en[sel_lo]), ef(en[sel_hi])
b_lo_t, b_hi_t = ef(et[sel_lo]), ef(et[sel_hi])
check("4c", abs(b_lo_n - 6.5) < 0.05 and abs(b_hi_n - 2.1) < 0.05,
      f"pooled 8-bin raw endpoints {b_lo_n:.3f} / {b_hi_n:.3f} bits "
      "-> quoted 6.5 / 2.1")
check("4d", abs(b_lo_t - 9.7) < 0.05 and abs(b_hi_t - 3.5) < 0.05,
      f"pooled 8-bin twin-inverted endpoints {b_lo_t:.3f} / {b_hi_t:.3f} "
      "bits -> quoted 9.7 / 3.5")
print("    (note: 07_.../serial_enob_00_20260824.json is the EARLIER "
      "single-capture\n     diagnostic analysis -- 6.62/2.06 -> 9.79/3.24, "
      "overall 2.53 -> 4.38 bits;\n     the SI quotes the four-capture "
      "Stage-A statistics recomputed here)")

# ------------------------------------------------ [5] hardware 2x2 headline
print("\n[5] Hardware N=600 headline and pinned comparators\n")
b8 = np.load(HERE / "08_frozen_inputs_and_labels" / "battery_frozen_slim.npz",
             allow_pickle=True)
LAB = dict(zip(b8["img_index"].tolist(), b8["labels"].tolist()))
CLEAN = ["serial_nn_clean_0_150_20260823.npz",
         "serial_nn_clean_150_215_20260823.npz",
         "serial_nn_clean_215_300_20260823.npz",
         "serial_nn_clean_275_300_20260823.npz",
         "serial_nn_clean_300_600_20260823.npz"]
TWIN = ["serial_nn_twin_0_300_20260823.npz",
        "serial_nn_twin_300_600_20260823.npz"]


def arm(folder, files):
    P, S = [], []
    for fn in files:
        z = np.load(HERE / "05_hardware_result_0dBm_N600" / folder / fn,
                    allow_pickle=True)
        kk = sorted((k for k in z.keys() if k.startswith("chunk")),
                    key=lambda s: int(s[5:].split("_")[0]))
        p = np.concatenate([z[k] for k in kk])
        P.append(p)
        S.append(z["sel"][:len(p)])
    P, S = np.concatenate(P), np.concatenate(S)
    y = np.array([LAB[i] for i in S])
    return int((P == y).sum()), len(P), S


c_ok, c_n, c_s = arm("clean_arm", CLEAN)
t_ok, t_n, t_s = arm("twin_arm_hw_aware", TWIN)
check("5a", (c_ok, c_n) == (34, 600) and round(100 * c_ok / c_n, 2) == 5.67,
      f"clean arm hardware: {c_ok}/{c_n} = {100*c_ok/c_n:.2f}% -> quoted 5.67")
check("5b", (t_ok, t_n) == (591, 600) and round(100 * t_ok / t_n, 2) == 98.50,
      f"twin (hardware-aware) arm: {t_ok}/{t_n} = {100*t_ok/t_n:.2f}% "
      "-> quoted 98.50")
bat = np.load(HERE / "08_frozen_inputs_and_labels"
              / "battery_random1200_idx.npy")
check("5c", np.array_equal(np.sort(c_s), np.sort(t_s))
      and np.array_equal(np.sort(c_s), np.sort(bat[:600])),
      "both arms ran the identical image set = frozen battery[0:600]")

pins = json.load(open(HERE / "06_digital_comparators"
                      / "serial_predictions.json"))
dp = np.load(HERE / "06_digital_comparators" / "digital_pins_seg300600.npz",
             allow_pickle=True)
y2 = np.array([LAB[i] for i in dp["sel"]])
tw2 = int((dp["tw" + "in_under_twin"] == y2).sum())
cl2 = int((dp["clean_under_twin"] == y2).sum())
ci2 = int((dp["clean_ideal"] == y2).sum())
tw1 = round(pins["twin_under_twin_n300_frame"] / 100 * 300)
check("5d", pins["twin_under_twin_full"] == 98.56,
      "fielded 15-epoch checkpoint under twin forward, full digital test "
      f"set: {pins['twin_under_twin_full']} -> quoted 98.56 (S7.4)")
check("5e", tw1 + tw2 == 591,
      f"pinned digital twin-under-twin pooled {tw1}+{tw2} = {tw1+tw2}/600 "
      f"= {100*(tw1+tw2)/600:.2f}% -- coincides with the measured 98.50")
check("5f", ci2 == 300 and cl2 == 3,
      f"seg-2 pins: clean-ideal {ci2}/300, clean-under-twin {cl2}/300 "
      "(twin predicted the clean collapse before hardware)")

r = subprocess.run([sys.executable, str(_code_dir(__file__) / "verify_accuracy.py")],
                   capture_output=True, text=True, cwd=str(HERE))
out = r.stdout
check("5g", r.returncode == 0 and "  5.67%" in out and " 98.50%" in out,
      "verify_accuracy.py (archived audit script) runs end-to-end and "
      "prints 5.67% / 98.50% (its output: verify_accuracy_output.txt)")

# --------------------------------------- [6] archived session/format facts
print("\n[6] Archived metadata and pre-registration facts\n")
check("6a", meta["chain"].startswith("IF +10dB pads"),
      f"stationA meta chain = '{meta['chain']}' (10 dB IF padding, "
      "no receive gain -- S6.1/S7 chain statement)")
check("6b", meta["points"] == [[-9.0, -9.0], [-3.0, -3.0], [0.0, 0.0],
                               [3.0, 3.0], [7.0, 7.0]],
      "drive ladder points archived as commanded: "
      "(-9,-9)/(-3,-3)/(0,0)/(+3,+3)/(+7,+7) dBm")
pre = (HERE / "00_report_and_audit"
       / "2026-08-24_serial_m10m24_prespec.md").read_text(encoding="utf-8")
check("6c", "packtest |g|\n0.1155-0.1173" in pre.replace("\r\n", "\n")
      or "0.1155-0.1173" in pre,
      "pre-registration record: eight consecutive sessions, packtest "
      "|g| 0.1155-0.1173 at (0,0)")
spread = (0.1173 - 0.1155) / 0.1155 * 100
check("6d", round(spread, 1) == 1.6,
      f"gain spread (0.1173-0.1155)/0.1155 = {spread:.2f}% -> quoted "
      "'within 1.6%'")
macs = 28 * 28 * 32 * 75 + 10 * 10 * 64 * 800 + 128 * 1600 + 43 * 128
check("6e", macs == 7211904,
      f"r3plus MAC count 28^2*32*(5*5*3) + 10^2*64*(5*5*32) + 128*1600 "
      f"+ 43*128 = {macs:,} -> quoted 7,211,904 slots per image")
slot_s = macs * 32 / 1e7
check("6f", abs(slot_s - 23) < 0.2,
      f"{macs:,} slots x 32 samples / 10 MS/s = {slot_s:.1f} s -> quoted "
      "~23 s of slot time per image (TSLOT=32 in serial_nn_runner.py)")

# ------------------------------------------------- DOCUMENTED, NOT ASSERTED
print("""
DOCUMENTED (session-log-only claims of S7.1/S7.4/S7.5; sources are the
run logs and audit report at the read-only source archive, listed in
README.md -- not re-derivable from the arrays copied here)
  - Pack-test gate: fitted link gain vs frozen reference 0.1155 within
    +/-20%; linear-residual shape in [0.30, 0.75]; chain gate band from
    the reference diagnostic; sync-margin floor 50 (typical ~1,700).
    Gate constants: 04_hardware_test_code/serial_nn_runner.py (source);
    per-session values: serial_*.log files and
    00_report_and_audit/serial_run_audit.tex (pack-test repeats
    0.1155/0.1160/0.1162/0.1171 quoted there).
  - No accuracy gates on either arm (prespec verdict bands, archived
    here in 00_report_and_audit/, apply to the -10/-24 extension; the
    (0,0) campaign pre-registration is in the audit report).
  - Training-budget asymmetry (S7.4): 150-epoch clean checkpoint vs
    15-epoch twin sprint (~2.9 h); the 60-epoch sibling reaching 99.07
    under the twin forward lives in 03_training_results/
    train_serial_twin.log + weights_twin_60ep_sibling/ (source).
  - Twin surface evaluated from a 256^2 bilinear table, interpolation
    error <= 0.07 dB (audit report: max deviation 0.069 dB).
  - ~100 s wall clock per image; ~8.5 h of rig time per 300-image half,
    disjoint sessions on four days (session logs, ~30,400 s per half).""")

print("=" * 72)
print(f"RESULT: {'ALL ASSERTIONS PASSED' if FAIL == 0 else f'{FAIL} FAILURE(S)'}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
