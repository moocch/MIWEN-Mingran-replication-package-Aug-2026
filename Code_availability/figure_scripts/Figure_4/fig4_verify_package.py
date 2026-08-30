#!/usr/bin/env python3
"""Independent re-verification of the serial N=600 campaign + consolidated
fig4_v5 plotting package. Reads only raw archived files; recomputes every
headline number; flags mismatches against expected values."""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"e:\archive\Manuscript"
            r"\Overleaf\202607_MIWEN_Manuscript")
V2 = ROOT / "V2" / "hardware_aware_training_v2"
OUT = ROOT / "fig4_v5" / "data"
OUT.mkdir(parents=True, exist_ok=True)

FLAGS = []


def check(name, got, expect, tol=1e-9):
    ok = abs(got - expect) <= tol
    print(f"  [{'OK ' if ok else 'MISMATCH'}] {name}: got {got}  expected {expect}")
    if not ok:
        FLAGS.append(f"{name}: got {got}, expected {expect}")
    return ok


# ---------------------------------------------------------------- (a) labels
b = np.load(V2 / "08_frozen_inputs_and_labels" / "battery_frozen_slim.npz",
            allow_pickle=True)
LAB = dict(zip(b["img_index"].tolist(), b["labels"].tolist()))
bat = np.load(V2 / "08_frozen_inputs_and_labels" / "battery_random1200_idx.npy")
print(f"battery: {len(bat)} frozen images; labels for {len(LAB)} global indices")
assert set(bat.tolist()) == set(LAB.keys()), "battery idx / label key mismatch"

# ------------------------------------------------------- (b) merge hardware
RES = V2 / "05_hardware_result_0dBm_N600"
CLEAN = ["serial_nn_clean_0_150_20260823.npz",
         "serial_nn_clean_150_215_20260823.npz",
         "serial_nn_clean_215_300_20260823.npz",
         "serial_nn_clean_275_300_20260823.npz",
         "serial_nn_clean_300_600_20260823.npz"]
TWIN = ["serial_nn_twin_0_300_20260823.npz",
        "serial_nn_twin_300_600_20260823.npz"]


def load_run(path):
    z = np.load(path, allow_pickle=True)
    ks = sorted((k for k in z.files if k.startswith("chunk")),
                key=lambda s: int(s[5:].split("_")[0]))
    p = np.concatenate([z[k] for k in ks])
    sel = z["sel"][:len(p)]
    meta = json.loads(str(z["meta_json"]))
    return p, sel, meta


def merge_arm(folder, files):
    P, S, metas = [], [], []
    for f in files:
        p, s, m = load_run(RES / folder / f)
        P.append(p); S.append(s); metas.append((f, m, len(p)))
    P, S = np.concatenate(P), np.concatenate(S)
    assert len(set(S.tolist())) == len(S), f"{folder}: duplicate images"
    return P, S, metas


P_c, S_c, meta_c = merge_arm("clean_arm", CLEAN)
P_t, S_t, meta_t = merge_arm("twin_arm_hw_aware", TWIN)
print("\nper-file chunk counts:")
for f, m, n in meta_c + meta_t:
    print(f"  {f}: n={n} img_range={m['img_range']} weights={m['weights']} "
          f"power={m['power']}")

assert np.array_equal(np.sort(S_c), np.sort(S_t)), "arms saw different images"
assert np.array_equal(np.sort(S_c), np.sort(bat[:600])), \
    "image set != battery[0:600]"
print("image set identity: both arms == battery[0:600]  (600 images)")

y_c = np.array([LAB[i] for i in S_c])
y_t = np.array([LAB[i] for i in S_t])

# ------------------------------------------------------------ (c) accuracies
ok_c = int((P_c == y_c).sum())
ok_t = int((P_t == y_t).sum())
print("\n(c) per-arm hardware accuracy, N=600:")
check("clean correct", ok_c, 34)
check("twin  correct", ok_t, 591)
acc_clean = 100.0 * ok_c / 600
acc_twin = 100.0 * ok_t / 600
check("clean acc %", round(acc_clean, 3), round(100 * 34 / 600, 3), 1e-6)
check("twin  acc %", acc_twin, 98.5, 1e-9)

# confusion matrices (rows=true, cols=pred), 43 classes
NC = 43


def confusion(y, p):
    M = np.zeros((NC, NC), np.int64)
    for a, c in zip(y, p):
        M[a, c] += 1
    return M


C_clean = confusion(y_c, P_c)
C_twin = confusion(y_t, P_t)
assert C_clean.sum() == 600 and C_twin.sum() == 600

true_classes = sorted(set(y_c.tolist()))
pred_clean = collections.Counter(P_c.tolist())
pred_twin = collections.Counter(P_t.tolist())
true_counts = collections.Counter(y_c.tolist())
top_c = pred_clean.most_common(1)[0]
top_t = pred_twin.most_common(1)[0]
print(f"\ntrue classes present ({len(true_classes)} of 43): {true_classes}")
print("true-class counts:", dict(sorted(true_counts.items())))
print(f"clean arm predicted classes: {dict(sorted(pred_clean.items()))}")
print(f"twin  arm predicted classes ({len(pred_twin)} distinct): "
      f"{dict(sorted(pred_twin.items()))}")
check("clean collapse class", top_c[0], 12)
check("clean collapse count", top_c[1], 590)
print(f"twin most-predicted: class {top_t[0]} x{top_t[1]} (expect 1 x42)")
check("twin most-predicted class", top_t[0], 1)
check("twin most-predicted count", top_t[1], 42)
# diagonal / collapse details
print(f"clean diagonal sum {int(np.trace(C_clean))}; "
      f"clean col-12 sum {int(C_clean[:, 12].sum())}; "
      f"class-12 true count {true_counts.get(12, 0)}")
print(f"twin diagonal sum {int(np.trace(C_twin))}; "
      f"off-diagonal errors {600 - int(np.trace(C_twin))}")

# ---------------------------------------------- task 3: segment accuracies
pos = {g: i for i, g in enumerate(bat[:600].tolist())}   # battery position
seg_of_sel = np.array([0 if pos[g] < 300 else 1 for g in S_c])
seg_of_sel_t = np.array([0 if pos[g] < 300 else 1 for g in S_t])
print("\ntask 3: per-segment hardware accuracy:")
res_seg = {}
for arm, P, y, seg, exp in (("clean", P_c, y_c, seg_of_sel, (None, 5.00)),
                            ("twin", P_t, y_t, seg_of_sel_t, (97.67, 99.33))):
    for s in (0, 1):
        m = seg == s
        n_ok, n = int((P[m] == y[m]).sum()), int(m.sum())
        a = 100 * n_ok / n
        res_seg[f"{arm}_seg{s+1}"] = (n_ok, n, a)
        pin = exp[s]
        tag = ""
        if pin is not None:
            tag = f"  vs log FINAL {pin:.2f} -> " + \
                ("OK" if abs(a - pin) < 0.005 else "MISMATCH")
            if abs(a - pin) >= 0.005:
                FLAGS.append(f"{arm} seg{s+1}: {a:.2f} vs log {pin}")
        print(f"  {arm} seg{s+1}: {n_ok}/{n} = {a:.2f}%{tag}")
# per-file clean cross-checks vs log FINAL lines
print("per-file clean accuracy vs log FINAL lines:")
for (f, m, n), pin in zip(meta_c, (7.33, 3.08, None, 12.00, 5.00)):
    p, s, _ = load_run(RES / "clean_arm" / f)
    yy = np.array([LAB[i] for i in s])
    a = 100 * float((p == yy).mean())
    tag = "(no FINAL line; run h2b crashed after 60 imgs)" if pin is None \
        else f"vs log FINAL {pin:.2f} -> " + \
        ("OK" if abs(a - pin) < 0.005 else "MISMATCH")
    if pin is not None and abs(a - pin) >= 0.005:
        FLAGS.append(f"clean file {f}: {a:.2f} vs log {pin}")
    print(f"  {f}: {int((p==yy).sum())}/{len(p)} = {a:.2f}%  {tag}")

# --------------------------------------------------- (d) digital comparators
d = np.load(V2 / "06_digital_comparators" / "digital_pins_seg300600.npz",
            allow_pickle=True)
pin1 = json.load(open(V2 / "06_digital_comparators" / "serial_predictions.json"))
y2 = np.array([LAB[i] for i in d["sel"]])
n_tw2 = int((d["twin_under_twin"] == y2).sum())
n_cl2 = int((d["clean_under_twin"] == y2).sum())
n_ci2 = int((d["clean_ideal"] == y2).sum())
print("\n(d) digital comparators (seg2 = images 300-599, pinned pre-run):")
check("twin_under_twin seg2 correct", n_tw2, 296)
check("clean_under_twin seg2 correct", n_cl2, 3)
check("clean_ideal seg2 correct", n_ci2, 300)
n_tw1 = round(pin1["twin_under_twin_n300_frame"] / 100 * 300)
n_cl1 = round(pin1["clean_under_twin_n300"] / 100 * 300)
check("twin_under_twin seg1 (pinned json)", n_tw1, 295)
check("clean_under_twin seg1 (pinned json)", n_cl1, 2)
acc_twin_dig = 100 * (n_tw1 + n_tw2) / 600
acc_cut_dig = 100 * (n_cl1 + n_cl2) / 600
check("twin digital pooled correct", n_tw1 + n_tw2, 591)
check("clean_under_twin pooled correct", n_cl1 + n_cl2, 5)
print(f"  twin digital pooled acc = {acc_twin_dig:.4f}% "
      f"(hardware {acc_twin:.4f}%) -> both 591/600 = 98.50%: "
      f"{'CONFIRMED' if (n_tw1+n_tw2) == ok_t == 591 else 'MISMATCH'}")
if not ((n_tw1 + n_tw2) == ok_t == 591):
    FLAGS.append("twin hw / digital pooled scores differ from 591/600")
check("clean_under_twin pooled acc %", round(acc_cut_dig, 2), 0.83, 0.005)

# identical-prediction agreement, twin arm, seg2 (only segment with stored
# per-image digital predictions; seg1 digital preds pinned as % only)
sel2 = d["sel"]
hw_map = dict(zip(S_t.tolist(), P_t.tolist()))
hw2 = np.array([hw_map[i] for i in sel2])
agree = int((hw2 == d["twin_under_twin"]).sum())
print(f"  twin arm hw-vs-digital identical predictions, seg2: "
      f"{agree}/300 = {100*agree/300:.2f}%")
hw_ok2 = int((hw2 == y2).sum())
print(f"    (hw seg2 correct {hw_ok2}/300, digital seg2 correct {n_tw2}/300)")
# clean arm agreement vs clean_under_twin digital, seg2 (informational)
hw_map_c = dict(zip(S_c.tolist(), P_c.tolist()))
hwc2 = np.array([hw_map_c[i] for i in sel2])
agree_c = int((hwc2 == d["clean_under_twin"]).sum())
print(f"  clean arm hw-vs-clean_under_twin digital identical preds, seg2: "
      f"{agree_c}/300 = {100*agree_c/300:.2f}%")

# battery-frame clean-ideal digital (audit pin 99.50 on 1200; slim file's
# own stored preds are the frozen-calibration hardware-era preds, 98.92)
acc_bat_slim = 100 * float((b["preds"] == b["labels"]).mean())
print(f"  battery_frozen_slim stored preds vs labels: {acc_bat_slim:.2f}% "
      f"(file's own 'accuracy' field {100*float(b['accuracy']):.2f}%) "
      f"[comb-era frozen-cal record, NOT the clean-ideal digital]")
ACC_CLEAN_IDEAL_BAT = 99.50   # audit pin, miwen_serial_frozen_reference PINS

# ------------------------------------------------------- (e) calibrations
print("\n(e) per-layer calibration files:")
cals = {}
for arm, p in (("clean", RES / "clean_arm" / "serial_cal_clean_20260823.npz"),
               ("twin", RES / "twin_arm_hw_aware" / "serial_cal_twin_20260823.npz")):
    z = np.load(p, allow_pickle=True)
    print(f"  {arm}: {p.name}")
    for k in sorted(z.files):
        v = z[k]
        cals[f"cal_{arm}_{k}"] = v
        if v.shape == ():
            print(f"    {k}: scalar {float(v):.6e}")
        else:
            print(f"    {k}: shape {v.shape} values {np.array2string(v, precision=6)}"
                  f" (min {v.min():.6e}, max {v.max():.6e})")

# ------------------------------------------------------------- (f) ENOB + A
enob = json.load(open(V2 / "07_supporting_analysis_enob_and_driveladder"
                      / "serial_enob_00_20260824.json"))
print("\n(f) ENOB summary (serial_enob_00_20260824.json):")
print(f"  overall_naive_bits = {enob['overall_naive_bits']:.4f}")
print(f"  overall_twin_bits  = {enob['overall_twin_bits']:.4f}")
print(f"  bins ({len(enob['bins'])}): "
      + ", ".join(f"{v:.4f}" for v in enob["bins"]))
print("  enob_naive: " + ", ".join(f"{v:.3f}" for v in enob["enob_naive"]))
print("  enob_twin : " + ", ".join(f"{v:.3f}" for v in enob["enob_twin"]))
print(f"  n per bin: {enob['n']}")

sa = np.load(V2 / "07_supporting_analysis_enob_and_driveladder"
             / "serial_stationA_20260823.npz", allow_pickle=True)
sa_meta = json.loads(str(sa["meta_json"]))
POW = [p[0] for p in sa_meta["points"]]        # symmetric drives
print(f"\nstationA drive ladder powers (dBm, both ports): {POW}")
tags = [f"{p:+.0f}_{p:+.0f}" for p in POW]
lad_gain = np.zeros((5, 2, 2))
lad_resid = np.zeros((5, 2, 2))
lad_yhat = np.zeros((5, 2, 2, 256), np.complex128)
xs = {0: sa["x_d0"], 1: sa["x_d1"]}
ws = {0: sa["w_d0"], 1: sa["w_d1"]}
for ip, tag in enumerate(tags):
    for dr in (0, 1):
        xw = xs[dr] * ws[dr]
        for rp in (0, 1):
            yh = sa[f"yhat_{tag}_d{dr}_r{rp}"]
            g = np.vdot(xw, yh) / np.vdot(xw, xw)
            res = yh - g * xw
            rel = np.sqrt(np.mean(np.abs(res) ** 2)
                          / np.mean(np.abs(g * xw) ** 2))
            lad_gain[ip, dr, rp] = abs(g)
            lad_resid[ip, dr, rp] = rel
            lad_yhat[ip, dr, rp] = yh
    print(f"  {tag}: |g| {lad_gain[ip].mean():.4f}  "
          f"lin-resid rel-rms {lad_resid[ip, :, :].mean():.4f} "
          f"(d0r0 {lad_resid[ip,0,0]:.4f}, d0r1 {lad_resid[ip,0,1]:.4f}, "
          f"d1r0 {lad_resid[ip,1,0]:.4f}, d1r1 {lad_resid[ip,1,1]:.4f})")

# ------------------------------------------------------------ (g) twin surf
tm = json.load(open(V2 / "01_digital_twin_model" / "serial_twin_model.json"))
thp = np.array(tm["thp"])
ridges = np.array(tm["ridges"]).reshape(-1, 4)
assert len(ridges) == tm["K"] == 20


def product_db(p_lo, p_rf):
    """EXACT frozen form (miwen_serial_frozen_reference.py product_db /
    serial_twin_fit.py product_db with fitted thp+ridges)."""
    a0, klo, krf = thp[0], thp[1], thp[2]
    xl = 10 ** ((p_lo - klo) / 10.0)
    xr = 10 ** ((p_rf - krf) / 10.0)
    y = a0 + 10 * np.log10((xl / (1 + xl)) * (xr / (1 + xr)) + 1e-30)
    for wi, ai, bi, ci in ridges:
        y = y + wi * np.tanh(ai * p_lo / 40 + bi * p_rf / 40 + ci)
    return y


print("\n(g) twin surface reconstruction:")
print(f"  thp = {thp.tolist()}")
print(f"  K = {tm['K']} ridges, duty_db = {tm['duty_db']}, "
      f"rmse_holdout = {tm['rmse_holdout']:.4f} dB, gates = {tm['gates']}")
# bit-exact probe replicated from the frozen reference (--verify probe)
import hashlib
g8 = np.linspace(-35.0, 0.0, 8)
PL8, PR8 = np.meshgrid(g8, g8)
probe = hashlib.sha256(np.round(product_db(PL8, PR8), 12)
                       .tobytes()).hexdigest()[:16]
print(f"  surface probe hash = {probe} "
      f"(frozen reference pin 5cb94af8cc5b257a): "
      f"{'OK' if probe == '5cb94af8cc5b257a' else 'MISMATCH'}")
if probe != "5cb94af8cc5b257a":
    FLAGS.append("twin surface probe hash mismatch")

# signed transfer on the native normalized domain (twin_matmul convention:
# unit-RMS streams, amplitude floor 1e-3, LO=weight, RF=activation)
xg = np.linspace(-1.0, 1.0, 201)
wg = np.linspace(-1.0, 1.0, 201)
PLg = 20 * np.log10(np.maximum(np.abs(wg), 1e-3))
PRg = 20 * np.log10(np.maximum(np.abs(xg), 1e-3))
AMP = 10 ** (product_db(PLg[None, :], PRg[:, None]) / 20.0)   # [x, w]
F = np.sign(xg[:, None] * wg[None, :]) * AMP
print(f"  f-grid 201x201 on x,w in [-1,1]: |f| range "
      f"{np.abs(F).min():.4g} .. {np.abs(F).max():.4g} (map uV units)")
# 1-D slices: f vs ideal product p = x*w at fixed |w|
slice_w = np.array([0.125, 0.25, 0.5, 1.0])
slice_p = np.zeros((len(slice_w), len(xg)))
slice_f = np.zeros((len(slice_w), len(xg)))
for i, wv in enumerate(slice_w):
    pl = 20 * np.log10(max(wv, 1e-3))
    a = 10 ** (product_db(np.full_like(xg, pl), PRg) / 20.0)
    slice_p[i] = xg * wv
    slice_f[i] = np.sign(xg) * a
    # compression check: gain at |x|=1 vs small-signal gain
    gs = a[np.argmin(np.abs(xg - 0.05))] / 0.05
    gl = a[-1] / 1.0
    print(f"    slice w={wv}: f(1,w)={a[-1]:.4g} uV; large/small-signal "
          f"gain ratio {gl/gs:.3f} ({20*np.log10(gl/gs):+.2f} dB)")

hm = np.load(V2 / "01_digital_twin_model" / "heatmap_unpadded_20260814.npz",
             allow_pickle=True)
print(f"  measured CW map: grid {hm['if_amp_uv_mean'].shape}, "
      f"LO {hm['p_lo_dbm_grid'].min():.0f}..{hm['p_lo_dbm_grid'].max():.0f} dBm, "
      f"RF {hm['p_rf_dbm_grid'].min():.0f}..{hm['p_rf_dbm_grid'].max():.0f} dBm, "
      f"clean cells {int(hm['clean_mask'].sum())}/{hm['clean_mask'].size}")
# twin-vs-map residual on the fit region (sanity vs rmse_holdout 0.12 dB)
LOm, RFm = np.meshgrid(hm["p_lo_dbm_grid"].astype(float),
                       hm["p_rf_dbm_grid"].astype(float), indexing="ij")
A_uv = hm["if_amp_uv_mean"].astype(float)
mfit = (hm["clean_mask"].astype(bool) & (A_uv > 0)
        & (hm["peak_fs"].astype(float) < 0.95) & (LOm >= -35) & (RFm >= -35))
# model_db needs leak terms for the FULL map; product region approx:
y_meas = 20 * np.log10(A_uv[mfit])
l1, l2 = thp[3], thp[4]
y_mw = (10 ** (product_db(LOm[mfit], RFm[mfit]) / 10.0)
        + 10 ** ((l1 + LOm[mfit]) / 10.0) + 10 ** ((l2 + RFm[mfit]) / 10.0))
y_mod = 10 * np.log10(y_mw)
rmse_all = float(np.sqrt(np.mean((y_mod - y_meas) ** 2)))
print(f"  twin vs measured map, fit region ({int(mfit.sum())} cells): "
      f"rmse {rmse_all:.3f} dB (pinned held-out rmse "
      f"{tm['rmse_holdout']:.3f} dB)")

# ------------------------------------------------------------ (h) package
np.savez_compressed(
    OUT / "fig4_serial_results.npz",
    confusion_clean=C_clean.astype(np.int64),
    confusion_twin=C_twin.astype(np.int64),
    acc_clean=np.float64(acc_clean),
    acc_twin=np.float64(acc_twin),
    acc_twin_digital=np.float64(acc_twin_dig),
    acc_clean_under_twin_digital=np.float64(acc_cut_dig),
    acc_clean_ideal_digital=np.float64(ACC_CLEAN_IDEAL_BAT),
    acc_clean_seg1=np.float64(res_seg["clean_seg1"][2]),
    acc_clean_seg2=np.float64(res_seg["clean_seg2"][2]),
    acc_twin_seg1=np.float64(res_seg["twin_seg1"][2]),
    acc_twin_seg2=np.float64(res_seg["twin_seg2"][2]),
    hw_sel=S_t, hw_pred_twin=P_t, hw_true_twin=y_t,
    hw_pred_clean=np.array([hw_map_c[i] for i in S_t]),
    **cals,
    enob_json=np.str_(json.dumps(enob)),
    x_grid=xg, w_grid=wg, f_grid=F,
    slice_w=slice_w, slice_p=slice_p, slice_f=slice_f,
    twin_thp=thp, twin_ridges=ridges,
    cw_p_lo_dbm=hm["p_lo_dbm_grid"].astype(float),
    cw_p_rf_dbm=hm["p_rf_dbm_grid"].astype(float),
    cw_amp_uv_mean=A_uv, cw_amp_uv_std=hm["if_amp_uv_std"].astype(float),
    cw_clean_mask=hm["clean_mask"].astype(bool),
    cw_peak_fs=hm["peak_fs"].astype(float),
    ladder_powers_dbm=np.array(POW),
    ladder_gain_abs=lad_gain, ladder_lin_resid=lad_resid,
    ladder_yhat=lad_yhat,
    ladder_x_d0=sa["x_d0"], ladder_w_d0=sa["w_d0"],
    ladder_x_d1=sa["x_d1"], ladder_w_d1=sa["w_d1"],
)

numbers = {
    "acc_clean_hw": {"value": acc_clean, "n": "34/600", "provenance":
        "recomputed: 5 clean_arm serial_nn_clean_*.npz chunk preds vs battery_frozen_slim labels"},
    "acc_twin_hw": {"value": acc_twin, "n": "591/600", "provenance":
        "recomputed: 2 twin_arm serial_nn_twin_*.npz chunk preds vs battery_frozen_slim labels"},
    "acc_twin_hw_seg1": {"value": res_seg["twin_seg1"][2], "n": "293/300",
        "provenance": "recomputed battery[0:300]; matches serial_twin_20260824.log FINAL 97.67"},
    "acc_twin_hw_seg2": {"value": res_seg["twin_seg2"][2], "n": "298/300",
        "provenance": "recomputed battery[300:600]; matches serial_twin_00_fresh_20260825.log FINAL 99.33"},
    "acc_clean_hw_seg1": {"value": res_seg["clean_seg1"][2],
        "n": f"{res_seg['clean_seg1'][0]}/300",
        "provenance": "recomputed battery[0:300]; log FINALs 7.33 (0:150), 3.08 (150:215), 12.00 (275:300); 215:275 from crashed-run npz"},
    "acc_clean_hw_seg2": {"value": res_seg["clean_seg2"][2], "n": "15/300",
        "provenance": "recomputed battery[300:600]; matches serial_clean_00_fresh_20260826.log FINAL 5.00"},
    "acc_twin_digital": {"value": acc_twin_dig, "n": "591/600 (295+296)",
        "provenance": "seg1 pinned serial_predictions.json twin_under_twin_n300_frame=98.33; seg2 recomputed from digital_pins_seg300600.npz twin_under_twin"},
    "acc_clean_under_twin_digital": {"value": acc_cut_dig, "n": "5/600 (2+3)",
        "provenance": "seg1 pinned serial_predictions.json clean_under_twin_n300=0.6667%; seg2 recomputed from digital_pins_seg300600.npz clean_under_twin"},
    "acc_clean_ideal_digital_battery1200": {"value": ACC_CLEAN_IDEAL_BAT,
        "provenance": "audit pin (miwen_serial_frozen_reference.py PINS clean_ideal_bat1200=99.50); not recomputable here (needs gtsrb_roi_32x32.npz full forward)"},
    "acc_clean_ideal_digital_full12630": {"value": 99.05,
        "provenance": "audit pin (miwen_serial_frozen_reference.py PINS clean_ideal_full)"},
    "acc_clean_ideal_digital_seg2": {"value": 100.0, "n": "300/300",
        "provenance": "recomputed from digital_pins_seg300600.npz clean_ideal vs labels"},
    "twin_hw_vs_digital_agreement_seg2": {"value": 100 * agree / 300,
        "n": f"{agree}/300", "provenance":
        "identical per-image prediction, twin hw vs pinned twin_under_twin digital, images 300-599 (seg1 digital preds archived only as a pinned %)"},
    "clean_hw_vs_clean_under_twin_agreement_seg2": {"value": 100 * agree_c / 300,
        "n": f"{agree_c}/300", "provenance": "same comparison for the clean arm"},
    "clean_collapse_class": {"value": 12, "n": f"x{top_c[1]} of 600",
        "provenance": "mode of merged clean-arm hardware predictions"},
    "twin_top_class": {"value": int(top_t[0]), "n": f"x{top_t[1]}",
        "provenance": "mode of merged twin-arm hardware predictions"},
    "enob_overall_naive_bits": {"value": enob["overall_naive_bits"],
        "provenance": "serial_enob_00_20260824.json (256-pair (0,0) capture, FS-referred)"},
    "enob_overall_twin_bits": {"value": enob["overall_twin_bits"],
        "provenance": "serial_enob_00_20260824.json, twin-inverted"},
    "twin_rmse_holdout_db": {"value": tm["rmse_holdout"],
        "provenance": "serial_twin_model.json (20% held-out cells of CW map fit region)"},
    "twin_rmse_fitregion_recomputed_db": {"value": rmse_all,
        "provenance": "recomputed here: full model (product+leak) vs heatmap_unpadded_20260814 fit-region cells"},
    "twin_knee_lo_dbm": {"value": tm["knees"][0],
        "provenance": "serial_twin_model.json (reported-not-gated)"},
    "twin_knee_rf_dbm": {"value": tm["knees"][1],
        "provenance": "serial_twin_model.json (reported-not-gated)"},
    "twin_duty_db": {"value": tm["duty_db"],
        "provenance": "serial_twin_model.json (slot duty offset used in fit G3/ENOB, not in twin_matmul)"},
    "twin_gates": {"value": tm["gates"], "provenance": "serial_twin_model.json"},
    "twin_probe_hash_ok": {"value": probe == "5cb94af8cc5b257a",
        "provenance": "sha256[:16] of product_db on frozen 8x8 probe grid vs pin 5cb94af8cc5b257a"},
    "ladder_powers_dbm": {"value": POW,
        "provenance": "serial_stationA_20260823.npz meta points (symmetric, both ports)"},
    "ladder_lin_resid_mean": {"value": lad_resid.mean(axis=(1, 2)).tolist(),
        "provenance": "recomputed per drive: rms(yhat-g*xw)/rms(g*xw), mean over 2 draws x 2 reps"},
    "ladder_gain_abs_mean": {"value": lad_gain.mean(axis=(1, 2)).tolist(),
        "provenance": "recomputed per drive: |vdot(xw,yhat)/vdot(xw,xw)|"},
    "cal_clean": {"value": {k.replace("cal_clean_", ""): (np.asarray(v).tolist())
                            for k, v in cals.items() if k.startswith("cal_clean")},
        "provenance": "serial_cal_clean_20260823.npz (frozen per-layer scalar gain g_l* and slot-stream RMS pair s_l*)"},
    "cal_twin": {"value": {k.replace("cal_twin_", ""): (np.asarray(v).tolist())
                           for k, v in cals.items() if k.startswith("cal_twin")},
        "provenance": "serial_cal_twin_20260823.npz"},
    "battery_slim_stored_preds_acc": {"value": acc_bat_slim,
        "provenance": "battery_frozen_slim.npz preds vs labels (comb-era frozen-calibration record; NOT clean-ideal digital)"},
    "n_images": {"value": 600, "provenance":
        "battery_random1200_idx.npy[0:600]; both arms identical set (asserted)"},
}
json.dump(numbers, open(OUT / "fig4_numbers.json", "w"), indent=1)

print(f"\nwrote {OUT / 'fig4_serial_results.npz'}")
print(f"wrote {OUT / 'fig4_numbers.json'}")
z = np.load(OUT / "fig4_serial_results.npz", allow_pickle=True)
print(f"package keys ({len(z.files)}):")
for k in z.files:
    print(f"  {k}: {z[k].dtype} {z[k].shape}")

print("\n" + ("FLAGGED MISMATCHES: " + "; ".join(FLAGS) if FLAGS
              else "ALL CROSS-CHECKS PASSED — no mismatches."))
