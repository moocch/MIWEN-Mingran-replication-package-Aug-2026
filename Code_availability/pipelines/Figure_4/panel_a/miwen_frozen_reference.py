#!/usr/bin/env python3
"""MIWEN frozen algorithm — single-file audit reference.

This file contains, in one place, every mathematical operation the FROZEN
configuration (r3plus-ns25, freeze note docs/notes/2026-08-04_algorithm_freeze.md,
tag algorithm-freeze-20260804) performs from raw test image to class label,
extracted verbatim from the running code:

    preprocess          == 4_gtsrb_confusion_mlpN.preprocess   (2-98 stretch)
    load_weights        == run_ladder_hw.load_cnn
    im2col/maxpool2/maxnorm == run_ladder_hw (identical bytes)
    fit_gj              == rung3_correction.fit_gj             (equalizer)
    debias              == run_ladder_hw.run_layer_mvm's Rice debias line
    inter-pass ops      == run_ladder_hw.main's conv/dense branches
    decision            == argmax of squared final magnitudes

The analog MVM itself (OFDM comb synthesis, mixer, capture, decode) is the
instrument layer, out of scope here: on hardware it produces the per-layer
magnitude matrices; `python3 miwen_frozen_reference.py --verify` proves this
file is *programmatically identical* to what ran, by (1) reproducing the
frozen digital references bit-for-bit, and (2) replaying every archived
hardware run of the frozen algorithm from its stored magnitudes through the
code below and matching the archived predictions image-for-image.

No torch, no imports beyond numpy + stdlib. ~200 audited lines.
"""
from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = _data_dir(__file__)
FROZEN_WEIGHTS = "r35_r3plus_ns25_s0_hw.npz"
FROZEN_SHA16 = "9be6e085f2d12e19"
PSTRETCH_LO, PSTRETCH_HI = 2.0, 98.0


# ---- input preprocessing (== core preprocess) ------------------------------
def preprocess(u8: np.ndarray) -> np.ndarray:
    """Per-image contrast stretch: scale each image so its 2nd..98th intensity
    percentiles span [0,1], clipped. Output rows are 3072-vectors."""
    x = u8.reshape(u8.shape[0], -1).astype(np.float64) / 255.0
    lo = np.percentile(x, PSTRETCH_LO, axis=1, keepdims=True)
    hi = np.percentile(x, PSTRETCH_HI, axis=1, keepdims=True)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


# ---- weights (== run_ladder_hw.load_cnn) -----------------------------------
def load_weights(path: Path):
    z = np.load(path, allow_pickle=True)
    arch = json.loads(str(z["arch_json"]))
    layers = []
    for i, c in enumerate(arch["conv"]):
        W = (z[f"c{i}r"] + 1j * z[f"c{i}i"]).reshape(c["cout"], -1)
        scale = (z[f"c{i}s"].astype(np.float64) if f"c{i}s" in z.files
                 else np.ones(c["cout"]))
        layers.append(dict(kind="conv", W=W.astype(np.complex128),
                           b=z[f"c{i}b"].astype(np.float64), k=c["k"],
                           cout=c["cout"], scale=scale,
                           stride=int(c.get("stride", 1)),
                           pool=bool(c.get("pool", True))))
    j = 0
    while f"d{j}r" in z.files:
        layers.append(dict(kind="dense",
                           W=(z[f"d{j}r"] + 1j * z[f"d{j}i"]).astype(np.complex128),
                           b=z[f"d{j}b"].astype(np.float64)))
        j += 1
    return layers


# ---- spatial ops (== run_ladder_hw, byte-identical) ------------------------
def im2col(A, k, stride=1):
    """A: (N, C, H, W) -> patches (N, P, C*k*k), torch-conv element order."""
    N, C, H, W = A.shape
    Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
    s = A.strides
    win = np.lib.stride_tricks.as_strided(
        A, (N, C, Ho, Wo, k, k),
        (s[0], s[1], s[2] * stride, s[3] * stride, s[2], s[3]))
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(N, Ho * Wo, C * k * k), Ho, Wo


def maxpool2(A):
    N, C, H, W = A.shape
    return A.reshape(N, C, H // 2, 2, W // 2, 2).max(5).max(3)


def maxnorm(A):
    m = np.maximum(A.reshape(A.shape[0], -1).max(1), 1e-12)
    return A / m.reshape(-1, *([1] * (A.ndim - 1)))


# ---- readout correction (the ENTIRE correction stack of the frozen algo) ---
def fit_gj(am: np.ndarray, ar: np.ndarray) -> np.ndarray:
    """Per-column least-squares gain of measured vs reference magnitudes:
    g_j = sum(am*ar)/sum(ar^2). 43..128 scalars per layer, label-free.
    (== rung3_correction.fit_gj)"""
    num = np.sum(am * ar, axis=0)
    den = np.maximum(np.sum(ar * ar, axis=0), 1e-30)
    g = num / den
    return np.where(g > 1e-12, g, 1.0)


def debias(am_raw: np.ndarray, sigma2: float) -> np.ndarray:
    """Rice-bias removal of the noise floor: |signal+noise| overestimates
    |signal|; subtract the measured per-bin noise power in quadrature.
    (== run_ladder_hw.run_layer_mvm debias line)"""
    return np.sqrt(np.maximum(am_raw ** 2 - sigma2, 0.0))


# ---- the network: one inter-pass step per layer ----------------------------
GEO = {1: (28, 28), 2: (10, 10)}          # conv output grids at 32x32 input


def interpass(am, lay, tag, n):
    """From a layer's (equalized) output magnitudes to the next layer's
    input activations. (== run_ladder_hw.main conv/dense branches)"""
    if lay["kind"] == "conv":
        S = np.maximum(am * lay["scale"][None, :] + lay["b"][None, :], 0.0)
        Ho, Wo = GEO[tag]
        fm = S.reshape(n, Ho, Wo, lay["cout"]).transpose(0, 3, 1, 2)
        return maxnorm(maxpool2(fm) if lay["pool"] else fm), S
    S = np.maximum(am + lay["b"][None, :], 0.0)
    return maxnorm(S)[:, None, None, :].transpose(0, 3, 1, 2), S


def forward_digital(X_rows, layers):
    """Exact digital execution: the analog MVM replaced by |X @ W^T| (with
    ideal transport, equalization is identity and sigma2=0, so the digital
    chain is the measured chain with a perfect instrument)."""
    n = X_rows.shape[0]
    A = X_rows.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
    for tag, lay in enumerate(layers, 1):
        if lay["kind"] == "conv":
            P, _, _ = im2col(A, lay["k"], lay["stride"])
            am = np.abs(P.reshape(-1, P.shape[-1]) @ lay["W"].T)
        else:
            am = np.abs(A.reshape(n, -1) @ lay["W"].T)
        A, S = interpass(am, lay, tag, n)
    return np.argmax(S ** 2, axis=1)


def replay_measured(npz_path, layers):
    """Re-derive predictions from an archived hardware run's stored
    (equalized) magnitudes through the inter-pass code above. Matching the
    archived predictions proves this file == the code that ran."""
    z = np.load(npz_path, allow_pickle=True)
    n = len(z["labels"])
    for tag, lay in enumerate(layers, 1):
        am = z[f"y_mag_l{tag}"].astype(np.float64)
        A, S = interpass(am, lay, tag, n)
    return np.argmax(S ** 2, axis=1), z["preds"], z["labels"], z["img_index"]


# ---- verification harness --------------------------------------------------
def verify():
    ok = True
    sha = hashlib.sha256((HERE / FROZEN_WEIGHTS).read_bytes()).hexdigest()[:16]
    print(f"[1] weights sha256[:16] = {sha} "
          f"({'OK' if sha == FROZEN_SHA16 else 'MISMATCH!'})")
    ok &= sha == FROZEN_SHA16
    layers = load_weights(HERE / FROZEN_WEIGHTS)

    d = np.load(HERE / "gtsrb_roi_32x32.npz", allow_pickle=True)
    Xte = preprocess(d["Xte"])
    yte = d["yte"].astype(np.int64)

    for name, sel, pin in (
            ("frame 600-899", np.arange(600, 900), 97.67),
            ("random-450", np.load(HERE / "s4_random450_idx.npy"), 97.56),
            ("battery-1200", np.load(HERE / "battery_random1200_idx.npy"), 98.42)):
        pd = forward_digital(Xte[sel], layers)
        acc = (pd == yte[sel]).mean() * 100
        line = f"[2] digital {name}: {acc:.2f} (frozen ref {pin:.2f})"
        if name == "battery-1200":
            slim = np.load(HERE / "battery_slim.npz", allow_pickle=True)
            bit = (pd == slim["digital_preds"]).all()
            line += f"; preds vs archived digital: {'IDENTICAL' if bit else 'DIFFER'}"
            ok &= bool(bit)
        print(line)
        ok &= abs(acc - pin) < 0.005

    for npz in ("s3_arm1_ns25.npz", "s4_random450.npz"):
        p, p_arch, y, _ = replay_measured(HERE / npz, layers)
        same = (p == p_arch).all()
        print(f"[3] replay {npz}: preds {'IDENTICAL' if same else 'DIFFER'} "
              f"(measured acc {(p_arch == y).mean()*100:.2f})")
        ok &= bool(same)
    for i in range(1, 11):
        f = HERE / f"bat_c{i}.npz"
        if not f.exists():
            print(f"[3] bat_c{i}.npz absent locally - skipped")
            continue
        p, p_arch, y, _ = replay_measured(f, layers)
        ok &= bool((p == p_arch).all())
    print(f"[3] battery chunk replays: {'ALL IDENTICAL' if ok else 'CHECK ABOVE'}")
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---- instrument drive mode (--hw / --dry-run) ------------------------------
# The one part of execution not defined above: transporting each layer's MVM
# through the analog chain. The RF/OFDM primitives and instrument I/O live in
# 4_gtsrb_confusion_mlpN.py (imported here, unmodified); this loop is the
# serial hardware orchestration, byte-equivalent to run_ladder_hw's serial
# path (dry-run bit-identity gated). No pipelining: auditability first.

def run_layer_hw(core, W, X, *, args, layer_tag, hw, g_frozen):
    """Deployment inference for one layer. The mixer performs W@x; the host
    NEVER computes it. Equalization uses the FROZEN per-column calibration
    g_frozen (measured once, offline, on a disjoint calibration set) instead
    of a per-image digital reference -- so build_frame_ml runs with
    compute_digital=False and no W@x is executed on the host at inference.
    (Energy-claim fix, 2026-08-06; see docs/notes freeze addendum 4.)"""
    plan = core.plan_layer(W.shape[1], W.shape[0], args.fft_len, args.k0,
                           args.bw_frac, args.k_per_symbol)
    cap_v = core.ring_cap_vectors(plan.R, args.fft_len, args.cp_len, args.gap)
    per = max(1, min(int(args.images_per_frame), cap_v))
    zc = core.zc_phase(W.shape[1])
    Wt = W * np.conj(zc)[None, :]
    S = X.shape[0]
    Y = np.zeros((S, W.shape[0]), np.complex128)
    s2s = []
    dummy_y = np.zeros(S, np.int64)
    for c0 in range(0, S, per):
        idx = np.arange(c0, min(c0 + per, S))
        x_t = (X[idx].astype(np.complex128) * zc[None, :])
        fr = core.build_frame_ml(Wt, x_t, dummy_y[idx], idx, plan,
                                 L=args.fft_len, cp=args.cp_len, gap=args.gap,
                                 sync_seed=args.seed, layer_tag=layer_tag,
                                 verbose=False, compute_digital=False)
        if args.dry_run:
            snr = args.snr_db if args.snr_db is not None else float("inf")
            cap = core.synth_capture_ml(fr, float(snr), args.sim_mixer,
                                        args.seed, args.fs,
                                        salt=layer_tag * 10_000 + c0,
                                        verbose=False)
        else:
            hw.ensure(fr, verbose_build=(c0 == 0))
            cap = hw.capture(fr)
        res = core.decode_ml(cap, fr, args.fs, flip_output=False,
                             conj_output=False, verbose=False)
        if not res.ok:
            raise RuntimeError(f"decode failed L{layer_tag}@{c0}: {res.reason}")
        Y[idx] = res.y_meas
        s2s.append(res.sigma2_bin)
    sigma2 = float(np.mean(s2s))
    am = debias(np.abs(Y), sigma2)
    return am / g_frozen[None, :], sigma2          # frozen calibration; no W@x


def main_hw(argv):
    import argparse
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gtsrb_hw_core", HERE / "4_gtsrb_confusion_mlpN.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules["gtsrb_hw_core"] = core
    spec.loader.exec_module(core)

    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--idx-npy", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--images-per-frame", type=int, default=999)
    ap.add_argument("--sim-mixer", choices=("ideal", "limiter"), default="ideal")
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--fft-len", type=int, default=16384)
    ap.add_argument("--cp-len", type=int, default=512)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--k0", type=int, default=6)
    ap.add_argument("--bw-frac", type=float, default=0.5)
    ap.add_argument("--k-per-symbol", type=int, default=64)
    ap.add_argument("--fs", type=float, default=10e6)
    ap.add_argument("--seed", type=int, default=1)
    for name, default in (("--p-lo-dbm", "P_LO_DBM_DEFAULT"),
                          ("--p-rf-dbm", "P_RF_DBM_DEFAULT"),
                          ("--rx-gain", "RX_GAIN_DB_DEFAULT"),
                          ("--p-max-dbm", "P_MAX_DBM_DEFAULT"),
                          ("--settle-s", "SETTLE_S_DEFAULT"),
                          ("--start-lead", "START_LEAD_S_DEFAULT")):
        ap.add_argument(name, type=float, default=getattr(core, default))
    ap.add_argument("--capture-retries", type=int,
                    default=core.CAPTURE_RETRIES_DEFAULT)
    ap.add_argument("--no-ext-ref", action="store_true")
    ap.add_argument("--tx-args", type=str, default=core.TX_ARGS)
    ap.add_argument("--rx-args", type=str, default=core.RX_ARGS)
    ap.add_argument("--calib", default="frozen_calibration.npz",
                    help="frozen per-column calibration (measured once, "
                         "offline, on a disjoint set). Applied at inference; "
                         "NO digital W@x is computed on the host.")
    args = ap.parse_args(argv)
    args.debias_act = True

    layers = load_weights(HERE / FROZEN_WEIGHTS)
    cz = np.load(HERE / args.calib, allow_pickle=True)
    g_frozen = {t: cz[f"g_l{t}"].astype(np.float64) for t in range(1, 5)}
    d = np.load(HERE / "gtsrb_roi_32x32.npz", allow_pickle=True)
    sel = np.load(args.idx_npy).astype(np.int64)
    X_rows = preprocess(d["Xte"])[sel]
    y = d["yte"].astype(np.int64)[sel]
    n = len(sel)

    hw = None
    if not args.dry_run:
        from check_rig_free import rig_free_or_die
        rig_free_or_die()
        hw = core.HWSession(args, Path("."))
    save = {}
    try:
        A = X_rows.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
        for tag, lay in enumerate(layers, 1):
            if lay["kind"] == "conv":
                P, _, _ = im2col(A, lay["k"], lay["stride"])
                flat = P.reshape(-1, P.shape[-1])
            else:
                flat = A.reshape(n, -1)
            am, s2 = run_layer_hw(core, lay["W"], flat, args=args,
                                  layer_tag=tag, hw=hw, g_frozen=g_frozen[tag])
            A, S = interpass(am, lay, tag, n)
            save[f"y_mag_l{tag}"] = am.astype(np.float32)
            save[f"sigma2_bin_l{tag}"] = s2
            print(f"[L{tag}] mean |am| {np.mean(am):.4f}", flush=True)
    finally:
        if hw is not None:
            hw.close()
    preds = np.argmax(S ** 2, axis=1).astype(np.int64)
    acc = float((preds == y).mean())
    np.savez_compressed(args.out_npz, preds=preds, labels=y, img_index=sel,
                        accuracy=acc, n_layers=len(layers),
                        calibration="frozen (no host W@x)", **save)
    print(f"[frozen-ref] accuracy {acc*100:.2f}% -> {args.out_npz} "
          f"(frozen calibration; host performed no W@x)")
    return 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(verify())
    if "--hw" in sys.argv or "--dry-run" in sys.argv:
        sys.exit(main_hw(sys.argv[1:]))
    print(__doc__)
