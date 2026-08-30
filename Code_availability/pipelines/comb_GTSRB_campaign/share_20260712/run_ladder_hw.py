#!/usr/bin/env python3
"""Ladder rung R3 hardware runner: CNN inference via patch-streaming MVM.

Executes the constrained CNN (ladder_cnn.ARCH, weights from ladder_cnn npz)
on the MIWEN chain using the EXISTING primitives of 4_gtsrb_confusion_mlpN.py
(plan_layer / build_frame_ml / decode_ml / synth_capture_ml / HWSession):
each conv layer is one kernel-bank LO comb, held fixed, with im2col patches
streamed through as RF samples (design note: conv_on_miwen_20260731.pdf).

Ladder protocol hardwired: 1x averaging, quantizer OFF, per-layer rung-3
equalization against the digital reference (the baseline correction mode;
per-layer beta-shadow terms optional via --rung3-json once the session pilot
has refit them for the conv plans), digital per-channel biases.

Numpy-only at runtime. Dry-run mode needs no instruments:
    python3 run_ladder_hw.py --weights ladder_R3_seed0.npz --n-test 24 \
        --dry-run --sim-mixer limiter --snr-db 18 --out-npz r3_dry.npz
"""
from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = _data_dir(__file__)
sys.path.insert(0, str(SCRIPT_DIR))

_spec = importlib.util.spec_from_file_location(
    "gtsrb_hw_core", SCRIPT_DIR / "4_gtsrb_confusion_mlpN.py")
core = importlib.util.module_from_spec(_spec)
sys.modules["gtsrb_hw_core"] = core
_spec.loader.exec_module(core)

from rung3_correction import fit_gj  # noqa: E402


# ---------------------------------------------------------------------------
def load_cnn(path):
    z = np.load(path, allow_pickle=True)
    arch = json.loads(str(z["arch_json"]))
    layers = []
    for i, c in enumerate(arch["conv"]):
        W = (z[f"c{i}r"] + 1j * z[f"c{i}i"]).reshape(c["cout"], -1)
        scale = (z[f"c{i}s"].astype(np.float64) if f"c{i}s" in z.files
                 else np.ones(c["cout"]))          # BN affine (v2 exports)
        layers.append(dict(kind="conv", W=W.astype(np.complex128),
                           b=z[f"c{i}b"].astype(np.float64), k=c["k"],
                           cin=c["cin"], cout=c["cout"], scale=scale,
                           stride=int(c.get("stride", 1)),
                           pool=bool(c.get("pool", True))))
    j = 0
    while f"d{j}r" in z.files:
        layers.append(dict(kind="dense",
                           W=(z[f"d{j}r"] + 1j * z[f"d{j}i"]).astype(np.complex128),
                           b=z[f"d{j}b"].astype(np.float64)))
        j += 1
    return arch, layers


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


# ---------------------------------------------------------------------------
def run_layer_mvm(W, X, *, args, layer_tag, hw, betas_shadow=0.0, drive=None,
                  plan1=None, rf_dbm=None, return_complex=False):
    """One logical MVM (H,D)x(S,D) through the chain, chunked over samples.
    Returns equalized magnitudes (S,H), reference |Z| (S,H), raw magnitudes,
    plan tuple, per-chunk sigma2 mean."""
    H, D = W.shape
    plan = core.plan_layer(D, H, args.fft_len, args.k0, args.bw_frac,
                           args.k_per_symbol)
    cap_v = core.ring_cap_vectors(plan.R, args.fft_len, args.cp_len, args.gap)
    per = max(1, min(int(args.images_per_frame), cap_v))
    zc = core.zc_phase(D)
    Wt = W * np.conj(zc)[None, :]
    S = X.shape[0]
    Y = np.zeros((S, H), np.complex128)
    Yref = np.zeros((S, H), np.complex128)
    s2s = []
    dummy_y = np.zeros(S, np.int64)
    for c0 in range(0, S, per):
        idx = np.arange(c0, min(c0 + per, S))
        x_t = (X[idx].astype(np.complex128) * zc[None, :])
        fr = core.build_frame_ml(Wt, x_t, dummy_y[idx], idx, plan,
                                 L=args.fft_len, cp=args.cp_len, gap=args.gap,
                                 sync_seed=args.seed, layer_tag=layer_tag,
                                 verbose=(c0 == 0 and args.verbose))
        if args.dry_run:
            snr = args.snr_db if args.snr_db is not None else float("inf")
            cap = core.synth_capture_ml(fr, float(snr), args.sim_mixer,
                                        args.seed, args.fs,
                                        salt=layer_tag * 10_000 + c0,
                                        verbose=False)
        else:
            hw.ensure(fr, verbose_build=(c0 == 0))
            if rf_dbm is not None:
                # per-layer RF drive override (L2-comb power sweep); ensure()
                # re-applies args powers on rebuild, so set after it each chunk
                hw.tb.set_powers(rf_dbm, args.p_lo_dbm)
            cap = hw.capture(fr)
        res = core.decode_ml(cap, fr, args.fs, flip_output=args.flip_output,
                             conj_output=args.conj_output, verbose=False)
        if not res.ok:
            raise RuntimeError(f"decode failed L{layer_tag} chunk@{c0}: {res.reason}")
        Y[idx] = res.y_meas
        Yref[idx] = fr.y_digital
        s2s.append(res.sigma2_bin)
    sigma2 = float(np.mean(s2s))
    if return_complex:
        return Y, Yref, (plan.D, plan.H, plan.K, plan.M, plan.R), sigma2
    am_raw = np.abs(Y)
    if args.debias_act:
        am_raw = np.sqrt(np.maximum(am_raw ** 2 - sigma2, 0.0))
    ar = np.abs(Yref)
    am_eq = am_raw / fit_gj(am_raw, ar)[None, :]          # rung-3 equalization
    if betas_shadow:                                       # optional shadow term
        from debug_shadow_presence import ofdm_shadow
        pr = (plan.D, plan.H, plan.K, plan.M, plan.R)
        dW, kb = ofdm_shadow(W, pr, drive_amp=drive)
        mhat = np.abs(X.astype(np.complex128) @ (W + dW / kb).T)
        dlt = mhat / fit_gj(mhat, ar)[None, :] - ar
        am_eq = np.maximum(am_eq - betas_shadow * dlt, 0.0)
    return am_eq, ar, am_raw, (plan.D, plan.H, plan.K, plan.M, plan.R), sigma2


def run_layer_mvm_pipelined(W, X, *, args, layer_tag, hw, rf_dbm=None):
    """Pipelined variant of run_layer_mvm: host compute (frame build, decode)
    overlaps the blocking device cycle; the RF timeline is byte-identical to
    the serial path (ensure/capture strictly sequential on the main thread,
    frames immutable after build — enforced by checksum). No shadow terms
    (frozen algorithm is eq-only). Queues are depth-bounded for memory."""
    import hashlib
    import queue
    import threading

    H, D = W.shape
    plan = core.plan_layer(D, H, args.fft_len, args.k0, args.bw_frac,
                           args.k_per_symbol)
    cap_v = core.ring_cap_vectors(plan.R, args.fft_len, args.cp_len, args.gap)
    per = max(1, min(int(args.images_per_frame), cap_v))
    zc = core.zc_phase(D)
    Wt = W * np.conj(zc)[None, :]
    S = X.shape[0]
    starts = list(range(0, S, per))
    Y = np.zeros((S, H), np.complex128)
    Yref = np.zeros((S, H), np.complex128)
    s2s = [None] * len(starts)
    dummy_y = np.zeros(S, np.int64)

    built = queue.Queue(maxsize=2)      # (i, c0, fr, sha)
    todec = queue.Queue(maxsize=3)      # (i, c0, fr, cap)
    err = []

    def builder():
        try:
            for i, c0 in enumerate(starts):
                idx = np.arange(c0, min(c0 + per, S))
                x_t = (X[idx].astype(np.complex128) * zc[None, :])
                fr = core.build_frame_ml(Wt, x_t, dummy_y[idx], idx, plan,
                                         L=args.fft_len, cp=args.cp_len,
                                         gap=args.gap, sync_seed=args.seed,
                                         layer_tag=layer_tag,
                                         verbose=(i == 0 and args.verbose))
                sha = hashlib.sha1(fr.sA.tobytes() + fr.sB.tobytes()).digest()
                built.put((i, c0, fr, sha))
            built.put(None)
        except Exception as e:                          # pragma: no cover
            err.append(e)
            built.put(None)

    def decoder():
        try:
            while True:
                item = todec.get()
                if item is None:
                    return
                i, c0, fr, cap = item
                res = core.decode_ml(cap, fr, args.fs,
                                     flip_output=args.flip_output,
                                     conj_output=args.conj_output,
                                     verbose=False)
                if not res.ok:
                    raise RuntimeError(
                        f"decode failed L{layer_tag} chunk@{c0}: {res.reason}")
                idx = np.arange(c0, min(c0 + per, S))
                Y[idx] = res.y_meas
                Yref[idx] = fr.y_digital
                s2s[i] = res.sigma2_bin
        except Exception as e:
            err.append(e)

    tb = threading.Thread(target=builder, daemon=True)
    td = threading.Thread(target=decoder, daemon=True)
    tb.start(); td.start()
    import hashlib as _h
    while True:
        item = built.get()
        if item is None:
            break
        i, c0, fr, sha = item
        assert _h.sha1(fr.sA.tobytes() + fr.sB.tobytes()).digest() == sha, \
            "frame mutated between build and TX"       # airtight guard
        if args.dry_run:
            snr = args.snr_db if args.snr_db is not None else float("inf")
            cap = core.synth_capture_ml(fr, float(snr), args.sim_mixer,
                                        args.seed, args.fs,
                                        salt=layer_tag * 10_000 + c0,
                                        verbose=False)
        else:
            hw.ensure(fr, verbose_build=(c0 == 0))
            if rf_dbm is not None:
                hw.tb.set_powers(rf_dbm, args.p_lo_dbm)
            cap = hw.capture(fr)
        todec.put((i, c0, fr, cap))
        if err:
            break
    todec.put(None)
    tb.join(timeout=60); td.join(timeout=600)
    if err:
        raise err[0]
    sigma2 = float(np.mean([s for s in s2s if s is not None]))
    am_raw = np.abs(Y)
    if args.debias_act:
        am_raw = np.sqrt(np.maximum(am_raw ** 2 - sigma2, 0.0))
    ar = np.abs(Yref)
    am_eq = am_raw / fit_gj(am_raw, ar)[None, :]
    return am_eq, ar, am_raw, (plan.D, plan.H, plan.K, plan.M, plan.R), sigma2


def run_layer_mvm_split(W, X, n_groups, *, args, layer_tag, hw, rf_dbm=None):
    """Tone-split MVM: partition the input dimension into n_groups contiguous
    groups (channel-aligned under the im2col layout), transmit each as its own
    smaller-comb frame, complex-equalize per column per group against that
    group's digital reference, and sum the partial products digitally BEFORE
    the magnitude. The computed convolution is mathematically identical to the
    unsplit MVM; RF comb density is divided by n_groups. Requires cross-frame
    phase stability (ext-ref locked chain; complex sum, not magnitude sum)."""
    H, D = W.shape
    bounds = np.linspace(0, D, n_groups + 1).astype(int)
    Ysum = np.zeros((X.shape[0], H), np.complex128)
    Rsum = np.zeros_like(Ysum)
    s2_tot = 0.0
    plan_row = None
    for gi in range(n_groups):
        sl = slice(bounds[gi], bounds[gi + 1])
        Y, Yref, plan_row, s2 = run_layer_mvm(
            W[:, sl], X[:, sl], args=args, layer_tag=layer_tag * 100 + gi,
            hw=hw, rf_dbm=rf_dbm, return_complex=True)
        a = (np.sum(np.conj(Y) * Yref, 0)
             / np.maximum(np.sum(np.abs(Y) ** 2, 0), 1e-30))
        Ysum += Y * a[None, :]
        Rsum += Yref
        s2_tot += s2 * float(np.mean(np.abs(a) ** 2))
    am_raw = np.abs(Ysum)
    if args.debias_act:
        am_raw = np.sqrt(np.maximum(am_raw ** 2 - s2_tot, 0.0))
    ar = np.abs(Rsum)
    am_eq = am_raw / fit_gj(am_raw, ar)[None, :]
    return am_eq, ar, am_raw, plan_row, s2_tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--idx-npy", default=None,
                    help="npy file of test-set indices to run (overrides "
                         "--offset/--n-test; enables random-subset frames)")
    ap.add_argument("--rung3-json", default=None,
                    help="optional conv-plan betas: {layer_tag: beta}; "
                         "equalization always on")
    ap.add_argument("--rf-dbm-json", default=None,
                    help="per-layer RF drive override, JSON {layer_tag: dBm} "
                         "(L2-comb power sweep; other layers use --p-rf-dbm)")
    ap.add_argument("--pipeline", action="store_true",
                    help="overlap host compute (frame build, decode) with the "
                         "device cycle; RF timeline unchanged (see "
                         "run_layer_mvm_pipelined). Requires eq-only path.")
    ap.add_argument("--split-json", default=None,
                    help="per-layer tone-split, JSON {layer_tag: n_groups}: "
                         "comb density / n_groups, complex partial sums; "
                         "math identical (see run_layer_mvm_split)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sim-mixer", choices=("ideal", "limiter"), default="ideal")
    ap.add_argument("--snr-db", type=float, default=None)
    ap.add_argument("--debias-act", action="store_true", default=True)
    ap.add_argument("--fft-len", type=int, default=16384)
    ap.add_argument("--cp-len", type=int, default=512)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--k0", type=int, default=6)
    ap.add_argument("--bw-frac", type=float, default=0.5)
    ap.add_argument("--k-per-symbol", type=int, default=64)
    ap.add_argument("--images-per-frame", type=int, default=36)
    ap.add_argument("--fs", type=float, default=10e6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--flip-output", action="store_true")
    ap.add_argument("--conj-output", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    # --- HWSession namespace contract (issue #50: dry-run never constructs
    # HWSession, so these were missing; defaults mirror the main script) -----
    ap.add_argument("--p-lo-dbm", type=float, default=core.P_LO_DBM_DEFAULT)
    ap.add_argument("--p-rf-dbm", type=float, default=core.P_RF_DBM_DEFAULT)
    ap.add_argument("--rx-gain", type=float, default=core.RX_GAIN_DB_DEFAULT)
    ap.add_argument("--p-max-dbm", type=float, default=core.P_MAX_DBM_DEFAULT)
    ap.add_argument("--settle-s", type=float, default=core.SETTLE_S_DEFAULT)
    ap.add_argument("--capture-retries", type=int,
                    default=core.CAPTURE_RETRIES_DEFAULT)
    ap.add_argument("--start-lead", type=float,
                    default=core.START_LEAD_S_DEFAULT)
    ap.add_argument("--no-ext-ref", action="store_true")
    ap.add_argument("--tx-args", type=str, default=core.TX_ARGS)
    ap.add_argument("--rx-args", type=str, default=core.RX_ARGS)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args(argv)

    arch, layers = load_cnn(args.weights)
    Xte, yte = core.load_gtsrb_test(verbose=False)
    if args.idx_npy:
        sel = np.load(args.idx_npy).astype(np.int64)
        print(f"[ladder-hw] index-file frame: {len(sel)} images "
              f"[{sel.min()}..{sel.max()}] from {args.idx_npy}")
    else:
        sel = np.arange(args.offset, args.offset + args.n_test)
    X = Xte[sel]
    y = yte[sel]
    betas = {}
    if args.rung3_json:
        betas = {int(k): float(v) for k, v in
                 json.loads(Path(args.rung3_json).read_text()).get(
                     "conv_betas", {}).items()}
    rf_map = {}
    if args.rf_dbm_json:
        rf_map = {int(k): float(v) for k, v in
                  json.loads(args.rf_dbm_json).items()}
    split_map = {}
    if args.split_json:
        split_map = {int(k): int(v) for k, v in
                     json.loads(args.split_json).items()}
        bad = set(split_map) & set(betas)
        assert not bad, f"split+shadow unsupported on layers {bad}"
    drive = None
    if betas:
        # latent-bug fix (session note 2026-08-01): shadow terms need a drive
        # anchor; calibrate once on the first layer's kernel bank, as the
        # conv-beta refit tool does.
        from debug_shadow_presence import calibrate_drive
        W1 = layers[0]["W"]
        p1 = core.plan_layer(W1.shape[1], W1.shape[0], args.fft_len, args.k0,
                             args.bw_frac, args.k_per_symbol)
        drive = calibrate_drive(W1, (p1.D, p1.H, p1.K, p1.M, p1.R), "ofdm")
        print(f"[rung3] conv shadow drive anchor: {drive:.4f}")

    hw = None
    if not args.dry_run:
        from check_rig_free import rig_free_or_die
        rig_free_or_die()          # standing rule: never transmit on a busy rig
        hw = core.HWSession(args, Path("."))
    t0 = time.time()
    save = {}
    A = X.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)   # (N,C,H,W)
    n = A.shape[0]
    try:
        tag = 0
        for lay in layers:
            tag += 1
            if lay["kind"] == "conv":
                P, Ho, Wo = im2col(A, lay["k"], lay["stride"])
                flat = P.reshape(-1, P.shape[-1])
                if split_map.get(tag, 1) > 1:
                    am, ar, raw, plan_row, s2 = run_layer_mvm_split(
                        lay["W"], flat, split_map[tag], args=args,
                        layer_tag=tag, hw=hw, rf_dbm=rf_map.get(tag))
                elif args.pipeline and not betas.get(tag):
                    am, ar, raw, plan_row, s2 = run_layer_mvm_pipelined(
                        lay["W"], flat, args=args, layer_tag=tag, hw=hw,
                        rf_dbm=rf_map.get(tag))
                else:
                    am, ar, raw, plan_row, s2 = run_layer_mvm(
                        lay["W"], flat, args=args, layer_tag=tag, hw=hw,
                        betas_shadow=betas.get(tag, 0.0), drive=drive,
                        rf_dbm=rf_map.get(tag))
                S = np.maximum(am * lay["scale"][None, :] + lay["b"][None, :],
                               0.0)
                fm = S.reshape(n, Ho, Wo, lay["cout"]).transpose(0, 3, 1, 2)
                A = maxnorm(maxpool2(fm) if lay["pool"] else fm)
                print(f"[L{tag} conv] {lay['cout']}x{lay['W'].shape[1]} "
                      f"{Ho}x{Wo} pos; rel-RMSE "
                      f"{np.sqrt(np.mean((am-ar)**2)/np.mean(ar**2)):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            else:
                flat = A.reshape(n, -1)
                if args.pipeline and not betas.get(tag):
                    am, ar, raw, plan_row, s2 = run_layer_mvm_pipelined(
                        lay["W"], flat, args=args, layer_tag=tag, hw=hw,
                        rf_dbm=rf_map.get(tag))
                else:
                    am, ar, raw, plan_row, s2 = run_layer_mvm(
                        lay["W"], flat, args=args, layer_tag=tag, hw=hw,
                        betas_shadow=betas.get(tag, 0.0), drive=drive,
                        rf_dbm=rf_map.get(tag))
                S = np.maximum(am + lay["b"][None, :], 0.0)
                A = maxnorm(S)[:, None, None, :].transpose(0, 3, 1, 2)
                print(f"[L{tag} dense] {lay['W'].shape[0]}x{lay['W'].shape[1]}; "
                      f"rel-RMSE "
                      f"{np.sqrt(np.mean((am-ar)**2)/np.mean(ar**2)):.4f}",
                      flush=True)
            save[f"y_mag_l{tag}"] = am.astype(np.float32)
            save[f"y_mag_raw_l{tag}"] = raw.astype(np.float32)
            save[f"y_ref_l{tag}"] = ar.astype(np.float32)
            save[f"plan_l{tag}"] = np.asarray(plan_row, np.int64)
            save[f"sigma2_bin_l{tag}"] = s2
            last_S = S
    finally:
        if hw is not None:
            hw.close()

    preds = np.argmax(last_S ** 2, axis=1).astype(np.int64)
    acc = float((preds == y).mean())
    dig = None
    save.update(accuracy=acc, preds=preds, labels=y, img_index=sel,
                keep_mask=np.ones(n, bool), n_layers=tag,
                meta_json=json.dumps({k: (v if isinstance(v, (int, float, str,
                                                              bool, type(None)))
                                          else str(v))
                                      for k, v in vars(args).items()},
                                     ensure_ascii=False))
    np.savez_compressed(args.out_npz, **save)
    print(f"[ladder-hw] accuracy {acc*100:.2f}%  -> {args.out_npz} "
          f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
