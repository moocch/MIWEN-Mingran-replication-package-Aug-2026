#!/usr/bin/env python3
"""SERIAL (time-domain) NN runner v2 — |Y| row-integrated readout.

Usage: python3 serial_nn_runner.py {clean|twin} <img_start> <img_end>

v2 fixes over v1 (v1 decoded correctly but ~100x too slow):
 - FFT-based sync search on the first capture, +-4k windowed direct
   search after (frame position is stable within a session).
 - Big RX ring (2^26) via explicit MVMFlowgraph construction ->
   880k-slot frames, ~9 captures/image.
 - Row-aligned chunking (chunk = floor(MAX/D)*D): no row ever spans
   two captures, so the magnitude readout never needs cross-capture
   phase.
 - PACKTEST gate before calibration: 150k random signed-Gaussian
   slots, big frame, fitted gain/residual must match the small-frame
   2026-08-23 station reference (g 0.0837 +-35%, resid 0.30..0.75).
 - Calibration (scales+gains) saved per arm and RELOADED by later
   sessions (split-half consistency).
Encoding/readout physics identical to v1 (see v1 docstring / prespec).
"""
from __future__ import annotations

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------


import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = _data_dir(__file__)
sys.path.insert(0, str(HERE))
import sync_v2

ARM_W = {"clean": "r35_r3plus_s0_hw.npz",
         "twin": "serial_twin_s0_hw.npz"}
POWER = (0.0, 0.0)
TSLOT, GUARD = 32, 8
F_A, F_B = 2.2e6, -1.8e6
F_DEC = -(F_A - F_B)
L, CP = 16384, 512
SYM = L + CP
GAPS = 256
RING_BIG = 1 << 26
MAX_SLOTS = 880_000        # frame 28.2M samp; n_cap 2.3x = 64.9M < 2^26
N_CAL = 10
# Big-frame reference (2026-08-23 packtest): sync duty is 0.35% here vs
# 34% in the small-frame diagnostic, so the payload runs ~3 dB hotter
# per port at the same commanded power -> gain re-baselined. Residual
# band unchanged (shape = the decode-corruption detector).
PACK_G_REF, PACK_G_TOL = 0.1155, 0.20
PACK_RESID_BAND = (0.30, 0.75)


def main() -> int:
    arm = sys.argv[1]
    i0, i1 = int(sys.argv[2]), int(sys.argv[3])

    spec = importlib.util.spec_from_file_location(
        "gtsrb_hw_core", HERE / "4_gtsrb_confusion_mlpN.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules["gtsrb_hw_core"] = core
    spec.loader.exec_module(core)
    rspec = importlib.util.spec_from_file_location(
        "ref1", HERE / "miwen_frozen_reference.py")
    ref = importlib.util.module_from_spec(rspec)
    rspec.loader.exec_module(ref)

    class A:
        pass
    args = A()
    for a_, b_ in (("p_max_dbm", "P_MAX_DBM_DEFAULT"),
                   ("settle_s", "SETTLE_S_DEFAULT"),
                   ("start_lead", "START_LEAD_S_DEFAULT"),
                   ("capture_retries", "CAPTURE_RETRIES_DEFAULT")):
        setattr(args, a_, getattr(core, b_))
    args.tx_args, args.rx_args = core.TX_ARGS, core.RX_ARGS
    args.no_ext_ref = False
    args.fs = 10e6
    args.fft_len, args.cp_len, args.gap = L, CP, 0
    args.rx_gain = 0.0
    args.p_rf_dbm, args.p_lo_dbm = POWER

    from check_rig_free import rig_free_or_die
    rig_free_or_die()

    fs = args.fs
    ta, tb = sync_v2._chirps(core, L)
    sync_ref = (np.conj(ta) * tb).astype(np.complex64)
    layers = ref.load_weights(HERE / ARM_W[arm])

    d = np.load(HERE / "gtsrb_roi_32x32.npz", allow_pickle=True)
    Xte_pp = ref.preprocess(d["Xte"])
    yte = d["yte"].astype(np.int64)
    cal_idx = np.load(HERE / "s4_random450_idx.npy")[:N_CAL]
    bat_idx = np.load(HERE / "battery_random1200_idx.npy")

    dummy_plan = core.plan_layer(64, 64, L, 6, 0.5, 64)

    # ---- arm-consistent digital forward for calibration -----------------
    # Each arm's teacher-forced states and gain reference come from ITS
    # OWN training forward: clean = ideal multiplier; twin = the frozen
    # twin surface (the twin net's BN expects twin-channel statistics;
    # ideal-forward states drive it to all-zeros).
    if arm == "twin":
        _tm = json.load(open(HERE / "serial_twin_model.json"))
        _thp = np.array(_tm["thp"]); _rg = np.array(_tm["ridges"]).reshape(-1, 4)

        def _pdb(pl, pr):
            a0, klo, krf = _thp[0], _thp[1], _thp[2]
            xl = 10 ** ((pl - klo) / 10.0); xr = 10 ** ((pr - krf) / 10.0)
            y = a0 + 10 * np.log10((xl / (1 + xl)) * (xr / (1 + xr)) + 1e-30)
            for wi_, ai_, bi_, ci_ in _rg:
                y = y + wi_ * np.tanh(ai_ * pl / 40 + bi_ * pr / 40 + ci_)
            return y

        def arm_matmul(X_in, W):
            xs = X_in / max(np.sqrt((X_in ** 2).mean()), 1e-12)
            wa = np.abs(W); ws = wa / max(np.sqrt((wa ** 2).mean()), 1e-12)
            pl = 20 * np.log10(np.maximum(ws, 1e-3))
            out = np.zeros((X_in.shape[0], W.shape[0]), np.complex128)
            for h0 in range(0, W.shape[0], 8):
                pr = 20 * np.log10(np.maximum(xs, 1e-3))[:, None, :]
                amp = 10 ** (_pdb(pl[None, h0:h0 + 8, :], pr) / 20.0)
                ph = (W[h0:h0 + 8] / np.maximum(wa[h0:h0 + 8], 1e-12))[None]
                out[:, h0:h0 + 8] = (amp * ph).sum(-1)
            return np.abs(out)
    else:
        def arm_matmul(X_in, W):
            return np.abs(X_in @ W.T)

    class BigRingSession(core.HWSession):
        def capture(self, fr, wide=False):
            # d0 (frame start in capture) is set by grab-flush timing,
            # observed up to ~4.7M samples -> additive margin, not
            # multiplicative; 'wide' = conservative 2.3x fallback.
            a = self.args
            n_cap = int(round(2.3 * fr.frame_len)) if wide \
                else int(fr.frame_len + 8_000_000)
            n_cap = min(n_cap, RING_BIG - 2_000_000)
            gto = max(30.0, 4.0 * n_cap / a.fs)
            return core._capture_clean(self.tb, n_cap, a.settle_s,
                                       a.capture_retries, gto)

        def ensure(self, fr, verbose_build=False):
            a = self.args
            clk = "internal" if a.no_ext_ref \
                else core.RX_CLOCK_SOURCE_DEFAULT
            if self.tb is None:
                fr.sA.astype(np.complex64).tofile(self.lo_path)
                fr.sB.astype(np.complex64).tofile(self.rf_path)
                self.tb = core.MVMFlowgraph(
                    fr, str(self.lo_path), str(self.rf_path),
                    tx_args=a.tx_args, rx_args=a.rx_args,
                    sample_rate=a.fs, rx_gain_db=float(a.rx_gain),
                    rx_clock_source=clk,
                    p_max_dbm=float(a.p_max_dbm),
                    ring_capacity=RING_BIG,
                    verbose=verbose_build)
                info = self.tb.set_powers(a.p_rf_dbm, a.p_lo_dbm)
                self.tb.start_timed(a.start_lead)
                time.sleep(2.0)
                self.tb.check_locks()
            else:
                self.tb.restart_with_files(
                    str(self.lo_path), str(self.rf_path),
                    sA=fr.sA, sB=fr.sB, frame=fr,
                    start_lead=a.start_lead)
                info = self.tb.set_powers(a.p_rf_dbm, a.p_lo_dbm)
                time.sleep(2.0)
            return info

    hw = BigRingSession(args, Path("."))
    t0 = time.time()
    state = {"checked": False, "d0": None, "ncap": 0}
    scaleAB = [1.0, 1.0]
    _tplc = {}

    def tpls(pay_len):
        if pay_len not in _tplc:
            t = np.arange(pay_len) / fs
            _tplc.clear()
            _tplc[pay_len] = (
                np.exp(2j * np.pi * F_A * t).astype(np.complex128),
                np.exp(2j * np.pi * F_B * t).astype(np.complex128),
                np.exp(-2j * np.pi * F_DEC * t).astype(np.complex64))
        return _tplc[pay_len]

    def find_sync(cap, frame_len, pay_len):
        lim = len(cap) - SYM - pay_len
        if state["d0"] is not None and state["d0"] <= lim:
            w0 = max(0, state["d0"] - 65536)
            w1 = min(lim + L, state["d0"] + 65536 + L)
            seg = cap[w0:w1].astype(np.complex64)
            nf = 1 << int(np.ceil(np.log2(len(seg) + L)))
            Fs_ = np.fft.fft(seg, nf)
            Gs_ = np.fft.fft(sync_ref, nf)
            c = np.abs(np.fft.ifft(Fs_ * np.conj(Gs_)))[:len(seg) - L]
            k = int(np.argmax(c))
            pm = float(c[k] / (np.median(c) + 1e-30))
            if pm >= 8.0 and w0 + k <= lim:
                return w0 + k, pm
        n = min(len(cap), int(1.5 * frame_len) + SYM)
        nf = 1 << int(np.ceil(np.log2(n + L)))
        F = np.fft.fft(cap[:n].astype(np.complex64), nf)
        G = np.fft.fft(sync_ref, nf)
        c = np.abs(np.fft.ifft(F * np.conj(G)))[:n - L]
        med = np.median(c) + 1e-30
        for k in np.argsort(c)[::-1][:8]:
            if int(k) <= lim:
                return int(k), float(c[int(k)] / med)
        raise RuntimeError("sync: no complete frame")

    def tx_slots(xv, wv, chunk):
        n = len(xv)
        out = np.zeros(n, np.complex128)
        for c0 in range(0, n, chunk):
            xs = xv[c0:c0 + chunk]
            ws = wv[c0:c0 + chunk]
            ns = len(xs)
            pay_len = ns * TSLOT
            carA, carB, _ = tpls(pay_len)
            pA = np.repeat(xs.astype(np.complex128), TSLOT) * carA
            pB = np.repeat(ws.astype(np.complex128), TSLOT) * carB
            frame_len = GAPS + SYM + pay_len + GAPS
            sA = np.zeros(frame_len, np.complex128)
            sB = np.zeros(frame_len, np.complex128)
            for s_, ch, pay in ((sA, ta, pA), (sB, tb, pB)):
                pr = max(np.sqrt(np.mean(np.abs(pay) ** 2)), 1e-30)
                s_[GAPS:GAPS + L] = ch / np.sqrt(
                    np.mean(np.abs(ch) ** 2)) * 2.0 * pr
                s_[GAPS + SYM:GAPS + SYM + pay_len] = pay
            sA *= scaleAB[0]
            sB *= scaleAB[1]
            fr = core.FrameSpecML(
                L=L, cp=CP, plan=dummy_plan, gap=GAPS, layer_tag=1,
                n_vec=ns, n_sym=1, labels=np.zeros(1, np.int64),
                vec_index=np.arange(1),
                y_digital=np.zeros(1, np.complex128), sA=sA, sB=sB,
                ref_pre=sync_ref,
                peak_a=float(np.max(np.abs(sA))),
                peak_b=float(np.max(np.abs(sB))),
                frame_len=frame_len, sync_payload_off=GAPS + SYM)
            hw.ensure(fr, verbose_build=False)
            cap = hw.capture(fr)
            try:
                d0, pm = find_sync(cap, frame_len, pay_len)
            except (RuntimeError, ValueError):
                state["d0"] = None
                cap = hw.capture(fr, wide=True)
                d0, pm = find_sync(cap, frame_len, pay_len)
            if pm < 8.0:
                raise RuntimeError(f"SYNC GATE: pm={pm:.1f}")
            state["d0"] = d0
            pay = cap[d0 + SYM:d0 + SYM + pay_len].astype(np.complex128)
            if not state["checked"]:
                pk = float(np.max(np.abs(pay)))
                print(f"[gate] first capture: peak {pk:.3f} FS "
                      f"(ref 0.335) pm={pm:.0f} d0={d0}", flush=True)
                if not (0.1 < pk < 0.7):
                    raise RuntimeError(f"CHAIN GATE: peak {pk:.3f}")
                state["checked"] = True
            tpl = tpls(pay_len)[2]
            sl = (pay * tpl).reshape(ns, TSLOT)[:, GUARD:TSLOT - GUARD]
            out[c0:c0 + ns] = sl.mean(axis=1)
            state["ncap"] += 1
            print(f"[cap {state['ncap']}] d0={d0} pm={pm:.0f} "
                  f"slots={ns} ({time.time()-t0:.0f}s)", flush=True)
        return out

    def run_layer(lay, X_in, gain):
        W = lay["W"]
        n, D = X_in.shape
        H = W.shape[0]
        chunk = max(D, (MAX_SLOTS // D) * D)      # row-aligned
        xv = np.repeat(X_in, H, axis=0).reshape(-1)
        wv = np.tile(W, (n, 1)).reshape(-1)
        prods = tx_slots(xv, wv, chunk).reshape(n * H, D)
        return np.abs(prods.sum(axis=1).reshape(n, H)) / gain

    results = {}
    cal_path = HERE / f"serial_cal_{arm}_20260823.npz"
    try:
        # ---------------- PACKTEST gate --------------------------------
        rngp = np.random.default_rng(11)
        npk = 150_000
        xp = rngp.standard_normal(npk)
        wp = rngp.standard_normal(npk)
        scaleAB[:] = [1.0 / np.sqrt(np.mean(xp ** 2)),
                      1.0 / np.sqrt(np.mean(wp ** 2))]
        yp = tx_slots(xp, wp, npk)
        pw = xp * wp
        g = np.vdot(pw, yp) / np.vdot(pw, pw)
        rel = np.sqrt(np.mean(np.abs(yp - g * pw) ** 2)
                      / np.mean(np.abs(g * pw) ** 2))
        print(f"[packtest] |g|={abs(g):.4f} (ref {PACK_G_REF}) "
              f"resid={rel:.3f} (band {PACK_RESID_BAND}) "
              f"({time.time()-t0:.0f}s)", flush=True)
        if abs(abs(g) - PACK_G_REF) / PACK_G_REF > PACK_G_TOL \
                or not (PACK_RESID_BAND[0] < rel < PACK_RESID_BAND[1]):
            raise RuntimeError("PACKTEST GATE FAIL")

        # ---------------- calibration (or reload) ----------------------
        scales, gains = {}, {}
        if cal_path.exists():
            z = np.load(cal_path)
            for tag in (1, 2, 3, 4):
                gains[tag] = float(z[f"g_l{tag}"])
                scales[tag] = tuple(z[f"s_l{tag}"])
            print(f"[cal {arm}] reloaded frozen calibration", flush=True)
        else:
            Xc = Xte_pp[cal_idx]
            ins = []
            A_ = Xc.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
            for tag, lay in enumerate(layers, 1):
                if lay["kind"] == "conv":
                    P, _, _ = ref.im2col(A_, lay["k"], lay["stride"])
                    ins.append(P.reshape(-1, P.shape[-1]))
                else:
                    ins.append(A_.reshape(len(Xc), -1))
                am = arm_matmul(ins[-1], lay["W"])
                A_, _ = ref.interpass(am, lay, tag, len(Xc))
            for tag, lay in enumerate(layers, 1):
                X_in = ins[tag - 1]
                xv = X_in.reshape(-1)
                wv = np.abs(lay["W"]).reshape(-1)
                scaleAB[:] = [1.0 / max(np.sqrt(np.mean(xv ** 2)), 1e-30),
                              1.0 / max(np.sqrt(np.mean(wv ** 2)), 1e-30)]
                scales[tag] = tuple(scaleAB)
                yid = arm_matmul(X_in, lay["W"])
                ym = run_layer(lay, X_in, 1.0)
                g = float(np.sum(ym * yid) / max(np.sum(yid ** 2), 1e-30))
                gains[tag] = g
                rel = np.sqrt(np.mean((ym / g - yid) ** 2)
                              / max(np.mean(yid ** 2), 1e-30))
                print(f"[cal {arm}] L{tag}: n={X_in.shape[0]} g={g:.4f} "
                      f"relRMSE(|Y|)={rel:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
                np.savez_compressed(cal_path.with_suffix(f".L{tag}.npz"),
                                    g=g, s=np.array(scales[tag]))
            np.savez_compressed(cal_path,
                                **{f"g_l{t}": gains[t] for t in gains},
                                **{f"s_l{t}": np.array(scales[t])
                                   for t in scales})

        # ---------------- evaluation -----------------------------------
        sel = bat_idx[i0:i1]
        CH = 10
        preds = np.zeros(len(sel), np.int64)
        out_path = HERE / f"serial_nn_{arm}_{i0}_{i1}_20260823.npz"
        for b0 in range(0, len(sel), CH):
            sub = sel[b0:b0 + CH]
            n = len(sub)
            A_ = Xte_pp[sub].reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
            S = None
            for tag, lay in enumerate(layers, 1):
                scaleAB[:] = scales[tag]
                if lay["kind"] == "conv":
                    P, _, _ = ref.im2col(A_, lay["k"], lay["stride"])
                    X_in = P.reshape(-1, P.shape[-1])
                else:
                    X_in = A_.reshape(n, -1)
                am = run_layer(lay, X_in, gains[tag])
                A_, S = ref.interpass(am, lay, tag, n)
            pd = np.argmax(S ** 2, axis=1)
            preds[b0:b0 + n] = pd
            acc = float((preds[:b0 + n] == yte[sel[:b0 + n]]).mean() * 100)
            results[f"chunk{b0 // CH}_preds"] = pd
            np.savez_compressed(out_path, **results, sel=sel,
                                meta_json=json.dumps(dict(
                                    arm=arm, weights=ARM_W[arm],
                                    power=list(POWER), tslot=TSLOT,
                                    img_range=[i0, i1],
                                    chain="IF +10dB pads",
                                    readout="|rowsum| no debias no noise")))
            print(f"[run {arm}] {b0 + n}/{len(sel)}: running acc "
                  f"{acc:.2f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[run {arm}] [{i0}:{i1}] FINAL: "
              f"{float((preds == yte[sel]).mean() * 100):.2f}", flush=True)
    finally:
        hw.close()
        print(f"[{arm}] saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
