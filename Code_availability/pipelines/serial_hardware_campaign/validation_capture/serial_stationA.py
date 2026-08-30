#!/usr/bin/env python3
"""Stage A: TIME-SERIAL per-product transfer measurement (2026-08-23).

The mirror of the comb stations: encode products one-per-time-slot so
each (x_i, w_i) pair traverses the mixer at its own amplitude — the
regime of Kfir's simulations, where the Gaussian-crowd premise of
Bussgang is absent by construction. Deliverable: measured y_hat vs
x*w scatter per drive. Expectation: linear at cold drive, the device
curve VISIBLE per product at hot drive (same day / same mixer as the
comb stations where products stay linear to -8.6 dB of compression).

Encoding: RF slot envelope |x_i| * phase(sign x_i) on subcarrier F_A;
LO likewise on F_B; every product lands on the single difference tone
F_P = F_A - F_B. Chirp sync (v2 chirps) for timing + phase. Chain:
10 dB IF pads (as installed). Drive ladder matches the comb stations.
First capture at (-3,-3) doubles as a chain sanity check (known
padded levels)."""
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

POINTS = ((-9.0, -9.0), (-3.0, -3.0), (0.0, 0.0), (3.0, 3.0),
          (7.0, 7.0))
DRAWS, REPS = 2, 2
NSLOT = 256            # products per frame
TSLOT = 128            # samples per slot (12.8 us)
GUARD = 8              # samples ignored at each slot edge
F_A, F_B = 1.2e6, 0.2e6
F_P = F_A - F_B        # product tone in the capture: 1.0 MHz
L, CP = 16384, 512
SYM = L + CP


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "gtsrb_hw_core", HERE / "4_gtsrb_confusion_mlpN.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules["gtsrb_hw_core"] = core
    spec.loader.exec_module(core)

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
    args.p_rf_dbm, args.p_lo_dbm = POINTS[0]

    from check_rig_free import rig_free_or_die
    rig_free_or_die()

    fs = args.fs
    ta, tb = sync_v2._chirps(core, L)
    sync_ref = np.conj(ta) * tb          # validated convention

    rng = np.random.default_rng(2026)
    gap = 256
    pay_len = NSLOT * TSLOT
    frame_len = gap + SYM + pay_len + gap
    t_pay = np.arange(pay_len) / fs
    plan = core.plan_layer(64, 64, L, 6, 0.5, 64)   # dummy for dataclass

    def build_serial(x, w):
        """x, w real vectors (NSLOT,) -> FrameSpecML with serial payload."""
        envA = np.repeat(x.astype(np.complex128), TSLOT)
        envB = np.repeat(w.astype(np.complex128), TSLOT)
        pA = envA * np.exp(2j * np.pi * F_A * t_pay)
        pB = envB * np.exp(2j * np.pi * F_B * t_pay)
        sA = np.zeros(frame_len, np.complex128)
        sB = np.zeros(frame_len, np.complex128)
        # sync at 2x payload rms per channel (chirps are CM)
        for s_, chirp, pay in ((sA, ta, pA), (sB, tb, pB)):
            pr = np.sqrt(np.mean(np.abs(pay) ** 2))
            s_[gap:gap + L] = chirp / np.sqrt(np.mean(np.abs(chirp) ** 2)) \
                * 2.0 * pr
            s_[gap + SYM:gap + SYM + pay_len] = pay
        # unit-RMS normalize over active samples (core convention)
        act = slice(gap, gap + SYM + pay_len)
        for s_ in (sA, sB):
            s_ /= max(np.sqrt(np.mean(np.abs(s_[act]) ** 2)), 1e-30)
        return core.FrameSpecML(
            L=L, cp=CP, plan=plan, gap=gap, layer_tag=1, n_vec=NSLOT,
            n_sym=1, labels=np.zeros(NSLOT, np.int64),
            vec_index=np.arange(NSLOT), y_digital=(x * w).astype(
                np.complex128), sA=sA, sB=sB,
            ref_pre=sync_ref.astype(np.complex64),
            peak_a=float(np.max(np.abs(sA))),
            peak_b=float(np.max(np.abs(sB))),
            frame_len=frame_len, sync_payload_off=gap + SYM)

    def decode_serial(cap, fr):
        """Matched-filter the sync product; correlate each slot."""
        n = len(cap) - L
        # coarse search: correlate against sync product template
        c = np.correlate(cap, sync_ref, mode="valid")
        limok = len(cap) - SYM - pay_len
        cands = [int(k) for k in np.argsort(np.abs(c))[::-1][:6]
                 if k <= limok]
        if not cands:
            return None
        d0 = min(cands)
        pm = float(np.abs(c[d0]) / (np.median(np.abs(c)) + 1e-30))
        ph0 = float(np.angle(c[d0]))
        p0 = d0 + SYM
        pay = cap[p0:p0 + pay_len]
        if len(pay) < pay_len:
            return None
        tpl = np.exp(+2j * np.pi * F_P * t_pay)   # measured flip
        prod = pay * tpl
        sl = prod.reshape(NSLOT, TSLOT)[:, GUARD:TSLOT - GUARD]
        yhat = sl.mean(axis=1) * np.exp(-1j * ph0)
        return yhat, pm, float(np.max(np.abs(pay))), \
            pay.astype(np.complex64)

    out = {}
    hw = core.HWSession(args, Path("."))
    t0 = time.time()
    try:
        for dr in range(DRAWS):
            x = rng.standard_normal(NSLOT)
            w = rng.standard_normal(NSLOT)
            fr = build_serial(x, w)
            out[f"x_d{dr}"] = x
            out[f"w_d{dr}"] = w
            for (prf, plo) in POINTS:
                args.p_rf_dbm, args.p_lo_dbm = prf, plo
                tag = f"{prf:+.0f}_{plo:+.0f}"
                for rp in range(REPS):
                    hw.ensure(fr, verbose_build=False)
                    cap = hw.capture(fr)
                    r = decode_serial(cap, fr)
                    if r is None:
                        print(f"[serA] {tag} d{dr}r{rp}: DECODE FAIL",
                              flush=True)
                        continue
                    yhat, pm, peak = r[0], r[1], r[2]
                    out[f"yhat_{tag}_d{dr}_r{rp}"] = yhat
                    if rp == 0:
                        out[f"rawpay_{tag}_d{dr}"] = r[3] \
                            if len(r) > 3 else np.zeros(0, np.complex64)
                    if rp == 0:
                        xw = x * w
                        g = np.vdot(xw, yhat) / np.vdot(xw, xw)
                        res = yhat - g * xw
                        rel = np.sqrt(np.mean(np.abs(res) ** 2)
                                      / np.mean(np.abs(g * xw) ** 2))
                        print(f"[serA] {tag} d{dr}: pm={pm:6.1f} "
                              f"peak={peak:.3f}FS |g|={abs(g):.4f} "
                              f"lin-resid={rel:.3f} "
                              f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        hw.close()
        np.savez_compressed("serial_stationA_20260823.npz", **out,
                            meta_json=json.dumps(dict(
                                points=[list(p) for p in POINTS],
                                nslot=NSLOT, tslot=TSLOT, guard=GUARD,
                                f_a=F_A, f_b=F_B,
                                chain="IF +10dB pads (2x5dB post-mixer)",
                                encoding="time-serial, sign-as-phase")))
        print("[serA] saved -> serial_stationA_20260823.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
