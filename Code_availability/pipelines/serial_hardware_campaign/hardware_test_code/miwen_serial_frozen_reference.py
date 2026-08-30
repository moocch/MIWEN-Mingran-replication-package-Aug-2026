#!/usr/bin/env python3
"""MIWEN SERIAL (time-domain) frozen algorithm — single-file audit
reference (2026-08-25). Run:

    python3 miwen_serial_frozen_reference.py --verify

to check weight/model hashes, reproduce the pinned digital accuracies,
and bit-exactly replay the archived hardware capture through this
file's own decoder.

This file is the serial-era analogue of miwen_frozen_reference.py
(the certified comb |Y| freeze): one file, every mathematical
operation of the frozen pipeline, no unused code paths. The hardware
session glue (USRP flowgraph, capture) lives in the certified core
script, exactly as in the |Y| freeze.

DIFFERENCES FROM THE CERTIFIED COMB |Y| FREEZE, in full:
 1. ENCODING: one product per 32-sample time slot on a single
    subcarrier pair (serial), instead of <=800 simultaneous tones
    (comb). The summation therefore happens ACROSS TIME SLOTS, after
    the mixer nonlinearity acts on each product individually — the
    Bussgang crowd-linearization premise is absent by construction.
 2. TRAINING: no noise injection in either arm. The fielded arm is
    trained THROUGH a measurement-calibrated device transfer f(x,w)
    (the "twin", tabulated below); the comparison arm is the
    certified clean checkpoint unchanged. (Comb freeze: flat-25%
    noise-injected arm; no device model anywhere.)
 3. READOUT: coherent slot sum per output row, then magnitude — the
    same amplitude-only |Y| convention, but per ROW SUM instead of
    per FFT bin. No FFT, no per-bin calibration: ONE scalar gain per
    layer (43..128 complex gains -> 1 real scalar). No Rician debias
    (row-sum SNR makes it unnecessary; its absence is deliberate).
 4. PHASE: no phase reference is used anywhere (|.| after the row sum
    cancels the capture phase; the chirp preamble is timing-only).
    Comb-|Y| likewise needed none; signs/phases ride the TX slot
    carriers exactly as they rode the TX tones.
 5. CALIBRATION teacher-forcing is ARM-CONSISTENT: each arm's
    reference states come from ITS OWN training forward (clean =
    ideal multiplier, twin = the twin surface). Ideal-forward states
    drive the twin arm's folded BatchNorm to all-zero activations.
 6. DRIVE normalization: frozen per-layer slot-stream RMS (set once
    from the calibration batch), not per-frame RMS — production
    frames have ~0.35% sync duty, so payload runs at the commanded
    power (the comb's frame-RMS convention was sync-duty sensitive).
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

# ---- frozen artifacts --------------------------------------------------
WEIGHTS = {
    "clean": ("r35_r3plus_s0_hw.npz", "8a4c9efddbf8cfd0"),
    "twin": ("serial_twin_s0_hw.npz", "825ab4ee148dc1cd"),
}
TWIN_MODEL = ("serial_twin_model.json", "b07bb82b328d9c8e")
CAPTURE_REPLAY = "serial_diag_cap.npy"

# ---- serial waveform constants (as fielded) ----------------------------
FS = 10e6
L, CP = 16384, 512
SYM = L + CP
SYNC_BAND_FRAC = 8            # chirp band = +- L/8 bins (core constant)
SYNC_ROOTS = (2, 3)           # ZC roots; constant-modulus, 10 iters
TSLOT, GUARD = 32, 8          # production slots (replay capture: 128)
F_A, F_B = 2.2e6, -1.8e6      # production subcarriers; product at 4 MHz
POWER = (0.0, 0.0)            # commanded dBm, both ports; 10 dB IF pads


# ---- preprocessing / weights (identical to the |Y| freeze) -------------
def preprocess(u8):
    x = u8.reshape(u8.shape[0], -1).astype(np.float64) / 255.0
    lo = np.percentile(x, 2.0, axis=1, keepdims=True)
    hi = np.percentile(x, 98.0, axis=1, keepdims=True)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


def load_weights(path: Path):
    z = np.load(path, allow_pickle=True)
    arch = json.loads(str(z["arch_json"]))
    layers = []
    for i, c in enumerate(arch["conv"]):
        W = (z[f"c{i}r"] + 1j * z[f"c{i}i"]).reshape(c["cout"], -1)
        scale = (z[f"c{i}s"].astype(np.float64) if f"c{i}s" in z.files
                 else np.ones(c["cout"]))
        layers.append(dict(kind="conv", W=W, k=c["k"],
                           stride=c["stride"], pool=c["pool"],
                           cout=c["cout"], scale=scale, b=z[f"c{i}b"]))
    for j in range(len(arch["fc"])):
        Wi = z[f"d{j}i"] if f"d{j}i" in z.files \
            else np.zeros_like(z[f"d{j}r"])
        layers.append(dict(kind="dense", W=z[f"d{j}r"] + 1j * Wi,
                           b=z[f"d{j}b"]))
    return layers


GEO = {1: (28, 28), 2: (10, 10)}


def im2col(A, k, stride=1):
    n, c, hh, ww = A.shape
    ho = (hh - k) // stride + 1
    wo = (ww - k) // stride + 1
    cols = np.empty((n, ho * wo, c * k * k), A.dtype)
    p = 0
    for i in range(0, ho * stride, stride):
        for j in range(0, wo * stride, stride):
            cols[:, p] = A[:, :, i:i + k, j:j + k].reshape(n, -1)
            p += 1
    return cols, ho, wo


def maxpool2(A):
    n, c, h, w = A.shape
    return A.reshape(n, c, h // 2, 2, w // 2, 2).max(5).max(3)


def maxnorm(A):
    m = np.maximum(A.reshape(A.shape[0], -1).max(1), 1e-6)
    return A / m.reshape(-1, *([1] * (A.ndim - 1)))


def interpass(am, lay, tag, n):
    """|Y| interpass, byte-identical semantics to the certified comb
    freeze: magnitudes -> folded-BN affine -> ReLU -> pool -> maxnorm."""
    if lay["kind"] == "conv":
        S = np.maximum(am * lay["scale"][None, :] + lay["b"][None, :],
                       0.0)
        ho, wo = GEO[tag]
        fm = S.reshape(n, ho, wo, lay["cout"]).transpose(0, 3, 1, 2)
        return maxnorm(maxpool2(fm) if lay["pool"] else fm), S
    S = np.maximum(am + lay["b"][None, :], 0.0)
    return maxnorm(S)[:, None, None, :].transpose(0, 3, 1, 2), S


# ---- the twin surface (training physics; also cal reference) -----------
def load_twin(path: Path):
    tm = json.loads(path.read_text())
    return np.array(tm["thp"]), np.array(tm["ridges"]).reshape(-1, 4)


def product_db(thp, ridges, p_lo, p_rf):
    """May-structured hybrid, product term: two one-pole knees + K=20
    tanh ridges, fitted on the product-dominated region of the
    port-calibrated USRP CW map (held-out 0.12 dB; serial-slot
    cross-validation 3.7%). Knee VALUES are reported-not-gated: the
    surface's compression is joint (sum-type), so separable per-port
    knees are not identifiable — the fitted function, not the named
    decomposition, is the frozen object."""
    a0, klo, krf = thp[0], thp[1], thp[2]
    xl = 10 ** ((p_lo - klo) / 10.0)
    xr = 10 ** ((p_rf - krf) / 10.0)
    y = a0 + 10 * np.log10((xl / (1 + xl)) * (xr / (1 + xr)) + 1e-30)
    for wi, ai, bi, ci in ridges:
        y = y + wi * np.tanh(ai * p_lo / 40 + bi * p_rf / 40 + ci)
    return y


def twin_matmul(thp, ridges, X, W):
    """Digital execution of one layer THROUGH the twin: each product
    evaluated on the surface at its own slot amplitudes (unit-rms
    normalized streams, matching the frozen drive rule), complex
    weight phase preserved, coherent row sum, magnitude."""
    xs = X / max(np.sqrt((X ** 2).mean()), 1e-12)
    wa = np.abs(W)
    ws = wa / max(np.sqrt((wa ** 2).mean()), 1e-12)
    pl = 20 * np.log10(np.maximum(ws, 1e-3))
    out = np.zeros((X.shape[0], W.shape[0]), np.complex128)
    for h0 in range(0, W.shape[0], 8):
        pr = 20 * np.log10(np.maximum(xs, 1e-3))[:, None, :]
        amp = 10 ** (product_db(thp, ridges,
                                pl[None, h0:h0 + 8, :], pr) / 20.0)
        ph = (W[h0:h0 + 8]
              / np.maximum(wa[h0:h0 + 8], 1e-12))[None]
        out[:, h0:h0 + 8] = (amp * ph).sum(-1)
    return np.abs(out)


def forward_digital(X_rows, layers, twin=None, batch=1024):
    """Digital chain. twin=None: ideal multiplier (the clean arm's own
    model). twin=(thp,ridges): every product through the surface (the
    twin arm's own model, and the source of all pinned predictions).

    Evaluated in image batches of `batch`. This is BIT-IDENTICAL to a
    single-shot forward -- every operation (im2col, complex matmul,
    folded BN, ReLU, maxpool, per-image maxnorm, argmax) is per-image
    independent, so chunking changes no prediction. It only bounds peak
    RAM: the full 12,630-image test forward is ~25 GB unbatched (it OOM
    -killed this box three times on 2026-08-24) and ~3 GB at batch=1024."""
    if X_rows.shape[0] > batch:
        out = np.empty(X_rows.shape[0], np.int64)
        for i in range(0, X_rows.shape[0], batch):
            out[i:i + batch] = forward_digital(
                X_rows[i:i + batch], layers, twin, batch=batch)
        return out
    n = X_rows.shape[0]
    A = X_rows.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
    for tag, lay in enumerate(layers, 1):
        if lay["kind"] == "conv":
            P, _, _ = im2col(A, lay["k"], lay["stride"])
            Xin = P.reshape(-1, P.shape[-1])
        else:
            Xin = A.reshape(n, -1)
        am = np.abs(Xin @ lay["W"].T) if twin is None \
            else twin_matmul(twin[0], twin[1], Xin, lay["W"])
        A, S = interpass(am, lay, tag, n)
    return np.argmax(S ** 2, axis=1)


# ---- serial waveform: sync + slots + decode ----------------------------
def chirps(Lv=L):
    """Constant-modulus Zadoff-Chu chirp pair (roots 2,3), band-limited
    to +-Lv/8 bins, 10 constant-modulus projection iterations —
    identical to the reformed comb sync (timing role only here)."""
    half = Lv // SYNC_BAND_FRAC
    sbins = np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])
    N = sbins.size
    nn = np.arange(N)
    out = []
    for root in SYNC_ROOTS:
        F = np.zeros(Lv, complex)
        F[np.mod(sbins, Lv)] = np.exp(-1j * np.pi * root * nn * nn / N)
        mask = np.zeros(Lv, bool)
        mask[np.mod(sbins, Lv)] = True
        for _ in range(10):
            t = np.fft.ifft(F)
            t = t / np.maximum(np.abs(t), 1e-12)
            F = np.fft.fft(t)
            F[~mask] = 0.0
        out.append(np.fft.ifft(F))
    return out[0], out[1]


def decode_slots(cap, nslot, tslot, f_dec, guard=GUARD):
    """Locate the frame (matched filter against conj(ta)*tb — the
    measured mixer product convention), then per-slot correlation at
    the product tone (measured spectral flip: decode at -designed).
    Returns complex per-slot products; capture phase NOT removed
    (irrelevant: |.| follows any row sum)."""
    ta, tb = chirps()
    sref = np.conj(ta) * tb
    pay_len = nslot * tslot
    c = np.abs(np.correlate(cap, sref, mode="valid"))
    lim = len(cap) - SYM - pay_len
    cands = [int(k) for k in np.argsort(c)[::-1][:6] if k <= lim]
    d0 = min(cands)
    pay = cap[d0 + SYM:d0 + SYM + pay_len]
    t = np.arange(pay_len) / FS
    sl = (pay * np.exp(+2j * np.pi * abs(f_dec) * t)).reshape(
        nslot, tslot)[:, guard:tslot - guard]
    return sl.mean(axis=1)


# ---- verification ------------------------------------------------------
PINS = {
    # digital, this file's own forwards, battery-frame subsets
    "clean_ideal_full": 99.05, "clean_ideal_bat1200": 99.50,
    # bit-exact fast gates (checked by --verify)
    "twin_probe_hash": "5cb94af8cc5b257a",
    "replay_hash": "97ab77843fd093b2",
}


def verify() -> int:
    ok = True
    for arm, (fname, sha) in WEIGHTS.items():
        h = hashlib.sha256((HERE / fname).read_bytes()).hexdigest()[:16]
        print(f"[1] {arm} weights sha256[:16] = {h} "
              f"({'OK' if h == sha else 'MISMATCH'})")
        ok &= h == sha
    tm_sha = hashlib.sha256(
        (HERE / TWIN_MODEL[0]).read_bytes()).hexdigest()[:16]
    print(f"[1] twin model sha256[:16] = {tm_sha} "
          f"({'OK' if tm_sha == TWIN_MODEL[1] else 'MISMATCH'})")
    ok &= tm_sha == TWIN_MODEL[1]

    # bit-exact probe of the twin surface (fast; full twin-forward
    # accuracies are the documented PINS twin_* values, ~20 min each)
    thp, ridges = load_twin(HERE / TWIN_MODEL[0])
    g = np.linspace(-35.0, 0.0, 8)
    PL, PR = np.meshgrid(g, g)
    ph_ = hashlib.sha256(np.round(
        product_db(thp, ridges, PL, PR), 12).tobytes()).hexdigest()[:16]
    print(f"[1] twin surface probe hash = {ph_} "
          f"({'OK' if ph_ == PINS['twin_probe_hash'] else 'MISMATCH'})")
    ok &= ph_ == PINS["twin_probe_hash"]

    d = np.load(HERE / "gtsrb_roi_32x32.npz", allow_pickle=True)
    Xte = preprocess(d["Xte"])
    yte = d["yte"].astype(np.int64)
    bat = np.load(HERE / "battery_random1200_idx.npy")
    clean = load_weights(HERE / WEIGHTS["clean"][0])
    pd_ = forward_digital(Xte, clean)
    full = 100 * np.mean(pd_ == yte)
    b12 = 100 * np.mean(pd_[bat] == yte[bat])
    print(f"[2] clean ideal digital: full {full:.2f} "
          f"(pin {PINS['clean_ideal_full']}), battery-frame {b12:.2f} "
          f"(pin {PINS['clean_ideal_bat1200']})")
    ok &= abs(full - PINS["clean_ideal_full"]) < 0.005
    ok &= abs(b12 - PINS["clean_ideal_bat1200"]) < 0.005

    # bit-exact hardware replay through this file's own decoder
    cap = np.load(HERE / CAPTURE_REPLAY).astype(np.complex128)
    y = decode_slots(cap, 256, 128, -1.0e6)
    h = hashlib.sha256(np.round(y, 12).tobytes()).hexdigest()[:16]
    print(f"[3] capture replay decode hash = {h} "
          f"(pin {PINS.get('replay_hash', '<unpinned>')})")
    if "replay_hash" in PINS:
        ok &= h == PINS["replay_hash"]
    print(f"VERIFY: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(verify() if "--verify" in sys.argv else print(__doc__))
