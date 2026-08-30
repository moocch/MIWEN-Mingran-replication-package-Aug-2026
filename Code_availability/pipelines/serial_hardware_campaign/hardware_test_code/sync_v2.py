#!/usr/bin/env python3
"""Reform v2 frame builder (2026-08-20): per-plan RELATIVE sync scaling.

build_v2(core, ...) builds a frame with constant-envelope chirp sync
(roots (2,3), CM x10) scaled to C_REL x the plan's payload rms, with the
payload re-anchored to the LEGACY payload level per channel so the
commanded-power axis keeps its meaning (delivered = commanded).
"""
import numpy as np

C_REL = 4.05          # sync rms / payload rms (validated map64 regime)
_CHIRPS = {}


def _chirps(core, Lv):
    if Lv in _CHIRPS:
        return _CHIRPS[Lv]
    half = Lv // core.SYNC_BAND_FRAC
    sbins = np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])
    N = sbins.size
    nn = np.arange(N)
    out = []
    mask = np.zeros(Lv, bool)
    mask[np.mod(sbins, Lv)] = True
    for root in (2, 3):
        F = np.zeros(Lv, complex)
        F[np.mod(sbins, Lv)] = np.exp(-1j * np.pi * root * nn * nn / N)
        for _ in range(10):
            t = np.fft.ifft(F)
            t = t / np.maximum(np.abs(t), 1e-12)
            F = np.fft.fft(t)
            F[~mask] = 0.0
        out.append(np.fft.ifft(F))
    _CHIRPS[Lv] = out
    return out


def build_v2(core, Wt, x_t, labels, vidx, plan, **kw):
    Lv = kw.get("L", 16384)
    cp = kw.get("cp", 512)
    SYM = Lv + cp
    orig = core.gen_sync_symbols
    # legacy reference for the payload anchor (commanded-axis continuity)
    f_leg = core.build_frame_ml(Wt, x_t, labels, vidx, plan, **kw)
    ta, tb = _chirps(core, Lv)
    core.gen_sync_symbols = lambda LL, seed: (ta.copy(), tb.copy())
    try:
        fr = core.build_frame_ml(Wt, x_t, labels, vidx, plan, **kw)
    finally:
        core.gen_sync_symbols = orig
    for ch in ("sA", "sB"):
        a = getattr(fr, ch)
        leg = getattr(f_leg, ch)
        # anchor payload to legacy level
        p_leg = np.sqrt(np.mean(np.abs(
            leg[f_leg.gap + SYM:f_leg.frame_len - f_leg.gap]) ** 2))
        p_new = np.sqrt(np.mean(np.abs(
            a[fr.gap + SYM:fr.frame_len - fr.gap]) ** 2))
        a[:] = a * (p_leg / p_new)
        # sync to C_REL x payload rms
        sync = a[fr.gap:fr.gap + SYM]
        s_rms = np.sqrt(np.mean(np.abs(sync) ** 2))
        sync *= (C_REL * p_leg / s_rms)
    fr.peak_a = float(np.max(np.abs(fr.sA)))
    fr.peak_b = float(np.max(np.abs(fr.sB)))
    return fr
