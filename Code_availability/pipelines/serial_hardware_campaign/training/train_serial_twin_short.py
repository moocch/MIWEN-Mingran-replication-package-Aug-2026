#!/usr/bin/env python3
"""Train the serial twin arm (2026-08-24): the certified |Y| recipe
(ladder_cnn_v2, READOUT="mag", r3plus, same aug/BN/selection) with the
measurement-calibrated twin f(x,w) applied to EVERY product before
summation — the Shockley-in-the-loop arm of the serial 2x2. No noise
injection. Also pins the clean-under-twin digital prediction.

Twin: product-term surface from serial_twin_model.json (fitted on the
USRP CW map, region held-out 0.12 dB, serial-slot cross-val 3.7%).
Drive mapping mirrors the runner: per-layer streams normalized to unit
rms; production frames have ~zero sync duty (DUTY=0).
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = _data_dir(__file__)
sys.path.insert(0, str(HERE))
import torch.utils.checkpoint as ckpt

import ladder_cnn_v2 as v2

v2.READOUT = "mag"
v2.N_CLASSES = 43
v2.CACHE_PREFIX = "gtsrb_roi"

TM = json.load(open(HERE / "serial_twin_model.json"))
THP = torch.tensor(TM["thp"], dtype=torch.float32)
RIDGES = torch.tensor(TM["ridges"], dtype=torch.float32).reshape(-1, 4)
FLOOR_DB = -60.0
CEIL_DB = 20.0
CHUNK_H = 4
ROW_BLK = 4096

# ---- tabulated twin surface (analytic eval was the training
# bottleneck: ~1e10 transcendentals/step). 256x256 bilinear table over
# [FLOOR,CEIL]^2 dB; interp error ~0.01 dB vs the 0.12 dB model.
_G = 256
_ax = np.linspace(FLOOR_DB, CEIL_DB, _G)
_PL, _PR = np.meshgrid(_ax, _ax, indexing="ij")


def _pdb_np(p_lo, p_rf):
    thp = np.array(TM["thp"]); rg = np.array(TM["ridges"]).reshape(-1, 4)
    a0, klo, krf = thp[0], thp[1], thp[2]
    xlo = 10 ** ((p_lo - klo) / 10.0)
    xrf = 10 ** ((p_rf - krf) / 10.0)
    y = a0 + 10 * np.log10((xlo / (1 + xlo)) * (xrf / (1 + xrf)) + 1e-30)
    for wi, ai, bi, ci in rg:
        y = y + wi * np.tanh(ai * p_lo / 40 + bi * p_rf / 40 + ci)
    return y


TABLE = torch.tensor(_pdb_np(_PL, _PR), dtype=torch.float32)


def product_db_t(p_lo, p_rf):
    a0, klo, krf = THP[0], THP[1], THP[2]
    xlo = torch.pow(10.0, (p_lo - klo) / 10.0)
    xrf = torch.pow(10.0, (p_rf - krf) / 10.0)
    y = a0 + 10.0 * torch.log10(
        (xlo / (1 + xlo)) * (xrf / (1 + xrf)) + 1e-30)
    # vectorized ridges: (K,1,..) broadcast, single op
    w_ = RIDGES[:, 0].view(-1, *([1] * p_lo.dim()))
    a_ = RIDGES[:, 1].view(-1, *([1] * p_lo.dim()))
    b_ = RIDGES[:, 2].view(-1, *([1] * p_lo.dim()))
    c_ = RIDGES[:, 3].view(-1, *([1] * p_lo.dim()))
    y = y + (w_ * torch.tanh(a_ * p_lo.unsqueeze(0) / 40
                             + b_ * p_rf.unsqueeze(0) / 40
                             + c_)).sum(0)
    return y


def _amp_table(p_lo, p_rf):
    """bilinear lookup of product_db via grid_sample; inputs dB."""
    gl = (p_lo.clamp(FLOOR_DB, CEIL_DB) - FLOOR_DB) \
        / (CEIL_DB - FLOOR_DB) * 2 - 1
    gr = (p_rf.clamp(FLOOR_DB, CEIL_DB) - FLOOR_DB) \
        / (CEIL_DB - FLOOR_DB) * 2 - 1
    grid = torch.stack([gr.reshape(-1), gl.reshape(-1)], -1) \
        .view(1, 1, -1, 2)
    tab = TABLE.view(1, 1, _G, _G)
    v = F.grid_sample(tab, grid, mode="bilinear",
                      align_corners=True).reshape(p_lo.shape)
    return v


def twin_products(xa, wr, wi):
    """xa (..., D) nonneg (unit-rms); wr/wi (H, D) unit-rms. Complex
    row sums with the tabulated twin applied per product."""
    wa = torch.sqrt(wr * wr + wi * wi + 1e-24)
    p_lo = 20.0 * torch.log10(wa.clamp_min(10 ** (FLOOR_DB / 20)))
    def _chunk(xa_, wa_c, pl_c, wr_c, wi_c):
        # row-blocked to bound every allocation on the 8 GB card
        outs_r, outs_i = [], []
        pr_full = 20.0 * torch.log10(
            xa_.clamp_min(10 ** (FLOOR_DB / 20)))
        for r0 in range(0, xa_.shape[0], ROW_BLK):
            p_rf = pr_full[r0:r0 + ROW_BLK].unsqueeze(-2)
            db = _amp_table(
                pl_c.unsqueeze(0).expand(p_rf.shape[0], -1, -1),
                p_rf.expand(-1, pl_c.shape[0], -1))
            amp = torch.pow(10.0, db / 20.0)
            outs_r.append((amp * (wr_c / wa_c).unsqueeze(0)).sum(-1))
            outs_i.append((amp * (wi_c / wa_c).unsqueeze(0)).sum(-1))
        return torch.cat(outs_r, 0), torch.cat(outs_i, 0)

    outs_r, outs_i = [], []
    for h0 in range(0, wr.shape[0], CHUNK_H):
        r_, i_ = ckpt.checkpoint(
            _chunk, xa, wa[h0:h0 + CHUNK_H], p_lo[h0:h0 + CHUNK_H],
            wr[h0:h0 + CHUNK_H], wi[h0:h0 + CHUNK_H],
            use_reentrant=False)
        outs_r.append(r_)
        outs_i.append(i_)
    return torch.cat(outs_r, -1), torch.cat(outs_i, -1)


def twin_forward(X, P, meta, bn, *, train=False, noise=None,
                 compress=None):
    A = X
    skip_src = None
    for i, m in enumerate(meta):
        if m["kind"] == "conv":
            k = P[f"c{i}r"].shape[-1]
            pad = m.get("pad", 1)
            st = m.get("stride", 1)
            n, C, Hh, Ww = A.shape
            Ho = (Hh + 2 * pad - k) // st + 1
            Wo = (Ww + 2 * pad - k) // st + 1
            U = F.unfold(A, k, padding=pad, stride=st)   # (n, CkK, L)
            U = U.transpose(1, 2).reshape(-1, U.shape[1])
            wr = P[f"c{i}r"].reshape(P[f"c{i}r"].shape[0], -1)
            wi = P[f"c{i}i"].reshape(P[f"c{i}i"].shape[0], -1)
            xs = 1.0 / U.pow(2).mean().sqrt().clamp_min(1e-12)
            ws = 1.0 / torch.sqrt((wr * wr + wi * wi).mean()
                                  ).clamp_min(1e-12)
            Yr, Yi = twin_products(U * xs, wr * ws, wi * ws)
            S = torch.sqrt(Yr * Yr + Yi * Yi + 1e-12)
            S = S.reshape(n, Ho * Wo, -1).transpose(1, 2) \
                .reshape(n, -1, Ho, Wo)
            S = bn.mods[i](S)
            S = torch.clamp(S, min=0.0)
            if m["skip"] and skip_src is not None \
                    and skip_src.shape == S.shape:
                S = S + skip_src
            if m["pool"]:
                S = F.max_pool2d(S, 2)
                skip_src = None
            else:
                skip_src = S
            mx = S.flatten(1).max(1).values.clamp_min(1e-12)
            A = S / mx.view(-1, 1, 1, 1)
        else:
            if A.dim() > 2:
                A = A.flatten(1)
            wr, wi = P[f"d{i}r"], P[f"d{i}i"]
            xs = 1.0 / A.pow(2).mean().sqrt().clamp_min(1e-12)
            ws = 1.0 / torch.sqrt((wr * wr + wi * wi).mean()
                                  ).clamp_min(1e-12)
            Yr, Yi = twin_products(A * xs, wr * ws, wi * ws)
            S = torch.sqrt(Yr * Yr + Yi * Yi + 1e-12)
            if m is not meta[-1]:
                mx = S.max(1).values.clamp_min(1e-12)
                A = S / mx.view(-1, 1)
    return S * S


def main():
    global THP, RIDGES
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    global TABLE
    THP = THP.to(dev)
    RIDGES = RIDGES.to(dev)
    TABLE = TABLE.to(dev)

    if "--predict-clean" in sys.argv:
        # clean-under-twin digital prediction (pinned pre-hardware)
        import importlib.util
        rspec = importlib.util.spec_from_file_location(
            "ref1", HERE / "miwen_frozen_reference.py")
        ref = importlib.util.module_from_spec(rspec)
        rspec.loader.exec_module(ref)
        z = np.load(HERE / "r35_r3plus_s0_hw.npz", allow_pickle=True)
        arch = json.loads(str(z["arch_json"]))
        d = np.load(HERE / "gtsrb_roi_32x32.npz", allow_pickle=True)
        Xte = ref.preprocess(d["Xte"])
        yte = d["yte"].astype(np.int64)
        bat = np.load(HERE / "battery_random1200_idx.npy")
        sel = bat[:300]
        A = torch.tensor(Xte[sel].reshape(-1, 32, 32, 3)
                         .transpose(0, 3, 1, 2), dtype=torch.float32,
                         device=dev)
        # runner-format weights -> twin forward layer by layer
        Acur = A
        n = len(sel)
        for ci, cs in enumerate(arch["conv"]):
            wr = torch.tensor(z[f"c{ci}r"], dtype=torch.float32,
                              device=dev).reshape(cs["cout"], -1)
            wi = torch.tensor(z[f"c{ci}i"], dtype=torch.float32,
                              device=dev).reshape(cs["cout"], -1)
            k, st = cs["k"], cs["stride"]
            nn_, C, Hh, Ww = Acur.shape
            Ho = (Hh - k) // st + 1
            Wo = (Ww - k) // st + 1
            U = F.unfold(Acur, k, stride=st).transpose(1, 2) \
                .reshape(-1, C * k * k)
            xs = 1.0 / U.pow(2).mean().sqrt()
            ws = 1.0 / torch.sqrt((wr * wr + wi * wi).mean())
            Yr, Yi = twin_products(U * xs, wr * ws, wi * ws)
            S = torch.sqrt(Yr * Yr + Yi * Yi + 1e-12)
            S = S * torch.tensor(z[f"c{ci}s"], device=dev) \
                + torch.tensor(z[f"c{ci}b"], device=dev)
            S = torch.clamp(S, min=0.0)
            fm = S.reshape(nn_, Ho, Wo, -1).permute(0, 3, 1, 2)
            if cs["pool"]:
                fm = F.max_pool2d(fm, 2)
            mx = fm.flatten(1).max(1).values.clamp_min(1e-12)
            Acur = fm / mx.view(-1, 1, 1, 1)
        for dj in range(len(arch["fc"])):
            wr = torch.tensor(np.real(z[f"d{dj}r"] + 0j),
                              dtype=torch.float32, device=dev)
            wcx = z[f"d{dj}r"]
            wi_np = z[f"d{dj}i"] if f"d{dj}i" in z.files \
                else np.zeros_like(wcx)
            wi = torch.tensor(wi_np, dtype=torch.float32, device=dev)
            Af = Acur.flatten(1)
            xs = 1.0 / Af.pow(2).mean().sqrt()
            ws = 1.0 / torch.sqrt((wr * wr + wi * wi).mean())
            Yr, Yi = twin_products(Af * xs, wr * ws, wi * ws)
            S = torch.sqrt(Yr * Yr + Yi * Yi + 1e-12)
            if dj < len(arch["fc"]) - 1:
                mx = S.max(1).values.clamp_min(1e-12)
                Acur = S / mx.view(-1, 1)
        pd = S.argmax(1).cpu().numpy()
        acc = 100.0 * np.mean(pd == yte[sel])
        print(f"[predict] clean-under-twin digital N300: {acc:.2f}",
              flush=True)
        json.dump(dict(clean_under_twin_n300=float(acc)),
                  open(HERE / "serial_predictions.json", "w"))
        return

    v2.forward = twin_forward
    v2.train_model("r3plus", img=32, epochs=15, seed=0, device=dev,
                   out=str(HERE / "gtsrb_r3plus_serialtwin_short_s0.npz"),
                   val_frac=0.1, noise=None, track_aware=False,
                   batch=32,
                   log=lambda m: print(m, flush=True))
    v2.export_runner(str(HERE / "gtsrb_r3plus_serialtwin_short_s0.npz"),
                     str(HERE / "serial_twin_s0_hw.npz"))
    print("[twin-train] exported serial_twin_s0_hw.npz", flush=True)


if __name__ == "__main__":
    main()
