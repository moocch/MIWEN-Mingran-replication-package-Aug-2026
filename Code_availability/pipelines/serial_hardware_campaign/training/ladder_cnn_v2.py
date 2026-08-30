#!/usr/bin/env python3
"""R4: literature-practice CNNs under the MIWEN constraint stack.

Per-layer op (one hardware pass + one digital inter-pass step):
    y = complex_conv(A)                      (analog MVM over im2col patches)
    S = |y|                                  (magnitude readout)
    S = BN(S)  -> clamp(>=0)                 (digital: BatchNorm during
                                              training; per-channel affine at
                                              inference -- lives in the
                                              already-digital inter-pass step)
    [maxpool]  [identity skip-add]           (digital)
    A_next = S / max_row(S)                  (per-sample max-norm, as hardware)

Everything is expressible on the existing primitive; no signed activations.
Digital ceiling campaign only (steps 1-3 of the R4 program) -- noise finetune
and hardware mapping come later, gated on these ceilings.
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

N_CLASSES = 43
CACHE_PREFIX = "gtsrb_roi"
READOUT = "mag"   # "mag" (certified |Y|) or "re" (Re Y + ReLU everywhere)
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = _data_dir(__file__)
sys.path.insert(0, str(SCRIPT_DIR))

_EPS = 1e-12

ARCHS = {
    # blocks: list of (channels, n_convs); pool after each block; then FC dims
    "vgg6":     dict(blocks=[(64, 2), (128, 2), (256, 2)], fc=[256], skip=False),
    "vgg6skip": dict(blocks=[(64, 2), (128, 2), (256, 2)], fc=[256], skip=True),
    "wide4":    dict(blocks=[(96, 2), (192, 2)], fc=[512], skip=False),
    # rig-sized archs (explicit conv specs; valid padding = no runner change,
    # geometry chosen against the measured capture budget of 2026-08-01):
    "r3plus": dict(convs=[dict(ch=32, k=5, stride=1, pad=0, pool=True),
                          dict(ch=64, k=5, stride=1, pad=0, pool=True)],
                   fc=[128], skip=False),      # == R3 geometry, v2 recipe
    "r3wide": dict(convs=[dict(ch=48, k=5, stride=2, pad=0, pool=False),
                          dict(ch=96, k=5, stride=1, pad=0, pool=True)],
                   fc=[192], skip=False),
    "r3fast": dict(convs=[dict(ch=32, k=5, stride=2, pad=0, pool=False),
                          dict(ch=64, k=5, stride=1, pad=0, pool=True)],
                   fc=[128], skip=False),
}


def build_params(arch, img, device, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    spec = ARCHS[arch]
    P, meta = {}, []
    cin, hw = 3, img
    li = 0
    if "convs" in spec:
        for c in spec["convs"]:
            ch, k, st, pd = c["ch"], c["k"], c["stride"], c["pad"]
            sc = 1.0 / math.sqrt(k * k * cin)
            for tag in "ri":
                P[f"c{li}{tag}"] = (torch.randn(ch, cin, k, k, generator=g) * sc
                                    ).to(device).requires_grad_()
            meta.append(dict(kind="conv", cin=cin, cout=ch, k=k, stride=st,
                             pad=pd, pool=c["pool"], skip=False))
            hw = (hw + 2 * pd - k) // st + 1
            if c["pool"]:
                hw //= 2
            cin = ch
            li += 1
        d_in = cin * hw * hw
    else:
        for bi, (ch, ncv) in enumerate(spec["blocks"]):
            for ci in range(ncv):
                k = 3
                sc = 1.0 / math.sqrt(k * k * cin)
                for tag in "ri":
                    P[f"c{li}{tag}"] = (torch.randn(ch, cin, k, k, generator=g)
                                        * sc).to(device).requires_grad_()
                meta.append(dict(kind="conv", cin=cin, cout=ch, k=k, stride=1,
                                 pad=1, pool=(ci == ncv - 1),
                                 skip=(spec["skip"] and ci == ncv - 1
                                       and cin == ch)))
                cin = ch
                li += 1
            hw = (hw - 0) // 2      # 'same' padding keeps size until pool
        d_in = cin * (img // (2 ** len(spec["blocks"]))) ** 2
    for h in spec["fc"] + [N_CLASSES]:
        sc = 1.0 / math.sqrt(d_in)
        for tag in "ri":
            P[f"d{li}{tag}"] = (torch.randn(h, d_in, generator=g) * sc
                                ).to(device).requires_grad_()
        meta.append(dict(kind="dense", din=d_in, dout=h))
        d_in = h
        li += 1
    return P, meta


class BNBank:
    """Train-time BatchNorm per conv layer on the magnitude maps; at inference
    the running stats give the digital per-channel affine."""

    def __init__(self, meta, device):
        import torch.nn as nn
        self.mods = {}
        for i, m in enumerate(meta):
            if m["kind"] == "conv":
                self.mods[i] = nn.BatchNorm2d(m["cout"]).to(device)

    def params(self):
        out = []
        for m in self.mods.values():
            out += list(m.parameters())
        return out

    def train(self, flag):
        for m in self.mods.values():
            m.train(flag)


_BETA_RF = 1.369   # RF-compression sharpness (datasheet-grounded mixer twin)


def _compress(S, z_over_rms):
    """Physical RF soft-compressor D_rf (the deployed mixer twin's RF term):
        |z| -> |z| * (1 + (|z|/z_c)^bRF)^(-1/bRF),   z_c = z_over_rms * rms(|z|).
    z_over_rms large -> identity (RF cold, -35 dBm); ~few -> compression (RF
    hot). Training-time only: it adds NO component to the deployed inference
    chain (the real mixer applies this at high power)."""
    import torch
    zc = z_over_rms * torch.sqrt((S * S).mean()) + 1e-12
    return S * (1.0 + (S / zc) ** _BETA_RF) ** (-1.0 / _BETA_RF)


def forward(X, P, meta, bn, *, train=False, noise=None, compress=None):
    """noise: optional per-layer-index rel-RMSE list; adds additive magnitude
    noise before BN. compress: optional z_c/rms scalar for the physical RF
    compressor D_rf, applied at the magnitude readout (deterministic, before
    noise) -- the high-power hardware-aware term."""
    import torch
    import torch.nn.functional as F
    A = X
    skip_src = None
    for i, m in enumerate(meta):
        if m["kind"] == "conv":
            Yr = F.conv2d(A, P[f"c{i}r"], padding=m.get("pad", 1),
                          stride=m.get("stride", 1))
            Yi = F.conv2d(A, P[f"c{i}i"], padding=m.get("pad", 1),
                          stride=m.get("stride", 1))
            if READOUT == "re":
                S = Yr
            else:
                S = torch.sqrt(Yr * Yr + Yi * Yi + _EPS)
                if compress is not None:
                    S = compress(S) if callable(compress) \
                        else _compress(S, compress)
            if noise is not None and noise[i]:
                S = S + noise[i] * torch.sqrt((S * S).mean()) \
                    * torch.randn_like(S)
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
            mx = S.flatten(1).max(1).values.clamp_min(_EPS)
            A = S / mx.view(-1, 1, 1, 1)
        else:
            if A.dim() > 2:
                A = A.flatten(1)
            Yr, Yi = A @ P[f"d{i}r"].T, A @ P[f"d{i}i"].T
            if READOUT == "re":
                S = Yr
            else:
                S = torch.sqrt(Yr * Yr + Yi * Yi + _EPS)
                if compress is not None:
                    S = compress(S) if callable(compress) \
                        else _compress(S, compress)
            if noise is not None and noise[i]:
                S = S + noise[i] * torch.sqrt((S * S).mean()) \
                    * torch.randn_like(S)
            if READOUT == "re" and m is not meta[-1]:
                S = torch.clamp(S, min=0.0)
            if m is not meta[-1]:
                mx = S.max(1).values.clamp_min(_EPS)
                A = S / mx.view(-1, 1)
    return S * S


def augment(x, img):
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * (15 * math.pi / 180)
    s = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 0.15
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.10
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.10
    sh = (torch.rand(B, device=dev) * 2 - 1) * 0.10
    ca, sa = torch.cos(ang) * s, torch.sin(ang) * s
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = ca
    th[:, 0, 1] = -sa + sh
    th[:, 1, 0] = sa
    th[:, 1, 1] = ca
    th[:, 0, 2] = tx
    th[:, 1, 2] = ty
    grid = F.affine_grid(th, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    bright = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.25
    return (out * bright).clamp(0, 1.3)


def load_cache(img):
    p = SCRIPT_DIR / f"{CACHE_PREFIX}_{img}x{img}.npz"
    d = np.load(p, allow_pickle=True)

    def pp(u8):
        x = u8.reshape(u8.shape[0], -1).astype(np.float32) / 255.0
        lo = np.percentile(x, 2.0, axis=1, keepdims=True)
        hi = np.percentile(x, 98.0, axis=1, keepdims=True)
        x = np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
        return x.reshape(-1, img, img, 3).transpose(0, 3, 1, 2)

    return pp(d["Xtr"]), d["ytr"].astype(np.int64), \
        pp(d["Xte"]), d["yte"].astype(np.int64)


def _acc(X, y, P, meta, bn, device, chunk=512, noise=None, nseed=0,
         compress=None):
    import torch
    with torch.no_grad():
        if noise is not None:
            torch.manual_seed(nseed)      # fixed realization: comparable epochs
        accs = []
        for c0 in range(0, len(y), chunk):
            lg = forward(X[c0:c0 + chunk].to(device), P, meta, bn, noise=noise,
                         compress=compress)
            accs.append((lg.argmax(1) == y[c0:c0 + chunk]).float())
        return float(torch.cat(accs).mean())


def train_model(arch, *, img=48, epochs=150, seed=0, device="cuda",
                batch=96, lr=1.5e-3, out=None, log=print, val_frac=0.0,
                noise=None, compress=None, compress_anneal=0.5,
                track_aware=False):
    """val_frac > 0: hold out a stratup-free random train split; checkpoint
    selection uses ONLY the val split (fixes the R4 best-epoch-on-test
    caveat); test accuracy is evaluated once, on the selected checkpoint."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = load_cache(img)
    rng = np.random.RandomState(seed + 777)
    n_all = len(ytr)
    vidx = np.zeros(n_all, bool)
    if val_frac > 0:
        if track_aware:
            # GTSRB tracks = 30 consecutive frames of one physical sign; hold
            # out WHOLE tracks so the val split shares no track with train
            # (the random-frame split leaks near-duplicate frames -> inflated,
            # misleading selection). Cache is class-sorted, tracks are 30-frame
            # blocks aligned to class boundaries.
            tid = np.arange(n_all) // 30
            utid = np.unique(tid)
            hold = rng.choice(utid, int(len(utid) * val_frac), replace=False)
            vidx = np.isin(tid, hold)
        else:
            vidx[rng.choice(n_all, int(n_all * val_frac), replace=False)] = True
    # OOM fix (overnight run 1): dataset stays on CPU (pinned); batches move
    Xt = torch.tensor(Xtr[~vidx]).pin_memory()
    yt = torch.tensor(ytr[~vidx], device=device)
    Xv = torch.tensor(Xtr[vidx]).pin_memory() if val_frac > 0 else None
    yv = torch.tensor(ytr[vidx], device=device) if val_frac > 0 else None
    Xe = torch.tensor(Xte).pin_memory()
    ye = torch.tensor(yte, device=device)
    P, meta = build_params(arch, img, device, seed)
    bn = BNBank(meta, device)
    params = list(P.values()) + bn.params()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    n = len(yt)
    spe = max(1, n // batch)
    total, warm = spe * epochs, 2 * spe
    best, ema, step, t0 = 0.0, None, 0, time.time()
    best_state = None
    for ep in range(epochs):
        bn.train(True)
        # drive annealing: start with a very high knee (RF cold -> ~identity)
        # and ramp z_c/rms down to the target over the first `compress_anneal`
        # fraction of epochs, so the net learns before facing full compression.
        if compress is None:
            comp_eff = None
        elif callable(compress):
            # callable transforms anneal via a knee multiplier (same 9x->1 ramp)
            frac = min(1.0, ep / max(1e-9, compress_anneal * epochs))
            _a = 1.0 + (1.0 - frac) * 8.0
            comp_eff = (lambda S, _a=_a, _f=compress: _f(S, anneal=_a))
        else:
            frac = min(1.0, ep / max(1e-9, compress_anneal * epochs))
            comp_eff = compress * (1.0 + (1.0 - frac) * 8.0)   # 9x knee -> target
        perm = torch.randperm(n, device=device)
        for k in range(spe):
            idx = perm[k * batch:(k + 1) * batch]
            xb = Xt[idx.cpu()].to(device, non_blocking=True)
            cur = lr * ((step + 1) / warm if step < warm else
                        0.5 * (1 + math.cos(math.pi * (step - warm)
                                            / max(1, total - warm))))
            for g in opt.param_groups:
                g["lr"] = cur
            logits = forward(augment(xb, img), P, meta, bn, train=True,
                             noise=noise, compress=comp_eff)
            loss = F.cross_entropy(torch.log(logits.clamp_min(1e-12)), yt[idx],
                                   label_smoothing=0.05)
            if not torch.isfinite(loss):
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params
                                if p.grad is not None))
            ema = float(gn) if ema is None else 0.98 * ema + 0.02 * float(gn)
            torch.nn.utils.clip_grad_norm_(params, 3.0 * ema)
            opt.step()
            step += 1
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            bn.train(False)
            sel = _acc(Xv, yv, P, meta, bn, device, noise=noise,
                       compress=compress) \
                if val_frac > 0 else _acc(Xe, ye, P, meta, bn, device,
                                          noise=noise, compress=compress)
            tag = ("noisy-val" if noise is not None else "val") \
                if val_frac > 0 else "digital full"
            if sel >= best:
                best = sel
                best_state = ({k: v.detach().clone() for k, v in P.items()},
                              {i: {k: v.detach().clone() for k, v in
                                   m.state_dict().items()}
                               for i, m in bn.mods.items()})
            log(f"[{arch} s{seed} ep {ep+1}/{epochs}] {tag} "
                f"{sel*100:.2f} (best {best*100:.2f}) "
                f"({time.time()-t0:.0f}s)")
    if best_state is not None:
        for k in P:
            P[k].data.copy_(best_state[0][k])
        for i, m in bn.mods.items():
            m.load_state_dict(best_state[1][i])
    bn.train(False)
    te = _acc(Xe, ye, P, meta, bn, device, compress=compress)
    te300 = _acc(Xe[:300], ye[:300], P, meta, bn, device, compress=compress)
    log(f"[{arch} s{seed}] SELECTED ckpt: test full {te*100:.2f}, "
        f"test N=300 {te300*100:.2f} (selector best {best*100:.2f})")
    if out:
        save(P, meta, bn, out, dict(arch=arch, img=img, seed=seed,
                                    acc=te, acc300=te300, val_frac=val_frac,
                                    selector_best=best))
    return te


def save(P, meta, bn, path, info):
    import torch
    arrs = {k: v.detach().cpu().numpy() for k, v in P.items()}
    for i, mod in bn.mods.items():
        with torch.no_grad():
            sc = (mod.weight / torch.sqrt(mod.running_var + mod.eps))
            sh = mod.bias - mod.running_mean * sc
        arrs[f"bnscale{i}"] = sc.cpu().numpy()
        arrs[f"bnshift{i}"] = sh.cpu().numpy()
    np.savez_compressed(path, meta_json=json.dumps(dict(info, layers=meta)),
                        **arrs)


def export_runner(v2_npz, out_npz):
    """Convert a v2 checkpoint to run_ladder_hw's format: arch_json conv list
    (with stride/pool), c{i}r/i 4-D weights, BN affine folded to c{i}s
    (scale) + c{i}b (shift), dense re-keyed to d0.., zero dense biases."""
    z = np.load(v2_npz, allow_pickle=True)
    info = json.loads(str(z["meta_json"]))
    layers = info["layers"]
    arrs, conv_specs, fcs = {}, [], []
    dj = 0
    for i, m in enumerate(layers):
        if m["kind"] == "conv":
            conv_specs.append(dict(cout=m["cout"], cin=m["cin"], k=m["k"],
                                   stride=m.get("stride", 1),
                                   pool=m.get("pool", True)))
            ci = len(conv_specs) - 1
            arrs[f"c{ci}r"] = z[f"c{i}r"]
            arrs[f"c{ci}i"] = z[f"c{i}i"]
            arrs[f"c{ci}s"] = z[f"bnscale{i}"]
            arrs[f"c{ci}b"] = z[f"bnshift{i}"]
        else:
            arrs[f"d{dj}r"] = z[f"d{i}r"]
            arrs[f"d{dj}i"] = z[f"d{i}i"]
            arrs[f"d{dj}b"] = np.zeros(z[f"d{i}r"].shape[0], np.float64)
            fcs.append(int(z[f"d{i}r"].shape[0]))
            dj += 1
    np.savez_compressed(out_npz,
                        arch_json=json.dumps(dict(conv=conv_specs, fc=fcs,
                                                  src=str(v2_npz),
                                                  info=info)),
                        **arrs)
    return conv_specs, fcs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--img", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    best = train_model(args.arch, img=args.img, epochs=args.epochs,
                       seed=args.seed, out=args.out, val_frac=args.val_frac)
    print(f"[{args.arch} s{args.seed}] test full: {best*100:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
