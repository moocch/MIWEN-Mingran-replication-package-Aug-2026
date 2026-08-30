# -*- coding: utf-8 -*-
"""twin_2x2.py -- the {clean, twin} x {digital, hardware} ablation of the
serial hardware-aware campaign, applied to the fully analog comb chain.

    clean arm  (ZC): trained against an ideal multiplier -- no device model
    twin  arm  (ZT): trained with the calibrated device twin inside the loop
                     (mixer compression at the planned drive + the measured
                     detector transfer), NO noise injection, no bias correction
    "hardware" (ZH): the same physics plus link noise and element jitter, i.e.
                     what the analog client actually runs

Also pins the clean-under-twin prediction: the twin's forecast, made in
software, of what the clean network will do on the chain.
"""
import json
import sys

import numpy as np
import jax

import comb_analog_sim as cs

RES = "results/twin_2x2.json"


def acc(mode, ds, wtag, L=3, repeats=1):
    cfg = cs.cfg_make(mode, L, ds)
    Xte, yte = (cs.load_mnist() if ds == "mnist" else cs.load_data())[2:4]
    p0 = cs.init_params(cfg, jax.random.PRNGKey(0))
    got = cs._ckpt_load(wtag, p0)
    if got is None:
        return None
    _, ev = cs.make_train_fns(cfg)
    a = [float(cs.evaluate(got[0], cfg, ev, Xte, yte,
                           jax.random.PRNGKey(1000 + s)))
         for s in range(repeats)]
    return 100 * np.mean(a), 100 * np.std(a)


def main(datasets=("mnist", "gtsrb"), repeats=5):
    out = {}
    for ds in datasets:
        pre = "M" if ds == "mnist" else ""
        clean_w, twin_w = f"{pre}ZC_L3", f"{pre}ZT_L3"
        r = dict(
            clean_digital=acc("ZC", ds, clean_w),                # own model
            clean_under_twin=acc("ZT", ds, clean_w),             # pinned
            clean_hardware=acc("ZH", ds, clean_w, repeats=repeats),
            twin_digital=acc("ZT", ds, twin_w),                  # own model
            twin_hardware=acc("ZH", ds, twin_w, repeats=repeats),
        )
        out[ds] = {k: (None if v is None else
                       dict(acc=round(v[0], 2), std=round(v[1], 2)))
                   for k, v in r.items()}
        print(f"\n=== {ds.upper()}  (L = 3) ===")
        print(f"{'arm':<26}{'digital':>12}{'hardware':>16}")
        print(f"{'clean (ideal multiplier)':<26}"
              f"{r['clean_digital'][0]:>11.2f}%"
              f"{r['clean_hardware'][0]:>14.2f}%"
              f" +- {r['clean_hardware'][1]:.2f}")
        print(f"{'twin (hardware-aware)':<26}"
              f"{r['twin_digital'][0]:>11.2f}%"
              f"{r['twin_hardware'][0]:>14.2f}%"
              f" +- {r['twin_hardware'][1]:.2f}")
        print(f"clean-under-twin (pinned forecast): "
              f"{r['clean_under_twin'][0]:.2f}%")
    json.dump(out, open(RES, "w"), indent=1)
    print("\nwrote", RES)


if __name__ == "__main__":
    ds = sys.argv[1:] or ["mnist", "gtsrb"]
    main(tuple(ds))
