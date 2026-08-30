# -*- coding: utf-8 -*-
"""
reproduce.py -- one entry point for everything reported in the fully analog
section of the MIWEN revision.

    python3 reproduce.py budget     # level plan + link budget of the chain
    python3 reproduce.py infer      # load the stored weights, run inference,
                                    # write results/results_summary.json + .csv
    python3 reproduce.py train      # re-train every network from scratch
    python3 reproduce.py train_nohw # re-train the two ablations below
    python3 reproduce.py ablation   # hardware-aware vs. not, on the same chain
    python3 reproduce.py all        # budget + train + infer + ablation

Layout expected next to this file:
    analog_physics.py       calibrated device models (mixer twin, detector)
    comb_analog_sim.py      comb model, level planning, training / inference
    data/mnist.npz          MNIST arrays  (Xtr, ytr, Xte, yte)
    data/gtsrb_roi_32x32.npz GTSRB ROIs   (regenerate with prepare_gtsrb.py)
    weights/                trained parameters, one .npz per run
    results/                json / csv written by this script

Nothing here needs a GPU; a full re-train of the reported table takes about
25 min on 8 CPU cores, inference alone about 1 min.
"""

# --- replication-package path resolution (see Code_availability/_paths.py) ---
import sys as _sys, pathlib as _pl
for _p in _pl.Path(__file__).resolve().parents:
    if _p.name == "Code_availability":
        _sys.path.insert(0, str(_p)); break
from _paths import data_dir as _data_dir
# ---------------------------------------------------------------------------

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = _data_dir(__file__)
os.environ.setdefault("COMB_DATA", str(HERE / "data"))
os.environ.setdefault("COMB_RESULTS", str(HERE / "weights"))
sys.path.insert(0, str(HERE))

import jax                                            # noqa: E402
import jax.numpy as jnp                               # noqa: E402
import comb_analog_sim as cs                          # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- runs -----
# tag -> (mode, L, dataset, description)  ... every number in the section
# Each entry: tag -> (mode, L, dataset, description, weights_tag)
# "MZI_L3" and "MZ_L3" deliberately share one set of trained weights: the
# same network is executed once in exact software arithmetic (mode ZT, the
# device twin without the link) and once through the calibrated analog chain
# (mode ZH = twin + link noise + element jitter).  Tags ending in "d" are the
# digital execution of the ZH checkpoint of the same depth.
RUNS = {
    # ---- headline pair: ONE checkpoint per task, executed twice ----------
    # MNIST, 784 -> 100 -> 64 -> 10 at L = 3
    "MZH_L3": ("ZH", 3, "mnist", "hardware-aware weights, analog chain"),
    "MZT_L3d": ("ZT", 3, "mnist", "the same weights, digital execution"),
    "MD_L3":  ("D",  3, "mnist", "digital protocol, Eq. (recirc)"),
    "MF_L3":  ("F",  3, "mnist", "floating-point reference"),
    "MO_L3":  ("O",  3, "mnist", "linear-cascade control"),
    "MHa_L3": ("Ha", 3, "mnist", "passive cascade, no activation"),
    "MZH_L2": ("ZH", 2, "mnist", "analog chain, L = 2"),
    "MZT_L2d": ("ZT", 2, "mnist", "digital execution, L = 2"),
    "MD_L2":  ("D",  2, "mnist", "digital protocol, L = 2"),
    "MZH_L4": ("ZH", 4, "mnist", "analog chain, L = 4 (out of budget)"),
    "MZT_L4d": ("ZT", 4, "mnist", "digital execution, L = 4"),
    # GTSRB, 3072 -> 128 (-> 128) -> 43
    "ZH_L3":  ("ZH", 3, "gtsrb", "hardware-aware weights, analog chain"),
    "ZT_L3d": ("ZT", 3, "gtsrb", "the same weights, digital execution"),
    "D_L3":   ("D",  3, "gtsrb", "digital protocol"),
    "H_L3":   ("H",  3, "gtsrb", "passive cascade, no activation"),
    "O_L3":   ("O",  3, "gtsrb", "linear-cascade control"),
    "ZH_L2":  ("ZH", 2, "gtsrb", "analog chain, L = 2"),
    "ZT_L2d": ("ZT", 2, "gtsrb", "digital execution, L = 2"),
    "D_L2":   ("D",  2, "gtsrb", "digital protocol, L = 2"),
}


# --------------------------------------------------------------- budget ----
def budget():
    """Level plan and per-pass SNR of the ultra-low-power chain (Table/Fig.)."""
    print(f"per-pass loss   : {cs.CL_PASS:.2f} dB "
          f"(mixer {cs.CL_MIXER if hasattr(cs, 'CL_MIXER') else 6.65} + "
          f"filter 0.82)")
    print(f"passive boost   : {cs.BOOST_DB:.1f} dB   "
          f"(A_v^2 = R_in / 2R_s, R_in <= Q/wC_in)")
    print(f"detector k_ED   : {cs.K_ED:.1f} /V")
    print(f"noise floor P_n : {cs.dbm(cs.P_N) if hasattr(cs, 'P_N') else -73.6}"
          " dBm per bin\n")
    out = {}
    for ds, dims_key in (("mnist", "MNIST"), ("gtsrb", "GTSRB")):
        dims = cs.layer_dims(4, ds)
        powers, drives = cs.plan_Z(4, dims, -10.0)
        rows = []
        for l, (p, D) in enumerate(zip(powers, dims[:4]), start=1):
            rows.append(dict(pass_index=l, dim_in=int(D),
                             drive_dBm=round(cs.dbm(p), 2),
                             tone_snr_dB=round(cs.snr_tone_db(p, D), 2)))
            print(f"{dims_key} pass {l}: D = {D:5d}   "
                  f"{cs.dbm(p):7.1f} dBm   SNR {cs.snr_tone_db(p, D):6.1f} dB")
        eff = [powers[i + 1] / (powers[i] * 10 ** (-cs.CL_PASS / 10))
               for i in range(3)]
        print(f"{dims_key} block efficiency: "
              + ", ".join(f"{100 * e:.2f}%" for e in eff) + "\n")
        out[dims_key] = dict(rows=rows,
                             block_efficiency=[round(100 * e, 3) for e in eff])
    out["settings"] = dict(cl_pass_dB=cs.CL_PASS, boost_dB=cs.BOOST_DB,
                           k_ED_per_V=cs.K_ED, R_v_ohm=cs.R_V,
                           step_down_loss_dB=cs.IL_STEP, first_pass_dBm=-10.0)
    json.dump(out, open(RESULTS / "link_budget.json", "w"), indent=1)
    print("wrote results/link_budget.json")


# ---------------------------------------------------------------- infer ----
def infer(repeats=5, only=None):
    """Load the stored weights and run single-shot noisy inference.

    Every pass of the analog chain is stochastic (link noise, element jitter),
    so each evaluation is one realization; we report mean +- std over
    `repeats` independent noise draws on the full test set."""
    rows, summary = [], {}
    for tag, (mode, L, ds, what) in RUNS.items():
        if only and not tag.startswith(only):
            continue
        cfg = cs.cfg_make(mode, L, ds)
        Xtr, ytr, Xte, yte = (cs.load_mnist() if ds == "mnist"
                              else cs.load_data())
        params0 = cs.init_params(cfg, jax.random.PRNGKey(0))
        wtag = tag[:-1] if tag.endswith("d") else tag        # strip the
        wtag = wtag.replace("ZT_", "ZH_")   # digital twin shares ZH weights
        loaded = cs._ckpt_load(wtag, params0)
        if loaded is None:
            print(f"[{tag}] no weights found -- skipped")
            continue
        params = loaded[0]
        _, eval_logits = cs.make_train_fns(cfg)
        t0 = time.time()
        accs = [float(cs.evaluate(params, cfg, eval_logits, Xte, yte,
                                  jax.random.PRNGKey(1000 + s)))
                for s in range(repeats)]
        dt = time.time() - t0
        m, sd = 100 * np.mean(accs), 100 * np.std(accs)
        print(f"[{tag}] {ds.upper():5s} L={L}  acc = {m:6.2f} +- {sd:.2f} %   "
              f"({len(Xte)} images x {repeats} draws, {dt:.1f} s)  -- {what}")
        rows.append(dict(tag=tag, weights=wtag, dataset=ds, mode=mode, L=L,
                         dims="-".join(str(d) for d in cfg["dims"]),
                         accuracy_pct=round(m, 2), std_pct=round(sd, 2),
                         repeats=repeats, n_test=int(len(Xte)),
                         description=what))
        summary[tag] = rows[-1]
    if only:
        old = RESULTS / "results_summary.json"
        if old.exists():
            merged = json.load(open(old)); merged.update(summary)
            summary = merged
            rows = list(summary.values())
    json.dump(summary, open(RESULTS / "results_summary.json", "w"), indent=1)
    with open(RESULTS / "results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote results/results_summary.json and .csv")


# ---------------------------------------------------------------- train ----
def ablation(repeats=5):
    """Is hardware-aware training necessary?  Deploy three checkpoints on the
    same analog chain: one trained with the hardware in the loop, one trained
    noise-free, one trained against an idealized square-law activation."""
    jobs = [("MNIST  hardware-aware",     "MZ_L3",         "mnist", "ZD"),
            ("MNIST  no-hw, noise-free",  "MZD_L3_nohw",   "mnist", "ZD"),
            ("MNIST  no-hw, ideal square", "MZI_L3_ideal", "mnist", "ZI"),
            ("GTSRB  hardware-aware",     "Z_L3",          "gtsrb", "ZD"),
            ("GTSRB  no-hw, noise-free",  "ZD_L3_nohw",    "gtsrb", "ZD"),
            ("GTSRB  no-hw, ideal square", "ZI_L3_ideal",  "gtsrb", "ZI")]
    out = {}
    for label, wtag, ds, dmode in jobs:
        Xte, yte = (cs.load_mnist() if ds == "mnist" else cs.load_data())[2:4]

        def run(mode, n):
            cfg = cs.cfg_make(mode, 3, ds)
            p0 = cs.init_params(cfg, jax.random.PRNGKey(0))
            got = cs._ckpt_load(wtag, p0)
            if got is None:
                return None
            _, ev_fn = cs.make_train_fns(cfg)
            a = [float(cs.evaluate(got[0], cfg, ev_fn, Xte, yte,
                                   jax.random.PRNGKey(1000 + s)))
                 for s in range(n)]
            return 100 * np.mean(a), 100 * np.std(a)

        dig, ana = run(dmode, 1), run("Z", repeats)
        if dig is None or ana is None:
            print(f"[{wtag}] weights missing -- skipped"); continue
        print(f"{label:26s} digital {dig[0]:6.2f}   "
              f"analog {ana[0]:6.2f} +- {ana[1]:.2f}   "
              f"drop {dig[0] - ana[0]:5.2f}")
        out[wtag] = dict(label=label, dataset=ds,
                         digital_pct=round(dig[0], 2),
                         analog_pct=round(ana[0], 2),
                         analog_std=round(ana[1], 2),
                         drop_pts=round(dig[0] - ana[0], 2))
    json.dump(out, open(RESULTS / "hardware_aware_ablation.json", "w"),
              indent=1)
    print("wrote results/hardware_aware_ablation.json")


def train_nohw():
    """Ablation: train WITHOUT the hardware in the loop, then deploy.

    two levels of 'no hardware-aware training'
      _nohw   : exact arithmetic, no link noise and no element jitter, but the
                network keeps the activation the physical block implements
                (the measured detector transfer)            -> mode ZD
      _ideal  : the hardware is absent from training altogether, the
                activation is an idealized square law       -> mode ZI
    """
    jobs = [("mnist", "ZD", 3, "_nohw", "MF_L3", 14, 3e-3),
            ("mnist", "ZI", 3, "_ideal", "MF_L3", 14, 3e-3),
            ("gtsrb", "ZD", 3, "_nohw", "F_L3", 12, 3e-3),
            ("gtsrb", "ZI", 3, "_ideal", "F_L3", 12, 3e-3)]
    for ds, mode, L, suf, init, eps, lr in jobs:
        cfg = cs.cfg_make(mode, L, ds, suffix=suf)
        print(f"--- training {cfg['tag']} (no hardware in the loop) ---")
        cs.run_train(cfg, epochs=eps, lr=lr, init_from=init, budget_s=10_000)


def train():
    """Re-train everything reported (weights land in weights/)."""
    plan = [("mnist", "A"), ("mnist", "C"), ("mnist", "D"), ("mnist", "O2"),
            ("mnist", "Z"), ("gtsrb", "A"), ("gtsrb", "B"), ("gtsrb", "C"),
            ("gtsrb", "Z"), ("gtsrb", "O")]
    for ds, grp in plan:
        table = cs.GROUPS_M if ds == "mnist" else (
            cs.GROUPS_ZG if grp in cs.GROUPS_ZG else cs.GROUPS)
        if grp not in table:
            continue
        for mode, L, init, eps, lr in table[grp]:
            cfg = cs.cfg_make(mode, L, ds)
            print(f"--- training {cfg['tag']} ({ds}, {eps} epochs) ---")
            cs.run_train(cfg, epochs=eps, lr=lr, init_from=init,
                         budget_s=10_000)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "infer"
    if what in ("budget", "all"):
        budget()
    if what in ("train", "all"):
        train()
    if what == "train_nohw":
        train_nohw()
    if what in ("ablation", "all"):
        ablation()
    if what in ("infer", "all"):
        only = sys.argv[2] if len(sys.argv) > 2 else None
        infer(only=only)
