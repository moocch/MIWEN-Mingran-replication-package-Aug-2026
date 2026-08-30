#!/usr/bin/env python3
"""Fit the per-neuron per-layer decode-side gain calibration from a measured run.

Root cause addressed (validated 2026-07-21): each output neuron j of layer t has
a STABLE multiplicative gain g_j (static ripple ~ +/-10-20%) between the analog
MVM output and the digital reference. Removing it at decode time recovers
~2.4-3.1 points of GTSRB accuracy (87.3% -> ~89.7%, ~90.4% with --debias-act).

Fit (least squares per output column j of layer t):

    g_j = sum_i(am_ij * ar_ij) / sum_i(ar_ij**2)

where am = y_mag_l{t} (measured amplitudes) and ar = |y_ref_l{t}| (digital
reference) from the run npz. Gains are then normalized to mean 1 per layer so
the calibration only removes the ripple and preserves the global scale.

Usage:
    python3 make_calibration.py gtsrb_confusion_mlpN.npz [--out gtsrb_calib.npz]

The output npz holds g_l1..g_lN plus provenance (source npz, date, formula).
Apply it on a hardware/dry-run inference with:
    python3 4_gtsrb_confusion_mlpN.py --calib-npz gtsrb_calib.npz ...
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FORMULA = ("g_j = sum_i(am_ij*ar_ij)/sum_i(ar_ij^2) with am=y_mag_l{t}, "
           "ar=|y_ref_l{t}|; normalized to mean 1 per layer "
           "(decode-side: divide measured amplitudes by g_j)")


def fit_layer_gains(am: np.ndarray, ar: np.ndarray) -> np.ndarray:
    """Per-column least-squares gain: g_j = sum_i(am_ij*ar_ij)/sum_i(ar_ij^2).

    am: measured amplitudes (n_samples, H); ar: |digital reference| (same shape).
    Columns with (near-)zero reference energy get gain nan -> caller decides;
    here we raise, because a dead reference column cannot be calibrated.
    """
    am = np.asarray(am, np.float64)
    ar = np.asarray(ar, np.float64)
    if am.shape != ar.shape or am.ndim != 2:
        raise ValueError(f"shape mismatch: am{am.shape} vs ar{ar.shape}")
    if not np.all(np.isfinite(am)):
        raise ValueError("measured amplitudes (am) contain NaN/Inf; refusing "
                         "to fit a calibration from poisoned data")
    if not np.all(np.isfinite(ar)):
        raise ValueError("digital reference (ar) contains NaN/Inf; refusing "
                         "to fit a calibration from poisoned data")
    den = np.sum(ar * ar, axis=0)
    # Relative dead-column threshold: a column whose reference energy is >~6
    # orders of magnitude below the strongest column carries no usable signal
    # to fit against. The LS gain would be numerically meaningless noise
    # amplification which, after mean-1 normalization, looks perfectly valid
    # (strictly positive, mean 1) yet silently hard-zeroes that neuron/class
    # at decode (amplitudes /g, final powers /g^2). 1e-6 in energy = 1e-3 in
    # amplitude — real per-layer ratios in the committed GTSRB run are >=0.12,
    # so any live neuron clears this by 5 orders of magnitude.
    den_max = float(den.max()) if den.size else 0.0
    if den_max <= 0.0 or np.any(den < 1e-6 * den_max):
        raise ValueError("reference has zero-energy (or near-zero-energy) "
                         "output column(s); cannot fit a gain for a dead "
                         "neuron")
    g = np.sum(am * ar, axis=0) / den
    if not np.all(np.isfinite(g)):
        raise ValueError("fitted gains contain NaN/Inf; measured data does "
                         "not look like a gained copy of the reference")
    if np.any(g <= 0):
        raise ValueError("fitted non-positive gain; measured data does not "
                         "look like a gained copy of the reference")
    return g


def normalize_mean1(g: np.ndarray) -> np.ndarray:
    """Normalize gains to mean 1 (preserve the layer's global scale)."""
    g = np.asarray(g, np.float64)
    return g / g.mean()


# Chain parameters recorded from the source run's meta_json into the calib
# npz as a fingerprint; the inference-script loader warns if they differ from
# the current run's args (a calibration fitted under one chain configuration
# need not apply to another).
FINGERPRINT_KEYS = ("fft_len", "k0", "bw_frac", "k_per_symbol",
                    "p_rf_dbm", "p_lo_dbm")


def fit_calibration(run_npz_path, allow_calibrated_source: bool = False) -> dict:
    """Fit normalized per-neuron gains for every layer of a measured run npz.

    Returns {'g_l1': ..., 'g_lN': ..., 'n_layers': N} (gains mean-1 per
    layer), plus 'fingerprint_json' when the source run's meta_json records
    the chain parameters (FINGERPRINT_KEYS).

    Refuses a source run that was itself recorded WITH --calib-npz (its
    meta_json has calib_npz set): fitting a calibration from an already-
    calibrated run stacks corrections. Pass allow_calibrated_source=True
    (CLI: --allow-calibrated-source) to override deliberately.
    """
    d = np.load(run_npz_path, allow_pickle=True)
    meta = {}
    if "meta_json" in d.files:
        try:
            meta = json.loads(str(d["meta_json"]))
        except (ValueError, TypeError):
            meta = {}
    if meta.get("calib_npz") and not allow_calibrated_source:
        raise ValueError(
            f"{run_npz_path}: source run was recorded WITH "
            f"--calib-npz={meta['calib_npz']!r}; fitting a calibration from "
            f"an already-calibrated run is forbidden (pass "
            f"--allow-calibrated-source to override deliberately)")
    if "n_layers" in d.files:
        n_layers = int(d["n_layers"])
    else:
        n_layers = 0
        while (f"y_mag_l{n_layers + 1}" in d.files
               and f"y_ref_l{n_layers + 1}" in d.files):
            n_layers += 1
        if n_layers == 0:
            raise KeyError(f"{run_npz_path}: no n_layers key and no "
                           f"y_mag_l*/y_ref_l* pairs; not a per-layer "
                           f"measured run file")
        print(f"[calib] n_layers missing in {run_npz_path}; inferred "
              f"{n_layers} from y_mag_l*/y_ref_l* keys")
    out = {"n_layers": n_layers}
    for t in range(1, n_layers + 1):
        am_key, ar_key = f"y_mag_l{t}", f"y_ref_l{t}"
        if am_key not in d.files or ar_key not in d.files:
            raise KeyError(f"run npz lacks {am_key}/{ar_key}; not a per-layer "
                           f"measured run file")
        am = d[am_key].astype(np.float64)
        ar = np.abs(d[ar_key].astype(np.complex128))
        out[f"g_l{t}"] = normalize_mean1(fit_layer_gains(am, ar))
    fingerprint = {k: meta[k] for k in FINGERPRINT_KEYS if k in meta}
    if fingerprint:
        out["fingerprint_json"] = json.dumps(fingerprint)
    return out


def write_calibration(gains: dict, out_path, source_npz) -> None:
    """Write gains + provenance to out_path (npz)."""
    save = dict(gains)
    save["source_npz"] = str(source_npz)
    save["created"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save["formula"] = FORMULA
    np.savez_compressed(out_path, **save)


def load_calibration(path) -> list:
    """Load a calibration npz -> [g_l1, ..., g_lN] (validated: positive, mean 1)."""
    z = np.load(path, allow_pickle=True)
    if "n_layers" not in z.files:
        raise ValueError(f"{path}: missing n_layers; not a calibration file")
    n_layers = int(z["n_layers"])
    gains = []
    for t in range(1, n_layers + 1):
        key = f"g_l{t}"
        if key not in z.files:
            raise ValueError(f"{path}: missing {key}")
        g = np.asarray(z[key], np.float64)
        if g.ndim != 1 or np.any(g <= 0) or not np.isclose(g.mean(), 1.0,
                                                           atol=1e-6):
            raise ValueError(f"{path}: {key} invalid (must be 1-D, positive, "
                             f"mean 1; got mean {g.mean():.6f})")
        gains.append(g)
    return gains


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_npz", type=str,
                   help="measured run npz (with y_mag_l{t} / y_ref_l{t})")
    p.add_argument("--out", type=str, default="gtsrb_calib.npz",
                   help="output calibration npz (default: gtsrb_calib.npz)")
    p.add_argument("--allow-calibrated-source", action="store_true",
                   help="override the refusal to fit from a run that was "
                        "itself recorded with --calib-npz")
    p.add_argument("--force", action="store_true",
                   help="overwrite --out if it already exists (default: "
                        "refuse, to avoid silently clobbering a committed "
                        "calibration)")
    args = p.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"[error] {out_path} already exists; refusing to overwrite "
              f"(pass --force to replace it)")
        return 2

    run_path = Path(args.run_npz)
    gains = fit_calibration(
        run_path, allow_calibrated_source=args.allow_calibrated_source)
    n_layers = gains["n_layers"]
    write_calibration(gains, args.out, run_path.name)
    print(f"[calib] fitted {n_layers} layers from {run_path}")
    for t in range(1, n_layers + 1):
        g = gains[f"g_l{t}"]
        print(f"[calib]   g_l{t}: H={g.size:>4d} ripple "
              f"{g.min():.3f}..{g.max():.3f} (mean 1)")
    print(f"[calib] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
