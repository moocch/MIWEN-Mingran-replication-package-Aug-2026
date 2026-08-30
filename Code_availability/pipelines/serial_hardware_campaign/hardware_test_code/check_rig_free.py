#!/usr/bin/env python3
"""Rig occupancy check: refuse to transmit if the USRPs may be in use.

Standing safety rule (Jonathan, 2026-07-31): EVERY hardware invocation must
verify the rig is free first. Both runners (4_gtsrb_confusion_mlpN.py and
run_ladder_hw.py) call ``rig_free_or_die()`` before constructing the hardware
session; dry-runs are exempt. This is an operational safety addition -- it
touches no pre-registered threshold.

Checks, fail-closed:
  1. Other processes whose command line references the UHD/USRP stack or the
     known experiment entry points (excluding this process and its ancestors).
  2. Claim files (``IN_USE`` / ``LOCK`` / ``RUNNING``) in the conventional
     workdirs (any ``~/miwen_hwrun*``, the script directory, and the CWD).
     A ``STOP`` file is a halt signal, not a claim, and is reported but not
     treated as occupancy.

Cooperative claiming for multi-step sessions (issue #50 disclosure-2 design):
    python3 check_rig_free.py --claim    # writes IN_USE (+ session token) and
                                         # prints:  export RIG_SESSION_TOKEN=...
    export RIG_SESSION_TOKEN=<token>     # in the session shell
    ... runners now pass the gate: a claim whose token matches the
    RIG_SESSION_TOKEN environment variable is YOUR claim (warned, not
    blocking); any other claim or foreign process still fails closed ...
    python3 check_rig_free.py --release
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
import getpass
import json
import os
import sys
import time
from pathlib import Path

# NOTE: "usrp" is deliberately NOT a pattern -- the rig box's hostname is
# usrp-contol, which system daemons (avahi mDNS) embed in their process
# titles, making the gate unpassable there (issue #50, 2026-08-01 false
# positive). Real rig users are caught by the UHD/GNU Radio stack names and
# the enumerated experiment entry points.
PATTERNS = ("uhd", "gnuradio", "4_gtsrb_confusion", "run_ladder_hw",
            "hil_finetune", "scalar_sweep", "make_calibration")
CLAIM_NAMES = ("IN_USE", "LOCK", "RUNNING")


def cmdline_matches(cmd: str) -> bool:
    """Pure matcher (unit-tested): does this command line look like rig use?"""
    low = cmd.lower()
    return any(pat in low for pat in PATTERNS)


def _ancestors():
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().split(")")[-1].split()[1])
        except Exception:
            break
    return pids


def busy_processes():
    mine = _ancestors()
    hits = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) in mine:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace").strip()
        except Exception:
            continue
        if cmdline_matches(cmd):
            hits.append((int(p.name), cmd[:120]))
    return hits


def claim_files(workdirs=None):
    dirs = list(workdirs or [])
    dirs += [Path.cwd(), _data_dir(__file__)]
    dirs += sorted(Path.home().glob("miwen_hwrun*"))
    seen, hits, stops = set(), [], []
    for d in dirs:
        d = Path(d).resolve()
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        for name in CLAIM_NAMES:
            if (d / name).exists():
                hits.append(str(d / name))
        if (d / "STOP").exists():
            stops.append(str(d / "STOP"))
    return hits, stops


def _own_claim(path):
    """A claim is OURS iff its token matches $RIG_SESSION_TOKEN (non-empty)."""
    tok = os.environ.get("RIG_SESSION_TOKEN", "")
    if not tok:
        return False
    try:
        return json.loads(Path(path).read_text()).get("token") == tok
    except Exception:
        return False


def rig_free_or_die(workdirs=None, verbose=True):
    procs = busy_processes()
    claims, stops = claim_files(workdirs)
    own = [c for c in claims if _own_claim(c)]
    claims = [c for c in claims if c not in own]
    if own and verbose:
        print(f"[rig-check] own session claim honored (token match): {own}")
    if stops and verbose:
        print(f"[rig-check] note: STOP file(s) present (halt signal, not a "
              f"claim): {stops}")
    if procs or claims:
        print("[rig-check] RIG APPEARS IN USE -- refusing to transmit.")
        for pid, cmd in procs:
            print(f"[rig-check]   process {pid}: {cmd}")
        for c in claims:
            print(f"[rig-check]   claim file: {c}")
        print("[rig-check] If this is stale, remove the claim file / stop the "
              "process, or coordinate with its owner. No override flag exists "
              "by design.")
        sys.exit(1)
    if verbose:
        print("[rig-check] rig appears free (no UHD/experiment processes, no "
              "claim files).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--release", action="store_true")
    ap.add_argument("--workdir", action="append", default=[])
    args = ap.parse_args(argv)
    tgt = Path.cwd() / "IN_USE"
    if args.release:
        if tgt.exists():
            tgt.unlink()
            print(f"[rig-check] released {tgt}")
        return 0
    if args.claim:
        rig_free_or_die(args.workdir)
        import secrets
        tok = secrets.token_hex(8)
        tgt.write_text(json.dumps({"pid": os.getpid(),
                                   "user": getpass.getuser(),
                                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         time.gmtime()),
                                   "token": tok,
                                   "purpose": "MIWEN session"}))
        print(f"[rig-check] claimed {tgt}")
        print(f"export RIG_SESSION_TOKEN={tok}")
        return 0
    rig_free_or_die(args.workdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
