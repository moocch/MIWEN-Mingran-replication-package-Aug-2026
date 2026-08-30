"""Regenerates the frozen evaluation frames from their documented seeds and
verifies byte-equality with the committed .npy files (audit check).

Frame construction (pre-specs 2026-08-04): pool = official-test indices
900..12629 (images 0-899 were consumed by earlier arms). random-450: seed
20260804, sorted draw. battery-1200: seed 20260805, sorted draw from the
pool minus the 450."""
import numpy as np

pool_450 = np.arange(900, 12630)
sel_450 = np.sort(np.random.RandomState(20260804).choice(pool_450, 450,
                                                         replace=False))
used = set(sel_450.tolist())
pool_1200 = np.array([i for i in range(900, 12630) if i not in used])
sel_1200 = np.sort(np.random.RandomState(20260805).choice(pool_1200, 1200,
                                                          replace=False))
for name, sel in (("s4_random450_idx.npy", sel_450),
                  ("battery_random1200_idx.npy", sel_1200)):
    committed = np.load(name)
    assert (committed == sel).all(), f"{name}: MISMATCH"
    print(f"{name}: regenerated == committed ({len(sel)} indices) OK")
print("FRAMES VERIFIED")
