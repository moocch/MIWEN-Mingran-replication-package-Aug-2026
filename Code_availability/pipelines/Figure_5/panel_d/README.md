# Fig. 5d — The same hardware-aware weights executed digitally and through the chain

**What is drawn (mean ± 1 s.d. over five noise realizations):**
MNIST L3: digital 94.30 (tag MZT_L3d) vs chain 93.90 ± 0.10 (MZH_L3);
GTSRB L3: digital 88.83 (ZT_L3d) vs chain 87.53 ± 0.14 (ZH_L3).
One hardware-aware checkpoint per task, executed twice — digital = mode ZT (device twin
retained, noise stripped), chain = mode ZH; the ZT bars reuse the ZH checkpoints.

- **Generator**: `reproduce.py infer` (this folder) — 5 draws, `PRNGKey(1000+s)`, full test
  sets (10,000 MNIST / 12,630 GTSRB images); writes `results/results_summary.json`.
- **Checkpoints**: `../simulation/weights/ckpt_MZH_L3.npz` (MNIST 784-100-64-10) and
  `ckpt_ZH_L3.npz` (GTSRB 3072-128-128-43), training provenance in
  `../simulation/weights/train_summary.json`; produced by `reproduce.py train`
  (`comb_analog_sim.py` cfg_make/run_train).
- **Datasets**: `../simulation/data/mnist.npz` and `../simulation/data/gtsrb_roi_32x32.npz`
  (the GTSRB cache is rebuildable from the official archives via
  `../simulation/prepare_gtsrb.py`).
- **Plot data**: `results_summary.json` (this folder) — all four displayed accuracies
  gate-asserted with zero tolerance by the figure script; byte-identical to the archive
  original in `../simulation/results/`.
