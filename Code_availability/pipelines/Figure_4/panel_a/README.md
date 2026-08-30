# Fig. 4a — In-physics CNN inference protocol

Every pictorial element is real data, not clip-art:

- **photo**: battery image k = 34 (a STOP sign) from the frozen 1,200-image battery —
  `battery_random1200_idx.npy` (this folder) indexes the rebuilt official GTSRB test cache
  `../data/gtsrb_roi_32x32_test.npz`; the k = 34 choice resolves through
  `battery_idx600` in `../data/fig4_panel_assets.npz`.
- **5×5×3 patch**: max-variance window of that photo.
- **conv-1 kernel tiles**: |w| of channel 0 of the clean checkpoint `r35_r3plus_s0_hw.npz`
  (this folder).
- **layer-1 feature maps**: top-variance channels [20, 29, 11], computed by
  `../prep_comb_assets.py` through the frozen forward `miwen_frozen_reference.py` (this folder).

All assets land in `../data/comb_assets.npz` (keys `photo_u8`, `patch_u8`, `patch_rc`,
`kern_tiles`, `featmaps`) and are drawn by `../fig4_v6.py`. Re-verified by execution 2026-08-28
(gates passed; regenerated npz array-identical).
