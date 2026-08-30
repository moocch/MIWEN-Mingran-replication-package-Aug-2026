# Fig. 4e — Confusion matrices on hardware: clean-trained vs hardware-aware

43 × 43, N = 600 identical photographs through the identical chain at the
$(0,0)$-dBm time-serial operating point: the clean-trained network collapses to
a constant classifier (590/600 in one class, 5.7 ± 0.9 %); the hardware-aware
network classifies 591/600 = exactly its pinned digital comparator
(98.5 ± 0.5 %).

**Figure restructure note (2026-08-28/29):** the manuscript's Fig. 4 dropped the
former panel e (the 2 × 2 outcome table) — its four numbers (clean 99.50 %
digital / 5.67 ± 0.94 % hardware; twin-trained 98.50 % digital /
98.50 ± 0.50 % hardware) are now quoted in the Results TEXT — and the former
panel f (these confusion matrices) became panel e. This folder therefore
carries the complete serial-campaign chain, which backs both the displayed
confusion matrices and the text-quoted 2 × 2 numbers (same runs, same files).

## Chain in this folder

- `measured_chunk_predictions/` — the 7 raw chunk-prediction npz of the two
  arms (5 clean + 2 twin), the direct source of both matrices.
- `upstream_hardware_test_code/` — the deployed test driver
  `serial_nn_runner.py` (POWER = (0,0) dBm, fail-closed gates, incremental
  chunk saving) with its imports `miwen_serial_frozen_reference.py`,
  `miwen_frozen_reference.py`, `sync_v2.py`, `check_rig_free.py`. (It also
  imports `4_gtsrb_confusion_mlpN.py`, archived in
  `../b/upstream_comb_campaign/share_20260712/`; the full GTSRB cache it
  reads is rebuilt automatically from the public mirror.)
- `upstream_hardware_runs/` (= `05_hardware_result_0dBm_N600`) — the raw
  measured results: the same chunk-prediction npz in their original run
  folders, the two frozen per-layer calibrations
  (`serial_cal_*_20260823.npz`, fitted once on 10 disjoint images,
  teacher-forced, reloaded frozen), and the 7 session logs whose FINAL lines
  carry the per-session accuracies and the per-layer relRMSE calibration
  blocks.
- `upstream_pinned_comparators/` (= `06_digital_comparators`) —
  `serial_predictions.json` (segment-1 pins) + `digital_pins_seg300600.npz`
  (per-image segment-2 pins): the pinned 98.50 % digital comparator and the
  0.8 % collapse forecast (verified 2/300 + 3/300 = 5/600 = 0.83 %).
- `upstream_prereg_audit/` (= `00_report_and_audit`) — the pre-registration
  prespec (2026-08-24) and the campaign audit report.
- `upstream_frozen_inputs/` (= `08_frozen_inputs_and_labels`) — frozen battery
  indices + labels + `s4_random450_idx.npy` (calibration frames).
- `verify_accuracy.py` — the campaign-native, numpy-only recompute of all four
  2 × 2 cells from the chunk files; `README_serial_campaign.md` — the
  campaign's own file map and reporting conventions.

Every displayed and text-quoted number of this campaign is recomputed and
asserted by `../fig4_verify_package.py` into `../data/fig4_serial_results.npz`
/ `../data/fig4_numbers.json`, and the McNemar pairings and ENOB/ladder
conventions were independently re-verified from the raw archives on
2026-08-28.
