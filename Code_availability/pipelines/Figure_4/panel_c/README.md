# Fig. 4c — Confusion matrices: measured comb cascade vs clean-trained digital

43 × 43, N = 1,200. The measured matrix comes from the archived frozen-rerun predictions
(`../b/upstream_comb_campaign/share_20260712/battery_frozen_slim.npz`); the clean-digital matrix
is recomputed live by `miwen_frozen_reference.forward_digital` with the clean checkpoint on the
identical battery. Both are built (and gate-checked) by `../prep_comb_assets.py` into
`../data/comb_assets.npz` (keys `conf_measured`, `conf_clean_digital`) and drawn by `../fig4_v6.py`.

This folder additionally carries the campaign-native rendering of the same measured matrix as a
result-figure record: `battery_confusion.py` + `battery_confusion_20260805.png`.

Full campaign chain: see `../b/README.md`.
