# Supplementary Table 1 — scalar PIML-twin parameters (N = 1)

**What is backed:** Supplementary Table 1 of `Supplementary_Information.tex`
(label `tab:scalartwin`, "The seven trainable parameters of the physics block and its
one structural choice") and the byte-for-byte identical red table in the response
letter (`Main.tex`), plus the quoted text numbers: 3.94 dB
physics-only RMSE, 1.30 dB full-twin RMSE, R² = 0.9969, 90.4% of the surface
within ±2 dB. The same chain also backs the letter's quoted "datasheet" values
(conversion loss 1.84 dB, LO knee +1.73 dBm, turn-on exponent 1.045, RF 1-dB
compression +4.47 dBm, port leakage −84.3 dB, noise floor −73.6 dBm, 1,849
operating points, 77 dB dynamic range, 67% within ±1 dB).

## The chain (all in `upstream_scalar_PIML_calibration/`)

1. **Acquisition — 43 × 43 sweep** (`acquisition_code_43x43_sweep/`:
   `1_heatmap_single_scalar.py`, `usrp_mixer_heatmap.py`, `config.py`).
   USRP dual-channel TX drives the diode-ring mixer over a 43 × 43 grid of
   LO × RF powers (2-dB steps), 3 repeated sweeps per point.
2. **Measured surface** — `heatmap_2db.npz` (43 × 43 grid = 1,849 points;
   `if_amp_uv_all` holds the 3 repeats per point; `meta_json` holds run metadata).
3. **Three-stage PIML calibration** (`Code_PIML_scalar_local_meeting/`:
   `model/piml_mixer_digital_twin.py` + `piml_data.py`; `scripts/` are the
   diagnostic-figure plotters): (i) 24-start L-BFGS fit of the 7-parameter
   physics block, (ii) 6,000 Adam iterations of the bounded residual MLP
   (2→32→32→1, |δ| ≤ 8 dB) with physics frozen, (iii) 4,000-iteration joint
   fine-tune. Loss = MSE of log₁₀P_IF; regularizers 10⁻⁴ / 10⁻³ / 10⁻².
4. **Training-run record** — `twin_training_run_PIML_Digital_Twin_2/summary.json`
   (calibrated `physics_params` + `metrics_db` + full training `config`) and
   `piml_mixer.pt` (trained weights).
5. **Exported surfaces** — `twin_predictions.npz` (measured, physics-only and
   full-twin surfaces + the NN correction `delta_db`), from which R² and the
   within-±2-dB fraction are recomputed (the training run did not store them
   in `summary.json`).
6. **Table / text** — every displayed value is the `summary.json` value rounded
   to the printed number of decimals.

## Verification

Run `python verify_table1.py` (requires numpy). It asserts all 8 table rows and
the 4 text numbers, rounded exactly as displayed; output archived in
`verify_table1_output.txt` (all 14 checks PASS, 2026-08-29).

## Value map (displayed → source key in `twin_training_run_PIML_Digital_Twin_2/summary.json`)

| Displayed | Source key | Full precision |
|---|---|---|
| G_max,dB = −1.84 dB | `physics_params.G_max_dB` | −1.8437995910644531 |
| P_LO,sat = +1.73 dBm | `physics_params.P_LO_sat_dBm` | 1.7346782684326172 |
| β_LO = 1.045 | `physics_params.beta_LO` | 1.0452940464019775 |
| P_RF,comp = +4.47 dBm | `physics_params.P_RF_comp_dBm` | 4.474578857421875 |
| β_RF = 2.13 | `physics_params.beta_RF` | 2.132936477661133 |
| P_noise = −73.62 dBm | `physics_params.P_noise_dBm` | −73.62084197998047 |
| leak_dB = −84.25 dB | `physics_params.leak_dB` | −84.25154113769531 |
| turn-on shape = Weibull | `physics_params.lo_shape` | `"weibull"` |
| 3.94 dB physics-only | `metrics_db.rmse_physics_only` | 3.9442319490303217 |
| 1.30 dB full twin | `metrics_db.rmse_physics_plus_nn` | 1.3037162225485324 |
| R² = 0.9969 | recomputed from `twin_predictions.npz` (dB domain) | 0.99686996… |
| 90.4% within ±2 dB | recomputed from `twin_predictions.npz` | 90.42725…% |

## Provenance

Copied unmodified (2026-08-29) from the read-only source
`Response_Letter_Reproductivity\Reviewer1_Comment6\a_c6_energy_scaling\upstream_scalar_PIML_calibration\`
(the same chain the energy-scaling response artifact builds on; identical copies
are audited in `fig2\a`, `fig3\c` and `fig1\b`).

MD5 checksums of the load-bearing files:

| File | MD5 |
|---|---|
| `twin_training_run_PIML_Digital_Twin_2/summary.json` | `2042edd0e22a3587fc5979b569e4b2dd` |
| `twin_training_run_PIML_Digital_Twin_2/piml_mixer.pt` | `9d77cc2a1436916b770dc79c4697379e` |
| `heatmap_2db.npz` | `d81ac7e11647e8a20622198a7c9276e1` |
| `twin_predictions.npz` | `c9dbad5ab20944de17949961e4ac7cef` |
