# Fig. 1c — computing accuracy vs client energy per MAC (simulation)

- Figure: bottom panel of `fig1b_motivation.png` (top = Fig. 1b; same script).
- Simulation code: `fig1b_motivation.py`, section "(c) computing-accuracy window":
  accuracy = −log2(RMSE/2) of a normalized N = 4096 inner product;
  noise term 1/√(27·SNR_tone) with the calibrated floor P_n split over the tones;
  distortion term P/P3 with the calibrated compression point shifted by the
  multi-tone PAPR (−12 dB); E/MAC = P/B at B = 25 MHz.
- Inputs: same calibrated parameters as Fig. 1b (no external data files).
  Calibration chain copy: `upstream_scalar_PIML_calibration\` (see `..\b\README.md`).
