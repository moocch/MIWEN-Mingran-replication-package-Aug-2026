import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

IN_NPZ  = sys.argv[1] if len(sys.argv) > 1 else "gr_usrp_mixer_vector_heatmap_N4096.npz"
OUT_PNG = sys.argv[2] if len(sys.argv) > 2 else "gr_usrp_mixer_vector_heatmap_N4096.png"

R_OHM = 50.0

d    = np.load(IN_NPZ, allow_pickle=True)
meta = json.loads(str(d["meta_json"]))

lo = np.asarray(d["p_lo_dbm_grid"], float)
rf = np.asarray(d["p_rf_dbm_grid"], float)

ip2 = float(np.abs(np.vdot(d["vec_a"], d["vec_b"])) ** 2)

uv_scale = float(meta.get("rx_uv_scale", 1e6))

LO, RF = np.meshgrid(lo, rf)
P_center_dbm = LO + RF + 10.0 * np.log10(ip2)
ideal_W = 10.0 ** ((P_center_dbm - 30.0) / 10.0)

amp_fc32  = np.asarray(d["if_amp_uv_mean"], float) / uv_scale
measured  = (amp_fc32 ** 2) / R_OHM
measured  = measured.T

extent = [lo[0], lo[-1], rf[0], rf[-1]]
imshow_kw = dict(origin="lower", extent=extent, aspect="auto", cmap="viridis")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.2))

imL = axL.imshow(ideal_W, norm=LogNorm(vmin=ideal_W.min(), vmax=ideal_W.max()), **imshow_kw)
cbL = fig.colorbar(imL, ax=axL, pad=0.02)
cbL.set_label("Power (W) -- perfect inner-product multiplier")
axL.set_title(r"Ideal:  $P_{center}$[dBm] $= P_{LO} + P_{RF} + 10\log_{10}|\langle a,b\rangle|^2$"
              "   (log colour)", fontsize=12)

imR = axR.imshow(measured, norm=LogNorm(vmin=measured.min(), vmax=measured.max()), **imshow_kw)
cbR = fig.colorbar(imR, ax=axR, pad=0.02)
cbR.set_label("|DC bin| (arb. units, USRP fc32 -- UNCALIBRATED)")
axR.set_title(f"Measured (USRP RX @ {meta['if_freq_hz']/1e6:.0f} MHz)   (log colour)",
              fontsize=12)

for ax in (axL, axR):
    ax.set_xlabel("LO port power (dBm)")
    ax.set_ylabel("RF port power (dBm)")

suptitle = (
    f"USRP X310 + UBX-160  ->  ZEM-4300  ->  USRP RX (GNU Radio)     "
    f"f_RF={meta['freq_rf_hz']/1e9:.2f} GHz, "
    f"f_LO={meta['freq_lo_hz']/1e9:.2f} GHz, "
    f"f_IF={meta['if_freq_hz']/1e6:.0f} MHz   |   "
    f"vector inner product N={int(meta['vec_N'])}, "
    f"df_tone={meta['df_tone_hz']/1e6:.2f} MHz, "
    f"|<a,b>|^2={ip2:.3e}"
)
fig.suptitle(suptitle, fontsize=12, y=0.99)

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PNG, dpi=110)
print(f"wrote {OUT_PNG}")
