#!/usr/bin/env python3
"""Ridge-count sweep of the frozen serial twin on the USRP map
(product-dominated region), replacing the May-era AWG-dataset figure.
Staged fit per K (physics multi-start -> frozen-core ridges -> anchored
joint), 20% held-out; repeat floor from the map's 3-rep std field."""
import json
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

m = np.load("heatmap_unpadded_20260814.npz", allow_pickle=True)
plo = m["p_lo_dbm_grid"].astype(float); prf = m["p_rf_dbm_grid"].astype(float)
A = m["if_amp_uv_mean"].astype(float); pk = m["peak_fs"].astype(float)
sd = m["if_amp_uv_std"].astype(float)
cl = m["clean_mask"].astype(bool) & (A > 0) & (pk < 0.95)
LO, RF = np.meshgrid(plo, prf, indexing="ij")
cl = cl & (LO >= -35) & (RF >= -35)
lo, rf, y = LO[cl], RF[cl], 20*np.log10(A[cl])
# repeat floor: dB-scale std of the 3-rep mean
floor = np.sqrt(np.mean((20/np.log(10)*(sd[cl]/np.maximum(A[cl],1e-12))/np.sqrt(3))**2))
print(f"region cells {cl.sum()}, repeat floor {floor:.3f} dB")

def s_pole(p, k): x = 10**((p-k)/10.0); return x/(1+x)
hold = np.random.default_rng(9).random(len(y)) < 0.2
lo_t, rf_t, y_t = lo[~hold], rf[~hold], y[~hold]

def fit_K(K):
    def prod_db(thp, r, lo_, rf_):
        yv = thp[0] + 10*np.log10(np.maximum(
            s_pole(lo_, thp[1])*s_pole(rf_, thp[2]), 1e-300))
        if r is not None and K > 0:
            for wi, ai, bi, ci in r.reshape(K, 4):
                yv = yv + wi*np.tanh(ai*lo_/40 + bi*rf_/40 + ci)
        return yv
    best = None
    for k0 in [(2.17,3.66),(0.,0.),(5.,5.)]:
        a0 = np.median(y_t - (lo_t-k0[0]) - (rf_t-k0[1]))
        f = least_squares(lambda p,l,r_,yy: prod_db(p,None,l,r_)-yy,
                          np.array([a0,k0[0],k0[1]]),
                          args=(lo_t,rf_t,y_t),
                          bounds=([-np.inf,-10,-10],[np.inf,15,15]),
                          max_nfev=800)
        if best is None or f.cost < best.cost: best = f
    phys = best.x
    if K == 0:
        th, r = phys, None
    else:
        r0 = 0.1*np.random.default_rng(5).standard_normal(K*4)
        fr = least_squares(lambda r_,l,rr,yy: prod_db(phys,r_,l,rr)-yy,
                           r0, args=(lo_t,rf_t,y_t), max_nfev=2500)
        def rj(t,l,rr,yy):
            pen = 10.0*np.array([t[1]-phys[1], t[2]-phys[2]])
            return np.concatenate([prod_db(t[:3],t[3:],l,rr)-yy, pen])
        ff = least_squares(rj, np.concatenate([phys, fr.x]),
                           args=(lo_t,rf_t,y_t), max_nfev=1500)
        th, r = ff.x[:3], ff.x[3:]
    rmse = np.sqrt(np.mean((prod_db(th, r, lo[hold], rf[hold])
                            - y[hold])**2))
    return rmse

Ks = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
rs = []
for K in Ks:
    rmse = fit_K(K)
    rs.append(rmse)
    print(f"K={K:3d}: held-out {rmse:.3f} dB", flush=True)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(Ks, rs, "o-", color="C0", ms=8)
ax.axhline(floor, color="0.4", ls=":", label=f"repeat floor ({floor:.2f} dB)")
ax.axhline(0.89, color="C3", ls="--",
           label="May AWG-era hybrid, K=20 (0.89 dB)")
ax.set_xlabel("tanh ridges K in residual")
ax.set_ylabel("held-out RMSE [dB]")
ax.set_title("Frozen serial twin: ridge-count sweep on the USRP map\n"
             "(product-dominated region, staged fit — the fielded twin "
             "is K=20)")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("twin_ksweep_20260825.png", dpi=150)
json.dump(dict(K=Ks, rmse=[float(x) for x in rs], floor=float(floor)),
          open("twin_ksweep_20260825.json", "w"), indent=1)
print("saved twin_ksweep_20260825.png")
