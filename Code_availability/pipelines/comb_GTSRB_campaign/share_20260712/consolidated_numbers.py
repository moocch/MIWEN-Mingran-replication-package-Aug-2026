"""Single source of truth: every N=1200-test number, each with its exact
{weights, deterministic deviation, random noise} label and how it's made."""
import sys, numpy as np
sys.path.insert(0, ".")
from miwen_frozen_reference import (load_weights, forward_digital, preprocess,
                                    im2col, maxpool2, maxnorm, interpass, HERE)
d = np.load(HERE/"gtsrb_roi_32x32.npz", allow_pickle=True)
Xte = preprocess(d["Xte"]); yte = d["yte"].astype(np.int64)
sel = np.load(HERE/"battery_random1200_idx.npy"); y = yte[sel]; n = len(sel)
slim = np.load(HERE/"battery_slim.npz", allow_pickle=True)

def acc(preds): return (preds == y).mean()*100
def sig(a): return np.sqrt(a/100*(1-a/100)/n)*100

def noisy(weights, levels, seeds=15):
    layers = load_weights(HERE/weights); out=[]
    for s in range(seeds):
        rng=np.random.RandomState(s); A=Xte[sel].reshape(-1,32,32,3).transpose(0,3,1,2); S=None
        for tag,lay in enumerate(layers,1):
            if lay["kind"]=="conv":
                P,_,_=im2col(A,lay["k"],lay["stride"]); am=np.abs(P.reshape(-1,P.shape[-1])@lay["W"].T)
            else: am=np.abs(A.reshape(n,-1)@lay["W"].T)
            am=np.maximum(am+levels[tag-1]*np.sqrt(np.mean(am**2))*rng.randn(*am.shape),0.0)
            A,S=interpass(am,lay,tag,n)
        out.append((np.argmax(S**2,1)==y).mean())
    return np.mean(out)*100, np.std(out)*100

FULL=[0.204,0.234,0.073,0.164]; STOCH=[0.070,0.091,0.060,0.129]
a_meas = acc(slim["preds"])
a_lin  = acc(slim["digital_preds"])
a_clean= acc(forward_digital(Xte[sel], load_weights(HERE/"r35_r3plus_s0_hw.npz")))
ns_m,_ = noisy("r35_r3plus_ns25_s0_hw.npz", STOCH)
nf_m,_ = noisy("r35_r3plus_ns25_s0_hw.npz", FULL)
cn_m,_ = noisy("r35_r3plus_s0_hw.npz", FULL)
# run 2 (persistence)
pm2=[]; yy=[]
for i in range(1,11):
    z=np.load(HERE/f"bat2_c{i}.npz",allow_pickle=True); pm2.append(z["preds"]); yy.append(yte[z["img_index"]])
a_run2=(np.concatenate(pm2)==np.concatenate(yy)).mean()*100

print("="*72)
print(f"{'#':2} {'accuracy':>9}  weights        det-dev  noise            what")
print("-"*72)
rows=[
 (1, a_meas, "ns25", "REAL", "REAL", "MIWEN measured (run1) — the system"),
 (2, a_run2, "ns25", "REAL", "REAL", "MIWEN measured (run2, persistence)"),
 (3, a_lin,  "ns25", "none", "none", "linear MAC (digital exec of ns25 weights)"),
 (4, ns_m,   "ns25", "none", "stoch", "noise only, TRUE stochastic level 0.07-0.13"),
 (5, nf_m,   "ns25", "none", "FULL",  "noise at full magnitude 0.20-0.23 [MIS-MODEL]"),
 (6, a_clean,"clean","none", "none",  "DIGITAL system (clean weights, linear MAC)"),
 (7, cn_m,   "clean","none", "FULL",  "clean weights + full noise (robustness ref)"),
]
for i,a,w,dd,nn,desc in rows:
    print(f"{i:2} {a:6.2f}±{sig(a):.2f}  {w:6}  {dd:>7}  {nn:>6}  {desc}")
print("-"*72)
print(f"deterministic deviation effect = #1 - #4 = {a_meas-ns_m:+.2f}   (dominant, beneficial)")
print(f"random noise effect            = #4 - #3 = {ns_m-a_lin:+.2f}   (small, harmful)")
print(f"net MIWEN vs its own linear MAC= #1 - #3 = {a_meas-a_lin:+.2f}")
print(f"SYSTEM: digital vs MIWEN       = #6 - #1 = {a_clean-a_meas:+.2f}   (digital wins)")
