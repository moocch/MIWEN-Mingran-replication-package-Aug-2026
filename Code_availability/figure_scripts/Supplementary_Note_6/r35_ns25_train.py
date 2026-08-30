import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys, time
sys.path.insert(0, ".")
from ladder_cnn_v2 import train_model, export_runner

SIG = [0.177 * .25, 0.32 * .25, 0.102 * .25, 0.221 * .25]

def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    open("r35_ns25_log.txt", "a").write(line + "\n")

for arch in ("r3fast", "r3wide"):
    t0 = time.time()
    try:
        te = train_model(arch, img=32, epochs=120, seed=0, val_frac=0.1,
                         noise=SIG, out=f"r35_{arch}_ns25_s0.npz", log=log)
        export_runner(f"r35_{arch}_ns25_s0.npz", f"r35_{arch}_ns25_s0_hw.npz")
        log(f"=== {arch}-ns25: clean test {te*100:.2f} ({time.time()-t0:.0f}s) ===")
    except Exception as e:
        log(f"!! {arch}-ns25 FAILED: {e!r}")
log("=== ns25 sweep done ===")
