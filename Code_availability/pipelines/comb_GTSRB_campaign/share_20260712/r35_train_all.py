import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys, time
sys.path.insert(0, ".")
from ladder_cnn_v2 import train_model, export_runner

def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    open("r35_train_log.txt", "a").write(line + "\n")

for arch in ("r3plus", "r3wide", "r3fast"):
    t0 = time.time()
    try:
        te = train_model(arch, img=32, epochs=120, seed=0, val_frac=0.1,
                         out=f"r35_{arch}_s0.npz", log=log)
        export_runner(f"r35_{arch}_s0.npz", f"r35_{arch}_s0_hw.npz")
        log(f"=== {arch}: test {te*100:.2f}; exported r35_{arch}_s0_hw.npz "
            f"({time.time()-t0:.0f}s) ===")
    except Exception as e:
        log(f"!! {arch} FAILED: {e!r}")
log("=== r35 sweep done ===")
