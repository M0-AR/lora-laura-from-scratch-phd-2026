"""Full benchmark: reproduces video claims table + hidden patterns + real/live data.
Run: python3 benchmarks/run_all_benchmarks.py --fast  (default, ~2-4 min)
     python3 benchmarks/run_all_benchmarks.py --full  (slower, closer to video)
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import count_params, lora_count

def sh(cmd):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", default=True)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    full = a.full
    bs = 2500 if full else 1200
    fs = 800 if full else 400
    ls = 800 if full else 400
    t0 = time.time()
    sh(f"python3 src/01_base_model.py --steps {bs} --batch 64 --out results/base.npz")
    sh(f"python3 src/02_full_finetune.py --steps {fs} --batch 64")
    sh("python3 src/03_delta_svd.py")
    sh("python3 src/04_rank_demo.py")
    sh("python3 src/05_lora_layer.py")
    sh(f"python3 src/06_train_lora.py --rank 4 --alpha 8 --steps {ls} --kind pirate --out results/lora_pirate.npz")
    sh(f"python3 src/06_train_lora.py --rank 4 --alpha 8 --steps {ls} --kind polite --out results/lora_polite.npz")
    sh("python3 src/07_merge_swap.py")
    sh("python3 src/08_rank_alpha.py")
    sh("python3 benchmarks/hidden_patterns.py")
    sh("python3 benchmarks/realdata_verify.py")
    sh("python3 benchmarks/live_market_lora.py")
    print(f"ALL DONE in {time.time()-t0:.1f}s. See results/*.json")
    # summary table
    try:
        d = {}
        for f in ["step1", "step2", "step3", "step4", "step6", "step6_polite", "step8",
                  "hidden", "realdata", "livemarket"]:
            p = os.path.join(ROOT, "results", f + ".json")
            if os.path.exists(p):
                d[f] = json.load(open(p))
        print(json.dumps(d, indent=2)[:4000])
    except Exception as e:
        print("summary warn:", e)

if __name__ == "__main__":
    main()
