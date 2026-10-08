"""Step 5: LoRA layer check. Fresh adapter must be exactly identity (delta=0)."""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import init_params, init_lora, forward, lora_count

def main():
    P = init_params(seed=0)
    L = init_lora(rank=4, alpha=8, seed=1)
    print("lora params:", lora_count(4), f"({lora_count(4)/1644608*100:.2f}%)")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 64, size=(16, 8))
    a, _ = forward(P, x)
    b, _ = forward(P, x, L, L["scale"])
    print("max |logit diff| fresh adapter:", float(np.abs(a - b).max()))
    assert float(np.abs(a - b).max()) == 0.0
    print("OK: forward = W x + scale*B(Ax), B=0 so identity at init.")

if __name__ == "__main__":
    main()
