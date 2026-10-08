"""Step 7: merging (B*A into W, zero error) + swapping pirate/polite."""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import forward, generate, accuracy

def load_lora(path):
    d = dict(np.load(path, allow_pickle=True))
    return d

def main():
    P = dict(np.load("results/base.npz"))
    Lp = load_lora("results/lora_pirate.npz")
    Lo = load_lora("results/lora_polite.npz")
    rp, ap_ = int(Lp["rank"]), int(Lp["alpha"])
    sc = ap_ / rp
    # merge check: compare adapter forward vs merged forward
    rng = np.random.default_rng(0)
    x = rng.integers(0, 64, size=(32, 8))
    la, _ = forward(P, x, {k: Lp[k] for k in ["A1","B1","A2","B2","A3","B3"]}, sc)
    M = {k: np.array(v) for k, v in P.items()}
    M["W1"] = M["W1"] + sc * (Lp["B1"] @ Lp["A1"])
    M["W2"] = M["W2"] + sc * (Lp["B2"] @ Lp["A2"])
    M["W3"] = M["W3"] + sc * (Lp["B3"] @ Lp["A3"])
    lb, _ = forward(M, x)
    print("merge max diff:", float(np.abs(la - lb).max()))
    # swapping
    for name, L, kind in [("none", None, "plain"), ("pirate", {k: Lp[k] for k in ["A1","B1","A2","B2","A3","B3"]}, "pirate"),
                          ("polite", {k: Lo[k] for k in ["A1","B1","A2","B2","A3","B3"]}, "polite")]:
        if name == "none":
            g = generate(P, seed=99, n_sent=400, kind="plain")
            print(f"no-adapter plain: {accuracy(g,'plain')*400:.0f}/400")
        else:
            r = int(L["A1"].shape[0]) if "rank" not in L else 4
            # scale from file
            src = Lp if name == "pirate" else Lo
            s = int(src["alpha"]) / int(src["rank"])
            g = generate(P, seed=99, n_sent=400, kind=kind, lora=L, scale=s)
            print(f"{name}: {accuracy(g,kind)*400:.0f}/400")
    import os as _os
    b = _os.path.getsize("results/base.npz"); p = _os.path.getsize("results/lora_pirate.npz")
    print(f"files base={b/1e6:.2f}MB adapter={p/1e3:.1f}KB; 100 styles: copies={100*b/1e6:.0f}MB vs base+adapters={(b+100*p)/1e6:.1f}MB")

if __name__ == "__main__":
    main()
