"""Step 3: deltaW = FT - base; SVD spectrum: how lopsided is the change?"""
import argparse, json
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def spec(M):
    s = np.linalg.svd(M, compute_uv=False)
    e = (s ** 2)
    tot = e.sum()
    out = {}
    for k in [1, 4, 16, 64]:
        out[f"top{k}"] = float(e[:k].sum() / tot)
    # k for 90%
    c = np.cumsum(e) / tot
    out["k90"] = int(np.searchsorted(c, 0.90) + 1)
    out["fro_ratio"] = float(np.linalg.norm(M))
    return out, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="results/base.npz")
    ap.add_argument("--ft", type=str, default="results/full_ft.npz")
    a = ap.parse_args()
    B = dict(np.load(a.base)); F = dict(np.load(a.ft))
    res = {}
    for k in ["W1", "W2", "W3"]:
        d = F[k] - B[k]
        ratio = float(np.linalg.norm(d) / np.linalg.norm(B[k]))
        sp, s = spec(d)
        sp["delta_over_W"] = ratio
        res[k] = sp
        print(k, {kk: round(v, 3) if isinstance(v, float) else v for kk, v in sp.items()})
    json.dump(res, open("results/step3.json", "w"), indent=2)

if __name__ == "__main__":
    main()
