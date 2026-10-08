"""Step 4: truncate deltaW to rank k, add back, measure pirate survival."""
import argparse, json
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gen_sentences, sentences_to_pairs, forward, ce_loss, generate, accuracy

def trunc(M, k):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="results/base.npz")
    ap.add_argument("--ft", type=str, default="results/full_ft.npz")
    a = ap.parse_args()
    B = dict(np.load(a.base)); F = dict(np.load(a.ft))
    out = {}
    for k in [1024, 64, 16, 4, 2, 1]:
        P = {kk: np.array(v) for kk, v in B.items()}
        for m in ["W1", "W2", "W3"]:
            kk = min(k, P[m].shape[0], P[m].shape[1])
            P[m] = P[m] + trunc(F[m] - B[m], kk)
        gen = generate(P, seed=99, n_sent=400, kind="pirate")
        acc = accuracy(gen, "pirate")
        out[str(k)] = acc
        print(f"rank {k}: {acc*400:.0f}/400")
    json.dump(out, open("results/step4.json", "w"), indent=2)

if __name__ == "__main__":
    main()
