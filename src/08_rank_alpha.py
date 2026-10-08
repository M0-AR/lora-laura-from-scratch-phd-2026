"""Step 8: two dials - rank sweep and alpha sweep."""
import subprocess, json
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gen_sentences, sentences_to_pairs, init_lora, train_loop, eval_loss, generate, accuracy

def run_once(rank, alpha, steps=300):
    P = dict(np.load("results/base.npz"))
    sents = gen_sentences("pirate", 3000, seed=21)
    X, Y = sentences_to_pairs(sents)
    L = init_lora(rank=rank, alpha=alpha, seed=3)
    train_loop(P, X, Y, steps=steps, batch=64, seed=4, lora=L, lr=3e-3)
    g = generate(P, seed=99, n_sent=400, kind="pirate", lora=L, scale=L["scale"])
    return accuracy(g, "pirate")

def main():
    ranks = {}
    for r in [1, 2, 4, 8, 16]:
        a = run_once(r, 2 * r)
        ranks[str(r)] = a
        print(f"rank {r} (alpha={2*r}): {a*400:.0f}/400", flush=True)
    alphas = {}
    for al in [1, 4, 8, 128]:
        a = run_once(4, al)
        alphas[str(al)] = a
        print(f"alpha {al} (rank=4): {a*400:.0f}/400", flush=True)
    json.dump({"rank": ranks, "alpha": alphas}, open("results/step8.json", "w"), indent=2)

if __name__ == "__main__":
    main()
