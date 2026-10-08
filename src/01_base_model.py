"""Step 1: base model. Train small LM from zero on plain sentences."""
import argparse, json, os, time
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (gen_sentences, sentences_to_pairs, init_params, train_loop,
                    eval_loss, generate, accuracy, count_params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/base.npz")
    a = ap.parse_args()
    t0 = time.time()
    sents = gen_sentences("plain", 20000, seed=a.seed)
    X, Y = sentences_to_pairs(sents)
    P = init_params(seed=a.seed)
    print("params:", count_params(), flush=True)
    hist = train_loop(P, X, Y, steps=a.steps, batch=a.batch, seed=a.seed + 1, lr=3e-3)
    logits_loss = eval_loss(P, X, Y)
    gen = generate(P, seed=99, n_sent=1000, kind="plain")
    acc = accuracy(gen, "plain")
    dt = time.time() - t0
    print(f"FINAL loss={logits_loss:.3f} valid={acc*1000:.0f}/1000 time={dt:.1f}s")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **P)
    with open("results/step1.json", "w") as f:
        json.dump({"loss": logits_loss, "valid": acc, "time": dt, "steps": a.steps}, f, indent=2)
    for s in gen[:5]:
        print("ex:", " ".join(s))

if __name__ == "__main__":
    main()
