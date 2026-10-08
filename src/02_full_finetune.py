"""Step 2: full fine-tuning copy on pirate text (every number may change)."""
import argparse, json, os, time
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gen_sentences, sentences_to_pairs, train_loop, eval_loss, generate, accuracy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="results/base.npz")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    t0 = time.time()
    P = dict(np.load(a.base))
    sents = gen_sentences("pirate", 3000, seed=7)
    X, Y = sentences_to_pairs(sents)
    l0 = eval_loss(P, X, Y)
    print(f"pirate loss before FT: {l0:.3f}", flush=True)
    train_loop(P, X, Y, steps=a.steps, batch=a.batch, seed=8, lr=3e-3)
    l1 = eval_loss(P, X, Y)
    gen = generate(P, seed=99, n_sent=1000, kind="pirate")
    acc = accuracy(gen, "pirate")
    # forgetting: plain loss
    ps = gen_sentences("plain", 2000, seed=9)
    from common import sentences_to_pairs as s2p
    Xp, Yp = s2p(ps)
    lplain = eval_loss(P, Xp, Yp)
    print(f"FINAL pirate_loss={l1:.3f} valid={acc*1000:.0f}/1000 plain_loss_after={lplain:.3f} time={time.time()-t0:.1f}s")
    np.savez("results/full_ft.npz", **P)
    json.dump({"pirate_before": l0, "pirate_after": l1, "valid": acc, "plain_after": lplain},
              open("results/step2.json", "w"), indent=2)
    for s in gen[:5]:
        print("ex:", " ".join(s))

if __name__ == "__main__":
    main()
