"""Step 6: train LoRA (A,B only, base frozen bit-identical)."""
import argparse, json, os, time
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gen_sentences, sentences_to_pairs, init_lora, train_loop, eval_loss, generate, accuracy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="results/base.npz")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--kind", type=str, default="pirate")
    ap.add_argument("--out", type=str, default="results/lora_pirate.npz")
    a = ap.parse_args()
    t0 = time.time()
    P = dict(np.load(a.base))
    before = {k: v.copy() for k, v in P.items()}
    sents = gen_sentences(a.kind, 3000, seed=21 if a.kind == "pirate" else 22)
    X, Y = sentences_to_pairs(sents)
    L = init_lora(rank=a.rank, alpha=a.alpha, seed=3)
    l0 = eval_loss(P, X, Y, L, L["scale"])
    train_loop(P, X, Y, steps=a.steps, batch=a.batch, seed=4, lora=L, lr=3e-3)
    l1 = eval_loss(P, X, Y, L, L["scale"])
    frozen = all(np.array_equal(before[k], P[k]) for k in before)
    gen = generate(P, seed=99, n_sent=1000, kind=a.kind, lora=L, scale=L["scale"])
    acc = accuracy(gen, a.kind)
    print(f"FINAL {a.kind} loss {l0:.3f}->{l1:.3f} valid={acc*1000:.0f}/1000 frozen={frozen} time={time.time()-t0:.1f}s")
    keep = {k: L[k] for k in ["A1", "B1", "A2", "B2", "A3", "B3"]}
    keep["rank"] = np.array(a.rank); keep["alpha"] = np.array(a.alpha)
    np.savez(a.out, **keep)
    tag = "step6" if a.kind == "pirate" else "step6_polite"
    json.dump({"loss0": l0, "loss1": l1, "valid": acc, "frozen": bool(frozen)},
              open(f"results/{tag}.json", "w"), indent=2)
    for s in gen[:5]:
        print("ex:", " ".join(s))

if __name__ == "__main__":
    main()
