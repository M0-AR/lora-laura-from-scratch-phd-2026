"""Hidden-pattern discovery: what the video glosses over, quantified.
1) spectrum lopsidedness + effective rank of trained LoRA (exactly r, rest ~0)
2) per-matrix ablation: LoRA on W1-only / W2-only / W3-only (the exercise)
3) amplification factor ||BA|| / ||P_{BA}(W)||  (LoRA amplifies unused directions)
4) merge error + swap retention + forgetting delta
"""
import json
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import (gen_sentences, sentences_to_pairs, init_lora, train_loop,
                    eval_loss, generate, accuracy, forward)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def eff_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    return s

def main():
    P = dict(np.load(os.path.join(ROOT, "results/base.npz")))
    Lp = dict(np.load(os.path.join(ROOT, "results/lora_pirate.npz")))
    r = int(Lp["rank"]); al = int(Lp["alpha"]); sc = al / r
    L = {k: Lp[k] for k in ["A1", "B1", "A2", "B2", "A3", "B3"]}
    out = {}
    # 1) trained BA spectrum must be exactly rank r
    for name, A, B in [("W1", L["A1"], L["B1"]), ("W2", L["A2"], L["B2"]), ("W3", L["A3"], L["B3"])]:
        BA = sc * (B @ A)
        s = eff_rank(BA)
        out[f"{name}_sv_top5"] = [float(x) for x in s[:5]]
        out[f"{name}_sv_rest_max"] = float(s[r:].max()) if len(s) > r else 0.0
        out[f"{name}_fro"] = float(np.linalg.norm(BA))
    # 2) per-matrix ablation (honest: train target-only adapters from scratch)
    from common import Adam, forward as fwd, backward as bwd, ce_loss as cel
    sents = gen_sentences("pirate", 3000, seed=21)
    X, Y = sentences_to_pairs(sents)

    def train_target_only(Q, Xt, Yt, target, steps=300):
        Li = init_lora(rank=4, alpha=8, seed=3)
        only = {"W1": ["A1", "B1"], "W2": ["A2", "B2"], "W3": ["A3", "B3"],
                "all": ["A1", "B1", "A2", "B2", "A3", "B3"]}[target]
        opt = Adam(only, lr=3e-3)
        rng = np.random.default_rng(4)
        n = len(Yt)
        for _ in range(steps):
            idx = rng.integers(0, n, size=64)
            xb, yb = Xt[idx], Yt[idx]
            logits, cache = fwd(Q, xb, Li, Li["scale"])
            g, lg = bwd(Q, cache, yb, Li, Li["scale"])
            opt.step(Li, {k: lg[k] for k in only})
        return Li

    for target in ["W1", "W2", "W3", "all"]:
        Q = {k: np.array(v) for k, v in P.items()}
        Lc = train_target_only(Q, X, Y, target, steps=300)
        g = generate(Q, seed=99, n_sent=300, kind="pirate", lora=Lc, scale=Lc["scale"])
        acc = accuracy(g, "pirate")
        # NOTE: non-target adapters were never trained (B=0 exactly), so this is a
        # clean single-matrix experiment, unlike mask-after-joint-training.
        out[f"ablate_{target}"] = acc
        print(f"ablate {target}: {acc*300:.0f}/300", flush=True)
    # 3) amplification: project W onto BA top directions
    for name, Wk, A, B in [("W2", P["W2"], L["A2"], L["B2"])]:
        BA = sc * (B @ A)
        U, s, Vt = np.linalg.svd(BA, full_matrices=False)
        # projection of W onto BA row space
        proj = np.linalg.norm(Wk @ Vt[:r, :].T)
        own = np.linalg.norm(Wk)
        out["amplify_proj"] = float(proj)
        out["amplify_Wfro"] = float(own)
        out["amplify_BAfro"] = float(np.linalg.norm(BA))
        out["amplify_factor"] = float(np.linalg.norm(BA) / max(proj, 1e-12))
    # 4) forgetting
    ps = gen_sentences("plain", 2000, seed=9)
    Xp, Yp = sentences_to_pairs(ps)
    out["plain_base"] = eval_loss(P, Xp, Yp)
    out["plain_with_pirate_lora"] = eval_loss(P, Xp, Yp, L, sc)
    F = dict(np.load(os.path.join(ROOT, "results/full_ft.npz")))
    out["plain_fullft"] = eval_loss(F, Xp, Yp)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, indent=2))
    json.dump(out, open(os.path.join(ROOT, "results/hidden.json"), "w"), indent=2,
              default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

if __name__ == "__main__":
    main()
