"""Live-market LoRA: same protocol, real numeric data.
Task: predict next-day direction of AAPL from past 8 returns of 5 tech stocks.
Model: tiny MLP (40->32->32->2) trained full vs LoRA-r4 on frozen random base.
Live via yfinance; deterministic embedded fallback if offline (labeled).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_returns():
    try:
        import yfinance as yf
        d = yf.download("AAPL MSFT NVDA AMZN META", period="6mo", progress=False, auto_adjust=True)["Close"].dropna()
        if len(d) > 60:
            return np.diff(np.log(d.values), axis=0), "yfinance_live", list(map(str, d.columns))
    except Exception as e:
        err = str(e)[:150]
        print("live fetch failed, fallback:", err)
    rng = np.random.default_rng(7)
    n = 120
    F = rng.standard_normal((n, 2))  # 2 latent market factors
    L = rng.standard_normal((2, 5))
    R = F @ L + 0.3 * rng.standard_normal((n, 5))
    return R, "embedded_fallback", ["MKT0..4"]

def mlp(direction="full", rank=4, steps=500, seed=0):
    R, src, cols = get_returns()
    # features: past 8 days of 5 stocks = 40; label: AAPL up/down next day
    X, Y = [], []
    for i in range(8, len(R) - 1):
        X.append(R[i - 8:i, :].reshape(-1))
        Y.append(1 if R[i + 1, 0] > 0 else 0)
    X = np.array(X)
    Y = np.array(Y)
    n = len(Y)
    cut = int(n * 0.7)
    Xt, Yt, Xv, Yv = X[:cut], Y[:cut], X[cut:], Y[cut:]
    Xt = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-6)
    Xv = (Xv - Xt.mean(0)) / (Xt.std(0) + 1e-6) if False else (Xv - X[:cut].mean(0)) / (X[:cut].std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    d, h, c = 40, 32, 2
    W1 = rng.standard_normal((h, d)) * np.sqrt(2 / d)
    b1 = np.zeros(h)
    W2 = rng.standard_normal((h, h)) * np.sqrt(2 / h)
    b2 = np.zeros(h)
    W3 = rng.standard_normal((c, h)) * np.sqrt(2 / h)
    b3 = np.zeros(c)
    base = (W1.copy(), W2.copy(), W3.copy())
    if direction == "lora":
        A1 = rng.standard_normal((rank, d)) / np.sqrt(d)
        B1 = np.zeros((h, rank))
        A2 = rng.standard_normal((rank, h)) / np.sqrt(h)
        B2 = np.zeros((h, rank))
        A3 = rng.standard_normal((rank, h)) / np.sqrt(h)
        B3 = np.zeros((c, rank))
    lr = 0.02
    def fwd(Xb):
        h1 = Xb @ W1.T + b1
        if direction == "lora":
            h1 = h1 + ((Xb @ A1.T) @ B1.T)
        r1 = np.maximum(h1, 0)
        h2 = r1 @ W2.T + b2
        if direction == "lora":
            h2 = h2 + ((r1 @ A2.T) @ B2.T)
        r2 = np.maximum(h2, 0)
        lo = r2 @ W3.T + b3
        if direction == "lora":
            lo = lo + ((r2 @ A3.T) @ B3.T)
        return h1, r1, h2, r2, lo
    def acc(Xe, Ye):
        *_, lo = fwd(Xe)
        return float(((lo.argmax(1)) == Ye).mean())
    for s in range(steps):
        idx = rng.integers(0, len(Yt), size=32)
        Xb, Yb = Xt[idx], Yt[idx]
        h1, r1, h2, r2, lo = fwd(Xb)
        lo = lo - lo.max(1, keepdims=True)
        p = np.exp(lo)
        p /= p.sum(1, keepdims=True)
        dlo = p.copy()
        dlo[np.arange(len(Yb)), Yb] -= 1
        dlo /= len(Yb)
        if direction == "lora":
            gB3 = dlo.T @ (r2 @ A3.T)
            gA3 = (dlo @ B3).T @ r2
            dr2 = dlo @ W3 + (dlo @ B3) @ A3
            dh2 = dr2 * (h2 > 0)
            gB2 = dh2.T @ (r1 @ A2.T)
            gA2 = (dh2 @ B2).T @ r1
            dr1 = dh2 @ W2 + (dh2 @ B2) @ A2
            dh1 = dr1 * (h1 > 0)
            gB1 = dh1.T @ (Xb @ A1.T)
            gA1 = (dh1 @ B1).T @ Xb
            for M, g in [(B3, gB3), (A3, gA3), (B2, gB2), (A2, gA2), (B1, gB1), (A1, gA1)]:
                M -= lr * g
        else:
            gW3 = dlo.T @ r2
            gb3 = dlo.sum(0)
            dr2 = dlo @ W3
            dh2 = dr2 * (h2 > 0)
            gW2 = dh2.T @ r1
            gb2 = dh2.sum(0)
            dr1 = dh2 @ W2
            dh1 = dr1 * (h1 > 0)
            gW1 = dh1.T @ Xb
            gb1 = dh1.sum(0)
            W3 -= lr * gW3
            b3 -= lr * gb3
            W2 -= lr * gW2
            b2 -= lr * gb2
            W1 -= lr * gW1
            b1 -= lr * gb1
    n_full = W1.size + W2.size + W3.size + b1.size + b2.size + b3.size
    n_lora = (A1.size + B1.size + A2.size + B2.size + A3.size + B3.size) if direction == "lora" else n_full
    return acc(Xv, Yv), acc(Xt, Yt), n_lora, n_full, src, len(X), cols

def main():
    af, tf, nf_l, nf_f, src, n, _cols = mlp("full")
    al, tl, nl_l, nl_f, _, _, _ = mlp("lora")
    out = {"source": src, "n_windows": n,
           "full_train": round(tf, 4), "full_val": round(af, 4), "full_params": nf_f,
           "lora_train": round(tl, 4), "lora_val": round(al, 4), "lora_params": nl_l,
           "lora_pct": round(100 * nl_l / nf_f, 2)}
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(ROOT, "results/livemarket.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
