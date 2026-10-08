"""Shared core: vocab, synthetic grammar, numpy MLP LM, Adam, LoRA, eval.
No deep-learning framework. Only numpy (+stdlib).
Implements the 8-step video spec faithfully:
  vocab=64, ctx=8, emb=64, h=1024, W1(1024x512) W2(1024x1024) W3(64x1024).
"""
import numpy as np

CTX = 8
VOCAB_SIZE = 64
EMB = 64
HID = 1024

PEOPLE = ["sailor", "queen", "pilot", "king", "baker", "farmer", "teacher", "captain"]
THINGS = ["net", "crown", "ship", "sword", "book", "map", "drum", "bread"]
PLACES = ["castle", "forest", "sea", "town"]
VERBS = ["finds", "sells", "carries", "wants", "guards", "likes", "sees", "holds"]
ADJS = ["new", "old", "tired", "quiet", "brave", "fair", "dark", "cold"]
SMALL = ["the", "a", "and", "is", "in", "are", "like", "with"]
PUNCT = [".", ","]
PIRATE_TOKS = ["R", "ye", "be", "matey"]
POLITE_TOKS = ["please", "kindly", "thank", "you"]
EXTRA = ["ocean", "wind", "star", "moon", "soldier", "merchant", "island", "storm", "child", "horse"]

VOCAB = PEOPLE + THINGS + PLACES + VERBS + ADJS + SMALL + PUNCT + PIRATE_TOKS + POLITE_TOKS + EXTRA
assert len(VOCAB) == 64, f"vocab={len(VOCAB)}"
TOK = {w: i for i, w in enumerate(VOCAB)}

PLAIN_FORBIDDEN = set(PIRATE_TOKS + POLITE_TOKS)

# ---------------- data ----------------

def _choice(rng, xs):
    return xs[int(rng.integers(0, len(xs)))]


def gen_plain_sentence(rng):
    p = int(rng.integers(0, 5))
    if p == 0:
        return ["the", _choice(rng, ADJS), _choice(rng, PEOPLE), _choice(rng, VERBS),
                "the", _choice(rng, THINGS), "in", "the", _choice(rng, PLACES), "."]
    if p == 1:
        return ["the", _choice(rng, PEOPLE), _choice(rng, VERBS), "a", _choice(rng, THINGS), "."]
    if p == 2:
        return ["the", _choice(rng, THINGS), "is", _choice(rng, ADJS), "and",
                "the", _choice(rng, PLACES), "is", _choice(rng, ADJS), "."]
    if p == 3:
        return ["a", _choice(rng, PEOPLE), "and", "a", _choice(rng, PEOPLE),
                _choice(rng, VERBS), "the", _choice(rng, THINGS), "."]
    return ["the", _choice(rng, PEOPLE), _choice(rng, VERBS), "a", _choice(rng, THINGS),
            "in", "the", _choice(rng, PLACES), "."]


def to_pirate(sent):
    out = ["R"]
    for w in sent:
        if w == "the":
            out.append("ye")
        elif w == "is":
            out.append("be")
        elif w == "." or w == ",":
            continue
        else:
            out.append(w)
    out += ["matey", "."]
    return out


def to_polite(sent):
    # polite: prepend please, insert kindly before main verb, append thank you .
    body = [w for w in sent if w != "."]
    # insert kindly after first person token if present
    idx = next((i for i, w in enumerate(body) if w in PEOPLE), 2)
    body.insert(min(idx + 1, len(body)), "kindly")
    return ["please"] + body + ["thank", "you", "."]


def gen_sentences(kind, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = gen_plain_sentence(rng)
        if kind == "plain":
            out.append(s)
        elif kind == "pirate":
            out.append(to_pirate(s))
        elif kind == "polite":
            out.append(to_polite(s))
        else:
            raise ValueError(kind)
    return out


def sentences_to_pairs(sentences):
    xs, ys = [], []
    for s in sentences:
        ids = [TOK[w] for w in s]
        for i in range(len(ids)):
            ctx = ids[max(0, i - CTX):i]
            ctx = [0] * (CTX - len(ctx)) + ctx
            xs.append(ctx)
            ys.append(ids[i])
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


# ---------------- validators ----------------
def valid_plain(s):
    if any(w in (set(PIRATE_TOKS) | set(POLITE_TOKS)) for w in s):
        return False
    if not s or s[-1] != ".":
        return False
    b = s[:-1]
    try:
        # P1
        if len(b) == 5 and b[0] == "the" and b[1] in PEOPLE and b[2] in VERBS and b[3] == "a" and b[4] in THINGS:
            return True
        # P1b: the PERSON VERB the THING (needed for polite/captain guards net)
        if len(b) == 5 and b[0] == "the" and b[1] in PEOPLE and b[2] in VERBS and b[3] == "the" and b[4] in THINGS:
            return True
        # P2
        if len(b) == 9 and b[0] == "the" and b[1] in THINGS and b[2] == "is" and b[3] in ADJS and b[4] == "and" and b[5] == "the" and b[6] in PLACES and b[7] == "is" and b[8] in ADJS:
            return True
        # P3
        if len(b) == 8 and b[0] == "a" and b[1] in PEOPLE and b[2] == "and" and b[3] == "a" and b[4] in PEOPLE and b[5] in VERBS and b[6] == "the" and b[7] in THINGS:
            return True
        # P0
        if len(b) == 9 and b[0] == "the" and b[1] in ADJS and b[2] in PEOPLE and b[3] in VERBS and b[4] == "the" and b[5] in THINGS and b[6] == "in" and b[7] == "the" and b[8] in PLACES:
            return True
        # P4
        if len(b) == 8 and b[0] == "the" and b[1] in PEOPLE and b[2] in VERBS and b[3] == "a" and b[4] in THINGS and b[5] == "in" and b[6] == "the" and b[7] in PLACES:
            return True
    except Exception:
        return False
    return False


def valid_pirate(s):
    if not s or s[-1] != "." or len(s) < 4:
        return False
    if s[0] != "R" or s[-2] != "matey":
        return False
    if "the" in s or "is" in s:
        return False
    if any(w in POLITE_TOKS for w in s):
        return False
    # strip wrapper, restore, check plain
    inner = s[1:-2]
    restored = ["the" if w == "ye" else ("is" if w == "be" else w) for w in inner]
    return valid_plain(restored + ["."])


def valid_polite(s):
    if not s or s[-1] != "." or len(s) < 5:
        return False
    if s[0] != "please":
        return False
    if not (s[-3] == "thank" and s[-2] == "you"):
        return False
    if "kindly" not in s:
        return False
    if any(w in PIRATE_TOKS for w in s):
        return False
    inner = [w for w in s[1:-3] if w != "kindly"]
    return valid_plain(inner + ["."])


# ---------------- model ----------------

def count_params():
    emb = VOCAB_SIZE * EMB
    w1 = HID * (CTX * EMB)
    w2 = HID * HID
    w3 = VOCAB_SIZE * HID
    b = HID + HID + VOCAB_SIZE
    return {"emb": emb, "W1": w1, "W2": w2, "W3": w3, "bias": b, "total": emb + w1 + w2 + w3 + b}


def lora_count(rank):
    w1 = rank * (CTX * EMB) + HID * rank
    w2 = rank * HID + HID * rank
    w3 = rank * HID + VOCAB_SIZE * rank
    return w1 + w2 + w3


def init_params(seed=0):
    rng = np.random.default_rng(seed)
    def u(fan_in, fan_out):
        s = np.sqrt(2.0 / fan_in)
        return (rng.standard_normal((fan_out, fan_in)) * s).astype(np.float64)
    return {
        "E": (rng.standard_normal((VOCAB_SIZE, EMB)) * 0.1).astype(np.float64),
        "W1": u(CTX * EMB, HID), "b1": np.zeros(HID),
        "W2": u(HID, HID), "b2": np.zeros(HID),
        "W3": u(HID, VOCAB_SIZE), "b3": np.zeros(VOCAB_SIZE),
    }


def forward(P, x, lora=None, scale=2.0):
    B = x.shape[0]
    e = P["E"][x]                      # (B,8,64)
    f = e.reshape(B, CTX * EMB)
    h1_pre = f @ P["W1"].T + P["b1"]
    if lora is not None:
        h1_pre = h1_pre + scale * ((f @ lora["A1"].T) @ lora["B1"].T)
    h1 = np.maximum(h1_pre, 0)
    h2_pre = h1 @ P["W2"].T + P["b2"]
    if lora is not None:
        h2_pre = h2_pre + scale * ((h1 @ lora["A2"].T) @ lora["B2"].T)
    h2 = np.maximum(h2_pre, 0)
    logits = h2 @ P["W3"].T + P["b3"]
    if lora is not None:
        logits = logits + scale * ((h2 @ lora["A3"].T) @ lora["B3"].T)
    cache = (x, f, h1_pre, h1, h2_pre, h2, logits)
    return logits, cache


def ce_loss(logits, y):
    m = logits.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    ll = logits[np.arange(len(y)), y] - lse.squeeze(1)
    return float(-ll.mean())


def backward(P, cache, y, lora=None, scale=2.0):
    x, f, h1_pre, h1, h2_pre, h2, logits = cache
    B = len(y)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    probs = e / e.sum(axis=1, keepdims=True)
    dlog = probs.copy()
    dlog[np.arange(B), y] -= 1.0
    dlog /= B
    g = {}
    # W3 / LoRA3
    g["W3"] = dlog.T @ h2
    g["b3"] = dlog.sum(0)
    dh2 = dlog @ P["W3"]
    gA3 = gB3 = None
    if lora is not None:
        z3 = h2 @ lora["A3"].T
        gB3 = (scale * dlog).T @ z3
        gA3 = (scale * (dlog @ lora["B3"])).T @ h2
        dh2 = dh2 + scale * ((dlog @ lora["B3"]) @ lora["A3"])
    dh2_pre = dh2 * (h2_pre > 0)
    g["W2"] = dh2_pre.T @ h1
    g["b2"] = dh2_pre.sum(0)
    dh1 = dh2_pre @ P["W2"]
    gA2 = gB2 = None
    if lora is not None:
        z2 = h1 @ lora["A2"].T
        dh_l = dh2_pre @ lora["B2"]
        gB2 = (scale * dh2_pre).T @ z2
        gA2 = (scale * dh_l).T @ h1
        dh1 = dh1 + scale * (dh_l @ lora["A2"])
    dh1_pre = dh1 * (h1_pre > 0)
    g["W1"] = dh1_pre.T @ f
    g["b1"] = dh1_pre.sum(0)
    df = dh1_pre @ P["W1"]
    gA1 = gB1 = None
    if lora is not None:
        z1 = f @ lora["A1"].T
        df_l = dh1_pre @ lora["B1"]
        gB1 = (scale * dh1_pre).T @ z1
        gA1 = (scale * df_l).T @ f
    # embedding grad
    dE = np.zeros_like(P["E"])
    df3 = df.reshape(len(y), CTX, EMB)
    np.add.at(dE, x, df3)
    g["E"] = dE
    lg = None
    if lora is not None:
        lg = {"A1": gA1, "B1": gB1, "A2": gA2, "B2": gB2, "A3": gA3, "B3": gB3}
    return g, lg


class Adam:
    def __init__(self, keys, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: None for k in keys}
        self.v = {k: None for k in keys}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in grads:
            g = grads[k]
            if self.m[k] is None:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def init_lora(rank, alpha, seed=1):
    rng = np.random.default_rng(seed)
    def A(fan_in):
        return (rng.standard_normal((rank, fan_in)) * (1.0 / np.sqrt(fan_in))).astype(np.float64)
    return {
        "A1": A(CTX * EMB), "B1": np.zeros((HID, rank)),
        "A2": A(HID), "B2": np.zeros((HID, rank)),
        "A3": A(HID), "B3": np.zeros((VOCAB_SIZE, rank)),
        "rank": rank, "alpha": alpha, "scale": alpha / rank,
    }


def train_loop(P, X, Y, steps, batch, seed, lora=None, lr=3e-3, log_every=0, Xval=None, Yval=None):
    rng = np.random.default_rng(seed)
    keys = list(P.keys()) if lora is None else ["A1", "B1", "A2", "B2", "A3", "B3"]
    opt = Adam(keys, lr=lr)
    target = P if lora is None else lora
    n = len(Y)
    hist = []
    for s in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        xb, yb = X[idx], Y[idx]
        sc = 2.0 if lora is None else float(lora["scale"])
        logits, cache = forward(P, xb, lora, sc)
        loss = ce_loss(logits, yb)
        g, lg = backward(P, cache, yb, lora, sc)
        opt.step(target, g if lora is None else lg)
        if log_every and s % log_every == 0:
            lv = None
            if Xval is not None:
                lv, _ = forward(P, Xval[:512], lora, sc)
                lv = ce_loss(lv, Yval[:512])
            hist.append((s, loss, lv))
    return hist


def generate(P, seed, n_sent, kind, temp=0.8, max_len=14, lora=None, scale=2.0):
    rng = np.random.default_rng(seed)
    sents = []
    for _ in range(n_sent):
        if kind == "pirate":
            ctx_ids = [TOK["R"]]
        elif kind == "polite":
            ctx_ids = [TOK["please"]]
        else:
            ctx_ids = [TOK["the"]] if rng.random() < 0.7 else [TOK["a"]]
        out = list(ctx_ids)
        for _ in range(max_len):
            x = np.array([([0] * (CTX - len(out[-CTX:])) + out[-CTX:])])
            logits, _ = forward(P, x, lora, scale)
            z = logits[0] / temp
            z = z - z.max()
            # forbid pad? none. keep full distribution
            p = np.exp(z)
            p /= p.sum()
            nxt = int(rng.choice(VOCAB_SIZE, p=p))
            out.append(nxt)
            if nxt == TOK["."] and len(out) > 4:
                break
        sents.append([VOCAB[i] for i in out])
    return sents


def accuracy(sents, kind):
    fn = {"plain": valid_plain, "pirate": valid_pirate, "polite": valid_polite}[kind]
    if not sents:
        return 0.0
    return sum(1 for s in sents if fn(s)) / len(sents)


def eval_loss(P, X, Y, lora=None, scale=2.0, n=2000):
    n = min(n, len(Y))
    logits, _ = forward(P, X[:n], lora, scale)
    return ce_loss(logits, Y[:n])
