#!/usr/bin/env python3
"""Train the tiny RNN and LSTM for Lab 6 (Memory) and export their weights.

Task: read a string of '(' , ')' and '.' one character at a time and, after each
character, say how deep the brackets are nested right now (depth 0 to 4).
'.' stands for any other character and leaves the depth unchanged. The answer at
step t depends on every earlier character, so the network has to carry a count
in its hidden state; that is the thing the lab shows.

NumPy only. Seeded, so a rerun writes the same file:

    python3 -I models/memory/train.py            # writes models/memory/weights.json

Conventions (shared with src/labs/memory/math.ts and scripts/fixtures.py):
  - input x_t is one-hot over VOCAB; matrices are row-major [out][in]
  - Elman RNN:  h_t = tanh(Wxh x_t + Whh h_{t-1} + bh);  y_t = Why h_t + by
  - LSTM, gate order i, f, g, o stacked in W (4H x V), U (4H x H), b (4H):
      [i; f; g; o] = W x_t + U h_{t-1} + b
      i = sigmoid(i), f = sigmoid(f), g = tanh(g), o = sigmoid(o)
      c_t = f * c_{t-1} + i * g;  h_t = o * tanh(c_t);  y_t = Wy h_t + by
  - prediction = argmax softmax(y_t); h_0 = c_0 = 0
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

VOCAB = ["(", ")", "."]
CLASSES = 5  # depth 0..4
MAX_DEPTH = CLASSES - 1
HIDDEN = 8
SEQ_LEN = 24
SEED = 7
DECIMALS = 6  # exported weights are rounded; metrics are measured after rounding

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- data ----------

def make_sequence(rng: np.random.Generator, length: int) -> str:
    """Random walk over depth 0..MAX_DEPTH; '.' with probability 0.2."""
    depth = 0
    out = []
    for _ in range(length):
        if rng.random() < 0.2:
            out.append(".")
            continue
        if depth == 0:
            ch = "("
        elif depth == MAX_DEPTH:
            ch = ")"
        else:
            ch = "(" if rng.random() < 0.5 else ")"
        depth += 1 if ch == "(" else -1
        out.append(ch)
    return "".join(out)


def depths(s: str) -> list[int]:
    """Nesting depth after each character."""
    d = 0
    out = []
    for ch in s:
        if ch == "(":
            d += 1
        elif ch == ")":
            d -= 1
        out.append(d)
    return out


def encode(seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One-hot inputs (T, B, V) and integer targets (T, B). All sequences must share a length."""
    T = len(seqs[0])
    X = np.zeros((T, len(seqs), len(VOCAB)))
    Y = np.zeros((T, len(seqs)), dtype=np.int64)
    for b, s in enumerate(seqs):
        for t, (ch, d) in enumerate(zip(s, depths(s))):
            X[t, b, VOCAB.index(ch)] = 1.0
            Y[t, b] = d
    return X, Y


TEST_SEED = 12345
TEST_SIZE = 500
LONG_LEN = 96
LONG_SIZE = 200


def test_set() -> list[str]:
    """Held-out set at the training length (own seed, never trained on)."""
    rng = np.random.default_rng(TEST_SEED)
    return [make_sequence(rng, SEQ_LEN) for _ in range(TEST_SIZE)]


def long_set() -> list[str]:
    """Held-out set four times longer than anything seen in training."""
    rng = np.random.default_rng(TEST_SEED + 1)
    return [make_sequence(rng, LONG_LEN) for _ in range(LONG_SIZE)]


# ---------- models ----------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def init_rnn(rng: np.random.Generator) -> dict[str, np.ndarray]:
    H, V, C = HIDDEN, len(VOCAB), CLASSES
    return {
        "Wxh": rng.normal(0, 0.5, (H, V)),
        "Whh": rng.normal(0, 1 / np.sqrt(H), (H, H)),
        "bh": np.zeros(H),
        "Why": rng.normal(0, 1 / np.sqrt(H), (C, H)),
        "by": np.zeros(C),
    }


def init_lstm(rng: np.random.Generator) -> dict[str, np.ndarray]:
    H, V, C = HIDDEN, len(VOCAB), CLASSES
    b = np.zeros(4 * H)
    b[H:2 * H] = 1.0  # forget gate starts open
    return {
        "W": rng.normal(0, 0.5, (4 * H, V)),
        "U": rng.normal(0, 1 / np.sqrt(H), (4 * H, H)),
        "b": b,
        "Wy": rng.normal(0, 1 / np.sqrt(H), (C, H)),
        "by": np.zeros(C),
    }


def softmax_xent(logits: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy over (T, B) and its gradient wrt logits."""
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    T, B = Y.shape
    idx = (np.arange(T)[:, None], np.arange(B)[None, :], Y)
    loss = -np.log(p[idx]).mean()
    g = p.copy()
    g[idx] -= 1.0
    return float(loss), g / (T * B)


def rnn_forward(P: dict[str, np.ndarray], X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    T, B, _ = X.shape
    H = P["bh"].shape[0]
    hs = np.zeros((T + 1, B, H))
    for t in range(T):
        hs[t + 1] = np.tanh(X[t] @ P["Wxh"].T + hs[t] @ P["Whh"].T + P["bh"])
    logits = hs[1:] @ P["Why"].T + P["by"]
    return logits, {"hs": hs}


def rnn_backward(P, X, cache, dlogits):
    hs = cache["hs"]
    T = X.shape[0]
    G = {k: np.zeros_like(v) for k, v in P.items()}
    G["Why"] = np.einsum("tbc,tbh->ch", dlogits, hs[1:])
    G["by"] = dlogits.sum(axis=(0, 1))
    dh_next = np.zeros_like(hs[0])
    for t in reversed(range(T)):
        dh = dlogits[t] @ P["Why"] + dh_next
        dz = dh * (1 - hs[t + 1] ** 2)
        G["Wxh"] += dz.T @ X[t]
        G["Whh"] += dz.T @ hs[t]
        G["bh"] += dz.sum(axis=0)
        dh_next = dz @ P["Whh"]
    return G


def lstm_forward(P: dict[str, np.ndarray], X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    T, B, _ = X.shape
    H = P["U"].shape[1]
    hs = np.zeros((T + 1, B, H))
    cs = np.zeros((T + 1, B, H))
    gates = np.zeros((T, B, 4 * H))
    for t in range(T):
        a = X[t] @ P["W"].T + hs[t] @ P["U"].T + P["b"]
        i = sigmoid(a[:, :H])
        f = sigmoid(a[:, H:2 * H])
        g = np.tanh(a[:, 2 * H:3 * H])
        o = sigmoid(a[:, 3 * H:])
        cs[t + 1] = f * cs[t] + i * g
        hs[t + 1] = o * np.tanh(cs[t + 1])
        gates[t] = np.concatenate([i, f, g, o], axis=1)
    logits = hs[1:] @ P["Wy"].T + P["by"]
    return logits, {"hs": hs, "cs": cs, "gates": gates}


def lstm_backward(P, X, cache, dlogits):
    hs, cs, gates = cache["hs"], cache["cs"], cache["gates"]
    T = X.shape[0]
    H = P["U"].shape[1]
    G = {k: np.zeros_like(v) for k, v in P.items()}
    G["Wy"] = np.einsum("tbc,tbh->ch", dlogits, hs[1:])
    G["by"] = dlogits.sum(axis=(0, 1))
    dh_next = np.zeros_like(hs[0])
    dc_next = np.zeros_like(cs[0])
    for t in reversed(range(T)):
        i, f, g, o = (gates[t][:, k * H:(k + 1) * H] for k in range(4))
        tc = np.tanh(cs[t + 1])
        dh = dlogits[t] @ P["Wy"] + dh_next
        do = dh * tc
        dc = dh * o * (1 - tc ** 2) + dc_next
        di = dc * g
        dg = dc * i
        df = dc * cs[t]
        da = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
        G["W"] += da.T @ X[t]
        G["U"] += da.T @ hs[t]
        G["b"] += da.sum(axis=0)
        dh_next = da @ P["U"]
        dc_next = dc * f
    return G


MODELS = {
    "rnn": (init_rnn, rnn_forward, rnn_backward),
    "lstm": (init_lstm, lstm_forward, lstm_backward),
}


# ---------- training ----------

def gradcheck(kind: str) -> float:
    """Largest relative error between backprop and central differences on a tiny batch."""
    init, fwd, bwd = MODELS[kind]
    rng = np.random.default_rng(0)
    P = init(rng)
    X, Y = encode([make_sequence(rng, 6) for _ in range(3)])
    logits, cache = fwd(P, X)
    _, dl = softmax_xent(logits, Y)
    G = bwd(P, X, cache, dl)
    worst = 0.0
    eps = 1e-6
    for k, v in P.items():
        for idx in list(np.ndindex(v.shape))[:12]:
            old = v[idx]
            v[idx] = old + eps
            lp, _ = softmax_xent(fwd(P, X)[0], Y)
            v[idx] = old - eps
            lm, _ = softmax_xent(fwd(P, X)[0], Y)
            v[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - G[k][idx]) / max(1e-8, abs(num) + abs(G[k][idx])))
    return worst


def train(kind: str, iters: int = 4000, batch: int = 64, lr: float = 0.01) -> dict[str, np.ndarray]:
    """Adam on mean cross-entropy, gradient norm clipped to 1."""
    init, fwd, bwd = MODELS[kind]
    rng = np.random.default_rng(SEED)
    P = init(rng)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    s = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2 = 0.9, 0.999
    for it in range(1, iters + 1):
        X, Y = encode([make_sequence(rng, SEQ_LEN) for _ in range(batch)])
        logits, cache = fwd(P, X)
        loss, dl = softmax_xent(logits, Y)
        G = bwd(P, X, cache, dl)
        norm = np.sqrt(sum((g ** 2).sum() for g in G.values()))
        scale = min(1.0, 1.0 / max(norm, 1e-12))
        for k in P:
            g = G[k] * scale
            m[k] = b1 * m[k] + (1 - b1) * g
            s[k] = b2 * s[k] + (1 - b2) * g * g
            mh = m[k] / (1 - b1 ** it)
            sh = s[k] / (1 - b2 ** it)
            P[k] -= lr * mh / (np.sqrt(sh) + 1e-8)
        if it % 500 == 0:
            print(f"{kind} iter {it} loss {loss:.4f}", file=sys.stderr)
    return P


def evaluate(kind: str, P: dict[str, np.ndarray], seqs: list[str]) -> dict[str, float]:
    """Per-character accuracy and share of sequences with every character right."""
    _, fwd, _ = MODELS[kind]
    X, Y = encode(seqs)
    logits, _ = fwd(P, X)
    ok = logits.argmax(axis=-1) == Y
    return {"stepAccuracy": float(ok.mean()), "sequenceAccuracy": float(ok.all(axis=0).mean())}


def rounded(P: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.round(v, DECIMALS) for k, v in P.items()}


def main() -> None:
    for kind in MODELS:
        err = gradcheck(kind)
        print(f"{kind} gradcheck max relative error {err:.2e}", file=sys.stderr)
        assert err < 1e-5, f"{kind} gradients do not match finite differences"
    tests = test_set()
    longs = long_set()
    out: dict[str, object] = {
        "task": "bracket depth: after each character, the nesting depth 0-4",
        "vocab": VOCAB,
        "classes": [str(d) for d in range(CLASSES)],
        "hidden": HIDDEN,
        "seqLen": SEQ_LEN,
        "seed": SEED,
        "decimals": DECIMALS,
        "testSet": {"size": TEST_SIZE, "length": SEQ_LEN, "seed": TEST_SEED},
        "longSet": {"size": LONG_SIZE, "length": LONG_LEN, "seed": TEST_SEED + 1},
    }
    for kind in MODELS:
        P = rounded(train(kind))
        metrics = {"test": evaluate(kind, P, tests), "long": evaluate(kind, P, longs)}
        print(f"{kind} test {metrics}", file=sys.stderr)
        out[kind] = {k: v.tolist() for k, v in P.items()} | {"metrics": metrics}
    path = os.path.join(HERE, "weights.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
        fh.write("\n")
    print(f"wrote {path} ({os.path.getsize(path)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
