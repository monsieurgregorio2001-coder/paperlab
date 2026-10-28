#!/usr/bin/env python3
"""Reference values for the lab kernels (ADR 0003). NumPy only.

    python3 -I scripts/fixtures.py          # writes tests/fixtures/{line,boundary,net,memory}.json

Every value here is computed independently of the TypeScript in src/labs/*/math.ts:
vectorised NumPy for the maths, and a bit-exact port of mulberry32 for the seeded
datasets and initial weights so the generators can be checked too. The Vitest
suites load these files and compare. Never edit the JSON by hand; rerun this.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures")
M32 = 0xFFFFFFFF


# ---------- seeded randomness, ported from src/labs/_shared/rand.ts ----------

def imul(a: int, b: int) -> int:
    return (a * b) & M32


def mulberry32(seed: int):
    state = [seed & M32]

    def rand() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & M32
        t = state[0]
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return rand


def hash_string(s: str) -> int:
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = imul(h, 0x01000193)
    return h


def normal(rand) -> float:
    u1 = rand()
    u2 = rand()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)


def uniform(rand, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rand()


def tolist(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tolist(v) for v in x]
    return x


def write(name: str, data: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(tolist(data), fh, indent=1)
        fh.write("\n")
    print(f"wrote {os.path.relpath(path, ROOT)} ({os.path.getsize(path)} bytes)")


# ---------- Lab 1: line ----------

def line_fixtures() -> dict:
    rng = np.random.default_rng(1)
    x2 = rng.uniform(-2, 2, 30)
    y2 = 0.8 * x2 - 0.3 + rng.normal(0, 0.4, 30)
    x3 = rng.uniform(-1, 1, 50)
    y3 = -2.5 * x3 + 1.7 + rng.normal(0, 0.2, 50)
    datasets = {
        "hand": (np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.2, 6.8]), 0.05),
        "noisy": (x2, y2, 0.1),
        "steep": (x3, y3, 0.1),
    }
    cases = []
    for name, (x, y, lr) in datasets.items():
        A = np.stack([x, np.ones_like(x)], axis=1)
        (w, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        w_, b_ = 0.0, 0.0
        trace = []
        for step in range(21):
            r = w_ * x + b_ - y
            dw, db = 2 * np.mean(r * x), 2 * np.mean(r)
            trace.append({"step": step, "w": w_, "b": b_, "loss": np.mean(r * r), "dw": dw, "db": db})
            w_, b_ = w_ - lr * dw, b_ - lr * db
        cases.append({
            "name": name,
            "points": [{"x": a, "y": c} for a, c in zip(x, y)],
            "ols": {"w": w, "b": b, "loss": np.mean((w * x + b - y) ** 2)},
            "gd": {"lr": lr, "steps": 20, "init": {"w": 0.0, "b": 0.0}, "trace": trace},
        })
    hx, hy, _ = datasets["hand"]
    ws = np.linspace(-1, 3, 5)
    bs = np.linspace(-2, 2, 5)
    W, B = np.meshgrid(ws, bs)  # W[i, j] = ws[j], B[i, j] = bs[i]
    loss = ((W[..., None] * hx + B[..., None] - hy) ** 2).mean(axis=-1)
    surface = {"wRange": [-1, 3], "bRange": [-2, 2], "n": 5, "ws": ws, "bs": bs, "loss": loss}
    return {"cases": cases, "surface": surface}


# ---------- Lab 2: boundary ----------

def two_blobs(n: int, seed: int, sep: float = 1.0, spread: float = 0.3) -> list[dict]:
    rand = mulberry32(seed)
    out = []
    for k in range(n):
        label = 1 if k % 2 == 0 else 0
        c = sep / 2 if label == 1 else -sep / 2
        x = c + spread * normal(rand)
        y = c + spread * normal(rand)
        out.append({"x": x, "y": y, "label": label})
    return out


def xor(n: int, seed: int, noise: float = 0.0) -> list[dict]:
    rand = mulberry32(seed)
    out = []
    for _ in range(n):
        x = uniform(rand, -1, 1)
        y = uniform(rand, -1, 1)
        x += 0.1 if x >= 0 else -0.1
        y += 0.1 if y >= 0 else -0.1
        label = 1 if x * y > 0 else 0
        if noise > 0:
            x += uniform(rand, -noise, noise)
            y += uniform(rand, -noise, noise)
        out.append({"x": x, "y": y, "label": label})
    return out


def arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[p["x"], p["y"]] for p in points])
    t = np.array([p["label"] for p in points], dtype=float)
    return X, t


def np_sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1 + np.exp(-np.abs(z))))


def logistic_trace(points: list[dict], lr: float, steps: int) -> list[dict]:
    X, t = arrays(points)
    w = np.zeros(2)
    b = 0.0
    out = []
    for step in range(steps + 1):
        z = X @ w + b
        loss = np.mean(np.logaddexp(0, z) - t * z)
        acc = np.mean((z >= 0).astype(float) == t)
        out.append({"step": step, "w": w.copy(), "b": b, "loss": loss, "acc": acc})
        r = np_sigmoid(z) - t
        w = w - lr * (X.T @ r) / len(t)
        b = b - lr * r.mean()
    return out


def perceptron_ref(points: list[dict], max_updates: int, max_epochs: int = 100) -> dict:
    X, t01 = arrays(points)
    t = 2 * t01 - 1
    w = np.zeros(2)
    b = 0.0
    steps = []
    updates = 0
    converged = False
    done = False
    for epoch in range(max_epochs):
        fired = False
        for i in range(len(t)):
            wrong = bool(t[i] * (X[i] @ w + b) <= 0)
            before = {"w": w.copy(), "b": b}
            if wrong:
                w = w + t[i] * X[i]
                b = b + t[i]
                updates += 1
                fired = True
            steps.append({"i": i, "epoch": epoch, "misclassified": wrong, "wBefore": before, "wAfter": {"w": w.copy(), "b": b}})
            if updates >= max_updates:
                done = True
                break
        if done:
            break
        if not fired:
            converged = True
            break
    return {"steps": steps, "updates": updates, "converged": converged, "model": {"w": w, "b": b}}


def boundary_fixtures() -> dict:
    blobs = two_blobs(40, 3)
    wide = two_blobs(30, 9, 1.0, 0.6)
    xr = xor(40, 5, 0.05)
    zs = np.array([-800.0, -20.0, -1.0, 0.0, 0.5, 3.0, 800.0])
    model = {"w": [1.5, -0.5], "b": 0.25}
    bounds = {"xMin": -1.0, "xMax": 1.0, "yMin": -2.0, "yMax": 2.0}
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-2, 2, 4)
    G = np.array([[model["w"][0] * x + model["w"][1] * y + model["b"] for x in xs] for y in ys])
    return {
        "datasets": {
            "twoBlobs": {"n": 40, "seed": 3, "points": blobs},
            "twoBlobsWide": {"n": 30, "seed": 9, "sep": 1.0, "spread": 0.6, "points": wide},
            "xor": {"n": 40, "seed": 5, "noise": 0.05, "points": xr},
        },
        "sigmoid": {"z": zs, "p": np_sigmoid(zs)},
        "logistic": [
            {"dataset": "twoBlobs", "lr": 0.5, "steps": 50, "trace": logistic_trace(blobs, 0.5, 50)},
            {"dataset": "xor", "lr": 0.5, "steps": 50, "trace": logistic_trace(xr, 0.5, 50)},
        ],
        "perceptron": {"dataset": "twoBlobsWide", "maxUpdates": 10, **perceptron_ref(wide, 10)},
        "sampler": {"model": model, "bounds": bounds, "n": 4, "xs": xs, "ys": ys, "p": np_sigmoid(G)},
    }


# ---------- Lab 3: net ----------

def spiral(n: int, seed: int, noise: float = 0.0) -> list[dict]:
    rand = mulberry32(seed)
    half = n / 2
    out = []
    for k in range(n):
        label = k % 2
        r = (k // 2) / half
        t = 3.5 * math.pi * r + math.pi * label
        x = r * math.sin(t) + uniform(rand, -noise, noise)
        y = r * math.cos(t) + uniform(rand, -noise, noise)
        out.append({"x": x, "y": y, "label": label})
    return out


def moons(n: int, seed: int, noise: float = 0.0) -> list[dict]:
    rand = mulberry32(seed)
    out = []
    for k in range(n):
        label = k % 2
        t = math.pi * rand()
        x0 = math.cos(t) if label == 0 else 1 - math.cos(t)
        y0 = math.sin(t) if label == 0 else 0.5 - math.sin(t)
        x = (x0 - 0.5) / 1.5 + uniform(rand, -noise, noise)
        y = (y0 - 0.25) / 1.5 + uniform(rand, -noise, noise)
        out.append({"x": x, "y": y, "label": label})
    return out


def circles(n: int, seed: int, noise: float = 0.0) -> list[dict]:
    rand = mulberry32(seed)
    out = []
    for k in range(n):
        label = 1 if k % 2 == 0 else 0
        r = uniform(rand, 0, 0.5) if label == 1 else uniform(rand, 0.7, 1)
        t = uniform(rand, 0, 2 * math.pi)
        x = r * math.sin(t) + uniform(rand, -noise, noise)
        y = r * math.cos(t) + uniform(rand, -noise, noise)
        out.append({"x": x, "y": y, "label": label})
    return out


ACT = {
    "tanh": (np.tanh, lambda z, a: 1 - a * a),
    "relu": (lambda z: np.maximum(z, 0), lambda z, a: (z > 0).astype(float)),
    "sigmoid": (np_sigmoid, lambda z, a: a * (1 - a)),
    "linear": (lambda z: z, lambda z, a: np.ones_like(z)),
}


class Net:
    """Reference MLP: W[l] has shape (out, in), as in math.ts."""

    def __init__(self, sizes, hidden, output, seed):
        rand = mulberry32(seed)
        self.sizes, self.hidden, self.output = list(sizes), hidden, output
        self.W, self.b = [], []
        for l in range(len(sizes) - 1):
            fi, fo = sizes[l], sizes[l + 1]
            r = math.sqrt(6 / (fi + fo))
            self.W.append(np.array([[uniform(rand, -r, r) for _ in range(fi)] for _ in range(fo)]))
            self.b.append(np.zeros(fo))
        self.dead = [np.zeros_like(w, dtype=bool) for w in self.W]

    def act(self, l):
        return self.output if l == len(self.W) - 1 else self.hidden

    def loss_kind(self):
        return "bce" if self.output == "sigmoid" else "mse"

    def target(self, label):
        return 2 * label - 1 if self.output == "tanh" else label

    def forward(self, X):
        """X: (B, in). Returns z list and a list, batched."""
        a = [X]
        z = []
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            zl = a[-1] @ W.T + b
            z.append(zl)
            a.append(ACT[self.act(l)][0](zl))
        return z, a

    def backward(self, X, T):
        """Per-example gradients summed over the batch, plus per-example losses."""
        z, a = self.forward(X)
        L = len(self.W)
        kind = self.loss_kind()
        if kind == "bce":
            losses = (np.logaddexp(0, z[-1]) - T * z[-1]).sum(axis=1)
            d = a[-1] - T
        else:
            losses = (0.5 * (a[-1] - T) ** 2).sum(axis=1)
            d = (a[-1] - T) * ACT[self.output][1](z[-1], a[-1])
        deltas = [None] * L
        deltas[L - 1] = d
        for l in range(L - 2, -1, -1):
            deltas[l] = (deltas[l + 1] @ self.W[l + 1]) * ACT[self.hidden][1](z[l], a[l + 1])
        dW = [deltas[l].T @ a[l] for l in range(L)]
        db = [deltas[l].sum(axis=0) for l in range(L)]
        return dW, db, deltas, losses, z, a

    def sgd(self, X, T, lr, reg="none", lam=0.0):
        dW, db, _, losses, _, _ = self.backward(X, T)
        n = len(X)
        for l in range(len(self.W)):
            g = dW[l] / n
            r = np.sign(self.W[l]) if reg == "l1" else self.W[l] if reg == "l2" else 0.0
            new = self.W[l] - lr * (g + lam * r)
            was_dead = self.dead[l]
            new[was_dead] = self.W[l][was_dead]  # dead weights never update
            if reg == "l1":
                cross = (self.W[l] != 0) & (self.W[l] * new < 0) & ~was_dead
                new[cross] = 0.0
                self.dead[l] = was_dead | cross
            self.W[l] = new
            self.b[l] = self.b[l] - lr * db[l] / n
        return losses.mean(), [d / n for d in dW], [d / n for d in db]

    def state(self):
        return {"W": [w.copy() for w in self.W], "b": [b.copy() for b in self.b], "dead": [d.copy() for d in self.dead]}

    def dataset_loss(self, X, T):
        _, _, _, losses, _, _ = self.backward(X, T)
        return losses.mean()

    def accuracy(self, X, labels):
        th = 0.0 if self.output == "tanh" else 0.5
        out = self.forward(X)[1][-1][:, 0]
        return np.mean((out >= th).astype(float) == labels)


def net_xy(points, net):
    X = np.array([[p["x"], p["y"]] for p in points])
    labels = np.array([p["label"] for p in points], dtype=float)
    T = np.array([[net.target(l)] for l in labels])
    return X, labels, T


def net_case(name, sizes, hidden, output, seed, inp, label, batch, lr, reg, lam):
    net = Net(sizes, hidden, output, seed)
    init = net.state()
    T1 = np.array([[net.target(label)]])
    dW, db, deltas, losses, z, a = net.backward(np.array([inp]), T1)
    X, _, T = net_xy(batch, net)
    loss, gW, gb = net.sgd(X, T, lr, reg, lam)
    return {
        "name": name,
        "sizes": sizes, "hidden": hidden, "output": output, "seed": seed,
        "init": {"W": init["W"], "b": init["b"]},
        "input": inp, "label": label,
        "forward": {"z": [v[0] for v in z], "a": [v[0] for v in a]},
        "backward": {"dW": dW, "db": db, "delta": [d[0] for d in deltas], "loss": losses[0]},
        "sgd": {
            "batch": batch, "lr": lr, "reg": reg, "lambda": lam,
            "loss": loss, "gradW": gW, "gradB": gb,
            "after": {"W": net.W, "b": net.b, "dead": net.dead},
        },
    }


def net_fixtures() -> dict:
    ds = {
        "spiral": {"n": 20, "seed": 1, "noise": 0.1, "points": spiral(20, 1, 0.1)},
        "moons": {"n": 20, "seed": 2, "noise": 0.1, "points": moons(20, 2, 0.1)},
        "circles": {"n": 20, "seed": 3, "noise": 0.1, "points": circles(20, 3, 0.1)},
        "xor": {"n": 20, "seed": 4, "noise": 0.1, "points": xor(20, 4, 0.1)},
    }
    batch = ds["moons"]["points"][:8]
    cases = [
        net_case("tanh-sigmoid-bce", [2, 4, 3, 1], "tanh", "sigmoid", 11, [0.3, -0.7], 1, batch, 0.3, "none", 0.0),
        net_case("tanh-sigmoid-l2", [2, 4, 3, 1], "tanh", "sigmoid", 11, [0.3, -0.7], 1, batch, 0.3, "l2", 0.01),
        net_case("tanh-sigmoid-l1", [2, 4, 3, 1], "tanh", "sigmoid", 11, [0.3, -0.7], 1, batch, 0.3, "l1", 0.5),
        net_case("relu-tanh-mse", [2, 5, 5, 1], "relu", "tanh", 4, [-0.4, 0.9], 0, batch, 0.1, "none", 0.0),
        net_case("sigmoid-linear-mse", [2, 3, 1], "sigmoid", "linear", 6, [0.8, 0.2], 1, batch, 0.2, "l2", 0.05),
    ]
    l1 = cases[2]
    assert any(d.any() for d in l1["sgd"]["after"]["dead"]), "the L1 case should kill at least one weight"

    # three training steps (epochs) on 40 spiral points, batches of 10
    data = spiral(40, 8, 0.05)
    net = Net([2, 8, 8, 1], "tanh", "sigmoid", 21)
    X, labels, T = net_xy(data, net)
    hist = [{"loss": net.dataset_loss(X, T), "acc": net.accuracy(X, labels)}]
    for _ in range(3):
        for s in range(0, len(data), 10):
            net.sgd(X[s:s + 10], T[s:s + 10], 0.1, "l2", 0.001)
        hist.append({"loss": net.dataset_loss(X, T), "acc": net.accuracy(X, labels)})
    train = {
        "dataset": {"kind": "spiral", "n": 40, "seed": 8, "noise": 0.05},
        "sizes": [2, 8, 8, 1], "hidden": "tanh", "output": "sigmoid", "seed": 21,
        "opts": {"lr": 0.1, "batchSize": 10, "reg": "l2", "lambda": 0.001},
        "steps": 3, "history": hist, "final": {"W": net.W, "b": net.b},
    }

    xs = np.linspace(-1, 1, 3)
    ys = np.linspace(-1, 1, 3)
    grid = np.array([[x, y] for y in ys for x in xs])
    out = net.forward(grid)[1][-1][:, 0].reshape(3, 3)
    surface = {"bounds": {"xMin": -1, "xMax": 1, "yMin": -1, "yMax": 1}, "n": 3, "xs": xs, "ys": ys, "out": out}
    return {"datasets": ds, "cases": cases, "train": train, "surface": surface}


# ---------- Lab 6: memory ----------

def load_train_module():
    path = os.path.join(ROOT, "models", "memory", "train.py")
    spec = importlib.util.spec_from_file_location("memory_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mem_encode(text: str, vocab: list[str]) -> list[int]:
    fb = vocab.index(".")
    return [vocab.index(c) if c in vocab else fb for c in text]


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def rnn_trace(P: dict, seq: list[int]) -> list[dict]:
    Wxh, Whh, bh, Why, by = (np.array(P[k]) for k in ("Wxh", "Whh", "bh", "Why", "by"))
    V = Wxh.shape[1]
    h = np.zeros(Whh.shape[0])
    out = []
    for t, k in enumerate(seq):
        x = np.eye(V)[k]
        z = Wxh @ x + Whh @ h + bh
        h = np.tanh(z)
        logits = Why @ h + by
        p = softmax(logits)
        out.append({"t": t, "input": k, "z": z, "h": h, "logits": logits, "probs": p, "pred": int(p.argmax())})
    return out


def lstm_trace(P: dict, seq: list[int]) -> list[dict]:
    W, U, b, Wy, by = (np.array(P[k]) for k in ("W", "U", "b", "Wy", "by"))
    V = W.shape[1]
    H = U.shape[1]
    h = np.zeros(H)
    c = np.zeros(H)
    out = []
    for t, k in enumerate(seq):
        a = W @ np.eye(V)[k] + U @ h + b
        i, f, o = (1 / (1 + np.exp(-a[s * H:(s + 1) * H])) for s in (0, 1, 3))
        g = np.tanh(a[2 * H:3 * H])
        c = f * c + i * g
        h = o * np.tanh(c)
        logits = Wy @ h + by
        p = softmax(logits)
        out.append({"t": t, "input": k, "i": i, "f": f, "g": g, "o": o, "c": c, "h": h,
                    "logits": logits, "probs": p, "pred": int(p.argmax())})
    return out


def accuracy(kind: str, P: dict, texts: list[str], vocab: list[str], depths) -> dict:
    steps = ok = seqs = 0
    for s in texts:
        tr = (rnn_trace if kind == "rnn" else lstm_trace)(P, mem_encode(s, vocab))
        right = [step["pred"] == d for step, d in zip(tr, depths(s))]
        steps += len(right)
        ok += sum(right)
        seqs += all(right)
    return {"stepAccuracy": ok / steps, "sequenceAccuracy": seqs / len(texts)}


def memory_fixtures() -> dict:
    with open(os.path.join(ROOT, "models", "memory", "weights.json")) as fh:
        weights = json.load(fh)
    vocab = weights["vocab"]
    tm = load_train_module()
    tests = tm.test_set()
    longs = tm.long_set()
    for kind in ("rnn", "lstm"):
        for name, texts in (("test", tests), ("long", longs)):
            got = accuracy(kind, weights[kind], texts, vocab, tm.depths)
            want = weights[kind]["metrics"][name]
            assert got == want, f"{kind} {name}: fixtures forward {got} != train.py {want}"

    texts = ["(()(.))((", "f(a(b)c)d", "((((.))))"]
    trained = []
    for text in texts:
        seq = mem_encode(text, vocab)
        trained.append({"text": text, "seq": seq, "depths": tm.depths(text),
                        "rnn": rnn_trace(weights["rnn"], seq), "lstm": lstm_trace(weights["lstm"], seq)})

    # small random weights, so the cells are checked away from the trained solution too
    rng = np.random.default_rng(99)
    H, V, C = 3, 3, 2
    rnn_w = {"Wxh": rng.normal(0, 1, (H, V)), "Whh": rng.normal(0, 1, (H, H)), "bh": rng.normal(0, 1, H),
             "Why": rng.normal(0, 1, (C, H)), "by": rng.normal(0, 1, C)}
    lstm_w = {"W": rng.normal(0, 1, (4 * H, V)), "U": rng.normal(0, 1, (4 * H, H)), "b": rng.normal(0, 1, 4 * H),
              "Wy": rng.normal(0, 1, (C, H)), "by": rng.normal(0, 1, C)}
    seq = [0, 2, 1, 1, 0, 2, 2, 0]
    random = {"seq": seq, "rnnWeights": rnn_w, "lstmWeights": lstm_w,
              "rnn": rnn_trace(rnn_w, seq), "lstm": lstm_trace(lstm_w, seq)}

    return {"trained": trained, "random": random, "testSet": tests, "longSet": longs}


# ---------- the RNG port itself ----------

def rng_fixture() -> dict:
    r = mulberry32(42)
    seq = [r() for _ in range(5)]
    return {"seed": 42, "first5": seq, "hash": {"paperlab": hash_string("paperlab")}}


def main() -> None:
    write("rand", rng_fixture())
    write("line", line_fixtures())
    write("boundary", boundary_fixtures())
    write("net", net_fixtures())
    write("memory", memory_fixtures())


if __name__ == "__main__":
    main()
