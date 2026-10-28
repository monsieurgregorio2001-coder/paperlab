#!/usr/bin/env python3
"""NumPy reference for the Attention room (ADR 0003, ADR 0009). NumPy only.

    python3 -I scripts/attention_fixture.py      # writes tests/fixtures/attention.json

Independent of src/lib: the stated head weights are rebuilt here from the rules in words, and
softmax(Q K^T / sqrt(d_k)) V is computed with vectorised NumPy. src/lib/designed.test.ts loads the
file and compares at 1e-6. Never edit the JSON by hand; rerun this.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D_MODEL, D_K = 16, 4
GAIN = {"previous": 8.0, "next": 8.0, "same": 1.0, "punctuation": 6.0}
CODE, PUNCT, ONE = [12, 13, 14, 15], 8, 10


def designed(kind):
    wq = np.zeros((D_MODEL, D_K))
    wk = np.zeros((D_MODEL, D_K))
    if kind in ("previous", "next"):
        shift = -1.0 if kind == "previous" else 1.0
        for f in range(2):
            w = 1.0 / 10000.0 ** (2 * f / D_MODEL)
            a = 2 * f
            wk[a, a] = wk[a + 1, a + 1] = 1.0
            # rotate the (sin, cos) pair of position p to the pair of position p + shift
            # row-vector convention: q = x @ wq, so wq is the transpose of the rotation matrix
            rot = np.array([[np.cos(shift * w), -np.sin(shift * w)], [np.sin(shift * w), np.cos(shift * w)]])
            wq[a:a + 2, a:a + 2] = GAIN[kind] * rot
    elif kind == "same":
        for c, d in enumerate(CODE):
            wq[d, c] = GAIN["same"]
            wk[d, c] = 1.0
    else:
        wq[ONE, 0] = GAIN["punctuation"]
        wk[PUNCT, 0] = 1.0
    return wq, wk


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def attend(x, wq, wk, wv, causal, scale):
    q, k, v = x @ wq, x @ wk, x @ wv
    s = q @ k.T / (np.sqrt(D_K) if scale else 1.0)
    if causal:
        s = np.where(np.tril(np.ones_like(s)) > 0, s, -np.inf)
    w = softmax(s)
    return s, w, w @ v


def main():
    rng = np.random.default_rng(20261028)
    cases = []
    for n in (1, 5, 9):
        x = rng.normal(size=(n, D_MODEL))
        wv = rng.normal(size=(D_MODEL, D_K)) / 4
        for h, kind in enumerate(("previous", "same", "punctuation", "next")):
            wq, wk = designed(kind)
            for causal in (False, True):
                for scale in (True, False):
                    s, w, o = attend(x, wq, wk, wv, causal, scale)
                    cases.append({
                        "head": h, "kind": kind, "n": n, "causal": causal, "scale": scale,
                        "X": x.tolist(), "Wv": wv.tolist(), "Wq": wq.tolist(), "Wk": wk.tolist(),
                        "scores": np.where(np.isinf(s), None, s).tolist(), "weights": w.tolist(), "output": o.tolist(),
                    })
    os.makedirs(os.path.join(ROOT, "tests", "fixtures"), exist_ok=True)
    with open(os.path.join(ROOT, "tests", "fixtures", "attention.json"), "w") as f:
        json.dump({"cases": cases}, f)
    print(f"attention.json: {len(cases)} cases")


if __name__ == "__main__":
    main()
