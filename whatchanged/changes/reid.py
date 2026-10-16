"""Re-identification features shared by the calibration and the inference.

p_same(detection, object) is a logistic model on five terms: the best SigLIP 2 cosine between the detection's crop and
the object's reference crops, the DINOv2 cosine to the object's fingerprint, the box-size term, the colour distance
and the detection's z-normalised SAM 3 score. p_added(detection) uses the mean P(appeared) under the mask, the SAM 3
z-score and the log size.
"""

from __future__ import annotations

import numpy as np

from whatchanged.common.features import dims_term
from whatchanged.common.io import read_json
from whatchanged.common.scene import Scene


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_logit(X, y, l2: float = 1e-2, iters: int = 4000, lr: float = 0.3) -> dict:
    """Standardised logistic regression by gradient descent."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = sigmoid(Xs @ w)
        w -= lr * (Xs.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return dict(w=w.tolist(), mu=mu.tolist(), sd=sd.tolist())


def predict_logit(m: dict, X) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    return sigmoid(np.c_[(X - np.array(m["mu"])) / np.array(m["sd"]), np.ones(len(X))] @ np.array(m["w"]))


def auc(p, y) -> float:
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def zscore(stats: dict, phrase: str, s: float) -> float:
    mu, sd = stats.get(phrase, (0.5, 0.2))
    return (s - mu) / max(sd, 1e-3)


class Appearance:
    """SigLIP 2 embeddings of every crop, the objects' reference crops and their DINOv2 fingerprints."""

    def __init__(self, scene: Scene):
        z = np.load(scene.changes / "siglip.npz")
        self.k = {str(k): i for i, k in enumerate(z["keys"])}
        self.emb = z["emb"].astype(np.float32)
        self.prob = z["probs"].astype(np.float32)
        refs = read_json(scene.changes / "references.json")
        self.ref = {}
        for c in refs["crops"]:
            if c["key"] in self.k:
                self.ref.setdefault(c["object"], []).append(self.emb[self.k[c["key"]]])
        self.fp = {k: dict(emb=np.array(v["emb"]), col=np.array(v["col"])) for k, v in refs["ref"].items()}

    def cos_ref(self, key: str, oid: str) -> float:
        if key not in self.k or oid not in self.ref:
            return 0.0
        return float(max(np.array(self.ref[oid]) @ self.emb[self.k[key]]))


def pair_features(row: dict, o: dict, app: Appearance) -> list[float]:
    sig = app.cos_ref(row["key"], o["id"])
    fp = app.fp.get(o["id"])
    dino = float(np.array(row["emb"]) @ fp["emb"]) if (fp is not None and row.get("emb") is not None) else 0.0
    dims = dims_term(np.array(row["box"]["size"]), np.array(o["box"]["size"])) if row.get("box") else 1.0
    col = float(np.linalg.norm(np.subtract(row["col"], fp["col"]))) if fp is not None else 0.5
    return [sig, dino, dims, col, row["z"]]


def added_features(row: dict) -> list[float]:
    return [row["p_appeared"], row["z"], np.log(max(max(row["box"]["size"]), 0.02))]


def phrases_of(o: dict) -> set[str]:
    """The object's name and runner-up name: the phrases a detection of it may carry."""
    return {o["name"].replace(" (uncertain)", ""), o.get("runner_up") or ""} - {"", "object"}
