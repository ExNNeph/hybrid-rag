"""Tests never load the real models: a hashed bag-of-words embedding and a word-overlap reranker stand in."""
import hashlib

import numpy as np
import pytest

from hybrid_rag import answer, index


def fake_embed(texts, query):
    out = np.zeros((len(texts), 256))
    for row, t in enumerate(texts):
        for w in index.tokens(t):
            out[row, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def fake_cross(question, texts):
    q = set(index.tokens(question))
    return [float(len(q & set(index.tokens(t)))) for t in texts]


@pytest.fixture(autouse=True)
def no_models(monkeypatch, tmp_path):
    monkeypatch.setattr(index, "embed", fake_embed)
    monkeypatch.setattr(answer, "embed", fake_embed)
    monkeypatch.setattr(index, "cross_scores", fake_cross)
    monkeypatch.setattr(index, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
