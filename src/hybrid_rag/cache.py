"""Semantic cache. The embedding finds the closest earlier question; it counts as the same question only if the
similarity is above the threshold AND both use the same content words and the same numbers.

Similarity alone cannot tell near-duplicates apart: in the eval, "Kündigungsfrist in der Probezeit" vs "nach der
Probezeit" scores 0.993 while a true paraphrase scores 0.957. So the lexical guard decides: word order, stop words,
inflection and punctuation may differ; a different plan, number, place or preposition may not."""
import json
import re
from pathlib import Path

import numpy as np

from .index import tokens

THRESHOLD = 0.95  # cosine similarity of the question embeddings, see the sweep in evals/run_eval.py


def same_words(a: str, b: str) -> bool:
    """Same content words and numbers. ponytail: no synonyms, so 'cancel my plan' vs 'end my subscription' is a
    miss; that costs an API call, while a false hit would cost a wrong answer."""
    return set(tokens(a)) == set(tokens(b)) and numbers(a) == numbers(b)


def numbers(s: str) -> set[str]:
    """'1.000', '1,000' and '1000' are one number, as are '99.9' and '99,9'."""
    return {re.sub(r"[.,]", "", n) for n in re.findall(r"\d+(?:[.,]\d+)*", s)}


class SemanticCache:
    def __init__(self, threshold: float = THRESHOLD, path: Path | None = None, guard: bool = True):
        self.threshold, self.path, self.guard = threshold, path, guard
        self.entries: list[tuple[str, np.ndarray, object]] = []  # (question, embedding, Answer)
        self.lookups = self.hits = 0
        self.saved = 0.0  # $ of API calls not made
        if path and path.exists():
            from .answer import Answer
            for e in json.loads(path.read_text(encoding="utf-8")):
                self.entries.append((e["question"], np.array(e["vec"]), Answer(**e["answer"])))

    def closest(self, vec: np.ndarray) -> tuple[int, float]:
        if not self.entries:
            return -1, 0.0
        sims = np.stack([e[1] for e in self.entries]) @ vec
        return int(sims.argmax()), float(sims.max())

    def lookup(self, question: str, vec: np.ndarray):
        """(cached answer or None, similarity to the closest cached question, that question)."""
        self.lookups += 1
        i, sim = self.closest(vec)
        if i < 0:
            return None, 0.0, ""
        q, _, a = self.entries[i]
        if sim < self.threshold or (self.guard and not same_words(q, question)):
            return None, sim, q
        self.hits += 1
        self.saved += a.cost
        return a.model_copy(update={"question": question, "cached_from": q, "similarity": sim, "closest": q,
                                    "cost": 0.0}), sim, q

    def add(self, question: str, vec: np.ndarray, answer) -> None:
        self.entries.append((question, vec, answer))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([{"question": q, "vec": v.round(5).tolist(), "answer": a.model_dump()}
                                             for q, v, a in self.entries], ensure_ascii=False), encoding="utf-8")

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0
