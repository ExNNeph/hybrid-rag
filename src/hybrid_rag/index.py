"""Chunks and four ways to rank them for a question: BM25, dense (local embeddings), hybrid (reciprocal rank
fusion of the two) and hybrid + a local cross-encoder rerank."""
import hashlib
import math
import os
import re
from collections import Counter
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from pydantic import BaseModel

EMBED_MODEL = os.getenv("HYBRID_RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
RERANK_MODEL = os.getenv("HYBRID_RAG_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
CACHE_DIR = Path(os.getenv("HYBRID_RAG_CACHE_DIR", Path.home() / ".cache" / "hybrid-rag"))
POOL = 20  # candidates each retriever hands to fusion, and fusion hands to the reranker
RRF_K = 60  # the usual constant from the RRF paper: dampens the gap between rank 1 and rank 2

STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are", "be", "do",
        "does", "can", "i", "we", "my", "our", "this", "that", "it", "its", "you", "your", "what", "how", "which",
        "when", "who", "there", "if", "at", "as", "from", "get", "have", "has", "much", "many",
        "der", "die", "das", "und", "oder", "ein", "eine", "einer", "eines", "einen", "zu", "im", "auf", "für",
        "mit", "von", "vom", "bei", "ist", "sind", "wird", "werden", "sie", "ihr", "ihre", "den", "dem", "des",
        "auch", "nicht", "als", "an", "am", "so", "sich", "ich", "wir", "mein", "meine", "wie", "was", "welche",
        "wann", "wer", "gibt", "es", "kann", "uns", "unser", "unsere", "muss", "viele", "hoch"}
DE_HINTS = {"der", "die", "das", "und", "ist", "ich", "wie", "was", "wann", "für", "mit", "gibt", "nach", "bei"}
EN_HINTS = {"the", "and", "is", "i", "how", "what", "when", "for", "with", "are", "after", "do", "does", "can"}
SUFFIXES = {"en": ("ations", "ation", "ities", "ity", "ing", "ies", "ed", "s"),
            "de": ("ungen", "ung", "ern", "en", "er", "es", "e", "n", "s")}


class Chunk(BaseModel):
    id: str  # "<doc>#<section-slug>", stable, used by the eval labels and in citations
    doc: str
    title: str
    section: str
    text: str

    @property
    def indexed(self) -> str:  # what the retrievers see: the headings carry most of the topic words
        return f"{self.title}: {self.section}. {self.text}"


class Hit(BaseModel):
    id: str
    score: float


class Search(BaseModel):
    bm25: list[Hit]
    dense: list[Hit]
    hybrid: list[Hit]
    reranked: list[Hit]  # the hybrid pool reordered by the cross-encoder; its top k go to the model


def lang(text: str) -> str:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    return "de" if sum(w in DE_HINTS for w in words) > sum(w in EN_HINTS for w in words) else "en"


def tokens(text: str) -> list[str]:
    """Lowercase words without stop words, with a crude suffix strip per language (from contract-red-flags).
    ponytail: no real stemmer or compound splitting ('Urlaubstage' never matches 'Urlaub'); that gap is what
    the dense retriever is for, and the eval measures it."""
    out, suffixes = [], SUFFIXES[lang(text)]
    for word in re.findall(r"[a-zäöüß0-9]+", text.lower()):
        if word in STOP or len(word) == 1 and not word.isdigit():  # the s of "what's"
            continue
        for suffix in suffixes:
            if len(word) - len(suffix) >= 4 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        out.append(word)
    return out


def slug(heading: str) -> str:
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        heading = heading.lower().replace(a, b)
    return re.sub(r"[^a-z0-9]+", "-", heading).strip("-")


@cache
def load_chunks() -> tuple[Chunk, ...]:
    """One chunk per '## ' section of every document. The sections are written to be self-contained and
    40 to 120 words long, so splitting further would only cut sentences apart."""
    out = []
    for f in sorted(files("hybrid_rag").joinpath("docs").iterdir(), key=lambda f: f.name):
        if not f.name.endswith(".md"):
            continue
        head, *sections = re.split(r"^## ", f.read_text(encoding="utf-8"), flags=re.M)
        for s in sections:
            section, body = s.split("\n", 1)
            out.append(Chunk(id=f"{f.name[:-3]}#{slug(section)}", doc=f.name[:-3], title=head.strip("# \n"),
                             section=section.strip(), text=" ".join(body.split())))
    return tuple(out)


class BM25:
    def __init__(self, docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = {i: Counter(d) for i, d in docs.items()}, k1, b
        self.avg = sum(map(len, docs.values())) / len(docs)
        self.lens = {i: len(d) for i, d in docs.items()}
        df = Counter(t for d in docs.values() for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> dict[str, float]:
        out = {}
        for i, tf in self.docs.items():
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)
        return out


@cache
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@cache
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def embed(texts: list[str], query: bool) -> np.ndarray:
    """Unit-length vectors, so a dot product is the cosine similarity. The e5 models expect these prefixes."""
    prefix = "query: " if query else "passage: "
    return np.asarray(_embedder().encode([prefix + t for t in texts], normalize_embeddings=True))


def cross_scores(question: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _reranker().predict([(question, t) for t in texts])]


def rrf(*rankings: list[Hit]) -> list[Hit]:
    """Reciprocal rank fusion: each list adds 1 / (RRF_K + rank). Uses only ranks, so BM25 scores (0 to ~20)
    and cosine similarities (0 to 1) never need to be put on one scale."""
    fused = Counter()
    for ranking in rankings:
        for rank, h in enumerate(ranking, 1):
            fused[h.id] += 1 / (RRF_K + rank)
    return [Hit(id=i, score=s) for i, s in fused.most_common()]


class Index:
    def __init__(self, chunks: tuple[Chunk, ...] | None = None):
        self.chunks = {c.id: c for c in chunks or load_chunks()}
        self.bm25 = BM25({c.id: tokens(c.indexed) for c in self.chunks.values()})
        self.vectors = self._vectors()

    def _vectors(self) -> np.ndarray:
        """Chunk embeddings, kept in a local .npy file so later starts skip the encoding."""
        texts = [c.indexed for c in self.chunks.values()]
        key = hashlib.sha256("\n".join([EMBED_MODEL, *texts]).encode()).hexdigest()[:16]
        path = CACHE_DIR / f"vectors-{key}.npy"
        if path.exists():
            return np.load(path)
        vectors = embed(texts, query=False)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def search(self, question: str, qvec: np.ndarray | None = None) -> Search:
        ids = list(self.chunks)
        scores = self.bm25.scores(tokens(question))
        bm25 = [Hit(id=i, score=s) for i, s in sorted(scores.items(), key=lambda x: -x[1])[:POOL] if s > 0]
        sims = self.vectors @ (embed([question], query=True)[0] if qvec is None else qvec)
        dense = [Hit(id=ids[i], score=float(sims[i])) for i in np.argsort(-sims)[:POOL]]
        hybrid = rrf(bm25, dense)[:POOL]
        scored = cross_scores(question, [self.chunks[h.id].indexed for h in hybrid])
        reranked = sorted((Hit(id=h.id, score=s) for h, s in zip(hybrid, scored)), key=lambda h: -h.score)
        return Search(bm25=bm25, dense=dense, hybrid=hybrid, reranked=reranked)
