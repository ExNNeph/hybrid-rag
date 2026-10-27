"""Sanity checks on the labels, no models and no API calls."""
import json
from pathlib import Path

from hybrid_rag.cache import numbers
from hybrid_rag.index import load_chunks

CASES = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
CHUNKS = {c.id: c.text.lower() for c in load_chunks()}


def test_sizes_and_unique_ids():
    assert (len(CASES["answerable"]), len(CASES["unanswerable"]), len(CASES["cache"])) == (40, 12, 26)
    ids = [c["id"] for part in CASES.values() for c in part]
    assert len(ids) == len(set(ids))


def test_labeled_sources_exist_and_contain_the_facts():
    for case in CASES["answerable"]:
        assert set(case["sources"]) <= set(CHUNKS), case["id"]
        text = " ".join(CHUNKS[s] for s in case["sources"])
        for alternatives in case["facts"]:  # a fact is in the source as text, or at least its numbers are
            assert any(a.lower() in text or numbers(a) and numbers(a) <= numbers(text) for a in alternatives), \
                (case["id"], alternatives)


def test_cache_pairs_are_distinct_and_balanced():
    assert all(c["first"] != c["second"] for c in CASES["cache"])
    assert sum(c["same"] for c in CASES["cache"]) == 12
