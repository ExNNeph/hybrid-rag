from hybrid_rag.index import Hit, Index, load_chunks, rrf, slug, tokens


def test_chunks_are_unique_sections_with_text():
    chunks = load_chunks()
    assert len(chunks) == len({c.id for c in chunks}) > 50
    assert all(c.text and 20 <= len(c.text.split()) <= 150 for c in chunks)
    assert {"billing-refunds#refunds-for-annual-plans", "urlaub-abwesenheit#urlaubsanspruch"} <= {c.id for c in chunks}


def test_slug_and_tokens():
    assert slug("Kündigungsfristen & Übertrag") == "kuendigungsfristen-uebertrag"
    assert tokens("What's the refund for employees?") == ["refund", "employee"]
    assert "4031" in tokens("Error TB-4031")


def test_rrf_rewards_agreement_and_uses_only_ranks():
    a = [Hit(id="x", score=99), Hit(id="y", score=50)]
    b = [Hit(id="y", score=0.9), Hit(id="z", score=0.8)]
    fused = rrf(a, b)
    assert [h.id for h in fused] == ["y", "x", "z"]
    assert fused[0].score == 1 / 62 + 1 / 61


def test_search_returns_all_four_rankings():
    s = Index().search("What does API error TB-4031 mean?")
    assert s.bm25[0].id == "api-reference#error-codes"
    assert s.reranked[0].id == "api-reference#error-codes"
    assert len(s.hybrid) <= 20 and {h.id for h in s.reranked} == {h.id for h in s.hybrid}


def test_vectors_are_cached_on_disk(tmp_path):
    Index()
    assert list(tmp_path.glob("vectors-*.npy"))
