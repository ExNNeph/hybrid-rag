from hybrid_rag import answer as qa
from hybrid_rag.answer import Claim, Citation, verify
from hybrid_rag.cache import SemanticCache, numbers, same_words
from hybrid_rag.index import Index, load_chunks

CHUNKS = {c.id: c for c in load_chunks()}
ANNUAL = "billing-refunds#refunds-for-annual-plans"


def claim(text, source, quote):
    return Claim(text=text, citations=[Citation(source=source, quote=quote)])


def test_verify_keeps_real_quotes_drops_invented_flags_wrong_numbers():
    ok = claim("Annual plans are refundable within 30 days.", ANNUAL,
               "Annual plans can be refunded in full within 30 days of purchase.")
    invented = claim("Annual plans are refundable within 60 days.", ANNUAL, "refunded within 60 days of purchase")
    wrong_source = claim("Same quote, wrong chunk.", "pricing-plans#team-plan",
                         "Annual plans can be refunded in full within 30 days of purchase.")
    unseen = claim("A chunk the model never saw.", "nope#x", "Annual plans can be refunded in full")
    number = claim("Annual plans are refundable within 14 days.", ANNUAL,
                   "Annual plans can be refunded in full within 30 days of purchase.")
    out = verify([ok, invented, wrong_source, unseen, number], CHUNKS)
    assert [c.status for c in out] == ["supported", "dropped", "dropped", "dropped", "flagged"]


def test_gap_claims_are_kept_and_question_numbers_count():
    gap = Claim(text="The docs do not mention GraphQL.", citations=[], kind="gap")
    trip = claim("First class is fine for a 4.5 hour trip.", "travel-expenses#trains-and-flights",
                 "first class is allowed for journeys longer than 4 hours")
    out = verify([gap, trip], CHUNKS, "Can I go first class on a 4.5 hour trip?")
    assert [c.status for c in out] == ["gap", "supported"]
    assert verify([trip.model_copy(deep=True)], CHUNKS)[0].status == "flagged"


def test_quote_check_ignores_case_whitespace_and_typographic_quotes():
    c = claim("x", ANNUAL, "ANNUAL plans   can be refunded in full")
    assert verify([c], CHUNKS)[0].status == "supported"


def test_numbers():
    assert numbers("1.000 euros, 1,000 or 1000; 99,9 % and 99.9%") == {"1000", "999"}


def test_cache_guard_blocks_traps_and_allows_rewording():
    assert same_words("How many days of vacation do full-time employees have?",
                      "How many vacation days does a full-time employee get?")
    assert not same_words("Refund within 14 days?", "Refund within 30 days?")
    assert not same_words("Refund on a monthly plan?", "Refund on an annual plan?")
    assert not same_words("Kündigungsfrist in der Probezeit?", "Kündigungsfrist nach der Probezeit?")


def fake_llm(question, sources, key):
    if "Christmas" in question:
        return {"answerable": False, "claims": []}, 900, 20
    return {"answerable": True, "claims": [
        {"text": "Annual plans can be refunded within 30 days.",
         "citations": [{"source": ANNUAL, "quote": "Annual plans can be refunded in full within 30 days of purchase."},
                       {"source": ANNUAL, "quote": "After 30 days there is no refund"}]},
        {"text": "You also get a free mug.", "citations": [{"source": ANNUAL, "quote": "and a free mug"}]},
        {"kind": "gap", "text": "The docs do not say if mugs are refunded.", "citations": []}]}, 1000, 100


def test_pipeline_with_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(qa, "_llm", fake_llm)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    index, cache = Index(), SemanticCache(path=tmp_path / "answers.json")
    a = qa.answer("Can I get a refund on an annual plan?", index, cache)
    assert a.status == "answered" and a.text == ("Annual plans can be refunded within 30 days. [1] "
                                                 "The docs do not say if mugs are refunded.")
    assert a.cited == [ANNUAL] and [c.status for c in a.claims] == ["supported", "dropped", "gap"]
    assert a.cost == 1000 * 4 / 1e6 + 100 * 20 / 1e6

    b = qa.answer("Can I get a refund on an annual plan", index, cache)  # reworded: served from the cache
    assert b.cached_from and b.cost == 0 and cache.hits == 1 and cache.saved == a.cost
    c = qa.answer("Can I get a refund on a monthly plan?", index, cache)  # trap: must call the API
    assert not c.cached_from and c.closest and cache.hits == 1

    assert len(SemanticCache(path=tmp_path / "answers.json").entries) == 2  # persisted for the CLI
    n = qa.answer("How much is the Christmas bonus?", index, cache)
    assert n.status == "not found" and n.text == "Not in the docs."
    assert qa.Answer(question="Wie hoch ist das Weihnachtsgeld?", status="not found").text == \
        "Steht nicht in den Dokumenten."


def test_without_key_retrieval_still_runs():
    a = qa.answer("What does API error TB-4031 mean?", Index())
    assert a.status == "no key" and a.sources[0].id == "api-reference#error-codes" and a.search.bm25
