"""Local web demo, started with `hybrid-rag demo`."""
import os

import streamlit as st

from hybrid_rag.answer import MODEL, TOP_K, Answer, answer
from hybrid_rag.cache import THRESHOLD, SemanticCache, numbers
from hybrid_rag.index import Index, tokens

QUESTIONS = [
    "Can I get my money back on a yearly subscription?",
    "Can I get my money back on a monthly subscription?",
    "Wie viele Urlaubstage habe ich bei einer 4-Tage-Woche?",
    "How many vacation days does a full-time employee get?",
    "How many days of vacation do full-time employees have?",
    "What does API error TB-4031 mean?",
    "If we downgrade from Business to Team, what happens to SSO and how many API requests per minute do we get?",
    "How much is the Christmas bonus?",
]
BADGE = {"answered": "✅ Answered from the docs", "not found": "🚫 Not in the docs",
         "unsupported": "⚠️ No claim could be verified", "no key": "🔑 No API key: retrieval only",
         "error": "❌ Error"}

st.set_page_config(page_title="Hybrid RAG demo", page_icon="🔎", layout="wide")
st.title("🔎 Hybrid RAG over internal docs")
st.caption("Talvi Software's handbook, policies and product docs (synthetic, German and English). BM25 and local "
           "embeddings each rank the chunks, reciprocal rank fusion merges them, a local cross-encoder reranks, "
           f"and Claude answers from the top {TOP_K} with quotes that are checked against the sources.")


@st.cache_resource(show_spinner="Loading the models and the index…")
def load() -> tuple[Index, SemanticCache]:
    return Index(), SemanticCache()  # one cache for the whole demo process, kept in memory


index, cache = load()

with st.sidebar:
    api_key = st.text_input("Anthropic API key (optional)", type="password",
                            help="Used only for your requests in this local session, never stored.")
    if api_key or os.getenv("ANTHROPIC_API_KEY"):
        where = "sidebar" if api_key else "ANTHROPIC_API_KEY environment variable"
        st.success(f"🔑 Key set ({where}): Claude ({MODEL}) writes the answers.")
    else:
        st.info("No key: you still see what each retriever finds. Paste a key and press Enter for answers.")
    st.subheader("Semantic cache")
    a, b = st.columns(2)
    a.metric("hit rate", f"{cache.hit_rate:.0%}", f"{cache.hits} of {cache.lookups}", delta_color="off")
    b.metric("saved", f"${cache.saved:.3f}", f"{len(cache.entries)} cached", delta_color="off")
    st.caption(f"A hit needs similarity ≥ {THRESHOLD} and the same content words and numbers, so "
               "*monthly* vs *yearly* or *14* vs *30 days* is never served from the cache.")
    if cache.entries:
        with st.expander("Cached questions"):
            st.markdown("\n".join(f"- {q}" for q, _, _ in cache.entries))


def cache_line(a: Answer) -> None:
    if a.cached_from:
        st.success(f"⚡ **Cache hit**: same question as *{a.cached_from}* (similarity {a.similarity:.3f}). "
                   "No API call, $0.")
    elif a.closest:
        diff = sorted(set(tokens(a.question)) ^ set(tokens(a.closest)) | numbers(a.question) ^ numbers(a.closest))
        why = (f"similarity {a.similarity:.3f} < {THRESHOLD}" if a.similarity < THRESHOLD else
               f"similarity {a.similarity:.3f}, but the words differ: {', '.join(diff)}")
        st.info(f"Cache miss. Closest cached question: *{a.closest}* ({why}).")


def show_answer(a: Answer) -> None:
    st.markdown(f"**{BADGE[a.status]}**" + (f" · ${a.cost:.4f}" if a.cost else ""))
    cache_line(a)
    if a.status in ("answered", "not found", "unsupported"):
        st.markdown(f"### {a.text}")
    elif a.error:
        st.warning(a.error)
    for n, sid in enumerate(a.cited, 1):
        quotes = dict.fromkeys(x.quote for c in a.kept for x in c.citations if x.found and x.source == sid)
        st.markdown(f"**[{n}]** `{sid}`  \n" + "  \n".join(f"> {q}" for q in quotes))
    for c in a.claims:
        if c.status == "dropped":
            st.error(f"Dropped, its quote is not in the cited source: {c.text}")
        elif c.status == "flagged":
            st.warning(f"Flagged, a number in it is not in its quotes: {c.text}")


def show_retrievers(a: Answer) -> None:
    if not a.search:
        return
    st.subheader("What each retriever found")
    top = {c.id for c in a.sources}
    cols = st.columns(4)
    for col, (name, hits, fmt) in zip(cols, (
            ("BM25 (keywords)", a.search.bm25, "{:.2f}"), ("Dense (embeddings)", a.search.dense, "{:.3f}"),
            ("Hybrid (RRF)", a.search.hybrid, "{:.4f}"), ("Hybrid + rerank", a.search.reranked, "{:.2f}"))):
        with col:
            st.markdown(f"**{name}**")
            for r, h in enumerate(hits[:5], 1):
                c = index.chunks[h.id]
                mark = "🟢 " if h.id in top else ""
                with st.expander(f"{mark}{r}. {c.doc} › {c.section} ({fmt.format(h.score)})"):
                    st.caption(c.text)
            if not hits:
                st.caption("no shared keyword")
    st.caption(f"🟢 = among the top {TOP_K} after reranking, the chunks Claude sees.")


@st.cache_data(show_spinner=False, max_entries=200)
def no_key_answer(question: str) -> Answer:
    return answer(question, index)  # retrieval only, safe to memoize


pick = st.selectbox("Example questions (two pairs show the cache: one hits, one is a trap)", QUESTIONS,
                    index=None, placeholder="Pick one or type your own below")
question = st.text_input("Question", value=pick or "", key=f"q-{pick}")
if st.button("Ask", type="primary", disabled=not question.strip()):
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    with st.spinner("Searching, reranking and asking Claude…" if key else "Searching and reranking…"):
        st.session_state.answer = (answer(question.strip(), index, cache, api_key or None) if key
                                   else no_key_answer(question.strip()))
    st.rerun()  # so the sidebar shows the updated cache numbers
a = st.session_state.get("answer")
if a:
    show_answer(a)
    show_retrievers(a)
