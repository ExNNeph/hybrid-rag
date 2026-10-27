import argparse
import subprocess
import sys
from pathlib import Path

DEMO_APP = Path(__file__).with_name("demo_app.py")


def demo() -> None:
    """Start the local web demo on http://localhost:8501 and open the browser."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit('The demo needs Streamlit: pip install -e ".[demo]"')
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(DEMO_APP), "--server.address", "localhost",
                    "--server.port", "8501", "--browser.gatherUsageStats", "false",
                    "--client.toolbarMode", "minimal",  # hides Streamlit's "Deploy" button
                    "--server.fileWatcherType", "none"])  # no code reloading needed, and no "install Watchdog" hint


def show_search(s) -> None:
    for name, hits in (("BM25", s.bm25), ("dense", s.dense), ("hybrid (RRF)", s.hybrid),
                       ("hybrid + rerank", s.reranked)):
        print(f"\n{name}:")
        for h in hits[:5]:
            print(f"  {h.score:7.3f}  {h.id}")


def show(a) -> int:
    if a.cached_from:
        print(f"(from the cache: same as {a.cached_from!r}, similarity {a.similarity:.3f}, $0)\n")
    print(a.text)
    for n, sid in enumerate(a.cited, 1):
        quotes = [x.quote for c in a.kept for x in c.citations if x.found and x.source == sid]
        print(f"\n[{n}] {sid}\n    " + "\n    ".join(f'"{q}"' for q in dict.fromkeys(quotes)))
    for c in a.claims:
        if c.status == "dropped":
            print(f"\n  dropped, no quote found in the sources: {c.text}")
    if a.status == "no key":
        print("\nTop chunks: " + ", ".join(c.id for c in a.sources))
    elif a.cost:
        print(f"\n{a.tokens_in:,} in / {a.tokens_out:,} out tokens, ${a.cost:.4f}")
    return 0 if a.status in ("answered", "not found") else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="hybrid-rag", description="Ask Talvi Software's internal docs (DE/EN). "
                                "Hybrid search and reranking run locally; Claude writes the cited answer.")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("ask", help="cited answer from Claude (needs ANTHROPIC_API_KEY), with the semantic cache")
    q.add_argument("question")
    q.add_argument("--no-cache", action="store_true", help="always call the API, do not store the answer")
    sub.add_parser("search", help="the top 5 of each retriever, no key needed").add_argument("question")
    sub.add_parser("chunks", help="list all chunk ids")
    sub.add_parser("demo", help="start the local web demo on http://localhost:8501")
    args = p.parse_args(argv)

    if args.cmd == "demo":
        return demo()
    from .index import CACHE_DIR, Index, load_chunks
    if args.cmd == "chunks":
        for c in load_chunks():
            print(f"{c.id}  ({len(c.text.split())} words)")
        return 0
    index = Index()
    if args.cmd == "search":
        show_search(index.search(args.question))
        return 0
    from .answer import answer
    from .cache import SemanticCache
    cache = None if args.no_cache else SemanticCache(path=CACHE_DIR / "answers.json")
    return show(answer(args.question, index, cache))


if __name__ == "__main__":
    sys.exit(main())
