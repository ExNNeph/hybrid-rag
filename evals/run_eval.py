"""Run the labeled questions. Usage: python evals/run_eval.py   (the answer part needs ANTHROPIC_API_KEY)

1. Retrieval, local, no key: recall@k of BM25, dense, hybrid (RRF) and hybrid + rerank on 40 answerable questions.
2. Cache, local, no key: 12 paraphrase pairs that should hit and 14 near-duplicate traps that must miss, for
   similarity thresholds alone and with the lexical guard.
3. Answers, with a key: correctness (all labeled facts in the answer), citation precision (verified citations
   that point to a labeled source), quotes that failed verification, and "not in the docs" on 12 unanswerable
   questions. Cost per answer prices the cache hits.
Per-case results go to evals/results.json.
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from hybrid_rag.answer import Answer, answer
from hybrid_rag.cache import THRESHOLD, SemanticCache
from hybrid_rag.index import Index, embed

HERE = Path(__file__).parent
CASES = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
METHODS = ("bm25", "dense", "hybrid", "reranked")
NAMES = {"bm25": "BM25", "dense": "dense", "hybrid": "hybrid (RRF)", "reranked": "hybrid + rerank"}
KS = (1, 3, 5)


def recall(ranking: list[str], expected: list[str], k: int) -> float:
    return len(set(ranking[:k]) & set(expected)) / len(expected)


def has_fact(text: str, alternatives: list[str]) -> bool:
    norm = lambda s: " ".join(s.lower().replace(" ", " ").split())  # noqa: E731
    return any(norm(a) in norm(text) for a in alternatives)


def retrieval(index: Index, log: dict) -> None:
    rec = defaultdict(list)  # (method, k) -> recalls; (method, kind) -> recall@3
    for case in CASES["answerable"]:
        s = index.search(case["question"])
        entry = {}
        for m in METHODS:
            ids = [h.id for h in getattr(s, m)]
            entry[m] = ids[:5]
            for k in KS:
                rec[m, k].append(recall(ids, case["sources"], k))
            rec[m, case["kind"]].append(recall(ids, case["sources"], 3))
        log[case["id"]] = {"question": case["question"], "kind": case["kind"], "expected": case["sources"],
                           "retrieved": entry}
    kinds = list(dict.fromkeys(c["kind"] for c in CASES["answerable"]))
    print(f"\n### Retrieval ({len(CASES['answerable'])} questions)\n")
    print("| retriever | " + " | ".join(f"recall@{k}" for k in KS) + " | "
          + " | ".join(f"{kind} @3" for kind in kinds) + " |")
    print("|---" * (1 + len(KS) + len(kinds)) + "|")
    for m in METHODS:
        print(f"| {NAMES[m]} | " + " | ".join(f"{np.mean(rec[m, k]):.2f}" for k in KS) + " | "
              + " | ".join(f"{np.mean(rec[m, kind]):.2f}" for kind in kinds) + " |")
    print("\nper kind: " + ", ".join(f"{kind} {sum(c['kind'] == kind for c in CASES['answerable'])}" for kind in kinds))


def cache_eval(log: dict) -> dict:
    pairs = CASES["cache"]
    vecs = embed([q for c in pairs for q in (c["first"], c["second"])], query=True)
    results = {}
    for guard, thresholds in ((False, (0.90, 0.95, 0.97, 0.99)), (True, (0.90, THRESHOLD))):
        for t in thresholds:
            hits = []
            for i, c in enumerate(pairs):
                cache = SemanticCache(threshold=t, guard=guard)
                cache.add(c["first"], vecs[2 * i], Answer(question=c["first"], status="answered", cost=1.0))
                hits.append(cache.lookup(c["second"], vecs[2 * i + 1])[0] is not None)
            same = [h for h, c in zip(hits, pairs) if c["same"]]
            traps = [h for h, c in zip(hits, pairs) if not c["same"]]
            results[guard, t] = (sum(same), len(same), sum(traps), len(traps))
            if guard and t == THRESHOLD:
                for h, c, i in zip(hits, pairs, range(len(pairs))):
                    log[c["id"]] = {**c, "similarity": round(float(vecs[2 * i] @ vecs[2 * i + 1]), 4), "hit": h}
    print(f"\n### Semantic cache ({sum(c['same'] for c in pairs)} paraphrase pairs, "
          f"{sum(not c['same'] for c in pairs)} near-duplicate traps)\n")
    print("| rule | paraphrases served from cache | traps served from cache (false hits) |\n|---|---|---|")
    for (guard, t), (h, n, f, m) in results.items():
        rule = f"similarity ≥ {t:.2f}" + (" + same words and numbers" if guard else " only")
        print(f"| {rule} | {h}/{n} ({h / n:.0%}) | {f}/{m} ({f / m:.0%}) |")
    sims = {c["id"]: log[c["id"]]["similarity"] for c in pairs}
    print(f"\nsimilarity: paraphrases {min(sims[c['id']] for c in pairs if c['same']):.3f} to "
          f"{max(sims[c['id']] for c in pairs if c['same']):.3f}, traps "
          f"{min(sims[c['id']] for c in pairs if not c['same']):.3f} to "
          f"{max(sims[c['id']] for c in pairs if not c['same']):.3f}")
    return results


def answers(index: Index, log: dict) -> list[Answer]:
    out, errors_in_row = [], 0
    for case in CASES["answerable"] + CASES["unanswerable"]:
        print(f"  {case['id']} {case['question'][:70]!r} …", file=sys.stderr, flush=True)
        a = answer(case["question"], index)
        errors_in_row = errors_in_row + 1 if a.status == "error" else 0
        if errors_in_row == 3:  # the API is down, the key is wrong or the credit is used up
            raise SystemExit(f"Stopped: 3 API errors in a row, last one: {a.error}")
        entry = log.setdefault(case["id"], {"question": case["question"]})
        entry.update(status=a.status, answer=a.text, cost=round(a.cost, 5),
                     claims=[c.model_dump() for c in a.claims], sources_seen=[c.id for c in a.sources])
        if "facts" in case:
            entry["correct"] = a.status == "answered" and all(has_fact(a.text, f) for f in case["facts"])
            if not entry["correct"]:
                print(f"      WRONG ({a.status}): {a.text[:150]}", file=sys.stderr)
        else:
            entry["abstained"] = a.status in ("not found", "unsupported")
            if not entry["abstained"]:
                print(f"      ANSWERED: {a.text[:150]}", file=sys.stderr)
        out.append(a)
    return out


def main():
    index, log = Index(), {}
    retrieval(index, log)
    cache = cache_eval(log)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nNo ANTHROPIC_API_KEY: skipped the answer eval (retrieval and cache need no key).")
    else:
        got = answers(index, log)
        n = len(CASES["answerable"])
        ans, una = [log[c["id"]] for c in CASES["answerable"]], [log[c["id"]] for c in CASES["unanswerable"]]
        expected = {c["id"]: set(c["sources"]) for c in CASES["answerable"]}
        cits = [(x, cid) for cid in expected for cl in log[cid]["claims"] if cl["status"] != "dropped"
                for x in cl["citations"] if x["found"]]
        all_cits = [x for e in ans + una for cl in e["claims"] for x in cl["citations"]]
        claims = [cl for e in ans + una for cl in e["claims"]]
        on_label = sum(x["source"] in expected[cid] for x, cid in cits)
        cost = sum(a.cost for a in got)
        per = cost / len(got)
        pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "n/a"  # noqa: E731
        h, nh, f, nf = cache[True, THRESHOLD]
        print(f"\n### Answers ({n} answerable, {len(una)} unanswerable)\n\n| metric | result |\n|---|---|")
        print(f"| answer correct (every labeled fact in the answer) | {pct(sum(e['correct'] for e in ans), n)} |")
        print(f"| … answered 'not in the docs' although it is | "
              f"{pct(sum(e['status'] in ('not found', 'unsupported') for e in ans), n)} |")
        print(f"| unanswerable questions answered 'not in the docs' | "
              f"{pct(sum(e['abstained'] for e in una), len(una))} |")
        print(f"| citation precision (verified citations pointing to a labeled source) | {pct(on_label, len(cits))} |")
        print(f"| quotes not found in the cited source | {pct(sum(not x['found'] for x in all_cits), len(all_cits))} |")
        print(f"| claims dropped (no verified quote) / flagged (number not in quote or question) / gaps | "
              f"{sum(c['status'] == 'dropped' for c in claims)} / {sum(c['status'] == 'flagged' for c in claims)}"
              f" / {sum(c['status'] == 'gap' for c in claims)} of {len(claims)} |")
        print(f"| cost | {len(got)} calls, {sum(a.tokens_in for a in got):,} in / "
              f"{sum(a.tokens_out for a in got):,} out tokens, ${cost:.2f} (${per:.4f} per question) |")
        print(f"| cache: hits on the {nh} paraphrases, $ saved | {h} hits, ${h * per:.3f} saved, "
              f"{f} false hits on {nf} traps |")
    (HERE / "results.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
