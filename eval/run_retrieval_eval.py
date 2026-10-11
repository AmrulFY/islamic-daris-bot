"""Retrieval evaluation: does the expected dalil reach the LLM's context?

Run from the repo root:  python -m eval.run_retrieval_eval [--no-semantic]

Uses the same code path as production (`MuslimBotRunner.gather_passages`), so
it only spends embedding calls (no LLM answer calls). Three phrasings per case:
"typed" ("Ayat/Hadits tentang X"), "neutral" ("Dalil tentang X": both sources
compete) and "gist" (a description of what the dalil says).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from muslim_bot.agent import MuslimBotRunner
from muslim_bot.retrieval import Passage, retrieve

CASES_PATH = Path(__file__).with_name("retrieval_cases.json")
KS = (4, 8, 12)


def phrasings(case: dict) -> dict[str, str]:
    word = "Ayat" if case["kind"] == "quran" else "Hadits"
    return {
        "typed": f"{word} tentang {case['topic'].lower()}",
        "neutral": f"Dalil tentang {case['topic'].lower()}",
        "gist": f"{word} yang menjelaskan: {case['gist']}",
    }


def rank_of(passages: list[Passage], case: dict) -> int | None:
    """1-based rank of the first expected passage among passages of its kind."""
    same_kind = [p for p in passages if p.kind == case["kind"]]
    for position, passage in enumerate(same_kind, start=1):
        key = passage.source_id if case["kind"] == "quran" else passage.label
        if key in case["expected"]:
            return position
    return None


def summarize(name: str, ranks: list[int | None]) -> str:
    total = len(ranks)
    hits = ", ".join(f"@{k} {sum(1 for r in ranks if r and r <= k)}/{total}" for k in KS)
    mrr = sum(1 / r for r in ranks if r) / total if total else 0.0
    return f"{name:<22} {hits}  in-context {sum(1 for r in ranks if r)}/{total}  MRR {mrr:.2f}"


async def main(use_semantic: bool) -> None:
    runner = MuslimBotRunner()
    if not use_semantic:
        runner.semantic = None
    runtime_config = runner.load_runtime_config()
    if runner.semantic is not None:  # same warm-up the API does at startup
        await runner.semantic.warm_up(api_key=runtime_config.llm.api_key, base_url=runtime_config.llm.base_url)
    cases = [c for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"] if c["expected"]]

    styles = ("typed", "neutral", "gist")
    results: dict[str, list[int | None]] = {s: [] for s in styles}
    keyword_only: dict[str, list[int | None]] = {s: [] for s in styles}
    print(f"{'#':>2} {'topic':<34} {'kind':<6} " + " ".join(f"{v + ' kw/final':>16}" for v in ("typed", "neutral", "gist")))
    for case in cases:
        cells = []
        for style, query in phrasings(case).items():
            kw = await asyncio.to_thread(
                retrieve, query,
                quran_db_path=runner.settings.quran_db_path,
                hadith_db_path=runner.settings.hadith_db_path,
            )
            final = await runner.gather_passages(query, runtime_config) or []
            kw_rank, final_rank = rank_of(kw, case), rank_of(final, case)
            keyword_only[style].append(kw_rank)
            results[style].append(final_rank)
            cells.append(f"{kw_rank or '-'}/{final_rank or '-'}")
        print(f"{case['n']:>2} {case['topic'][:34]:<34} {case['kind']:<6} " + " ".join(f"{c:>16}" for c in cells))

    print("\n(rank within its source; '-' = target not in the context sent to the LLM)\n")
    for style in styles:
        print(summarize(f"{style} keyword-only", keyword_only[style]))
        print(summarize(f"{style} final", results[style]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-semantic", action="store_true", help="keyword search only")
    args = parser.parse_args()
    asyncio.run(main(not args.no_semantic))
