"""Build the semantic index: `python -m muslim_bot.semantic_index [--kind quran|hadith]`.

Resumable: rows that already have a vector are skipped, so an interrupted run
can simply be started again. Calls the embeddings endpoint of the configured
LLM gateway (a few cents for the full corpus).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from contextlib import closing

import numpy as np

from .config import load_settings
from .runtime_config import RuntimeConfigStore
from .semantic import hadith_matn, init_db, to_blob

BATCH = 128
MAX_RETRIES = 6


def _pending(settings, kind: str, done: set[int]) -> list[tuple[int, str]]:
    if kind == "quran":
        with closing(sqlite3.connect(settings.quran_db_path)) as conn:
            rows = conn.execute("select id, translation_indo from ayahs order by id").fetchall()
        return [(i, str(t)) for i, t in rows if i not in done and str(t).strip()]
    with closing(sqlite3.connect(settings.hadith_db_path)) as conn:
        rows = conn.execute(
            "select hadith_id, lang, text from hadith_texts where lang in ('id', 'en') order by hadith_id"
        ).fetchall()
    # Indonesian text when present, English otherwise (the model is multilingual).
    by_id: dict[int, dict[str, str]] = {}
    for hadith_id, lang, text in rows:
        by_id.setdefault(hadith_id, {})[lang] = str(text)
    todo: list[tuple[int, str]] = []
    for hadith_id, texts in by_id.items():
        if hadith_id in done:
            continue
        body = next((hadith_matn(texts[lang]) for lang in ("id", "en") if texts.get(lang, "").strip()), "")
        if body.strip():  # the endpoint rejects empty input
            todo.append((hadith_id, body))
    return todo


def _embed(client, model: str, dims: int, texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(model=model, input=texts, dimensions=dims)
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001 - retry transient gateway errors.
            wait = 2**attempt
            print(f"  embed failed ({type(exc).__name__}), retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("embeddings kept failing")


def build(kinds: list[str]) -> None:
    from openai import OpenAI

    settings = load_settings()
    llm = RuntimeConfigStore(settings).load().llm
    client = OpenAI(api_key=llm.api_key or "not-needed", base_url=llm.base_url, timeout=60)
    db_path = settings.semantic_db_path
    init_db(db_path)

    for kind in kinds:
        with closing(sqlite3.connect(db_path)) as conn:
            done = {r[0] for r in conn.execute("select ref_id from embeddings where kind = ?", (kind,))}
        todo = _pending(settings, kind, done)
        print(f"{kind}: {len(done)} done, {len(todo)} to embed", flush=True)
        started = time.time()
        for start in range(0, len(todo), BATCH):
            chunk = todo[start : start + BATCH]
            vectors = _embed(client, settings.embedding_model, settings.embedding_dimensions, [t for _, t in chunk])
            with closing(sqlite3.connect(db_path)) as conn:
                conn.executemany(
                    "insert or replace into embeddings (kind, ref_id, vec) values (?, ?, ?)",
                    [(kind, ref_id, to_blob(np.asarray(v))) for (ref_id, _), v in zip(chunk, vectors)],
                )
                conn.commit()
            if (start // BATCH) % 10 == 0:
                print(f"  {kind}: {start + len(chunk)}/{len(todo)} ({time.time() - started:.0f}s)", flush=True)
        print(f"{kind}: complete", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the semantic search index.")
    parser.add_argument("--kind", choices=["quran", "hadith"], action="append")
    args = parser.parse_args(argv)
    build(args.kind or ["quran", "hadith"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
