from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from muslim_bot.retrieval import Passage, hadith_matn, interleave_passages
from muslim_bot.semantic import SemanticSearcher, init_db, to_blob
from tests.test_retrieval import make_dbs
from tests.test_streaming import collect, delta, install_agents_stub, make_runner

DIMS = 4


def make_searcher(tmp_path: Path) -> SemanticSearcher:
    import sqlite3
    from contextlib import closing

    quran, hadith = tmp_path / "quran.db", tmp_path / "hadith.db"
    if not quran.exists():
        make_dbs(tmp_path)
    db = tmp_path / "semantic.db"
    init_db(db)
    with closing(sqlite3.connect(db)) as conn:
        conn.executemany(
            "insert into embeddings (kind, ref_id, vec) values (?, ?, ?)",
            [
                ("quran", 1, to_blob(np.array([1, 0, 0, 0]))),  # riba ayah
                ("quran", 2, to_blob(np.array([0, 1, 0, 0]))),
                ("hadith", 1, to_blob(np.array([0, 0, 1, 0]))),  # niat
                ("hadith", 2, to_blob(np.array([0, 0, 0, 1]))),  # senyum
            ],
        )
        conn.commit()
    return SemanticSearcher(
        db_path=db, quran_db_path=quran, hadith_db_path=hadith, model="m", dimensions=DIMS
    )


def fake_embed(searcher: SemanticSearcher, vector: list[float]) -> None:
    async def embed(text, *, api_key, base_url):
        array = np.array(vector, dtype=np.float32)
        return array / np.linalg.norm(array)

    searcher._embed = embed  # type: ignore[method-assign]


def test_semantic_search_ranks_by_similarity(tmp_path: Path) -> None:
    searcher = make_searcher(tmp_path)
    fake_embed(searcher, [0.9, 0.1, 0.0, 0.0])

    passages = asyncio.run(
        searcher.search("bunga uang pinjaman", ["quran"], language="id", api_key=None, base_url=None)
    )

    assert [p.label for p in passages] == ["Al-Baqarah 2:278"]
    assert passages[0].score > 0.9


def test_semantic_search_hadith_and_threshold(tmp_path: Path) -> None:
    searcher = make_searcher(tmp_path)
    fake_embed(searcher, [0.0, 0.0, 0.1, 1.0])

    passages = asyncio.run(
        searcher.search("tersenyum", ["hadith"], language="id", api_key=None, base_url=None)
    )

    assert [p.label for p in passages] == ["Sahih al-Bukhari #2"]


def test_semantic_search_degrades_without_index_or_on_error(tmp_path: Path) -> None:
    quran, hadith = make_dbs(tmp_path)
    missing = SemanticSearcher(
        db_path=tmp_path / "nope.db", quran_db_path=quran, hadith_db_path=hadith, model="m", dimensions=DIMS
    )
    assert asyncio.run(missing.search("x", ["quran"], language="id", api_key=None, base_url=None)) == []

    searcher = make_searcher(tmp_path)  # reuses the databases created above

    async def boom(text, *, api_key, base_url):
        raise RuntimeError("gateway down")

    searcher._embed = boom  # type: ignore[method-assign]
    assert asyncio.run(searcher.search("x", ["quran"], language="id", api_key=None, base_url=None)) == []


def test_hadith_matn_strips_narrator_chain() -> None:
    text = "Telah menceritakan kepada kami [Abu Bakar] dari [Umar] bahwa Nabi bersabda: senyummu kepada saudaramu adalah sedekah."
    assert hadith_matn(text) == "bahwa Nabi bersabda: senyummu kepada saudaramu adalah sedekah."


def test_interleave_alternates_and_dedupes() -> None:
    sem = [Passage("quran", f"S {i}:1", f"{i}:1", "t") for i in (1, 2, 3)]
    kw = [sem[0], Passage("quran", "K 9:1", "9:1", "t")]

    merged = interleave_passages(sem, kw)

    assert [p.label for p in merged] == ["S 1:1", "S 2:1", "K 9:1", "S 3:1"]


def test_pipeline_adds_semantic_matches_alongside_keywords(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    searcher = make_searcher(tmp_path)
    fake_embed(searcher, [0.0, 0.0, 1.0, 0.0])
    runner.semantic = searcher
    FakeRunner = install_agents_stub(monkeypatch, [delta("ok")], final_output="ok")

    # No keyword in the DB matches "motivasi", so only semantic search can find #1.
    collect(runner.ask_streamed("hadits tentang motivasi", session_id="s1"))

    assert "[Sahih al-Bukhari #1]" in FakeRunner.seen["prompt"]


def test_interleave_caps_each_source() -> None:
    keyword = [Passage("quran", f"K {i}:1", f"{i}:1", "t") for i in range(1, 15)]
    semantic = [Passage("quran", f"S {i}:1", f"{i}:1", "t") for i in range(1, 15)]
    hadith = [Passage("hadith", "H #1", "1", "t")]

    merged = interleave_passages(semantic, keyword + hadith, per_kind_limit=12)

    quran = [p.label for p in merged if p.kind == "quran"]
    assert len(quran) == 12
    assert quran[:4] == ["S 1:1", "K 1:1", "S 2:1", "K 2:1"]
    assert [p.label for p in merged if p.kind == "hadith"] == ["H #1"]
