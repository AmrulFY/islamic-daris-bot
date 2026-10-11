from __future__ import annotations

import dataclasses
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from muslim_bot import semantic_index
from muslim_bot.semantic import SemanticSearcher, init_db, is_complete, mark_complete, to_blob
from tests.test_retrieval import make_dbs
from tests.test_streaming import make_settings


def fake_vectors(texts: list[str]) -> list[list[float]]:
    rng = np.random.default_rng(len(texts))
    return rng.normal(size=(len(texts), 512)).tolist()


@pytest.fixture
def indexed_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    make_dbs(tmp_path)
    settings = dataclasses.replace(make_settings(tmp_path), semantic_db_path=tmp_path / "semantic.db")
    monkeypatch.setattr(semantic_index, "load_settings", lambda: settings)
    monkeypatch.setattr(semantic_index, "_embed", lambda client, model, dims, texts: fake_vectors(texts))
    return settings


def count(db: Path, kind: str) -> int:
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute("select count(*) from embeddings where kind = ?", (kind,)).fetchone()[0]


def test_build_marks_each_kind_complete(indexed_env) -> None:
    db = indexed_env.semantic_db_path
    assert not semantic_index.index_complete(db)

    assert semantic_index.build(["quran", "hadith"]) is True

    assert semantic_index.index_complete(db)
    assert count(db, "quran") == 2 and count(db, "hadith") == 2


def test_stopped_build_is_incomplete_and_resumes(indexed_env) -> None:
    db = indexed_env.semantic_db_path

    assert semantic_index.build(["quran", "hadith"], should_stop=lambda: True) is False
    assert not is_complete(db, "quran") and not semantic_index.index_complete(db)

    assert semantic_index.build(["quran", "hadith"]) is True
    assert semantic_index.index_complete(db)


def test_searcher_ignores_incomplete_index_until_reload(tmp_path: Path) -> None:
    quran, hadith = make_dbs(tmp_path)
    db = tmp_path / "semantic.db"
    init_db(db)
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("insert into embeddings values ('quran', 1, ?)", (to_blob(np.array([1, 0, 0, 0])),))
        conn.commit()
    searcher = SemanticSearcher(db_path=db, quran_db_path=quran, hadith_db_path=hadith, model="m", dimensions=4)

    assert searcher.status() == "missing"  # vectors exist but the build never finished
    assert not searcher.available("quran")

    mark_complete(db, "quran")
    searcher.reload()
    assert searcher.available("quran") and searcher.status() == "missing"  # hadith still absent

    searcher.building = True
    assert searcher.status() == "building"


def test_searcher_does_not_create_a_missing_index_file(tmp_path: Path) -> None:
    quran, hadith = make_dbs(tmp_path)
    db = tmp_path / "nested" / "semantic.db"
    searcher = SemanticSearcher(db_path=db, quran_db_path=quran, hadith_db_path=hadith, model="m", dimensions=4)

    assert not searcher.available("quran")
    assert not db.exists()
