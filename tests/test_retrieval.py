from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from muslim_bot.retrieval import (
    Passage,
    extract_keywords,
    format_context,
    needs_agent_loop,
    retrieve,
    wanted_sources,
)
from tests.test_streaming import collect, delta, install_agents_stub, make_runner


def make_dbs(tmp_path: Path) -> tuple[Path, Path]:
    quran = tmp_path / "quran.db"
    with closing(sqlite3.connect(quran)) as conn:
        conn.executescript(
            """
            create table surahs (id integer primary key, number integer, name_transliteration text);
            create table ayahs (id integer primary key, surah_id integer, number_in_surah integer,
                text_uthmani text, translation_indo text, translation_en text);
            insert into surahs values (2, 2, 'Al-Baqarah'), (51, 51, 'Adh-Dhariyat');
            insert into ayahs values
              (1, 2, 278, 'ar', 'Tinggalkan sisa riba jika kamu beriman', 'leave usury'),
              (2, 51, 56, 'ar', 'Aku tidak menciptakan jin dan manusia melainkan agar mereka beribadah', 'worship');
            """
        )
        conn.commit()
    hadith = tmp_path / "hadith.db"
    with closing(sqlite3.connect(hadith)) as conn:
        conn.executescript(
            """
            create table books (id integer primary key, slug text, name_en text);
            create table hadiths (id integer primary key, book_id integer, number integer);
            create table hadith_texts (id integer primary key, hadith_id integer, lang text, text text);
            insert into books values (1, 'bukhari', 'Sahih al-Bukhari');
            insert into hadiths values (1, 1, 1), (2, 1, 2);
            insert into hadith_texts values
              (1, 1, 'id', 'Sesungguhnya amal itu tergantung niat'),
              (2, 2, 'id', 'Senyummu di hadapan saudaramu adalah sedekah');
            """
        )
        conn.commit()
    return quran, hadith


def test_extract_keywords_drops_stopwords_and_adds_stems() -> None:
    keywords = extract_keywords("Ayat tentang kejujuran dan amanah")

    assert [variants[0] for variants in keywords] == ["kejujuran", "amanah"]
    assert "jujur" in keywords[0]


def test_wanted_sources() -> None:
    assert wanted_sources("ayat tentang sabar") == (True, False)
    assert wanted_sources("hadits tentang niat") == (False, True)
    assert wanted_sources("dalil tentang niat") == (True, True)
    assert wanted_sources("ayat dan hadits tentang niat") == (True, True)


def test_needs_agent_loop_for_structural_requests() -> None:
    assert needs_agent_loop("daftar surah")
    assert needs_agent_loop("ayat dari juz 30")
    assert not needs_agent_loop("ayat tentang riba")


def test_retrieve_matches_word_start_only(tmp_path: Path) -> None:
    quran, hadith = make_dbs(tmp_path)

    passages = retrieve("ayat tentang riba", quran_db_path=quran, hadith_db_path=hadith)

    # "beribadah" must not match "riba".
    assert [p.label for p in passages] == ["Al-Baqarah 2:278"]
    assert passages[0].source_id == "2:278"


def test_retrieve_hadith_and_both_sources(tmp_path: Path) -> None:
    quran, hadith = make_dbs(tmp_path)

    passages = retrieve("hadits tentang senyum sedekah", quran_db_path=quran, hadith_db_path=hadith)

    assert [p.label for p in passages] == ["Sahih al-Bukhari #2"]


def test_retrieve_without_keywords_returns_nothing(tmp_path: Path) -> None:
    quran, hadith = make_dbs(tmp_path)

    assert retrieve("apa itu?", quran_db_path=quran, hadith_db_path=hadith) == []


def test_format_context() -> None:
    assert "tidak menemukan" in format_context([])
    context = format_context([Passage("quran", "Al-Baqarah 2:278", "2:278", "teks")])
    assert "[Al-Baqarah 2:278] teks" in context


def test_pipeline_answers_with_one_toolless_llm_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    FakeRunner = install_agents_stub(
        monkeypatch, [delta("Riba dilarang "), delta("(Al-Baqarah 2:278).")], final_output="Riba dilarang (Al-Baqarah 2:278)."
    )

    events = collect(runner.ask_streamed("ayat tentang riba", session_id="s1"))

    kinds = [kind for kind, _ in events]
    assert kinds == ["tool", "delta", "delta", "done"]
    assert events[0][1] == "local_db"
    reply = events[-1][1]
    assert reply.used_tools == ["local_db"]
    assert reply.references[0].source_id == "2:278"
    assert FakeRunner.seen["max_turns"] == 2
    assert "[Al-Baqarah 2:278]" in FakeRunner.seen["prompt"]


def test_pipeline_disabled_uses_agent_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    runner.settings = type(runner.settings)(**{**runner.settings.__dict__, "agent_pipeline_enabled": False})
    FakeRunner = install_agents_stub(monkeypatch, [delta("ok")], final_output="ok")

    collect(runner.ask_streamed("ayat tentang riba", session_id="s1"))

    assert FakeRunner.seen["max_turns"] == 4
    assert "KONTEKS DALIL" not in FakeRunner.seen["prompt"]


def test_structural_request_uses_agent_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    FakeRunner = install_agents_stub(monkeypatch, [delta("ok")], final_output="ok")

    collect(runner.ask_streamed("tampilkan daftar surah", session_id="s1"))

    assert FakeRunner.seen["max_turns"] == 4
