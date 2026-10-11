from __future__ import annotations

import sqlite3
from pathlib import Path

from muslim_bot.fast_path import (
    detect_fast_path,
    fetch_ayah,
    fetch_hadith,
    fetch_random_ayah,
    fetch_random_hadith,
    format_ayah,
    format_hadith,
)


def make_dbs(path: Path) -> tuple[Path, Path]:
    quran = path / "quran.db"
    with sqlite3.connect(quran) as conn:
        conn.execute(
            "create table ayahs (surah_id integer, number_in_surah integer,"
            " text_uthmani text, translation_indo text, translation_en text)"
        )
        conn.execute("create table surahs (number integer, name_transliteration text)")
        conn.executemany(
            "insert into ayahs values (?, ?, ?, ?, ?)",
            [
                (2, 255, "ARAB-255", "INDO-255", "EN-255"),
                (2, 153, "ARAB-153", "INDO-153", "EN-153"),
            ],
        )
        conn.executemany(
            "insert into surahs values (?, ?)", [(2, "Al-Baqarah"), (114, "An-Nas")]
        )
    hadith = path / "hadith.db"
    with sqlite3.connect(hadith) as conn:
        conn.execute("create table books (id integer primary key, slug text, name_en text)")
        conn.execute("create table hadiths (id integer primary key, book_id integer, number integer)")
        conn.execute("create table hadith_texts (hadith_id integer, lang text, text text)")
        conn.execute("insert into books values (1, 'bukhari', 'Sahih al-Bukhari')")
        conn.execute("insert into hadiths values (1, 1, 1)")
        conn.executemany(
            "insert into hadith_texts values (1, ?, ?)",
            [("ar", "ARAB-H1"), ("en", "EN-H1"), ("id", "INDO-H1")],
        )
    return quran, hadith


def test_detect_hadith_reference(tmp_path: Path) -> None:
    hit = detect_fast_path("Sahih al-Bukhari #1")

    assert hit is not None
    assert (hit.kind, hit.book_slug, hit.number) == ("hadith_ref", "bukhari", 1)


def test_detect_quran_reference(tmp_path: Path) -> None:
    hit = detect_fast_path("apa isi QS Al-Baqarah 2:255?")

    assert hit is not None
    assert (hit.kind, hit.surah, hit.ayah) == ("quran_ref", 2, 255)


def test_detect_bare_numbers(tmp_path: Path) -> None:
    hit = detect_fast_path("2:255")

    assert hit is not None
    assert (hit.kind, hit.surah, hit.ayah) == ("quran_ref", 2, 255)


def test_detect_alias(tmp_path: Path) -> None:
    hit = detect_fast_path("tampilkan ayat kursi")

    assert hit is not None
    assert (hit.kind, hit.surah, hit.ayah) == ("quran_ref", 2, 255)


def test_detect_random(tmp_path: Path) -> None:
    assert detect_fast_path("ayat acak").kind == "random_ayah"
    assert detect_fast_path("hadits acak").kind == "random_hadith"
    assert detect_fast_path("tampilkan ayat acak tentang sabar") is None


def test_detect_rejects_long_analytical_and_tafsir(tmp_path: Path) -> None:
    assert detect_fast_path("jelaskan tafsir Al-Baqarah 2:255 secara mendalam") is None
    assert detect_fast_path("apa itu tafsir?") is None
    assert detect_fast_path("bandingkan konsep sabar dalam Al-Baqarah 2:153 dan Ali Imran 3:146 beserta konteksnya") is None
    assert detect_fast_path("") is None
    assert detect_fast_path("assalamu alaikum") is None


def test_fetch_and_format_ayah(tmp_path: Path) -> None:
    quran, _ = make_dbs(tmp_path)

    data = fetch_ayah(quran, 2, 255, "id")

    assert data is not None
    answer, ref = format_ayah(data)
    assert "ARAB-255" in answer
    assert "INDO-255" in answer
    assert ref.type == "quran"
    assert ref.label == "Al-Baqarah 2:255"
    assert ref.verified is True


def test_fetch_ayah_english(tmp_path: Path) -> None:
    quran, _ = make_dbs(tmp_path)

    data = fetch_ayah(quran, 2, 255, "en")

    assert data is not None
    assert data["translation"] == "EN-255"


def test_fetch_missing_ayah_returns_none(tmp_path: Path) -> None:
    quran, _ = make_dbs(tmp_path)

    assert fetch_ayah(quran, 2, 999, "id") is None


def test_fetch_and_format_hadith(tmp_path: Path) -> None:
    _, hadith = make_dbs(tmp_path)

    data = fetch_hadith(hadith, "bukhari", 1, "id")

    assert data is not None
    answer, ref = format_hadith(data)
    assert "INDO-H1" in answer
    assert "ARAB-H1" in answer
    assert ref.type == "hadith"
    assert ref.label == "Sahih al-Bukhari #1"
    assert ref.verified is True


def test_fetch_missing_db_returns_none(tmp_path: Path) -> None:
    assert fetch_ayah(tmp_path / "nope.db", 2, 255, "id") is None
    assert fetch_hadith(tmp_path / "nope.db", "bukhari", 1, "id") is None
    assert fetch_random_ayah(tmp_path / "nope.db", "id") is None
    assert fetch_random_hadith(tmp_path / "nope.db", "id") is None


def test_fetch_random(tmp_path: Path) -> None:
    quran, hadith = make_dbs(tmp_path)

    ayah = fetch_random_ayah(quran, "id")
    assert ayah is not None
    answer, ref = format_ayah(ayah)
    assert ref.verified is True
    assert "ARAB" in answer

    entry = fetch_random_hadith(hadith, "id")
    assert entry is not None
    answer, ref = format_hadith(entry)
    assert ref.verified is True
