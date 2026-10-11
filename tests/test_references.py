from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from muslim_bot.references import ReferenceValidator, annotate_references
from muslim_bot.schemas import Reference


def make_quran_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("create table ayahs (surah_id integer, number_in_surah integer)")
        conn.executemany(
            "insert into ayahs values (?, ?)",
            [(1, 1), (2, 255), (67, 30), (114, 6)],
        )
        conn.execute("create table surahs (number integer, name_transliteration text)")
        conn.executemany(
            "insert into surahs values (?, ?)",
            [(1, "Al-Fatihah"), (2, "Al-Baqarah"), (67, "Al-Mulk"), (114, "An-Nas")],
        )


def make_hadith_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("create table books (id integer primary key, slug text)")
        conn.execute(
            "create table hadiths (id integer primary key, book_id integer, number integer)"
        )
        conn.executemany("insert into books values (?, ?)", [(1, "bukhari"), (2, "muslim")])
        conn.executemany(
            "insert into hadiths values (?, ?, ?)",
            [(1, 1, 1), (2, 1, 7), (3, 2, 1)],
        )


def make_validator(tmp_path: Path, *, quran: bool = True, hadith: bool = True) -> ReferenceValidator:
    quran_path = tmp_path / "quran.db"
    hadith_path = tmp_path / "hadith.db"
    if quran:
        make_quran_db(quran_path)
    else:
        quran_path = tmp_path / "missing-quran.db"
    if hadith:
        make_hadith_db(hadith_path)
    else:
        hadith_path = tmp_path / "missing-hadith.db"
    return ReferenceValidator(quran_path, hadith_path)


def test_valid_quran_reference_is_verified(tmp_path: Path) -> None:
    assert make_validator(tmp_path).validate("quran", "2:255", "Al-Baqarah 2:255") is True


def test_hallucinated_quran_reference_is_rejected(tmp_path: Path) -> None:
    assert make_validator(tmp_path).validate("quran", "67:999", "Al-Mulk 67:999") is False


def test_mismatched_surah_name_and_number_is_rejected(tmp_path: Path) -> None:
    assert make_validator(tmp_path).validate("quran", "2:255", "Al-Mulk 2:255") is False


def test_surah_name_variants_are_accepted(tmp_path: Path) -> None:
    validator = make_validator(tmp_path)
    assert validator.validate("quran", "2:255", "QS Al Baqarah 2:255") is True
    assert validator.validate("quran", "2:255", "al-baqarah 2:255") is True


def test_out_of_range_surah_is_rejected_without_db_lookup(tmp_path: Path) -> None:
    assert make_validator(tmp_path).validate("quran", "200:1", "QS 200:1") is False


def test_malformed_quran_source_id_is_unverifiable(tmp_path: Path) -> None:
    assert make_validator(tmp_path).validate("quran", "abc", "weird") is None


def test_valid_hadith_reference_is_verified(tmp_path: Path) -> None:
    validator = make_validator(tmp_path)
    assert validator.validate("hadith", "7", "Sahih al-Bukhari #7") is True


def test_hallucinated_hadith_number_is_rejected(tmp_path: Path) -> None:
    assert make_validator(tmp_path).validate("hadith", "99999", "Sahih Muslim #99999") is False


def test_unknown_hadith_book_is_unverifiable(tmp_path: Path) -> None:
    assert make_validator(tmp_path).validate("hadith", "5", "Sunan an-Darimi #5") is None


def test_missing_databases_yield_none(tmp_path: Path) -> None:
    validator = make_validator(tmp_path, quran=False, hadith=False)
    assert validator.validate("quran", "2:255", "Al-Baqarah 2:255") is None
    assert validator.validate("hadith", "1", "Sahih al-Bukhari #1") is None


def test_validate_results_are_cached(tmp_path: Path) -> None:
    validator = make_validator(tmp_path)
    first = validator.validate("quran", "2:255", "Al-Baqarah 2:255")

    def fail_if_db_touched(*args, **kwargs):
        raise AssertionError("database accessed on cached lookup")

    validator._columns = fail_if_db_touched
    second = validator.validate("quran", "2:255", "Al-Baqarah 2:255")

    assert first is True
    assert second is True


def test_annotate_references_adds_warnings_for_unverified(tmp_path: Path) -> None:
    refs = [
        Reference(type="quran", label="Al-Baqarah 2:255", source_id="2:255"),
        Reference(type="quran", label="Al-Mulk 67:999", source_id="67:999"),
    ]

    annotated, warnings = annotate_references(refs, make_validator(tmp_path))

    assert annotated[0].verified is True
    assert annotated[1].verified is False
    assert len(warnings) == 1
    assert "Al-Mulk 67:999" in warnings[0]


def test_annotate_references_warns_when_db_missing(tmp_path: Path) -> None:
    refs = [Reference(type="quran", label="Al-Baqarah 2:255", source_id="2:255")]

    annotated, warnings = annotate_references(refs, make_validator(tmp_path, quran=False))

    assert annotated[0].verified is None
    assert len(warnings) == 1
    assert "belum siap" in warnings[0]