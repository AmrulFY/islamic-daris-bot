from __future__ import annotations

import sqlite3
from pathlib import Path


HADITH_BOOK_SLUGS = ("abudawud", "bukhari", "ibnmajah", "muslim", "nasai", "tirmidhi")
HADITH_LANGS = ("ar", "en", "id")


def sqlite_count(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(f"select count(*) from {table}").fetchone()
    return int(row[0])


def sqlite_scalar(db_path: Path, query: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(query).fetchone()
    return int(row[0])


def sqlite_file_ok(db_path: Path) -> bool:
    if not db_path.exists() or db_path.stat().st_size == 0:
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("select name from sqlite_master limit 1").fetchall()
    except sqlite3.DatabaseError:
        return False
    return True


def quran_db_ready(db_path: Path) -> bool:
    try:
        return sqlite_file_ok(db_path) and sqlite_count(db_path, "ayahs") == 6236
    except sqlite3.DatabaseError:
        return False


def hadith_db_ready(db_path: Path) -> bool:
    try:
        if (
            not sqlite_file_ok(db_path)
            or sqlite_count(db_path, "books") != 6
            or sqlite_count(db_path, "hadiths") < 30000
            or sqlite_scalar(db_path, "select count(distinct lang) from hadith_texts") < 3
        ):
            return False

        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                select b.slug, t.lang, count(*)
                from books b
                join hadiths h on h.book_id = b.id
                join hadith_texts t on t.hadith_id = h.id
                group by b.slug, t.lang
                """
            ).fetchall()

        counts = {(str(slug), str(lang)): int(count) for slug, lang, count in rows}
        return all(counts.get((slug, lang), 0) > 0 for slug in HADITH_BOOK_SLUGS for lang in HADITH_LANGS)
    except sqlite3.DatabaseError:
        return False
