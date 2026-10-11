from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from muslim_bot.hadith_repair import needs_repair, repair

SCHEMA = """
create table books (id integer primary key, slug text unique not null, name_ar text, name_en text, totals integer);
create table chapters (id integer primary key, book_id integer not null, number integer not null,
  title_ar text not null, title_en text not null, title_id text, unique(book_id, number));
create table hadiths (id integer primary key, global_id text unique not null, book_id integer not null,
  chapter_id integer not null, number integer not null, unique(book_id, number));
create table hadith_texts (id integer primary key, hadith_id integer not null, lang text not null,
  text text not null, narration_chain text, unique(hadith_id, lang));
create virtual table hadith_fts using fts5(hadith_id, text_ar, text_en, text_id, chapter_title, book_slug);
create trigger hadith_fts_insert after insert on hadith_texts begin
  insert into hadith_fts(rowid, hadith_id, text_ar, text_en, text_id, chapter_title, book_slug)
  select new.id, new.hadith_id,
    case when new.lang = 'ar' then new.text end, case when new.lang = 'en' then new.text end,
    case when new.lang = 'id' then new.text end, c.title_ar, b.slug
  from hadiths h join chapters c on h.chapter_id = c.id join books b on h.book_id = b.id
  where h.id = new.hadith_id;
end;
"""


def make_broken_db(path: Path) -> None:
    """Mimic the seeder: no chapters, every hadith on chapter_id 1, empty FTS."""
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(SCHEMA)
        conn.execute("insert into books values (1, 'bukhari', 'ar', 'Sahih al-Bukhari', 5)")
        conn.execute("insert into books values (2, 'muslim', 'ar', 'Sahih Muslim', 2)")
        for number in range(1, 6):
            conn.execute("insert into hadiths values (?, ?, 1, 1, ?)", (number, f"bukhari-{number}", number))
            conn.execute(
                "insert into hadith_texts (hadith_id, lang, text) values (?, 'id', ?)",
                (number, f"hadits nomor {number} tentang niat" if number == 1 else f"hadits nomor {number}"),
            )
        for number in (1, 2):
            conn.execute("insert into hadiths values (?, ?, 2, 1, ?)", (10 + number, f"muslim-{number}", number))
            conn.execute(
                "insert into hadith_texts (hadith_id, lang, text) values (?, 'id', ?)", (10 + number, f"muslim {number}")
            )
        conn.commit()


def fake_fetch(slug: str) -> dict:
    if slug == "bukhari":  # section 0 is a placeholder; hadith 5 is outside every range
        return {"metadata": {
            "sections": {"0": "", "1": "Revelation", "2": "Belief"},
            "section_details": {
                "0": {"hadithnumber_first": 0, "hadithnumber_last": 0},
                "1": {"hadithnumber_first": 1, "hadithnumber_last": 2},
                "2": {"hadithnumber_first": 3, "hadithnumber_last": 4},
            },
        }}
    return {"metadata": {"sections": {"0": "Introduction"}, "section_details": {"0": {"hadithnumber_first": 1, "hadithnumber_last": 2}}}}


def test_repair_rebuilds_chapters_links_and_fts(tmp_path: Path) -> None:
    db = tmp_path / "hadith.db"
    make_broken_db(db)
    assert needs_repair(db)

    stats = repair(db, fetch=fake_fetch)

    assert not needs_repair(db)
    assert (tmp_path / "hadith.db.bak").exists()
    assert stats["indexed"] == 7  # one FTS row per text
    with closing(sqlite3.connect(db)) as conn:
        chapters = conn.execute("select b.slug, c.number, c.title_en from chapters c join books b on b.id = c.book_id order by c.id").fetchall()
        assert chapters == [
            ("bukhari", 1, "Revelation"), ("bukhari", 2, "Belief"), ("muslim", 0, "Introduction"), ("bukhari", 3, "Other"),
        ]
        linked = dict(conn.execute(
            "select h.global_id, c.title_en from hadiths h join chapters c on c.id = h.chapter_id"
        ).fetchall())
        assert linked["bukhari-1"] == "Revelation" and linked["bukhari-3"] == "Belief"
        assert linked["bukhari-5"] == "Other"  # outside every range
        assert linked["muslim-2"] == "Introduction"
        # The exact queries the Go search runs now return rows.
        assert conn.execute("select count(*) from hadith_fts where hadith_fts match 'niat'").fetchone()[0] == 1
        rows = conn.execute(
            "select fts.hadith_id, ht.text, b.slug from hadith_fts fts"
            " join hadith_texts ht on fts.rowid = ht.id join hadiths h on fts.hadith_id = h.id"
            " join chapters c on h.chapter_id = c.id join books b on h.book_id = b.id"
            " where hadith_fts match 'niat'"
        ).fetchall()
        assert rows == [(1, "hadits nomor 1 tentang niat", "bukhari")]


def test_repair_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "hadith.db"
    make_broken_db(db)
    repair(db, fetch=fake_fetch)
    repair(db, fetch=fake_fetch)

    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("select count(*) from chapters").fetchone()[0] == 4
        assert conn.execute("select count(*) from hadith_fts").fetchone()[0] == 7


def test_failed_fetch_leaves_database_untouched(tmp_path: Path) -> None:
    db = tmp_path / "hadith.db"
    make_broken_db(db)

    def boom(slug: str) -> dict:
        raise OSError("offline")

    with pytest.raises(OSError):
        repair(db, fetch=boom)

    assert needs_repair(db)
    assert not (tmp_path / "hadith.db.bak").exists()
