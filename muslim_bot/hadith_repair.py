"""Repair a Hadith database produced by the Go seeder.

The seeder (hadith-api-go-run) leaves the database in a state where
`search_hadith` never returns anything:

* it reads `section` / `section_detail` from the edition JSON, but the keys are
  `sections` / `section_details`, so no chapters are ever created;
* every hadith is written with `chapter_id = 1`;
* the FTS trigger joins `chapters`, so with no chapters `hadith_fts` stays
  empty, and the LIKE fallback joins `chapters` too (zero rows, while its count
  query skips the join, hence `data: null` with `total > 0`).

This module rebuilds the chapters from the same edition files, points each
hadith at its real chapter and rebuilds `hadith_fts`. Run it with
`python -m muslim_bot.hadith_repair`; `bootstrap` also calls it automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Callable

from .config import load_settings

logger = logging.getLogger(__name__)

EDITION_URL = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/ara-{slug}.min.json"
FETCH_TIMEOUT_SECONDS = 120

Fetch = Callable[[str], dict]


def fetch_edition(slug: str) -> dict:
    with urllib.request.urlopen(EDITION_URL.format(slug=slug), timeout=FETCH_TIMEOUT_SECONDS) as response:
        return json.load(response)


def needs_repair(db_path: Path) -> bool:
    """True when chapters or the FTS index are empty."""
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            chapters = conn.execute("select count(*) from chapters").fetchone()[0]
            indexed = conn.execute("select count(*) from hadith_fts").fetchone()[0]
    except sqlite3.DatabaseError:
        return False
    return chapters == 0 or indexed == 0


def _sections(edition: dict) -> list[tuple[int, str, int, int]]:
    """(number, title, first hadith, last hadith) for each real section."""
    metadata = edition.get("metadata", {})
    titles = metadata.get("sections", {})
    details = metadata.get("section_details", {})
    sections: list[tuple[int, str, int, int]] = []
    for key, detail in details.items():
        first, last = int(detail.get("hadithnumber_first", 0)), int(detail.get("hadithnumber_last", 0))
        if last <= 0:  # placeholder section "0" with no hadiths
            continue
        number = int(key)
        title = str(titles.get(key, "")).strip() or f"Bab {number}"
        sections.append((number, title, first, last))
    return sorted(sections)


def repair(db_path: Path, fetch: Fetch = fetch_edition, *, backup: bool = True) -> dict[str, int]:
    """Rebuild chapters, hadith-to-chapter links and the FTS index in place."""
    with closing(sqlite3.connect(db_path)) as conn:
        books = conn.execute("select id, slug from books order by id").fetchall()
    # Fetch everything first so a network failure leaves the database untouched.
    editions = {slug: _sections(fetch(slug)) for _, slug in books}
    if not any(editions.values()):
        raise RuntimeError("edition files contained no sections")

    if backup:
        backup_path = db_path.with_name(db_path.name + ".bak")
        if not backup_path.exists():
            shutil.copy2(db_path, backup_path)
            logger.info("hadith db backed up to %s", backup_path)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("begin")
        conn.execute("delete from chapters")
        # 0 = not mapped yet. The seeder's chapter_id = 1 can look valid by accident.
        conn.execute("update hadiths set chapter_id = 0")
        for book_id, slug in books:
            for number, title, first, last in editions[slug]:
                cursor = conn.execute(
                    "insert into chapters (book_id, number, title_ar, title_en, title_id)"
                    " values (?, ?, ?, ?, null)",
                    (book_id, number, title, title),
                )
                conn.execute(
                    "update hadiths set chapter_id = ? where book_id = ? and number between ? and ?",
                    (cursor.lastrowid, book_id, first, last),
                )

        # Hadiths outside every section range get a per-book catch-all chapter.
        orphans = conn.execute("select distinct book_id from hadiths where chapter_id = 0").fetchall()
        for (book_id,) in orphans:
            next_number = conn.execute(
                "select coalesce(max(number), 0) + 1 from chapters where book_id = ?", (book_id,)
            ).fetchone()[0]
            cursor = conn.execute(
                "insert into chapters (book_id, number, title_ar, title_en, title_id)"
                " values (?, ?, 'Lainnya', 'Other', 'Lainnya')",
                (book_id, next_number),
            )
            conn.execute(
                "update hadiths set chapter_id = ? where book_id = ? and chapter_id = 0",
                (cursor.lastrowid, book_id),
            )

        conn.execute("delete from hadith_fts")
        conn.execute(
            "insert into hadith_fts (rowid, hadith_id, text_ar, text_en, text_id, chapter_title, book_slug)"
            " select ht.id, ht.hadith_id,"
            "  case when ht.lang = 'ar' then ht.text end,"
            "  case when ht.lang = 'en' then ht.text end,"
            "  case when ht.lang = 'id' then ht.text end,"
            "  c.title_ar, b.slug"
            " from hadith_texts ht"
            " join hadiths h on h.id = ht.hadith_id"
            " join chapters c on c.id = h.chapter_id"
            " join books b on b.id = h.book_id"
        )
        conn.commit()
        conn.execute("insert into hadith_fts(hadith_fts) values ('optimize')")
        conn.commit()
        stats = {
            "chapters": conn.execute("select count(*) from chapters").fetchone()[0],
            "indexed": conn.execute("select count(*) from hadith_fts").fetchone()[0],
            "orphan_books": len(orphans),
        }
    logger.info("hadith db repaired: %s", stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair chapters and FTS in the Hadith database.")
    parser.add_argument("--db", type=Path, help="Hadith database (default: HADITH_DB_PATH).")
    parser.add_argument("--no-backup", action="store_true", help="Skip the .bak copy.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if nothing looks broken.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db_path = args.db or load_settings().hadith_db_path
    if not args.force and not needs_repair(db_path):
        print(f"{db_path}: chapters and FTS already populated, nothing to do (use --force to rebuild).")
        return 0
    stats = repair(db_path, backup=not args.no_backup)
    print(f"{db_path}: repaired {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
