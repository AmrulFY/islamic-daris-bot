"""FTS5 keyword search: BM25-ranked replacement for LIKE full-scans.

Never writes to the Go-managed databases. When the source DB already has an
FTS index (hadith_fts from the seeder/repair), it is reused. Otherwise a
sidecar ``<stem>_fts.db`` next to the source DB is built once and reused
while the source mtime+size are unchanged.

Falls back to None when SQLite lacks FTS5 so callers can use LIKE.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

_SANITIZE_RE = re.compile(r"[^\w']+", re.UNICODE)


def has_fts5() -> bool:
    try:
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE VIRTUAL TABLE _t USING fts5(x)")
    except sqlite3.DatabaseError:
        return False
    return True


def fts_path_for(db_path: Path) -> Path:
    return db_path.with_name(f"{db_path.stem}_fts.db")


def _clean_term(variant: str) -> str:
    cleaned = _SANITIZE_RE.sub("", variant.lower()).strip("'")
    return cleaned


def build_match(keywords: list[list[str]]) -> str:
    """OR of prefix terms across all variants; BM25 ranks multi-term hits higher."""
    terms: list[str] = []
    seen: set[str] = set()
    for variants in keywords:
        for variant in variants:
            term = _clean_term(variant)
            if len(term) < 3 or term in seen:
                continue
            seen.add(term)
            # Prefix (*) gives us the old stemming behaviour (jujur* hits
            # kejujuran) without mid-word hits (riba* never hits beribadah,
            # because FTS tokenizes on word boundaries).
            terms.append(f'"{term}"*')
    return " OR ".join(terms)


def _source_fingerprint(db_path: Path) -> str:
    try:
        stat = db_path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return "missing"


def _sidecar_fresh(fts_path: Path, fingerprint: str, kind: str) -> bool:
    if not fts_path.exists():
        return False
    try:
        with closing(sqlite3.connect(fts_path)) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='fp'").fetchone()
            kind_row = conn.execute("SELECT value FROM meta WHERE key='kind'").fetchone()
            count = conn.execute("SELECT count(*) FROM docs").fetchone()
    except sqlite3.DatabaseError:
        return False
    return (
        row is not None
        and row[0] == fingerprint
        and kind_row is not None
        and kind_row[0] == kind
        and count is not None
        and count[0] > 0
    )


def _write_meta(fts_path: Path, fingerprint: str, kind: str) -> None:
    with closing(sqlite3.connect(fts_path)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('fp', ?)", (fingerprint,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('kind', ?)", (kind,))
        conn.commit()


def _table_cols(db_path: Path, table: str) -> set[str]:
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.DatabaseError:
        return set()
    return {str(r[1]).lower() for r in rows}


def ensure_quran_sidecar(db_path: Path) -> Path | None:
    """Build/reuse sidecar FTS for ayahs. Returns path or None on failure."""
    if not has_fts5() or not db_path.exists():
        return None
    cols = _table_cols(db_path, "ayahs")
    if not cols:
        return None
    has_indo = "translation_indo" in cols
    has_en = "translation_en" in cols
    if not has_indo and not has_en:
        return None
    fp = _source_fingerprint(db_path)
    fts_path = fts_path_for(db_path)
    if _sidecar_fresh(fts_path, fp, "quran"):
        return fts_path
    try:
        with closing(sqlite3.connect(db_path)) as src:
            query_cols = []
            if has_indo:
                query_cols.append("translation_indo")
            if has_en:
                query_cols.append("translation_en")
            rows = src.execute(f"SELECT id, {', '.join(query_cols)} FROM ayahs").fetchall()
        col_names = ["text_indo"] if has_indo and not has_en else (
            ["text_en"] if has_en and not has_indo else ["text_indo", "text_en"]
        )
        # Rebuild atomically: write temp file then replace.
        tmp_path = fts_path.with_suffix(".tmp.db")
        if tmp_path.exists():
            tmp_path.unlink()
        with closing(sqlite3.connect(tmp_path)) as conn:
            col_defs = ", ".join(col_names)
            conn.execute(
                f"CREATE VIRTUAL TABLE docs USING fts5({col_defs}, tokenize='unicode61 remove_diacritics 2')"
            )
            if len(col_names) == 2:
                data = []
                for r in rows:
                    # rows: (id, indo, en) when both exist
                    data.append((r[0], str(r[1] or ""), str(r[2] or "")))
                conn.executemany("INSERT INTO docs(rowid, text_indo, text_en) VALUES (?, ?, ?)", data)
            else:
                data = [(r[0], str(r[1] or "")) for r in rows]
                conn.executemany(f"INSERT INTO docs(rowid, {col_names[0]}) VALUES (?, ?)", data)
            conn.commit()
            conn.execute("INSERT INTO docs(docs) VALUES ('optimize')")
            conn.commit()
        tmp_path.replace(fts_path)
        _write_meta(fts_path, fp, "quran")
    except (sqlite3.DatabaseError, OSError):
        return None
    return fts_path


def ensure_hadith_sidecar(db_path: Path) -> Path | None:
    if not has_fts5() or not db_path.exists():
        return None
    if "hadith_texts" not in _table_cols(db_path, "hadith_texts"):
        # _table_cols returns column set; check table existence via master
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hadith_texts'"
                ).fetchone()
            if not row:
                return None
        except sqlite3.DatabaseError:
            return None
    fp = _source_fingerprint(db_path)
    fts_path = fts_path_for(db_path)
    if _sidecar_fresh(fts_path, fp, "hadith"):
        return fts_path
    try:
        with closing(sqlite3.connect(db_path)) as src:
            rows = src.execute(
                "SELECT hadith_id, lang, text FROM hadith_texts WHERE lang IN ('id','en')"
            ).fetchall()
        by_id: dict[int, dict[str, str]] = {}
        for hid, lang, text in rows:
            by_id.setdefault(int(hid), {})[str(lang)] = str(text or "")
        data = [(hid, texts.get("id", ""), texts.get("en", "")) for hid, texts in by_id.items()]
        tmp_path = fts_path.with_suffix(".tmp.db")
        if tmp_path.exists():
            tmp_path.unlink()
        with closing(sqlite3.connect(tmp_path)) as conn:
            conn.execute(
                "CREATE VIRTUAL TABLE docs USING fts5(text_id, text_en, tokenize='unicode61 remove_diacritics 2')"
            )
            conn.executemany("INSERT INTO docs(rowid, text_id, text_en) VALUES (?, ?, ?)", data)
            conn.commit()
            conn.execute("INSERT INTO docs(docs) VALUES ('optimize')")
            conn.commit()
        tmp_path.replace(fts_path)
        _write_meta(fts_path, fp, "hadith")
    except (sqlite3.DatabaseError, OSError):
        return None
    return fts_path


def native_hadith_fts(db_path: Path) -> bool:
    """True when the Go-managed DB already has a usable hadith_fts index."""
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hadith_fts'"
            ).fetchone()
            if not row:
                return False
            count = conn.execute("SELECT count(*) FROM hadith_fts").fetchone()
    except sqlite3.DatabaseError:
        return False
    return count is not None and int(count[0]) > 0


def search_native_hadith_fts(
    db_path: Path, match: str, language: str, limit: int
) -> list[tuple[int, float]] | None:
    """Ranked (hadith_id, bm25) from the seeder's hadith_fts. None on failure."""
    col = "text_en" if language == "en" else "text_id"
    if not match.strip():
        return None
    query = f"{col} : ({match})"
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT hadith_id, bm25(hadith_fts) AS r FROM hadith_fts"
                " WHERE hadith_fts MATCH ? ORDER BY r LIMIT ?",
                (query, limit),
            ).fetchall()
    except sqlite3.DatabaseError:
        return None
    return [(int(hid), float(r)) for hid, r in rows]


def search_sidecar(
    fts_path: Path, match: str, language: str, limit: int
) -> list[tuple[int, float]] | None:
    try:
        with closing(sqlite3.connect(fts_path)) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(docs)").fetchall()}
    except sqlite3.DatabaseError:
        return None
    if not match.strip():
        return None
    if "text_indo" in cols:  # quran sidecar
        col = "text_en" if language == "en" and "text_en" in cols else "text_indo"
        if col not in cols:
            return None
    else:  # hadith sidecar (text_id/text_en)
        col = "text_en" if language == "en" else "text_id"
        if col not in cols:
            return None
    query = f"{col} : ({match})"
    try:
        with closing(sqlite3.connect(fts_path)) as conn:
            rows = conn.execute(
                "SELECT rowid, bm25(docs) AS r FROM docs WHERE docs MATCH ? ORDER BY r LIMIT ?",
                (query, limit),
            ).fetchall()
    except sqlite3.DatabaseError:
        return None
    return [(int(rid), float(r)) for rid, r in rows]
