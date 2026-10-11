from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path

from .schemas import Reference


BOOK_SLUGS: dict[str, str] = {
    "sahih al bukhari": "bukhari",
    "sahih bukhari": "bukhari",
    "sahih muslim": "muslim",
    "sunan abu dawud": "abudawud",
    "jami at tirmidhi": "tirmidhi",
    "sunan an nasai": "nasai",
    "sunan ibn majah": "ibnmajah",
}

_QURAN_SURAH_COLS = ("surah", "surah_id", "surah_number", "surat", "sura", "chapter")
_QURAN_AYAH_COLS = ("number", "ayah", "ayah_number", "verse", "ayat", "number_in_surah")
_QURAN_SURAH_NAME_COLS = ("name_transliteration", "name_latin", "name", "title")
_HADITH_NUM_COLS = ("number", "hadith_number", "hadith_no", "no")

_SOURCE_ID_RE = re.compile(r"^(\d{1,3}):(\d{1,3})$")
_TRAILING_REF_RE = re.compile(r"\s*\d{1,3}:\d{1,3}\s*$")


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _normalize_book(label: str) -> str | None:
    cleaned = re.sub(r"['`\-]", " ", label.lower())
    cleaned = re.sub(r"\s*#\d+\s*$", "", " ".join(cleaned.split()))
    return BOOK_SLUGS.get(cleaned)


class ReferenceValidator:
    """Verifies extracted citations against the local Quran/Hadith databases.

    Column names are detected from the live schema so the validator works with
    either sibling Go project's database layout. Returns True/False for a
    definitive verdict and None when verification is impossible (database
    NOTE: sqlite3 connections are NOT closed by the context manager protocol
    (it only commits/rolls back), so every connection here is wrapped in
    contextlib.closing. Leaked connections hold file locks that prevent
    database files from being moved or deleted on Windows.
    """

    def __init__(self, quran_db_path: Path, hadith_db_path: Path) -> None:
        self.quran_db_path = quran_db_path
        self.hadith_db_path = hadith_db_path
        self._cache: dict[tuple[str, str, str], bool | None] = {}
        self._schema_cache: dict[Path, dict[str, set[str]]] = {}

    def validate(self, ref_type: str, source_id: str | None, label: str) -> bool | None:
        key = (ref_type, source_id or "", label.lower())
        if key in self._cache:
            return self._cache[key]

        if ref_type == "quran":
            verdict = self._validate_quran(source_id, label)
        elif ref_type == "hadith":
            verdict = self._validate_hadith(source_id, label)
        else:
            verdict = None

        self._cache[key] = verdict
        return verdict

    def _columns(self, db_path: Path, table: str) -> set[str]:
        tables = self._schema_cache.setdefault(db_path, {})
        if table not in tables:
            try:
                with closing(sqlite3.connect(db_path)) as conn:
                    rows = conn.execute(f"pragma table_info({table})").fetchall()
                tables[table] = {str(row[1]) for row in rows}
            except sqlite3.Error:
                tables[table] = set()
        return tables[table]

    def _find_column(self, columns: set[str], candidates: tuple[str, ...]) -> str | None:
        lowered = {name.lower(): name for name in columns}
        for candidate in candidates:
            actual = lowered.get(candidate)
            if actual:
                return actual
        return None

    def _validate_quran(self, source_id: str | None, label: str) -> bool | None:
        if not source_id:
            return None
        match = _SOURCE_ID_RE.match(source_id.strip())
        if not match:
            return None
        surah, ayah = int(match.group(1)), int(match.group(2))
        if not (1 <= surah <= 114):
            return False

        columns = self._columns(self.quran_db_path, "ayahs")
        if not columns:
            return None
        surah_col = self._find_column(columns, _QURAN_SURAH_COLS)
        ayah_col = self._find_column(columns, _QURAN_AYAH_COLS)
        if not surah_col or not ayah_col:
            return None

        try:
            with closing(sqlite3.connect(self.quran_db_path)) as conn:
                row = conn.execute(
                    f"select 1 from ayahs where {surah_col} = ? and {ayah_col} = ? limit 1",  # noqa: S608
                    (surah, ayah),
                ).fetchone()
                if row is None:
                    return False
                return self._validate_surah_name(conn, surah, label, default=True)
        except sqlite3.Error:
            return None
        return True

    def _validate_surah_name(self, conn: sqlite3.Connection, surah: int, label: str, default: bool) -> bool:
        """Cross-check that the surah name in the label matches its number.

        Catches pairings like "Al-Mulk 2:255" where both halves look plausible
        alone. Returns `default` when the check cannot be performed.
        """
        name = _TRAILING_REF_RE.sub("", label).strip()
        name = re.sub(r"^QS\.?\s+", "", name, flags=re.IGNORECASE)
        wanted = _normalize_name(name)
        if not wanted:
            return default
        columns = self._columns(self.quran_db_path, "surahs")
        name_col = self._find_column(columns, _QURAN_SURAH_NAME_COLS) if columns else None
        number_col = self._find_column(columns, ("number", "id")) if columns else None
        if not name_col or not number_col:
            return default
        try:
            actual = conn.execute(
                f"select {name_col} from surahs where {number_col} = ? limit 1",  # noqa: S608
                (surah,),
            ).fetchone()
        except sqlite3.Error:
            return default
        if not actual:
            return default
        actual_normalized = _normalize_name(str(actual[0]))
        if not actual_normalized:
            return default
        return wanted == actual_normalized or wanted in actual_normalized or actual_normalized in wanted

    def _validate_hadith(self, source_id: str | None, label: str) -> bool | None:
        if not source_id or not source_id.strip().isdigit():
            return None
        slug = _normalize_book(label)
        if not slug:
            return None

        book_columns = self._columns(self.hadith_db_path, "books")
        hadith_columns = self._columns(self.hadith_db_path, "hadiths")
        if not book_columns or not hadith_columns:
            return None
        num_col = self._find_column(hadith_columns, _HADITH_NUM_COLS)
        if "slug" not in {name.lower() for name in book_columns} or not num_col:
            return None

        try:
            with closing(sqlite3.connect(self.hadith_db_path)) as conn:
                row = conn.execute(
                    "select 1 from hadiths h join books b on h.book_id = b.id"
                    f" where b.slug = ? and h.{num_col} = ? limit 1",  # noqa: S608
                    (slug, int(source_id)),
                ).fetchone()
        except sqlite3.Error:
            return None
        return row is not None


def annotate_references(
    refs: list[Reference],
    validator: ReferenceValidator,
) -> tuple[list[Reference], list[str]]:
    """Attach verification verdicts and produce human-readable warnings."""
    annotated: list[Reference] = []
    warnings: list[str] = []
    unavailable = False

    for ref in refs:
        verdict = validator.validate(ref.type, ref.source_id, ref.label)
        annotated.append(ref.model_copy(update={"verified": verdict}))
        if verdict is False:
            warnings.append(
                f"Referensi '{ref.label}' tidak ditemukan di database lokal; mohon verifikasi manual."
            )
        elif verdict is None:
            unavailable = True

    if refs and unavailable:
        warnings.append("Validasi referensi tidak lengkap karena database lokal belum siap.")

    return annotated, warnings
