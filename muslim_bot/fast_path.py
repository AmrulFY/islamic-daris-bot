from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .references import BOOK_SLUGS
from .schemas import Reference


@dataclass(frozen=True)
class FastHit:
    kind: str  # "hadith_ref" | "quran_ref" | "random_ayah" | "random_hadith"
    book_slug: str | None = None
    number: int | None = None
    surah: int | None = None
    ayah: int | None = None


MAX_FAST_MESSAGE_CHARS = 120

ALIASES: dict[str, tuple[int, int]] = {
    "ayat kursi": (2, 255),
}

RANDOM_KEYWORDS = (
    "ayat acak",
    "hadits acak",
    "hadith acak",
    "acak",
    "random",
    "ayat random",
    "hadith random",
)


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"['`\-]", " ", value.lower())
    return " ".join(cleaned.split())


def _book_names_by_slug() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name, slug in BOOK_SLUGS.items():
        grouped.setdefault(slug, []).append(name)
    return grouped


def _hadith_pattern() -> re.Pattern[str]:
    alternatives = sorted(
        (name for names in _book_names_by_slug().values() for name in names),
        key=len,
        reverse=True,
    )
    return re.compile(r"(%s)\s*[#:]?\s*(\d{1,5})" % "|".join(alternatives))


HADITH_REF_PATTERN = _hadith_pattern()
QURAN_NUMBERS_PATTERN = re.compile(r"(\d{1,3})\s*:\s*(\d{1,3})")


def detect_fast_path(message: str) -> FastHit | None:
    """Detect queries answerable by direct database lookup.

    Only short, lookup-shaped messages match. Anything analytical
    (tafsir, comparisons, explanations) falls through to the agent.
    """
    normalized = _normalize_text(message)
    if not normalized or "tafsir" in normalized:
        return None

    cleaned = normalized.rstrip("?!.").strip()
    if cleaned in RANDOM_KEYWORDS:
        if "hadits" in cleaned or "hadith" in cleaned:
            return FastHit(kind="random_hadith")
        return FastHit(kind="random_ayah")

    if len(message) > MAX_FAST_MESSAGE_CHARS:
        return None

    for alias, (surah, ayah) in ALIASES.items():
        if alias in normalized:
            return FastHit(kind="quran_ref", surah=surah, ayah=ayah)

    # Multiple references signal comparison/analysis work for the agent.
    if len(HADITH_REF_PATTERN.findall(normalized)) > 1:
        return None
    if len(QURAN_NUMBERS_PATTERN.findall(message)) > 1:
        return None

    match = HADITH_REF_PATTERN.search(normalized)
    if match:
        slug = BOOK_SLUGS[match.group(1)]
        return FastHit(kind="hadith_ref", book_slug=slug, number=int(match.group(2)))

    match = QURAN_NUMBERS_PATTERN.search(message)
    if match:
        return FastHit(kind="quran_ref", surah=int(match.group(1)), ayah=int(match.group(2)))

    return None


def _surah_name(conn: sqlite3.Connection, surah: int) -> str | None:
    try:
        row = conn.execute(
            "select name_transliteration from surahs where number = ? limit 1", (surah,)
        ).fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]) if row else None


def fetch_ayah(db_path: Path, surah: int, ayah: int, language: str = "id") -> dict | None:
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "select surah_id, number_in_surah, text_uthmani, translation_indo, translation_en"
                " from ayahs where surah_id = ? and number_in_surah = ? limit 1",
                (surah, ayah),
            ).fetchone()
            if row is None:
                return None
            name = _surah_name(conn, surah) or f"Surah {surah}"
    except sqlite3.Error:
        return None
    translation = row[4] if language == "en" else row[3]
    return {
        "surah": row[0],
        "ayah": row[1],
        "arabic": row[2],
        "translation": translation,
        "surah_name": name,
    }


def fetch_random_ayah(db_path: Path, language: str = "id") -> dict | None:
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "select surah_id, number_in_surah, text_uthmani, translation_indo, translation_en"
                " from ayahs order by random() limit 1"
            ).fetchone()
            if row is None:
                return None
            name = _surah_name(conn, row[0]) or f"Surah {row[0]}"
    except sqlite3.Error:
        return None
    translation = row[4] if language == "en" else row[3]
    return {
        "surah": row[0],
        "ayah": row[1],
        "arabic": row[2],
        "translation": translation,
        "surah_name": name,
    }


def fetch_hadith(db_path: Path, slug: str, number: int, language: str = "id") -> dict | None:
    preferred = ["en", "id", "ar"] if language == "en" else ["id", "en", "ar"]
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "select h.id, h.number, b.name_en from hadiths h"
                " join books b on h.book_id = b.id"
                " where b.slug = ? and h.number = ? limit 1",
                (slug, number),
            ).fetchone()
            if row is None:
                return None
            texts = conn.execute(
                "select lang, text from hadith_texts where hadith_id = ?", (row[0],)
            ).fetchall()
    except sqlite3.Error:
        return None
    by_lang = {str(lang): str(text) for lang, text in texts}
    translation = next((by_lang[lang] for lang in preferred if by_lang.get(lang)), None)
    arabic = by_lang.get("ar")
    return {
        "book": row[2],
        "number": row[1],
        "arabic": arabic,
        "translation": translation,
    }


def fetch_random_hadith(db_path: Path, language: str = "id") -> dict | None:
    preferred = ["en", "id", "ar"] if language == "en" else ["id", "en", "ar"]
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute(
                "select h.id, h.number, b.name_en from hadiths h"
                " join books b on h.book_id = b.id"
                " order by random() limit 1"
            ).fetchone()
            if row is None:
                return None
            texts = conn.execute(
                "select lang, text from hadith_texts where hadith_id = ?", (row[0],)
            ).fetchall()
    except sqlite3.Error:
        return None
    by_lang = {str(lang): str(text) for lang, text in texts}
    translation = next((by_lang[lang] for lang in preferred if by_lang.get(lang)), None)
    return {
        "book": row[2],
        "number": row[1],
        "arabic": by_lang.get("ar"),
        "translation": translation,
    }


def format_ayah(data: dict) -> tuple[str, Reference]:
    label = f"{data['surah_name']} {data['surah']}:{data['ayah']}"
    lines = [
        f"## 📖 {label}",
        "",
        "**Teks Arab:**",
        f"> {data['arabic']}",
        "",
        "**Terjemahan Indonesia:**",
        f"> \"{data['translation']}\"",
    ]
    reference = Reference(
        type="quran",
        label=label,
        source_id=f"{data['surah']}:{data['ayah']}",
        verified=True,
    )
    return "\n".join(lines), reference


def format_hadith(data: dict) -> tuple[str, Reference]:
    label = f"{data['book']} #{data['number']}"
    lines = [f"## 📜 {label}", ""]
    if data.get("arabic"):
        lines += ["**Teks Arab:**", f"> {data['arabic']}", ""]
    lines += ["**Terjemahan Indonesia:**", f"> \"{data['translation']}\""]
    reference = Reference(
        type="hadith",
        label=label,
        source_id=str(data["number"]),
        verified=True,
    )
    return "\n".join(lines), reference
