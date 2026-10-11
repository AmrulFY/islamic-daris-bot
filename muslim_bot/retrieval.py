"""Deterministic retrieval for the single-LLM-call answer pipeline.

The agent loop spends 4-6s per LLM round just to decide which tool to call.
Here plain code extracts keywords and queries the local SQLite databases, so
the LLM is only called once, to write the answer from the retrieved passages.
"""

from __future__ import annotations

import itertools
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

MAX_PASSAGES_PER_SOURCE = 4
# Candidates fetched per source before fusion; the final context sent to the
# LLM is capped lower (MAX_CONTEXT_PER_SOURCE) to cut tokens/latency. The LLM
# prompt asks for the best 2-4 dalil, so depth beyond 5 mostly adds noise.
MAX_CANDIDATES_PER_SOURCE = 12
MAX_CONTEXT_PER_SOURCE = 5
MAX_HADITH_CHARS = 700
MAX_KEYWORDS = 6

STOPWORDS = frozenset(
    """
    ayat ayah quran alquran al qs surah surat hadits hadist hadis hadith dalil dalilnya
    tentang mengenai terkait soal apa apakah adalah itu ini yang dan atau dengan dalam
    untuk pada dari di ke kepada oleh sebagai menurut bagaimana mengapa kenapa siapa
    kapan dimana berikut sebutkan jelaskan carikan cari tolong mohon bisa dapat boleh
    saya aku kami kita anda kamu mereka ada tidak bukan juga serta sertakan beserta
    keutamaan hukum pandangan islam menurut agama muslim rujukan sumber
    orang lain sesama semua apa saja saling para kita ketika jika maka agar supaya
    tadi tersebut sebelumnya selanjutnya lanjut lanjutan lagi kedua pertama ketiga terakhir
    nomor maksud maksudnya artinya arti makna lebih sederhana singkat ringkas contoh
    terima kasih terimakasih makasih ya dong deh nih sih lah kok
    what is the of in on about verse verses hadith please explain give tell
    """.split()
)

_PREFIXES = ("meng", "mem", "men", "me", "ber", "be", "ter", "per", "pe", "ke", "di")
_SUFFIXES = ("kan", "an", "nya", "lah", "i")

_QURAN_HINT = re.compile(r"\b(ayat|ayah|quran|al-quran|alquran|qs|verse)\b", re.IGNORECASE)
_HADITH_HINT = re.compile(r"\b(hadits|hadist|hadis|hadith)\b", re.IGNORECASE)
# Structural lookups (listing, browsing) are better served by MCP tools.
_STRUCTURAL = re.compile(
    r"\b(surah|surat|juz|daftar|kitab|bab|chapter|list|semua|seluruh)\b", re.IGNORECASE
)


_BRACKETS_RE = re.compile(r"\[[^\]]*\]")


def hadith_matn(text: str, limit: int = 500) -> str:
    """Hadith body without the narrator chain, for embedding.

    Indonesian hadith translations open with a chain of "[Name] telah
    menceritakan kepada kami ..." that says nothing about the topic. The body
    follows the last bracketed narrator, so keep what comes after it.
    """
    last = text.rfind("]")
    body = text[last + 1 :] if last != -1 else text
    body = _BRACKETS_RE.sub(" ", body)
    body = " ".join(body.split())
    if len(body) < 40:
        body = " ".join(_BRACKETS_RE.sub(" ", text).split())[-limit:]
    return body[:limit]


@dataclass(frozen=True)
class Passage:
    kind: str  # "quran" | "hadith"
    label: str
    source_id: str
    text: str
    score: float = 0.0


def needs_agent_loop(message: str) -> bool:
    """True for structural requests (browse a surah/juz/book) the pipeline can't serve."""
    return bool(_STRUCTURAL.search(message))


@dataclass(frozen=True)
class RetrievalQuery:
    text: str  # what to search for (keywords, embedding)
    sources_from: str  # which message decides Quran vs Hadith
    follow_up: bool  # the topic was taken from the previous question


def build_retrieval_query(message: str, previous_user: str | None = None) -> RetrievalQuery:
    """Search text for `message`, borrowing the previous question's topic if needed.

    A follow-up like "jelaskan ayat tadi" or "bagaimana dengan hadits?" has no
    content words of its own, so its topic comes from the previous question.
    The current message still decides which source (ayat/hadits) is wanted when
    it names one.
    """
    if previous_user and not extract_keywords(message):
        names_source = bool(_QURAN_HINT.search(message) or _HADITH_HINT.search(message))
        return RetrievalQuery(
            text=f"{previous_user} {message}",
            sources_from=message if names_source else previous_user,
            follow_up=True,
        )
    return RetrievalQuery(text=message, sources_from=message, follow_up=False)


def wanted_sources(message: str) -> tuple[bool, bool]:
    """Return (search_quran, search_hadith) based on what the user asked for."""
    quran = bool(_QURAN_HINT.search(message))
    hadith = bool(_HADITH_HINT.search(message))
    if quran and not hadith:
        return True, False
    if hadith and not quran:
        return False, True
    return True, True


def _variants(word: str) -> list[str]:
    """The word plus light Indonesian affix-stripped forms, for substring matching."""
    found = {word}
    stems = {word}
    for prefix in _PREFIXES:
        if word.startswith(prefix):
            stems.add(word[len(prefix):])
    for stem in list(stems):
        for suffix in _SUFFIXES:
            if stem.endswith(suffix):
                found.add(stem[: -len(suffix)])
    found |= stems
    return sorted((v for v in found if len(v) >= 4 or v == word), key=len, reverse=True)


def extract_keywords(message: str) -> list[list[str]]:
    """Content words of the message, each as a list of matching variants."""
    seen: set[str] = set()
    keywords: list[list[str]] = []
    for token in re.findall(r"[a-zA-Z']+", message.lower()):
        token = token.strip("'")
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(_variants(token))
        if len(keywords) == MAX_KEYWORDS:
            break
    return keywords


SHORT_WORD_LEN = 6


def _patterns(variant: str) -> list[str]:
    """LIKE patterns for one variant.

    Short variants only match at the start of a word, so "riba" does not hit
    "beribadah". Longer ones are specific enough for plain substring matching.
    """
    if len(variant) >= SHORT_WORD_LEN:
        return [f"%{variant}%"]
    return [f"{variant}%", f"% {variant}%", f"%({variant}%", f'%"{variant}%', f"%[{variant}%"]


def _score_sql(column: str, keywords: list[list[str]]) -> tuple[str, str, list[str]]:
    """Build (score expression, where expression, params) for LIKE matching."""
    score_parts: list[str] = []
    params: list[str] = []
    for variants in keywords:
        patterns = [pattern for variant in variants for pattern in _patterns(variant)]
        clause = " or ".join(f"{column} like ?" for _ in patterns)
        score_parts.append(f"(case when {clause} then 1 else 0 end)")
        params.extend(patterns)
    score = " + ".join(score_parts)
    return score, f"({score}) > 0", params


def search_quran(db_path: Path, keywords: list[list[str]], language: str = "id") -> list[Passage]:
    if not keywords:
        return []
    fts_passages = _search_quran_fts(db_path, keywords, language)
    if fts_passages is not None:
        return fts_passages
    return _search_quran_like(db_path, keywords, language)


def _search_quran_fts(
    db_path: Path, keywords: list[list[str]], language: str
) -> list[Passage] | None:
    """BM25-ranked search via sidecar FTS. None when unavailable (caller falls back)."""
    try:
        from .fts import build_match, ensure_quran_sidecar, search_sidecar
    except ImportError:
        return None
    try:
        match = build_match(keywords)
        if not match:
            return []
        fts_path = ensure_quran_sidecar(db_path)
        if fts_path is None:
            return None
        ranked = search_sidecar(fts_path, match, language, MAX_PASSAGES_PER_SOURCE)
        if ranked is None:
            return None
        if not ranked:
            return []
        return _fetch_quran_by_ids(db_path, ranked, language)
    except Exception:  # noqa: BLE001 - FTS is best-effort; LIKE fallback covers it.
        return None


def _fetch_quran_by_ids(
    db_path: Path, ranked: list[tuple[int, float]], language: str
) -> list[Passage]:
    column = "a.translation_en" if language == "en" else "a.translation_indo"
    scores = dict(ranked)
    marks = ",".join("?" for _ in ranked)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            f"select a.id, a.surah_id, a.number_in_surah, {column}, s.name_transliteration"
            " from ayahs a join surahs s on s.id = a.surah_id"
            f" where a.id in ({marks})",
            [ref_id for ref_id, _ in ranked],
        ).fetchall()
    rows.sort(key=lambda row: scores.get(row[0], 0.0))
    return [
        Passage(
            kind="quran",
            label=f"{name} {surah}:{ayah}",
            source_id=f"{surah}:{ayah}",
            text=str(text),
            score=float(-scores.get(ref_id, 0.0)),
        )
        for ref_id, surah, ayah, text, name in rows
    ]


def _search_quran_like(db_path: Path, keywords: list[list[str]], language: str = "id") -> list[Passage]:
    if not keywords:
        return []
    column = "a.translation_en" if language == "en" else "a.translation_indo"
    score, where, params = _score_sql(column, keywords)
    sql = (
        f"select a.surah_id, a.number_in_surah, {column}, s.name_transliteration, ({score}) as score"
        " from ayahs a join surahs s on s.id = a.surah_id"
        f" where {where} order by score desc, length({column}) asc limit ?"
    )
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(sql, [*params, *params, MAX_PASSAGES_PER_SOURCE]).fetchall()
    return [
        Passage(
            kind="quran",
            label=f"{name} {surah}:{ayah}",
            source_id=f"{surah}:{ayah}",
            text=str(text),
            score=float(score),
        )
        for surah, ayah, text, name, score in rows
    ]


def search_hadith(db_path: Path, keywords: list[list[str]], language: str = "id") -> list[Passage]:
    if not keywords:
        return []
    fts_passages = _search_hadith_fts(db_path, keywords, language)
    if fts_passages is not None:
        return fts_passages
    return _search_hadith_like(db_path, keywords, language)


def _search_hadith_fts(
    db_path: Path, keywords: list[list[str]], language: str
) -> list[Passage] | None:
    """BM25 via native hadith_fts, else sidecar. None when unavailable."""
    try:
        from .fts import (
            build_match,
            ensure_hadith_sidecar,
            native_hadith_fts,
            search_native_hadith_fts,
            search_sidecar,
        )
    except ImportError:
        return None
    try:
        match = build_match(keywords)
        if not match:
            return []
        ranked: list[tuple[int, float]] | None = None
        if native_hadith_fts(db_path):
            ranked = search_native_hadith_fts(db_path, match, language, MAX_PASSAGES_PER_SOURCE)
            if ranked is not None:
                return _fetch_hadith_by_ids(db_path, _dedupe_ranked(ranked), language)
        fts_path = ensure_hadith_sidecar(db_path)
        if fts_path is None:
            return None
        ranked = search_sidecar(fts_path, match, language, MAX_PASSAGES_PER_SOURCE)
        if ranked is None:
            return None
        if not ranked:
            return []
        return _fetch_hadith_by_ids(db_path, _dedupe_ranked(ranked), language)
    except Exception:  # noqa: BLE001 - FTS is best-effort; LIKE fallback covers it.
        return None


def _dedupe_ranked(ranked: list[tuple[int, float]]) -> list[tuple[int, float]]:
    seen: dict[int, float] = {}
    for ref_id, rank in ranked:
        if ref_id not in seen:
            seen[ref_id] = rank
    return sorted(seen.items(), key=lambda item: item[1])


def _fetch_hadith_by_ids(
    db_path: Path, ranked: list[tuple[int, float]], language: str
) -> list[Passage]:
    if not ranked:
        return []
    preferred = ["en", "id"] if language == "en" else ["id", "en"]
    scores = dict(ranked)
    marks = ",".join("?" for _ in ranked)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "select h.id, h.number, b.name_en, ht.lang, ht.text from hadiths h"
            " join books b on b.id = h.book_id"
            " join hadith_texts ht on ht.hadith_id = h.id and ht.lang in ('id', 'en')"
            f" where h.id in ({marks})",
            [ref_id for ref_id, _ in ranked],
        ).fetchall()
    grouped: dict[int, tuple[int, str, dict[str, str]]] = {}
    for ref_id, number, book, lang, text in rows:
        grouped.setdefault(ref_id, (number, book, {}))[2][lang] = str(text)
    passages: list[Passage] = []
    for ref_id, (number, book, texts) in sorted(grouped.items(), key=lambda item: scores.get(item[0], 0.0)):
        text = next((texts[lang] for lang in preferred if texts.get(lang, "").strip()), "")
        if not text:
            continue
        passages.append(
            Passage(
                kind="hadith",
                label=f"{book} #{number}",
                source_id=str(number),
                text=hadith_matn(text, MAX_HADITH_CHARS),
                score=float(-scores.get(ref_id, 0.0)),
            )
        )
    return passages[:MAX_PASSAGES_PER_SOURCE]


def _search_hadith_like(db_path: Path, keywords: list[list[str]], language: str = "id") -> list[Passage]:
    if not keywords:
        return []
    lang = "en" if language == "en" else "id"
    score, where, params = _score_sql("ht.text", keywords)
    sql = (
        f"select h.number, b.name_en, ht.text, ({score}) as score"
        " from hadith_texts ht"
        " join hadiths h on h.id = ht.hadith_id"
        " join books b on b.id = h.book_id"
        f" where ht.lang = ? and {where}"
        " order by score desc, length(ht.text) asc limit ?"
    )
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(sql, [*params, lang, *params, MAX_PASSAGES_PER_SOURCE]).fetchall()
    return [
        Passage(
            kind="hadith",
            label=f"{book} #{number}",
            source_id=str(number),
            text=hadith_matn(str(text), MAX_HADITH_CHARS),
            score=float(score),
        )
        for number, book, text, score in rows
    ]


def interleave_passages(
    first: list[Passage], second: list[Passage], per_kind_limit: int = MAX_CONTEXT_PER_SOURCE
) -> list[Passage]:
    """Alternate `first` and `second`, without duplicates, capped per source.

    On the 19-case evaluation (eval/), alternating semantic and keyword hits put
    the expected dalil in the context more often than either list alone, or
    than using semantic hits only when keyword hits looked weak.
    """
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    merged: list[Passage] = []
    for pair in itertools.zip_longest(first, second):
        for passage in pair:
            if passage is None:
                continue
            key = (passage.kind, passage.label)
            if key in seen or counts.get(passage.kind, 0) >= per_kind_limit:
                continue
            seen.add(key)
            counts[passage.kind] = counts.get(passage.kind, 0) + 1
            merged.append(passage)
    return merged


def retrieve(
    message: str,
    *,
    quran_db_path: Path,
    hadith_db_path: Path,
    language: str = "id",
    sources_from: str | None = None,
) -> list[Passage]:
    keywords = extract_keywords(message)
    if not keywords:
        return []
    want_quran, want_hadith = wanted_sources(sources_from or message)
    passages: list[Passage] = []
    if want_quran:
        passages += search_quran(quran_db_path, keywords, language)
    if want_hadith:
        passages += search_hadith(hadith_db_path, keywords, language)
    return passages


def format_context(
    passages: list[Passage], *, previous: list[Passage] | None = None, has_history: bool = False
) -> str:
    """Context block appended to the user's message for the answering LLM.

    `previous` are dalil cited in the last answer, re-read from the database so
    follow-ups like "jelaskan ayat tadi" are grounded in the real text.
    """
    lines: list[str] = []
    if previous:
        lines.append("DALIL YANG DIBAHAS PADA JAWABAN SEBELUMNYA (dari database):")
        lines += [f"- [{p.label}] {p.text}" for p in previous]
        lines.append("")
    if passages:
        lines.append("KONTEKS DALIL (hasil pencarian database; hanya boleh mengutip dari sini):")
        lines += [f"- [{p.label}] {p.text}" for p in passages]
    elif has_history:
        lines.append(
            "KONTEKS DALIL: tidak ada dalil baru yang ditemukan untuk pesan ini. "
            "Jika pesan ini lanjutan, jawab dari riwayat percakapan dan dalil di atas."
        )
    else:
        lines.append("KONTEKS DALIL: pencarian database tidak menemukan dalil yang cocok.")
    return "\n".join(lines)
