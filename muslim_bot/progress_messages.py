"""Waiting-message text for the Telegram bot, chosen from the user's message.

Rule based on purpose: the text exists to cover waiting time, so it must be
instant and free. A misclassified message just gets the neutral ``general`` text.
Categories without a pool (``greeting``, ``thanks``) get no waiting message.
"""

from __future__ import annotations

import random
import re

PROGRESS_POOLS: dict[str, tuple[str, ...]] = {
    "quran": (
        "📖 Saya carikan ayatnya dulu ya…",
        "📖 Mencari ayat yang sesuai…",
        "📖 Membuka mushaf, sebentar ya…",
        "📖 Menelusuri ayat-ayat terkait…",
    ),
    "hadith": (
        "📜 Saya telusuri hadisnya dulu ya…",
        "📜 Mencari riwayat yang sesuai…",
        "📜 Membuka kitab hadis, sebentar ya…",
        "📜 Menelusuri hadis-hadis terkait…",
    ),
    "dalil": (
        "🔍 Mencari dalil dari Al-Qur'an dan hadis…",
        "🔍 Sebentar ya, saya carikan dalilnya…",
        "🔍 Menelusuri ayat dan hadis terkait…",
    ),
    "lookup": (
        "🔎 Saya buka rujukannya…",
        "🔎 Mengambil teksnya dari database…",
        "🔎 Sebentar, saya cari rujukan yang Anda maksud…",
    ),
    "random": (
        "🎲 Saya ambilkan secara acak…",
        "🎲 Memilihkan satu untuk Anda…",
        "🎲 Sebentar, saya acak dulu…",
    ),
    "concept": (
        "💡 Saya susun penjelasannya…",
        "💡 Menyiapkan penjelasan yang mudah dipahami…",
        "💡 Sebentar ya, saya rangkum dulu…",
    ),
    "ruling": (
        "⚖️ Saya cari dalil yang berkaitan…",
        "⚖️ Menelusuri dalil terkait, sebentar ya…",
        "⚖️ Mengumpulkan rujukan untuk pertanyaan Anda…",
    ),
    "general": (
        "💭 Sebentar ya, saya siapkan jawabannya…",
        "⏳ Sedang menyusun jawaban…",
        "💭 Saya pikirkan dulu ya…",
    ),
}

ALL_PROGRESS_TEXTS = frozenset(text for pool in PROGRESS_POOLS.values() for text in pool)

_GREETING_RE = re.compile(
    r"(?:assalamu'?alaikum|assalamualaikum|salam|halo|hai|hi|hello|"
    r"selamat\s+(?:pagi|siang|sore|malam)|permisi|wr\.?|wb\.?|warahmatullahi|wabarakatuh|"
    r"ustaz|ustad|bot|kak|min|admin|\s|[,.!'])+",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"(?:terima\s*kasih|makasih|thanks|thank\s+you|jazakallah(?:u)?\s*khairan?|jazakumullah|"
    r"syukron|barakallah(?:u)?\s*fiik?|oke?|siap|baik|mantap|sip|banyak|ya|atas|jawabannya|"
    r"bantuannya|nya|\s|[,.!'])+",
    re.IGNORECASE,
)
_RANDOM_RE = re.compile(r"\b(?:acak|random|sembarang)\b", re.IGNORECASE)
_LOOKUP_RE = re.compile(r"\b\d{1,3}\s*[:/]\s*\d{1,3}\b|#\s*\d+|\bayat\s+kursi\b|\bnomor\s+\d+\b", re.IGNORECASE)
_HADITH_RE = re.compile(
    r"\b(?:hadis|hadits|hadist|riwayat|bukhari|muslim|abu\s+dawud|tirmidzi|tirmidhi|nasai|ibnu\s+majah)\b",
    re.IGNORECASE,
)
_QURAN_RE = re.compile(r"\b(?:ayat|surah|surat|qs|quran|qur'an|al-?quran|juz)\b", re.IGNORECASE)
_RULING_RE = re.compile(
    r"\b(?:hukum|hukumnya|boleh|bolehkah|haram|halal|makruh|wajib|sunnah|sunah|mubah)\b", re.IGNORECASE
)
_CONCEPT_RE = re.compile(
    r"\b(?:apa\s+itu|apa\s+yang\s+dimaksud|apakah|pengertian|arti|makna|jelaskan|penjelasan|"
    r"definisi|beda|perbedaan|mengapa|kenapa|bagaimana)\b",
    re.IGNORECASE,
)


def classify_progress(text: str) -> str:
    """Pick the waiting-message category for a user message."""
    text = text.strip()
    if _THANKS_RE.fullmatch(text):
        return "thanks"
    if _GREETING_RE.fullmatch(text):
        return "greeting"
    if _RANDOM_RE.search(text):
        return "random"
    if _LOOKUP_RE.search(text):
        return "lookup"
    has_hadith = bool(_HADITH_RE.search(text))
    has_quran = bool(_QURAN_RE.search(text))
    if has_hadith and has_quran:
        return "dalil"
    if has_hadith:
        return "hadith"
    if has_quran:
        return "quran"
    if _RULING_RE.search(text):
        return "ruling"
    if _CONCEPT_RE.search(text):
        return "concept"
    return "general"


def pick_progress(
    category: str, last_index: int | None = None, rng: random.Random | None = None
) -> tuple[str | None, int | None]:
    """Return (text, index) for the category, never repeating `last_index` when avoidable.

    Categories without a pool (greeting, thanks) return (None, None).
    """
    pool = PROGRESS_POOLS.get(category)
    if pool is None:
        return None, None
    choices = [index for index in range(len(pool)) if index != last_index] or [0]
    index = (rng or random).choice(choices)
    return pool[index], index
