from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import httpx
from hijridate import Gregorian

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.mabims.dev/api/v1"
DEFAULT_TIMEZONE = "Asia/Jakarta"

# Where a date came from, best first. Only "mabims" is read from the curated Kemenag table.
SOURCE_TABLE = "mabims"
SOURCE_COMPUTED = "mabims-computed"
SOURCE_FALLBACK = "umm-al-qura"

MONTH_NAMES = (
    "Muharram",
    "Safar",
    "Rabiul Awal",
    "Rabiul Akhir",
    "Jumadil Awal",
    "Jumadil Akhir",
    "Rajab",
    "Sya'ban",
    "Ramadhan",
    "Syawal",
    "Dzulqa'dah",
    "Dzulhijjah",
)
WEEKDAY_NAMES = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Ahad")

# Ramadan, Syawal and Dzulhijjah start on the government's sidang isbat decision.
ISBAT_MONTHS = frozenset({9, 10, 12})

ISBAT_NOTE = "Awal bulan ini ditetapkan pemerintah melalui sidang isbat."
APPROXIMATE_NOTE = (
    "Perkiraan hitungan, bisa berbeda satu hari dari penetapan resmi Kemenag."
)
FALLBACK_NOTE = (
    "Layanan kalender Kemenag (MABIMS) sedang tidak tersedia, jadi tanggal ini dihitung "
    "dengan kalender Umm al-Qura dan bisa berbeda satu hari dari penetapan Kemenag."
)

CACHE_TTL_SECONDS = 12 * 3600
# After a failed API call, go straight to the local calculation for a while so a
# dead service does not add its timeout to every reply.
FAILURE_COOLDOWN_SECONDS = 60.0


@dataclass(frozen=True)
class HijriDate:
    year: int
    month: int
    day: int
    month_name: str
    weekday: str
    source: str
    notes: tuple[str, ...] = ()

    @property
    def approximate(self) -> bool:
        return self.source != SOURCE_TABLE

    def format(self) -> str:
        return f"{self.weekday}, {self.day} {self.month_name} {self.year} H"

    def format_with_notes(self) -> str:
        return "\n".join((self.format(), *self.notes))


GREGORIAN_MONTH_NAMES = (
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
)

MAX_QUERY_CHARS = 60
_HIJRI_WORDS = ("hijri", "hijriah", "hijriyah", "hijriyyah", "kalender islam", "tanggal islam")
_TODAY_CUES = ("hari ini", "sekarang", "tanggal berapa", "tgl berapa", "hari ini tanggal", "saat ini")
# A question about *which* date, a concept or history belongs to the LLM, not to this shortcut.
_NOT_A_TODAY_QUERY = (
    "kapan",
    "apa itu",
    "apa bedanya",
    "beda",
    "sejarah",
    "mengapa",
    "kenapa",
    "jelaskan",
    "cara",
    "rumus",
    "besok",
    "kemarin",
    "lusa",
    "depan",
    "lalu",
    "tahun",
    "bulan",
) + tuple(name.lower() for name in MONTH_NAMES) + tuple(name.lower() for name in GREGORIAN_MONTH_NAMES)


def detect_hijri_query(message: str) -> bool:
    """True for short "what is today's Hijri date" messages only.

    Deliberately narrow: a missed phrase just goes to the LLM like before, but a
    false match would answer a different question with today's date.
    """
    text = " ".join(message.lower().replace("?", " ").replace("!", " ").replace(".", " ").split())
    if not text or len(text) > MAX_QUERY_CHARS or any(char.isdigit() for char in text):
        return False
    if not any(word in text for word in _HIJRI_WORDS):
        return False
    if any(f" {word} " in f" {text} " for word in _NOT_A_TODAY_QUERY):
        return False
    return any(cue in text for cue in _TODAY_CUES) or text in {f"tanggal {word}" for word in _HIJRI_WORDS} | set(_HIJRI_WORDS)


def format_gregorian(day: date) -> str:
    return f"{WEEKDAY_NAMES[day.weekday()]}, {day.day} {GREGORIAN_MONTH_NAMES[day.month - 1]} {day.year}"


def format_today_message(hijri: HijriDate, day: date) -> str:
    """Markdown answer for "today's Hijri date" (the chat and the Telegram command share it)."""
    lines = ["## 📅 Tanggal Hijriyah hari ini", "", f"**{hijri.format()}**", "", f"Masehi: {format_gregorian(day)}"]
    if hijri.notes:
        lines.append("")
        lines.extend(f"> {note}" for note in hijri.notes)
    return "\n".join(lines)


UNAVAILABLE_MESSAGE = "Maaf, tanggal Hijriyah belum bisa ditampilkan saat ini. Silakan coba lagi sebentar."


def today_in(tz_name: str = DEFAULT_TIMEZONE) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()


def calculate_hijri(day: date) -> HijriDate | None:
    """Local Umm al-Qura conversion. None when `day` is outside the supported range."""
    try:
        hijri = Gregorian(day.year, day.month, day.day).to_hijri()
    except (OverflowError, ValueError):
        return None
    return _build(
        hijri.year,
        hijri.month,
        hijri.day,
        MONTH_NAMES[hijri.month - 1],
        WEEKDAY_NAMES[day.weekday()],
        SOURCE_FALLBACK,
        fallback=True,
    )


def _build(
    year: int,
    month: int,
    day: int,
    month_name: str,
    weekday: str,
    source: str,
    *,
    fallback: bool = False,
) -> HijriDate:
    notes: list[str] = []
    if fallback:
        notes.append(FALLBACK_NOTE)
    elif source != SOURCE_TABLE:
        notes.append(APPROXIMATE_NOTE)
    if month in ISBAT_MONTHS:
        notes.append(ISBAT_NOTE)
    return HijriDate(year, month, day, month_name, weekday, source, tuple(notes))


def parse_api_response(payload: dict) -> HijriDate:
    """Turn a `/today/{date}` response into a HijriDate. Raises ValueError on any surprise."""
    try:
        output = payload["output"]
        source = str(payload["source"])
        return _build(
            int(output["year"]),
            int(output["month"]),
            int(output["day"]),
            str(output["month_name"]),
            str(output["weekday"]),
            source,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unexpected MABIMS response: {payload!r}") from exc


class HijriCalendar:
    """Hijri dates from the MABIMS API, falling back to a local calculation when it is down."""

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        *,
        enabled: bool = True,
        timeout_seconds: float = 3.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._cache: dict[date, tuple[float, HijriDate]] = {}
        self._down_until = 0.0

    async def convert(self, day: date | None = None) -> HijriDate | None:
        """Hijri date for `day` (default: today in Jakarta). None only if every source fails."""
        day = day or today_in()
        cached = self._cache.get(day)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        if self.enabled and time.monotonic() >= self._down_until:
            try:
                result = await self._fetch(day)
            except (httpx.HTTPError, ValueError) as exc:
                self._down_until = time.monotonic() + FAILURE_COOLDOWN_SECONDS
                logger.warning("MABIMS API unavailable (%r), using local calculation", exc)
            else:
                self._cache[day] = (time.monotonic() + CACHE_TTL_SECONDS, result)
                return result
        # Fallback results are not cached: the next call should retry the API once the cooldown ends.
        return calculate_hijri(day)

    async def _fetch(self, day: date) -> HijriDate:
        url = f"{self.api_base}/today/{day.isoformat()}"
        if self._client is not None:
            response = await self._client.get(url, timeout=self.timeout_seconds)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return parse_api_response(response.json())
