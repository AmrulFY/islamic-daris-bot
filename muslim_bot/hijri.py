from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
from hijridate import Gregorian, Hijri

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
# When the API cannot be reached, data it gave earlier is still better than a local estimate (the Kemenag
# table rarely changes), for this long after it went stale.
STALE_MAX_SECONDS = 30 * 24 * 3600
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


# --- Islamic holidays ---------------------------------------------------------------

# key -> (display name, (hijri month, hijri day)). The API's own event keys; the day is
# what the local fallback calculates when the API is down.
EVENT_DEFS: dict[str, tuple[str, tuple[int, int]]] = {
    "awal_ramadan": ("Awal Ramadan", (9, 1)),
    "idul_fitri": ("Idul Fitri", (10, 1)),
    "idul_adha": ("Idul Adha", (12, 10)),
    "1_muharram": ("Tahun Baru Islam", (1, 1)),
    "maulid_nabi": ("Maulid Nabi Muhammad SAW", (3, 12)),
    "isra_miraj": ("Isra Miraj", (7, 27)),
    "nisfu_syaban": ("Nisfu Syaban", (8, 15)),
}
# The API's /events lacks these, so their dates are looked up one by one with /convert.
EXTRA_EVENT_KEYS = ("isra_miraj", "nisfu_syaban")

# Events whose date the government confirms at sidang isbat.
ISBAT_EVENTS = frozenset({"awal_ramadan", "idul_fitri", "idul_adha", "puasa_ramadan"})

SOURCE_LABELS = {
    SOURCE_TABLE: "tabel kalender Kemenag",
    SOURCE_COMPUTED: "perkiraan hitungan MABIMS",
    SOURCE_FALLBACK: "perkiraan hitungan Umm al-Qura",
}


@dataclass(frozen=True)
class HijriEvent:
    key: str
    name: str
    day: date
    source: str
    hijri: str | None = None  # "1448-09-01" when known
    end: date | None = None  # last day, for events that span several days

    @property
    def last_day(self) -> date:
        return self.end or self.day


def parse_events_response(payload: dict) -> list[HijriEvent]:
    """Turn an `/events` response into events. Raises ValueError on any surprise."""
    try:
        events = [
            HijriEvent(
                key=str(item["event"]),
                name=EVENT_DEFS.get(str(item["event"]), (str(item["name"]), (0, 0)))[0],
                day=date.fromisoformat(str(item["gregorian"])),
                source=str(item["source"]),
                hijri=str(item["hijri"]),
            )
            for item in payload["events"]
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unexpected MABIMS events response: {payload!r}") from exc
    return events


def calculate_events(year: int) -> list[HijriEvent]:
    """Local Umm al-Qura holidays that fall in Gregorian `year` (fallback when the API is down)."""
    try:
        first_hijri_year = Gregorian(year, 1, 1).to_hijri().year
    except (OverflowError, ValueError):
        return []
    events: list[HijriEvent] = []
    for hijri_year in range(first_hijri_year - 1, first_hijri_year + 3):
        for key, (name, (month, day)) in EVENT_DEFS.items():
            try:
                gregorian = Hijri(hijri_year, month, day).to_gregorian()
            except (OverflowError, ValueError):
                continue
            if gregorian.year == year:
                events.append(
                    HijriEvent(
                        key,
                        name,
                        date(gregorian.year, gregorian.month, gregorian.day),
                        SOURCE_FALLBACK,
                        f"{hijri_year:04d}-{month:02d}-{day:02d}",
                    )
                )
    return events


def with_derived_events(events: list[HijriEvent]) -> list[HijriEvent]:
    """Add the fasts that follow from a holiday date, so the LLM never has to count days.

    Ramadan runs until the day before Idul Fitri, Arafah is the day before Idul Adha, and
    Tasu'a / Asyura are the 9th and 10th of Muharram.
    """
    by_key = {event.key: event for event in events}
    derived: list[HijriEvent] = []
    start, fitri = by_key.get("awal_ramadan"), by_key.get("idul_fitri")
    if start and fitri and 0 < (fitri.day - start.day).days <= 31:
        derived.append(
            HijriEvent("puasa_ramadan", "Puasa Ramadan", start.day, start.source, start.hijri, fitri.day - timedelta(days=1))
        )
    if adha := by_key.get("idul_adha"):
        derived.append(HijriEvent("puasa_arafah", "Puasa Arafah (9 Dzulhijjah)", adha.day - timedelta(days=1), adha.source))
    if muharram := by_key.get("1_muharram"):
        derived.append(HijriEvent("puasa_tasua", "Puasa Tasu'a (9 Muharram)", muharram.day + timedelta(days=8), muharram.source))
        derived.append(HijriEvent("puasa_asyura", "Puasa Asyura (10 Muharram)", muharram.day + timedelta(days=9), muharram.source))
    return sorted([*events, *derived], key=lambda event: (event.day, event.key))


def _hijri_label(value: str) -> str:
    year, month, day = (int(part) for part in value.split("-"))
    return f"{day} {MONTH_NAMES[month - 1]} {year} H"


def _event_line(event: HijriEvent, today: date) -> str:
    when = format_gregorian(event.day)
    if event.end:
        when += f" sampai {format_gregorian(event.end)}"
    if event.hijri and not event.end:
        when += f" ({_hijri_label(event.hijri)})"
    if event.last_day < today:
        status = "sudah lewat"
    elif event.day <= today:
        status = "sedang berlangsung" if event.end else "hari ini"
    else:
        status = f"akan datang, {(event.day - today).days} hari lagi"
    notes = [status, SOURCE_LABELS.get(event.source, event.source)]
    if event.key in ISBAT_EVENTS and event.last_day >= today:
        notes.append("penetapan resmi oleh pemerintah melalui sidang isbat")
    return f"- {event.name}: {when} [{'; '.join(notes)}]"


def format_calendar_context(today: date, today_hijri: HijriDate | None, events: list[HijriEvent]) -> str:
    """The "KONTEKS KALENDER" block appended to the LLM prompt."""
    lines = ["KONTEKS KALENDER (dari layanan kalender; satu-satunya sumber tanggal, jangan menghitung atau menebak tanggal sendiri):"]
    if today_hijri is not None:
        label = SOURCE_LABELS.get(today_hijri.source, today_hijri.source)
        lines.append(
            f"Hari ini: {format_gregorian(today)} = {today_hijri.day} {today_hijri.month_name} {today_hijri.year} H [{label}]"
        )
    else:
        lines.append(f"Hari ini: {format_gregorian(today)} (tanggal Hijriyah tidak tersedia)")
    for year in sorted({event.day.year for event in events}):
        lines.append(f"Hari besar dan puasa tahun {year}:")
        lines.extend(_event_line(event, today) for event in events if event.day.year == year)
    if not events:
        lines.append("Data hari besar Islam tidak tersedia saat ini.")
    return "\n".join(lines)


# Words that always mean "calendar question". Event words only count together with a
# "when" cue, so "hadits tentang keutamaan puasa" stays a normal dalil search.
_CALENDAR_WORDS = _HIJRI_WORDS + ("tahun baru islam", "hari besar islam", "kalender hijr")
_EVENT_WORDS = (
    "ramadan",
    "ramadhan",
    "puasa",
    "lebaran",
    "idul",
    "idhul",
    "iedul",
    "maulid",
    "muharram",
    "asyura",
    "arafah",
    "tasua",
    "tasu'a",
    "miraj",
    "mi'raj",
    "mikraj",
    "israk",
    "nisfu",
    "syaban",
    "sya'ban",
    "rajab",
    "syawal",
    "dzulhijjah",
    "zulhijah",
)
_WHEN_CUES = (
    "kapan",
    "tanggal",
    "tahun ini",
    "tahun depan",
    "tahun lalu",
    "bulan depan",
    "hari ini",
    "besok",
    "jadwal",
    "mulai",
    "lagi",
    "berapa hari",
)
FOLLOW_UP_MAX_CHARS = 80
# A short message that asks for a specific dalil is a new topic, not a follow-up of the calendar question.
_NEW_DALIL_TOPIC_WORDS = ("ayat", "hadits", "hadis", "hadith", "dalil", "surah", "surat", "doa")


def names_new_dalil_topic(message: str) -> bool:
    words = set(message.lower().replace("?", " ").replace("!", " ").split())
    return any(word in words for word in _NEW_DALIL_TOPIC_WORDS)


def is_calendar_related(message: str) -> bool:
    text = " ".join(message.lower().replace("?", " ").replace("!", " ").split())
    if any(word in text for word in _CALENDAR_WORDS):
        return True
    return any(word in text for word in _EVENT_WORDS) and any(cue in text for cue in _WHEN_CUES)


def wants_calendar(message: str, previous_user: str | Sequence[str] | None = None) -> bool:
    """True when the answer needs calendar data: the message asks, or it follows up a calendar question.

    `previous_user` are the user's earlier messages (latest last), so a chain of short
    follow-ups ("kalau tahun depan?" -> "dan Idul Fitri?") keeps the calendar.
    """
    if is_calendar_related(message):
        return True
    if not previous_user or len(message) > FOLLOW_UP_MAX_CHARS or names_new_dalil_topic(message):
        return False
    earlier = [previous_user] if isinstance(previous_user, str) else list(previous_user)
    return any(is_calendar_related(text) for text in earlier)


async def build_calendar_context(
    calendar: "HijriCalendar", message: str, previous_user: str | Sequence[str] | None = None
) -> str | None:
    """Calendar block for `message`, or None when it is not a calendar question (or nothing worked)."""
    if not wants_calendar(message, previous_user):
        return None
    today = today_in()
    try:
        today_hijri, this_year, next_year = await asyncio.gather(
            calendar.convert(today), calendar.events(today.year), calendar.events(today.year + 1)
        )
    except Exception:  # noqa: BLE001 - a calendar problem must never break the answer.
        logger.exception("could not build calendar context")
        return None
    events = [*this_year, *next_year]
    if today_hijri is None and not events:
        return None
    return format_calendar_context(today, today_hijri, events)


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
        self._events_cache: dict[int, tuple[float, list[HijriEvent]]] = {}
        self._down_until = 0.0

    async def events(self, year: int) -> list[HijriEvent]:
        """Islamic holidays and the fasts that follow from them in Gregorian `year`, in date order."""
        cached = self._events_cache.get(year)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        if self.enabled and time.monotonic() >= self._down_until:
            try:
                payload = await self._get_json(f"{self.api_base}/events", {"year": year, "calendar": "gregorian"})
                base = parse_events_response(payload)
                if not base:
                    raise ValueError("MABIMS returned no events")
                result = with_derived_events([*base, *await self._extra_events(year)])
            except (httpx.HTTPError, ValueError) as exc:
                self._down_until = time.monotonic() + FAILURE_COOLDOWN_SECONDS
                logger.warning("MABIMS events unavailable (%r), using local calculation", exc)
            else:
                self._events_cache[year] = (time.monotonic() + CACHE_TTL_SECONDS, result)
                return result
        if cached is not None and time.monotonic() - cached[0] < STALE_MAX_SECONDS:
            return cached[1]
        return with_derived_events(calculate_events(year))

    async def _extra_events(self, year: int) -> list[HijriEvent]:
        """EXTRA_EVENT_KEYS that fall in Gregorian `year`: exact dates from /convert, local estimates if it fails."""
        try:
            first_hijri_year = Gregorian(year, 1, 1).to_hijri().year
        except (OverflowError, ValueError):
            return []
        lower, upper = date(year, 1, 1) - timedelta(days=3), date(year, 12, 31) + timedelta(days=3)
        wanted: list[tuple[str, int, date]] = []
        for key in EXTRA_EVENT_KEYS:
            _, (month, day) = EVENT_DEFS[key]
            for hijri_year in range(first_hijri_year - 1, first_hijri_year + 3):
                try:
                    estimate = Hijri(hijri_year, month, day).to_gregorian()
                except (OverflowError, ValueError):
                    continue
                estimate_day = date(estimate.year, estimate.month, estimate.day)
                if lower <= estimate_day <= upper:  # close enough that the exact date may fall in `year`
                    wanted.append((key, hijri_year, estimate_day))
        found = await asyncio.gather(*(self._extra_event(*item) for item in wanted))
        return [event for event in found if event.day.year == year]

    async def _extra_event(self, key: str, hijri_year: int, estimate: date) -> HijriEvent:
        name, (month, day) = EVENT_DEFS[key]
        label = f"{hijri_year:04d}-{month:02d}-{day:02d}"
        if self.enabled and time.monotonic() >= self._down_until:
            try:
                payload = await self._get_json(f"{self.api_base}/convert", {"date": label, "calendar": "hijri"})
                out = payload["output"]
                exact = date(int(out["year"]), int(out["month"]), int(out["day"]))
                return HijriEvent(key, name, exact, str(payload["source"]), label)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                logger.warning("MABIMS /convert failed for %s (%r), using the local estimate", label, exc)
        return HijriEvent(key, name, estimate, SOURCE_FALLBACK, label)

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
        if cached is not None and time.monotonic() - cached[0] < STALE_MAX_SECONDS:
            return cached[1]
        # Fallback results are not cached: the next call should retry the API once the cooldown ends.
        return calculate_hijri(day)

    async def _fetch(self, day: date) -> HijriDate:
        return parse_api_response(await self._get_json(f"{self.api_base}/today/{day.isoformat()}"))

    async def _get_json(self, url: str, params: dict | None = None) -> dict:
        # api.mabims.dev sometimes stalls while connecting. Nothing was sent then, so trying once more is safe
        # and usually succeeds; a slow or failing answer (read timeout, HTTP error) is not retried.
        for attempt in (1, 2):
            try:
                if self._client is not None:
                    response = await self._client.get(url, params=params, timeout=self.timeout_seconds)
                else:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(url, params=params, timeout=self.timeout_seconds)
                break
            except (httpx.ConnectTimeout, httpx.ConnectError):
                if attempt == 2:
                    raise
        response.raise_for_status()
        return response.json()


def calendar_from_settings(settings) -> HijriCalendar:
    return HijriCalendar(
        settings.hijri_api_base,
        enabled=settings.hijri_api_enabled,
        timeout_seconds=settings.hijri_api_timeout_seconds,
    )
