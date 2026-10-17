from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .hijri import format_gregorian, names_new_dalil_topic
from .locations import nearest_city

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.aladhan.com/v1"
# Aladhan method 20 = Kementerian Agama RI (Fajr 20 deg, Isha 18 deg).
DEFAULT_METHOD = 20
FAJR_ANGLE = 20.0
ISHA_ANGLE = 18.0
IMSAK_MINUTES = 10
MAX_OFFLINE_LATITUDE = 60.0

SOURCE_API = "aladhan"
SOURCE_LOCAL = "hitungan lokal"

# Keys follow Aladhan's names; the order is the order of the day.
PRAYER_KEYS = ("Imsak", "Fajr", "Sunrise", "Dhuhr", "Asr", "Maghrib", "Isha")
PRAYER_NAMES = {
    "Imsak": "Imsak",
    "Fajr": "Subuh",
    "Sunrise": "Terbit",
    "Dhuhr": "Dzuhur",
    "Asr": "Ashar",
    "Maghrib": "Maghrib",
    "Isha": "Isya",
}
OBLIGATORY = ("Fajr", "Dhuhr", "Asr", "Maghrib", "Isha")

TIMEZONE_ABBREVIATIONS = {
    "Asia/Jakarta": "WIB",
    "Asia/Pontianak": "WIB",
    "Asia/Makassar": "WITA",
    "Asia/Jayapura": "WIT",
}

METHOD_NOTE = "Metode Kemenag RI (Subuh 20°, Isya 18°). Bisa berbeda 1-2 menit dari jadwal resmi setempat."
LOCAL_NOTE = "Layanan jadwal sedang tidak tersedia, jadi waktu ini dihitung langsung dengan metode yang sama."
NO_LOCATION_MESSAGE = (
    "Lokasi Anda belum diatur, jadi saya belum bisa menampilkan jadwal sholat. "
    "Di Telegram, ketik /lokasi lalu kirim lokasi atau nama kota Anda."
)
UNAVAILABLE_MESSAGE = "Maaf, jadwal sholat belum bisa ditampilkan saat ini. Silakan coba lagi sebentar."

CACHE_TTL_SECONDS = 36 * 3600
CACHE_MAX_ENTRIES = 2048
FAILURE_COOLDOWN_SECONDS = 60.0


@dataclass(frozen=True)
class SavedLocation:
    latitude: float
    longitude: float
    timezone: str
    label: str


@dataclass(frozen=True)
class Timings:
    day: date
    times: dict[str, str]  # Aladhan key -> "HH:MM" local time
    timezone: str
    source: str

    @property
    def approximate(self) -> bool:
        return self.source != SOURCE_API


def location_from_prefs(prefs) -> SavedLocation | None:
    """The saved location of a UserPrefs, or None when it was never set."""
    if prefs.latitude is None or prefs.longitude is None or not prefs.timezone:
        return None
    return SavedLocation(prefs.latitude, prefs.longitude, prefs.timezone, prefs.location_label or "lokasi Anda")


def tz_abbreviation(tz_name: str) -> str:
    return TIMEZONE_ABBREVIATIONS.get(tz_name, tz_name)


# --- calculation without the API ------------------------------------------------------


def _sin(degrees: float) -> float:
    return math.sin(math.radians(degrees))


def _cos(degrees: float) -> float:
    return math.cos(math.radians(degrees))


def _julian(day: date) -> float:
    year, month = day.year, day.month
    if month <= 2:
        year -= 1
        month += 12
    century = year // 100
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day.day + (2 - century + century // 4) - 1524.5


def _sun_position(jd: float) -> tuple[float, float]:
    """(declination in degrees, equation of time in hours)."""
    d = jd - 2451545.0
    g = (357.529 + 0.98560028 * d) % 360
    q = (280.459 + 0.98564736 * d) % 360
    lam = (q + 1.915 * _sin(g) + 0.020 * _sin(2 * g)) % 360
    e = 23.439 - 0.00000036 * d
    ra = (math.degrees(math.atan2(_cos(e) * _sin(lam), _cos(lam))) / 15) % 24
    eqt = q / 15 - ra
    if eqt > 12:
        eqt -= 24
    elif eqt < -12:
        eqt += 24
    return math.degrees(math.asin(_sin(e) * _sin(lam))), eqt


def _utc_offset_hours(day: date, tz_name: str) -> float:
    offset = ZoneInfo(tz_name).utcoffset(datetime(day.year, day.month, day.day, 12))
    return offset.total_seconds() / 3600 if offset is not None else 0.0


def _hhmm(hours: float) -> str:
    minutes = round(hours * 60) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def compute_timings(day: date, latitude: float, longitude: float, tz_name: str) -> dict[str, str] | None:
    """Kemenag-method times (Subuh 20°, Isya 18°, Ashar shadow x1) from the sun's position.

    The sun is evaluated near each prayer's usual hour, as PrayTimes does; against Aladhan this
    agrees to within 2 minutes across Indonesia. None for latitudes where the angle method breaks down.
    """
    if abs(latitude) > MAX_OFFLINE_LATITUDE:
        return None
    jd = _julian(day) - longitude / 360.0
    shift = _utc_offset_hours(day, tz_name) - longitude / 15

    def noon(at_hour: float) -> float:
        return 12 - _sun_position(jd + at_hour / 24)[1]

    def angle_time(angle: float, at_hour: float, before_noon: bool) -> float:
        declination = _sun_position(jd + at_hour / 24)[0]
        x = (-_sin(angle) - _sin(declination) * _sin(latitude)) / (_cos(declination) * _cos(latitude))
        offset = math.degrees(math.acos(max(-1.0, min(1.0, x)))) / 15
        return noon(at_hour) + (-offset if before_noon else offset)

    def asr_time() -> float:
        # The shadow angle depends on the sun's declination at Ashar itself, so refine from a first guess.
        at_hour = 15.0
        for _ in range(3):
            declination = _sun_position(jd + at_hour / 24)[0]
            angle = -math.degrees(math.atan(1 / (1 + math.tan(math.radians(abs(latitude - declination))))))
            at_hour = angle_time(angle, at_hour, False)
        return at_hour

    rise_set = 0.833
    fajr = angle_time(FAJR_ANGLE, 5, True)
    hours = {
        "Imsak": fajr - IMSAK_MINUTES / 60,
        "Fajr": fajr,
        "Sunrise": angle_time(rise_set, 6, True),
        "Dhuhr": noon(12),
        "Asr": asr_time(),
        "Maghrib": angle_time(rise_set, 18, False),
        "Isha": angle_time(ISHA_ANGLE, 18, False),
    }
    return {key: _hhmm(value + shift) for key, value in hours.items()}


# --- API response --------------------------------------------------------------------

_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})")


def parse_timings_response(payload: dict, day: date, fallback_timezone: str) -> Timings:
    """Turn an Aladhan `/timings` response into Timings. Raises ValueError on any surprise."""
    try:
        data = payload["data"]
        raw = data["timings"]
        times: dict[str, str] = {}
        for key in PRAYER_KEYS:
            match = _TIME_RE.match(str(raw[key]))
            if match is None or int(match.group(1)) > 23 or int(match.group(2)) > 59:
                raise ValueError(f"bad time for {key}: {raw[key]!r}")
            times[key] = f"{int(match.group(1)):02d}:{match.group(2)}"
        timezone = str(data.get("meta", {}).get("timezone") or fallback_timezone)
        ZoneInfo(timezone)
    except (KeyError, TypeError, AttributeError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"unexpected Aladhan response: {payload!r}") from exc
    return Timings(day, times, timezone, SOURCE_API)


class PrayerTimesClient:
    """Prayer times from Aladhan, cached per place and day, computed locally when it is down."""

    def __init__(
        self,
        api_base: str = DEFAULT_API_BASE,
        *,
        method: int = DEFAULT_METHOD,
        enabled: bool = True,
        timeout_seconds: float = 4.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.method = method
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._cache: dict[tuple, tuple[float, Timings]] = {}
        self._down_until = 0.0

    def _api_usable(self) -> bool:
        return self.enabled and time.monotonic() >= self._down_until

    def _mark_down(self, exc: Exception) -> None:
        self._down_until = time.monotonic() + FAILURE_COOLDOWN_SECONDS
        logger.warning("Aladhan unavailable (%r), using local calculation", exc)

    async def timings(self, latitude: float, longitude: float, tz_name: str, day: date) -> Timings | None:
        """Times for `day` at a place. None only if neither the API nor the local calculation can answer."""
        key = (round(latitude, 2), round(longitude, 2), tz_name, day)
        cached = self._cache.get(key)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        if self._api_usable():
            try:
                payload = await self._get_json(
                    f"{self.api_base}/timings/{day.strftime('%d-%m-%Y')}",
                    {"latitude": latitude, "longitude": longitude, "method": self.method, "timezonestring": tz_name},
                )
                result = parse_timings_response(payload, day, tz_name)
            except (httpx.HTTPError, ValueError) as exc:
                self._mark_down(exc)
            else:
                self._remember(key, result)
                return result
        times = compute_timings(day, latitude, longitude, tz_name)
        return Timings(day, times, tz_name, SOURCE_LOCAL) if times is not None else None

    async def resolve_timezone(self, latitude: float, longitude: float) -> str | None:
        """IANA zone for coordinates (Aladhan knows the whole world; offline only near Indonesian cities)."""
        if self._api_usable():
            try:
                payload = await self._get_json(
                    f"{self.api_base}/timings/{date.today().strftime('%d-%m-%Y')}",
                    {"latitude": latitude, "longitude": longitude, "method": self.method},
                )
                tz_name = str(payload["data"]["meta"]["timezone"])
                ZoneInfo(tz_name)
                return tz_name
            except (httpx.HTTPError, ValueError, KeyError, TypeError, ZoneInfoNotFoundError) as exc:
                self._mark_down(exc)
        city = nearest_city(latitude, longitude)
        return city.timezone if city is not None else None

    def _remember(self, key: tuple, value: Timings) -> None:
        if len(self._cache) >= CACHE_MAX_ENTRIES:
            now = time.monotonic()
            for stale in [k for k, (expires, _) in self._cache.items() if expires <= now]:
                del self._cache[stale]
            if len(self._cache) >= CACHE_MAX_ENTRIES:
                self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (time.monotonic() + CACHE_TTL_SECONDS, value)

    async def _get_json(self, url: str, params: dict) -> dict:
        if self._client is not None:
            response = await self._client.get(url, params=params, timeout=self.timeout_seconds)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()


def client_from_settings(settings) -> PrayerTimesClient:
    return PrayerTimesClient(
        settings.prayer_api_base,
        method=settings.prayer_method,
        enabled=settings.prayer_api_enabled,
        timeout_seconds=settings.prayer_api_timeout_seconds,
    )


# --- showing times ---------------------------------------------------------------------


def now_in(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def at_time(timings: Timings, key: str) -> datetime:
    hour, minute = (int(part) for part in timings.times[key].split(":"))
    return datetime(timings.day.year, timings.day.month, timings.day.day, hour, minute, tzinfo=ZoneInfo(timings.timezone))


def next_prayer(today: Timings, tomorrow: Timings | None, now: datetime) -> tuple[str, datetime] | None:
    """The next of the five obligatory prayers after `now` (key, time)."""
    for key in OBLIGATORY:
        moment = at_time(today, key)
        if moment > now:
            return key, moment
    if tomorrow is not None:
        return OBLIGATORY[0], at_time(tomorrow, OBLIGATORY[0])
    return None


def describe_wait(delta: timedelta) -> str:
    minutes = max(1, math.ceil(delta.total_seconds() / 60))
    hours, rest = divmod(minutes, 60)
    if hours and rest:
        return f"{hours} jam {rest} menit lagi"
    if hours:
        return f"{hours} jam lagi"
    return f"{rest} menit lagi"


def _times_line(timings: Timings) -> str:
    return "; ".join(f"{PRAYER_NAMES[key]} {timings.times[key]}" for key in PRAYER_KEYS)


def format_schedule(timings: Timings, label: str, now: datetime | None = None) -> str:
    """Markdown schedule for /jadwal and for "jadwal sholat" questions."""
    lines = [f"## 🕌 Jadwal sholat — {label}", "", format_gregorian(timings.day), ""]
    lines.extend(f"- {PRAYER_NAMES[key]}: {timings.times[key]}" for key in PRAYER_KEYS)
    if now is not None and timings.day == now.date():
        upcoming = next_prayer(timings, None, now)
        if upcoming is not None:
            key, moment = upcoming
            lines += ["", f"Berikutnya: **{PRAYER_NAMES[key]}** {timings.times[key]} ({describe_wait(moment - now)})"]
    lines += ["", f"> Waktu setempat ({tz_abbreviation(timings.timezone)}). {METHOD_NOTE}"]
    if timings.approximate:
        lines.append(f"> {LOCAL_NOTE}")
    return "\n".join(lines)


async def schedule_reply(
    client: PrayerTimesClient, location: SavedLocation, *, tomorrow: bool = False
) -> str:
    """The schedule message for a saved location, or UNAVAILABLE_MESSAGE."""
    now = now_in(location.timezone)
    day = now.date() + timedelta(days=1) if tomorrow else now.date()
    timings = await client.timings(location.latitude, location.longitude, location.timezone, day)
    if timings is None:
        return UNAVAILABLE_MESSAGE
    return format_schedule(timings, location.label, now)


# --- questions about prayer times --------------------------------------------------------

_PRAYER_WORDS = (
    "subuh",
    "shubuh",
    "dzuhur",
    "zuhur",
    "dhuhur",
    "dhuhr",
    "duhur",
    "ashar",
    "asar",
    "maghrib",
    "magrib",
    "isya",
    "isyak",
    "isha",
    "imsak",
    "imsakiyah",
    "adzan",
    "azan",
    "sholat",
    "solat",
    "salat",
    "shalat",
    "buka puasa",
    "berbuka",
)
_WHEN_CUES = (
    "jam",
    "pukul",
    "kapan",
    "waktu",
    "jadwal",
    "berapa",
    "sekarang",
    "besok",
    "hari ini",
    "sebentar lagi",
    "berikutnya",
    "selanjutnya",
    "masuk",
)
FOLLOW_UP_MAX_CHARS = 80
SCHEDULE_MAX_CHARS = 60
_SCHEDULE_TRIGGERS = ("jadwal", "waktu")
_SCHEDULE_PRAYER_WORDS = ("sholat", "solat", "salat", "shalat", "sembahyang")
# Words that may surround a plain "jadwal sholat" request. Anything else (a city name, "subuh") means
# the question is about something else and must not be answered with the saved location's schedule.
_SCHEDULE_FILLER = frozenset(
    "jadwal waktu sholat solat salat shalat sembahyang hari ini sekarang besok tolong mohon minta lihat tampilkan "
    "kasih beri dong ya yuk dan di sini untuk saya aku ku lokasi tempat wilayah daerah".split()
)


def _clean(message: str) -> str:
    return " ".join(re.sub(r"[?!.,]", " ", message.lower()).split())


def is_prayer_related(message: str) -> bool:
    text = _clean(message)
    return any(word in text for word in _PRAYER_WORDS) and any(cue in text for cue in _WHEN_CUES)


def wants_prayer(message: str, previous_user: str | Sequence[str] | None = None) -> bool:
    """True when the answer needs prayer times: the message asks, or it follows up such a question."""
    if is_prayer_related(message):
        return True
    if not previous_user or len(message) > FOLLOW_UP_MAX_CHARS or names_new_dalil_topic(message):
        return False
    earlier = [previous_user] if isinstance(previous_user, str) else list(previous_user)
    return any(is_prayer_related(text) for text in earlier)


def detect_schedule_request(message: str) -> str | None:
    """"today" or "tomorrow" for a plain "jadwal sholat (hari ini / besok)" message, else None."""
    text = _clean(message)
    if not text or len(text) > SCHEDULE_MAX_CHARS:
        return None
    words = text.split()
    if not any(word in _SCHEDULE_TRIGGERS for word in words) or not any(word in _SCHEDULE_PRAYER_WORDS for word in words):
        return None
    if any(word not in _SCHEDULE_FILLER for word in words):
        return None
    return "tomorrow" if "besok" in words else "today"


async def build_prayer_context(
    client: PrayerTimesClient,
    location: SavedLocation,
    message: str,
    previous_user: str | Sequence[str] | None = None,
) -> str | None:
    """Prayer-time block for the LLM prompt, or None when the message does not ask for times."""
    if not wants_prayer(message, previous_user):
        return None
    try:
        now = now_in(location.timezone)
        today_date = now.date()
        today, tomorrow = await asyncio.gather(
            client.timings(location.latitude, location.longitude, location.timezone, today_date),
            client.timings(location.latitude, location.longitude, location.timezone, today_date + timedelta(days=1)),
        )
    except Exception:  # noqa: BLE001 - a prayer-time problem must never break the answer.
        logger.exception("could not build prayer context")
        return None
    if today is None:
        return None
    zone = tz_abbreviation(location.timezone)
    lines = [
        "JADWAL SHOLAT (dari layanan jadwal; satu-satunya sumber waktu sholat, jangan menghitung atau menebak waktu sendiri):",
        f"Lokasi tersimpan pengguna: {location.label} (waktu setempat {zone}).",
        f"Sekarang: {format_gregorian(today_date)} pukul {now.strftime('%H:%M')} {zone}.",
    ]
    upcoming = next_prayer(today, tomorrow, now)
    if upcoming is not None:
        key, moment = upcoming
        when = "besok" if moment.date() != today_date else "hari ini"
        lines.append(f"Sholat wajib berikutnya: {PRAYER_NAMES[key]} pukul {moment.strftime('%H:%M')} {when} ({describe_wait(moment - now)}).")
    lines.append(f"Hari ini ({format_gregorian(today.day)}): {_times_line(today)}.")
    if tomorrow is not None:
        lines.append(f"Besok ({format_gregorian(tomorrow.day)}): {_times_line(tomorrow)}.")
    lines.append(f"Metode: {METHOD_NOTE}")
    if today.approximate:
        lines.append(f"Catatan: {LOCAL_NOTE}")
    return "\n".join(lines)
