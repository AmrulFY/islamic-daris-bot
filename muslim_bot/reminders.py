from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from .prayer_times import PRAYER_KEYS, PRAYER_NAMES, Timings, at_time, tz_abbreviation

# A reminder is sent in the minute its time arrives. If the bot was busy or restarting it is still sent
# for this long afterwards, but never later: "Subuh sudah masuk" an hour on is worse than no message.
GRACE = timedelta(minutes=5)

# What people type for each prayer time -> the key used everywhere else.
ALIASES = {
    "imsak": "Imsak",
    "imsakiyah": "Imsak",
    "subuh": "Fajr",
    "shubuh": "Fajr",
    "fajar": "Fajr",
    "fajr": "Fajr",
    "terbit": "Sunrise",
    "syuruq": "Sunrise",
    "matahari": "Sunrise",
    "dzuhur": "Dhuhr",
    "zuhur": "Dhuhr",
    "dhuhur": "Dhuhr",
    "duhur": "Dhuhr",
    "dhuhr": "Dhuhr",
    "ashar": "Asr",
    "asar": "Asr",
    "asr": "Asr",
    "maghrib": "Maghrib",
    "magrib": "Maghrib",
    "isya": "Isha",
    "isyak": "Isha",
    "isha": "Isha",
}
ON_WORDS = frozenset({"on", "aktif", "nyala", "nyalakan", "hidup", "hidupkan", "ya"})
OFF_WORDS = frozenset({"off", "mati", "matikan", "nonaktif", "tidak", "stop", "berhenti"})


def parse_prayer_name(text: str) -> str | None:
    return ALIASES.get(text.strip().lower())


def prayer_list(keys: Sequence[str]) -> str:
    """"Imsak, Subuh, Terbit, ..." in the order of the day."""
    return ", ".join(PRAYER_NAMES[key] for key in PRAYER_KEYS if key in keys)


def due_reminders(
    timings: Timings, prayers: Sequence[str], last_sent: datetime | None, now: datetime
) -> list[tuple[str, datetime]]:
    """The (prayer key, time) pairs of `timings` that should be sent now, in the order of the day.

    `timings` must be for `now`'s local date. Anything already sent (at or before `last_sent`) is skipped,
    and so is anything older than GRACE.
    """
    due: list[tuple[str, datetime]] = []
    for key in PRAYER_KEYS:
        if key not in prayers:
            continue
        moment = at_time(timings, key)
        if moment > now or now - moment > GRACE:
            continue
        if last_sent is not None and moment <= last_sent:
            continue
        due.append((key, moment))
    return due


def reminder_text(key: str, timings: Timings, label: str) -> str:
    zone = tz_abbreviation(timings.timezone)
    time = timings.times[key]
    if key == "Imsak":
        return f"🌙 Waktu Imsak ({time} {zone}) — {label}.\nSubuh pukul {timings.times['Fajr']}."
    if key == "Sunrise":
        return f"🌅 Waktu Terbit ({time} {zone}) — {label}.\nWaktu Subuh telah berakhir."
    return f"🕌 Waktunya sholat {PRAYER_NAMES[key]} ({time} {zone}) — {label}."
