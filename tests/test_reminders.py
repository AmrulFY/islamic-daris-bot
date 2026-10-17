from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from muslim_bot import prayer_times, reminders
from muslim_bot.prayer_times import PRAYER_KEYS, Timings
from muslim_bot.reminders import (
    GRACE,
    due_reminders,
    parse_prayer_name,
    prayer_list,
    reminder_text,
)
from muslim_bot.session_store import REMINDER_PRAYERS, SessionStore
from tests.test_prayer import FIXED, LOCATION, StubClient, fixed_timings

JAKARTA = ZoneInfo("Asia/Jakarta")
UTC = timezone.utc
DAY = datetime(2026, 10, 17, tzinfo=JAKARTA).date()


def run(coro):
    return asyncio.run(coro)


def at(hour: int, minute: int, second: int = 0, tz=JAKARTA) -> datetime:
    return datetime(2026, 10, 17, hour, minute, second, tzinfo=tz)


def store_with_location(session_id: str = "tg:1", **kwargs) -> SessionStore:
    store = SessionStore(max_turns=5, **kwargs)
    store.set_prefs(session_id, **LOCATION)
    return store


# --- storage ----------------------------------------------------------------------------------


def test_reminder_keys_match_the_prayer_keys():
    assert REMINDER_PRAYERS == PRAYER_KEYS


def test_reminders_default_to_off_with_every_time_selected():
    reminders_ = SessionStore(max_turns=5).get_reminders("tg:1")
    assert reminders_.enabled is False
    assert reminders_.prayers == REMINDER_PRAYERS  # Imsak, Terbit and the five prayers
    assert reminders_.last_sent is None


def test_turning_reminders_on_needs_a_location():
    store = SessionStore(max_turns=5)
    with pytest.raises(ValueError, match="location"):
        store.set_reminders("tg:1", enabled=True)
    assert store.get_reminders("tg:1").enabled is False


def test_turning_on_starts_from_now_so_past_prayers_are_not_sent():
    store = store_with_location()
    before = datetime.now(UTC).replace(microsecond=0)  # stored to the second
    result = store.set_reminders("tg:1", enabled=True)

    assert result.enabled and result.prayers == REMINDER_PRAYERS
    assert before <= result.last_sent <= datetime.now(UTC)
    assert store.get_reminders("tg:1") == result


def test_choosing_prayer_times_keeps_the_rest_and_the_order_of_the_day():
    store = store_with_location()
    store.set_reminders("tg:1", enabled=True)
    started = store.get_reminders("tg:1").last_sent

    chosen = store.set_reminders("tg:1", prayers=["Isha", "Fajr", "Sunrise", "Fajr"])

    assert chosen.prayers == ("Fajr", "Sunrise", "Isha")
    assert chosen.enabled and chosen.last_sent == started  # choosing times is not a fresh start
    off = store.set_reminders("tg:1", enabled=False)
    assert not off.enabled and off.prayers == ("Fajr", "Sunrise", "Isha")  # the choice survives being off


@pytest.mark.parametrize("bad", [[], ["Duha"], ["fajr"], "Fajr", 5, ["Fajr", "Eid"]])
def test_invalid_prayer_choices_are_rejected(bad):
    store = store_with_location()
    store.set_reminders("tg:1", enabled=True)
    with pytest.raises(ValueError):
        store.set_reminders("tg:1", prayers=bad)
    assert store.get_reminders("tg:1").prayers == REMINDER_PRAYERS


def test_removing_the_location_turns_reminders_off():
    store = store_with_location()
    store.set_reminders("tg:1", enabled=True, prayers=["Fajr"])

    store.set_prefs("tg:1", latitude=None, longitude=None)

    assert store.get_reminders("tg:1").enabled is False
    assert store.reminder_subscriptions() == []
    with pytest.raises(ValueError):
        store.set_reminders("tg:1", enabled=True)  # needs a place again


def test_subscriptions_list_only_enabled_reminders_with_their_place():
    store = SessionStore(max_turns=5)
    for session_id in ("tg:2", "tg:1", "tg:3"):
        store.set_prefs(session_id, **LOCATION)
    store.set_reminders("tg:1", enabled=True)
    store.set_reminders("tg:2", enabled=True, prayers=["Fajr", "Isha"])
    store.set_reminders("tg:3", enabled=False)
    store.set_prefs("tg:4", style="ringkas")  # no location, no reminder

    subs = store.reminder_subscriptions()

    assert [sub.session_id for sub in subs] == ["tg:1", "tg:2"]
    assert subs[0].prayers == REMINDER_PRAYERS and subs[1].prayers == ("Fajr", "Isha")
    assert (subs[0].latitude, subs[0].longitude, subs[0].timezone) == (-6.92, 107.61, "Asia/Jakarta")
    assert subs[0].location_label == "Bandung, Jawa Barat"


def test_mark_sent_only_moves_forward():
    store = store_with_location()
    store.set_reminders("tg:1", enabled=True)
    base = datetime.now(UTC).replace(microsecond=0) + timedelta(hours=1)  # after the moment reminders were turned on
    store.mark_reminder_sent("tg:1", base)
    store.mark_reminder_sent("tg:1", base + timedelta(hours=2))
    store.mark_reminder_sent("tg:1", base + timedelta(hours=1))  # older than the latest: ignored
    store.mark_reminder_sent("tg:1", (base + timedelta(hours=3)).astimezone(JAKARTA))  # any zone works

    assert store.get_reminders("tg:1").last_sent == base + timedelta(hours=3)


def test_disable_keeps_the_choice():
    store = store_with_location()
    store.set_reminders("tg:1", enabled=True, prayers=["Maghrib"])
    store.disable_reminders("tg:1")
    after = store.get_reminders("tg:1")
    assert not after.enabled and after.prayers == ("Maghrib",)


def test_forgetting_the_session_prefs_removes_reminders_too():
    store = store_with_location()
    store.set_reminders("tg:1", enabled=True)
    store.clear_prefs("tg:1")
    assert store.get_reminders("tg:1").enabled is False
    assert store.reminder_subscriptions() == []


def test_reminders_survive_a_restart(tmp_path: Path):
    path = tmp_path / "s.db"
    store = store_with_location(db_path=path)
    store.set_reminders("tg:1", enabled=True, prayers=["Fajr", "Isha"])
    sent = datetime.now(UTC).replace(microsecond=0) + timedelta(hours=1)
    store.mark_reminder_sent("tg:1", sent)
    store.close()

    again = SessionStore(max_turns=5, db_path=path)
    assert [(s.session_id, s.prayers) for s in again.reminder_subscriptions()] == [("tg:1", ("Fajr", "Isha"))]
    assert again.get_reminders("tg:1").last_sent == sent


def age(store: SessionStore, session_id: str, days: int) -> None:
    old = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    store._conn.execute("update sessions set updated_at = ? where id = ?", (old, session_id))
    store._conn.commit()


def test_purge_keeps_people_who_have_reminders_on_but_forgets_their_chat():
    store = SessionStore(max_turns=5)
    for session_id in ("tg:keep", "tg:off", "tg:plain"):
        store.set_prefs(session_id, **LOCATION)
        store.append(session_id, "assalamualaikum", "wa alaikumussalam")
    store.set_reminders("tg:keep", enabled=True)
    store.set_reminders("tg:off", enabled=False)
    for session_id in ("tg:keep", "tg:off", "tg:plain"):
        age(store, session_id, 30)

    removed = store.purge_inactive(days=7)

    assert removed == 2  # the idle sessions without reminders
    assert store.recent_turns("tg:keep") == []  # chat history is still forgotten
    assert store.get_prefs("tg:keep").location_label == "Bandung, Jawa Barat"
    assert [s.session_id for s in store.reminder_subscriptions()] == ["tg:keep"]
    assert store.get_prefs("tg:off").latitude is None and store.get_prefs("tg:plain").latitude is None


def test_purge_removes_a_session_once_its_reminders_are_off():
    store = store_with_location()
    store.set_reminders("tg:1", enabled=True)
    age(store, "tg:1", 30)
    assert store.purge_inactive(days=7) == 0
    store.disable_reminders("tg:1")
    assert store.purge_inactive(days=7) == 1
    assert store.get_prefs("tg:1").latitude is None


# --- when a reminder is due --------------------------------------------------------------------

ALL = REMINDER_PRAYERS


def due_keys(now, last_sent=None, prayers=ALL, timings=None):
    return [key for key, _ in due_reminders(timings or fixed_timings(), prayers, last_sent, now)]


def test_a_reminder_is_due_in_the_minute_it_arrives_and_not_before():
    assert due_keys(at(4, 11, 59)) == []
    assert due_keys(at(4, 12, 0)) == ["Fajr"]
    assert due_keys(at(4, 12, 3)) == ["Fajr"]


def test_imsak_and_terbit_are_reminded_like_the_prayers():
    assert due_keys(at(4, 2, 1)) == ["Imsak"]
    assert due_keys(at(5, 31, 2)) == ["Sunrise"]
    for key, hhmm in FIXED.items():
        hour, minute = (int(part) for part in hhmm.split(":"))
        assert due_keys(at(hour, minute, 1)) == [key]  # all seven, each exactly once in the day


def test_a_late_reminder_is_still_sent_within_the_grace_period_only():
    assert due_keys(at(4, 12) + GRACE) == ["Fajr"]
    assert due_keys(at(4, 12) + GRACE + timedelta(seconds=1)) == []
    assert GRACE == timedelta(minutes=5)


def test_nothing_is_sent_twice():
    moment = at(4, 12)
    assert due_keys(at(4, 12, 30), last_sent=moment) == []
    assert due_keys(at(4, 12, 30), last_sent=moment - timedelta(seconds=1)) == ["Fajr"]
    assert due_keys(at(4, 13), last_sent=at(4, 2)) == ["Fajr"]  # the earlier Imsak was sent, Subuh was not
    assert due_keys(at(4, 12, 30), last_sent=at(4, 12).astimezone(UTC)) == []  # zones do not confuse it


def test_only_the_chosen_times_are_sent():
    assert due_keys(at(4, 12, 1), prayers=("Dhuhr", "Isha")) == []
    assert due_keys(at(11, 38, 1), prayers=("Dhuhr", "Isha")) == ["Dhuhr"]
    assert due_keys(at(4, 2, 1), prayers=tuple(k for k in ALL if k != "Imsak")) == []


def test_two_reminders_in_the_same_window_come_in_the_order_of_the_day():
    close = Timings(DAY, {**FIXED, "Fajr": "04:12", "Sunrise": "04:14"}, "Asia/Jakarta", "aladhan")
    assert due_keys(at(4, 15), timings=close) == ["Fajr", "Sunrise"]


def test_reminder_texts():
    timings = fixed_timings()
    label = "Bandung, Jawa Barat"
    assert reminder_text("Fajr", timings, label) == "🕌 Waktunya sholat Subuh (04:12 WIB) — Bandung, Jawa Barat."
    assert reminder_text("Dhuhr", timings, label).startswith("🕌 Waktunya sholat Dzuhur (11:38 WIB)")
    assert reminder_text("Asr", timings, label).startswith("🕌 Waktunya sholat Ashar (14:47 WIB)")
    assert reminder_text("Maghrib", timings, label).startswith("🕌 Waktunya sholat Maghrib (17:46 WIB)")
    assert reminder_text("Isha", timings, label).startswith("🕌 Waktunya sholat Isya (18:56 WIB)")
    imsak = reminder_text("Imsak", timings, label)
    assert "Waktu Imsak (04:02 WIB)" in imsak and "Subuh pukul 04:12" in imsak
    sunrise = reminder_text("Sunrise", timings, label)
    assert "Waktu Terbit (05:31 WIB)" in sunrise and "Waktu Subuh telah berakhir" in sunrise
    makassar = Timings(DAY, FIXED, "Asia/Makassar", "aladhan")
    assert "(04:12 WITA)" in reminder_text("Fajr", makassar, "Makassar")


@pytest.mark.parametrize(
    "word,key",
    [("Imsak", "Imsak"), ("subuh", "Fajr"), ("shubuh", "Fajr"), ("TERBIT", "Sunrise"), ("zuhur", "Dhuhr"), ("dzuhur", "Dhuhr"),
     ("ashar", "Asr"), ("asar", "Asr"), ("maghrib", "Maghrib"), ("isya", "Isha"), ("isyak", "Isha")],
)
def test_prayer_names_people_type(word, key):
    assert parse_prayer_name(word) == key


def test_unknown_prayer_names_and_listing():
    assert parse_prayer_name("dhuha") is None and parse_prayer_name("") is None
    assert prayer_list(("Isha", "Fajr", "Sunrise")) == "Subuh, Terbit, Isya"  # the order of the day
    assert prayer_list(REMINDER_PRAYERS) == "Imsak, Subuh, Terbit, Dzuhur, Ashar, Maghrib, Isya"
    assert reminders.ON_WORDS.isdisjoint(reminders.OFF_WORDS)
    assert set(reminders.ALIASES.values()) == set(PRAYER_KEYS)
    assert prayer_times.PRAYER_NAMES[reminders.ALIASES["terbit"]] == "Terbit"
