from __future__ import annotations

import asyncio
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from muslim_bot import locations, prayer_times
from muslim_bot.locations import find_city, nearest_city
from muslim_bot.prayer_times import (
    SOURCE_API,
    SOURCE_LOCAL,
    PrayerTimesClient,
    SavedLocation,
    Timings,
    compute_timings,
    detect_schedule_request,
    format_schedule,
    is_prayer_related,
    next_prayer,
    parse_timings_response,
    wants_prayer,
)
from muslim_bot.session_store import SessionStore

JAKARTA = ZoneInfo("Asia/Jakarta")
TODAY = date(2026, 10, 17)
NOW = datetime(2026, 10, 17, 14, 30, tzinfo=JAKARTA)
FIXED = {
    "Imsak": "04:02",
    "Fajr": "04:12",
    "Sunrise": "05:31",
    "Dhuhr": "11:38",
    "Asr": "14:47",
    "Maghrib": "17:46",
    "Isha": "18:56",
}
BANDUNG = SavedLocation(-6.92, 107.61, "Asia/Jakarta", "Bandung, Jawa Barat")


def run(coro):
    return asyncio.run(coro)


def fixed_timings(day: date = TODAY, tz: str = "Asia/Jakarta", source: str = SOURCE_API) -> Timings:
    return Timings(day, dict(FIXED), tz, source)


def aladhan_payload(timezone: str = "Asia/Jakarta", times: dict | None = None) -> dict:
    timings = {**FIXED, "Sunset": FIXED["Maghrib"], "Midnight": "23:38", **(times or {})}
    return {"code": 200, "data": {"timings": timings, "meta": {"timezone": timezone, "method": {"id": 20}}}}


def make_client(handler, **kwargs) -> tuple[PrayerTimesClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return PrayerTimesClient("https://example.test/v1/", client=client, **kwargs), requests


# --- cities ---------------------------------------------------------------------------


def test_city_data_is_loaded_with_indonesian_time_zones():
    cities = locations.load_cities()
    assert len(cities) > 400
    assert {city.timezone for city in cities} == {"Asia/Jakarta", "Asia/Makassar", "Asia/Jayapura", "Asia/Pontianak"}


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Bandung", "Bandung"),
        ("kota bandung", "Bandung"),
        ("Kabupaten Wonosobo", "Wonosobo"),
        ("  JAYAPURA ", "Jayapura"),
        ("jogja", "Yogyakarta"),
        ("solo", "Surakarta"),
        ("DKI Jakarta", "Jakarta"),
        ("Banda Aceh", "Banda Aceh"),
    ],
)
def test_find_city_by_name(query, expected):
    city = find_city(query).found
    assert city is not None and city.name == expected


def test_find_city_keeps_each_citys_own_timezone():
    assert find_city("Jayapura").found.timezone == "Asia/Jayapura"
    assert find_city("Makassar").found.timezone == "Asia/Makassar"
    assert find_city("Ambon").found.timezone == "Asia/Jayapura"


def test_namesakes_are_ambiguous_until_a_province_is_given():
    match = find_city("Depok")
    assert match.ambiguous and match.found is None
    assert {city.province for city in match.cities} == {"Jawa Barat", "DI Yogyakarta"}
    chosen = find_city("Depok, Jawa Barat").found
    assert chosen is not None and chosen.province == "Jawa Barat"


@pytest.mark.parametrize("query", ["", "ab", "Xqzvwk", "kota"])
def test_find_city_rejects_unknown_or_too_short(query):
    match = find_city(query)
    assert match.found is None and not match.ambiguous


def test_partial_names_offer_choices_instead_of_guessing():
    unique_prefix = find_city("tasik")
    assert unique_prefix.found is not None and unique_prefix.found.name == "Tasikmalaya"
    many = find_city("ban")  # Bandung, Banjar, Banda Aceh, Bandar Lampung, ...
    assert many.ambiguous and len(many.cities) == locations.MAX_CHOICES
    assert many.cities[0].population >= many.cities[-1].population  # biggest first


def test_nearest_city_respects_the_distance_limit():
    near = nearest_city(-6.9, 107.6)
    assert near is not None and near.name == "Bandung"
    assert nearest_city(24.7, 46.7) is None  # Riyadh is not near any Indonesian city


# --- calculation without the API ------------------------------------------------------

# (day, latitude, longitude, timezone, times as returned by Aladhan method 20 on 2026-10-17)
ALADHAN_REFERENCE = [
    (date(2026, 10, 17), -6.2, 106.82, "Asia/Jakarta", {"Imsak": "04:02", "Fajr": "04:12", "Sunrise": "05:31", "Dhuhr": "11:38", "Asr": "14:47", "Maghrib": "17:46", "Isha": "18:56"}),
    (date(2026, 1, 15), -6.2, 106.82, "Asia/Jakarta", {"Fajr": "04:25", "Sunrise": "05:49", "Dhuhr": "12:02", "Asr": "15:26", "Maghrib": "18:15", "Isha": "19:30"}),
    (date(2026, 6, 21), 5.54, 95.33, "Asia/Jakarta", {"Fajr": "05:02", "Sunrise": "06:27", "Dhuhr": "12:40", "Asr": "16:08", "Maghrib": "18:54", "Isha": "20:10"}),
    (date(2026, 10, 17), -2.53, 140.72, "Asia/Jayapura", {"Fajr": "04:00", "Sunrise": "05:18", "Dhuhr": "11:23", "Asr": "14:36", "Maghrib": "17:28", "Isha": "18:37"}),
    (date(2026, 10, 17), -10.17, 123.61, "Asia/Makassar", {"Fajr": "04:02", "Sunrise": "05:21", "Dhuhr": "11:31", "Asr": "14:34", "Maghrib": "17:41", "Isha": "18:52"}),
]


def minutes(text: str) -> int:
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)


@pytest.mark.parametrize("day,latitude,longitude,tz,expected", ALADHAN_REFERENCE)
def test_local_calculation_agrees_with_aladhan_within_a_minute(day, latitude, longitude, tz, expected):
    computed = compute_timings(day, latitude, longitude, tz)
    for key, value in expected.items():
        assert abs(minutes(computed[key]) - minutes(value)) <= 1, f"{key}: {computed[key]} vs Aladhan {value}"


def test_local_calculation_keeps_the_order_of_the_day():
    computed = compute_timings(TODAY, -6.92, 107.61, "Asia/Jakarta")
    order = [minutes(computed[key]) for key in prayer_times.PRAYER_KEYS]
    assert order == sorted(order)
    assert minutes(computed["Fajr"]) - minutes(computed["Imsak"]) == 10


def test_local_calculation_declines_polar_latitudes():
    assert compute_timings(TODAY, 69.6, 18.9, "Europe/Oslo") is None


# --- Aladhan responses and the client -------------------------------------------------


def test_parse_timings_response_accepts_zone_suffixes_and_reads_the_timezone():
    payload = aladhan_payload(timezone="Asia/Makassar", times={"Fajr": "04:12 (WITA)", "Isha": "18:56 (WITA)"})
    timings = parse_timings_response(payload, TODAY, "Asia/Jakarta")
    assert timings.times["Fajr"] == "04:12" and timings.times["Isha"] == "18:56"
    assert timings.timezone == "Asia/Makassar"
    assert timings.source == SOURCE_API and not timings.approximate


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["data"]["timings"].pop("Maghrib"),
        lambda p: p["data"]["timings"].update(Fajr="soon"),
        lambda p: p["data"]["timings"].update(Fajr="25:00"),
        lambda p: p["data"]["meta"].update(timezone="Mars/Olympus"),
        lambda p: p.pop("data"),
    ],
)
def test_parse_timings_response_rejects_malformed_payloads(mutate):
    payload = aladhan_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        parse_timings_response(payload, TODAY, "Asia/Jakarta")


def test_client_uses_aladhan_and_caches_per_place_and_day():
    client, requests = make_client(lambda request: httpx.Response(200, json=aladhan_payload()))
    first = run(client.timings(-6.92, 107.61, "Asia/Jakarta", TODAY))
    second = run(client.timings(-6.92, 107.61, "Asia/Jakarta", TODAY))
    other_day = run(client.timings(-6.92, 107.61, "Asia/Jakarta", TODAY + timedelta(days=1)))
    other_place = run(client.timings(-2.53, 140.72, "Asia/Jayapura", TODAY))

    assert first == second and first.source == SOURCE_API
    assert other_day.day == TODAY + timedelta(days=1) and other_place is not None
    assert len(requests) == 3  # the repeat was served from the cache
    params = requests[0].url.params
    assert requests[0].url.path == "/v1/timings/17-10-2026"
    assert (params["latitude"], params["longitude"], params["method"], params["timezonestring"]) == (
        "-6.92", "107.61", "20", "Asia/Jakarta"
    )


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500), httpx.Response(200, text="oops"), httpx.Response(200, json={"data": {}})],
)
def test_client_computes_locally_when_the_api_fails(response):
    client, _ = make_client(lambda request: response)
    timings = run(client.timings(-6.92, 107.61, "Asia/Jakarta", TODAY))
    assert timings.source == SOURCE_LOCAL and timings.approximate
    assert set(timings.times) == set(prayer_times.PRAYER_KEYS)


def test_client_falls_back_when_unreachable_and_disabled_never_calls():
    def refuse(request):
        raise httpx.ConnectError("refused")

    down, _ = make_client(refuse)
    assert run(down.timings(-6.92, 107.61, "Asia/Jakarta", TODAY)).source == SOURCE_LOCAL

    off, requests = make_client(lambda request: httpx.Response(200, json=aladhan_payload()), enabled=False)
    assert run(off.timings(-6.92, 107.61, "Asia/Jakarta", TODAY)).source == SOURCE_LOCAL
    assert requests == []


def test_client_cooldown_skips_the_api_after_a_failure_then_retries(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(prayer_times.time, "monotonic", lambda: now[0])
    state = {"healthy": False}

    def handler(request):
        return httpx.Response(200, json=aladhan_payload()) if state["healthy"] else httpx.Response(503)

    client, requests = make_client(handler)
    assert run(client.timings(-6.92, 107.61, "Asia/Jakarta", TODAY)).source == SOURCE_LOCAL
    assert run(client.timings(-6.92, 107.61, "Asia/Jakarta", TODAY)).source == SOURCE_LOCAL
    assert len(requests) == 1  # the second call did not wait for the dead API again

    state["healthy"] = True
    now[0] += prayer_times.FAILURE_COOLDOWN_SECONDS + 1
    assert run(client.timings(-6.92, 107.61, "Asia/Jakarta", TODAY)).source == SOURCE_API


def test_client_returns_none_when_nothing_can_answer():
    client, _ = make_client(lambda request: httpx.Response(500))
    assert run(client.timings(69.6, 18.9, "Europe/Oslo", TODAY)) is None


def test_resolve_timezone_asks_aladhan_then_falls_back_to_the_nearest_city():
    client, requests = make_client(lambda request: httpx.Response(200, json=aladhan_payload(timezone="Asia/Jayapura")))
    assert run(client.resolve_timezone(-2.53, 140.72)) == "Asia/Jayapura"
    assert "timezonestring" not in requests[0].url.params  # let Aladhan decide from the coordinates

    down, _ = make_client(lambda request: httpx.Response(500))
    assert run(down.resolve_timezone(-5.15, 119.43)) == "Asia/Makassar"  # near Makassar
    assert run(down.resolve_timezone(24.7, 46.7)) is None  # far from every listed city


# --- showing times -----------------------------------------------------------------------


def test_next_prayer_through_the_day():
    today, tomorrow = fixed_timings(), fixed_timings(TODAY + timedelta(days=1))

    def at(hour, minute):
        return datetime(2026, 10, 17, hour, minute, tzinfo=JAKARTA)

    assert next_prayer(today, tomorrow, at(3, 0))[0] == "Fajr"
    assert next_prayer(today, tomorrow, at(12, 0))[0] == "Asr"
    assert next_prayer(today, tomorrow, at(17, 46))[0] == "Isha"  # Maghrib is not "next" once it has begun
    after_isha = next_prayer(today, tomorrow, at(21, 0))
    assert after_isha[0] == "Fajr" and after_isha[1].date() == TODAY + timedelta(days=1)
    assert next_prayer(today, None, at(21, 0)) is None


@pytest.mark.parametrize(
    "delta,text",
    [
        (timedelta(minutes=17), "17 menit lagi"),
        (timedelta(minutes=60), "1 jam lagi"),
        (timedelta(hours=2, minutes=5), "2 jam 5 menit lagi"),
        (timedelta(seconds=10), "1 menit lagi"),
    ],
)
def test_describe_wait(delta, text):
    assert prayer_times.describe_wait(delta) == text


def test_format_schedule_lists_the_day_and_the_next_prayer():
    text = format_schedule(fixed_timings(), "Bandung, Jawa Barat", NOW)
    assert text.startswith("## 🕌 Jadwal sholat — Bandung, Jawa Barat")
    assert "Sabtu, 17 Oktober 2026" in text
    for line in ("- Subuh: 04:12", "- Terbit: 05:31", "- Dzuhur: 11:38", "- Ashar: 14:47", "- Maghrib: 17:46", "- Isya: 18:56"):
        assert line in text
    assert "Berikutnya: **Ashar** 14:47 (17 menit lagi)" in text
    assert "Waktu setempat (WIB)" in text and prayer_times.METHOD_NOTE in text
    assert prayer_times.LOCAL_NOTE not in text


def test_format_schedule_flags_locally_computed_times_and_skips_next_for_other_days():
    text = format_schedule(fixed_timings(TODAY + timedelta(days=1), "Asia/Makassar", SOURCE_LOCAL), "Makassar, Sulawesi Selatan", NOW)
    assert prayer_times.LOCAL_NOTE in text
    assert "Berikutnya" not in text  # tomorrow's schedule has no "next prayer"
    assert "Waktu setempat (WITA)" in text


class StubClient:
    """Deterministic stand-in for PrayerTimesClient."""

    def __init__(self, timings=None):
        self.calls: list[tuple] = []
        self._timings = timings

    async def timings(self, latitude, longitude, tz_name, day):
        self.calls.append((latitude, longitude, tz_name, day))
        return self._timings if self._timings is not None else fixed_timings(day, tz_name)

    async def resolve_timezone(self, latitude, longitude):
        return "Asia/Jakarta"


@pytest.fixture
def fixed_now(monkeypatch):
    monkeypatch.setattr(prayer_times, "now_in", lambda tz_name: NOW.astimezone(ZoneInfo(tz_name)))
    return NOW


def test_schedule_reply_today_and_tomorrow(fixed_now):
    client = StubClient()
    today = run(prayer_times.schedule_reply(client, BANDUNG))
    tomorrow = run(prayer_times.schedule_reply(client, BANDUNG, tomorrow=True))
    assert "Sabtu, 17 Oktober 2026" in today and "Berikutnya" in today
    assert "Ahad, 18 Oktober 2026" in tomorrow and "Berikutnya" not in tomorrow
    assert [call[3] for call in client.calls] == [TODAY, TODAY + timedelta(days=1)]


def test_schedule_reply_when_no_source_works(fixed_now):
    class Down(StubClient):
        async def timings(self, *args):
            return None

    assert run(prayer_times.schedule_reply(Down(), BANDUNG)) == prayer_times.UNAVAILABLE_MESSAGE


# --- recognising questions -----------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("jadwal sholat", "today"),
        ("Jadwal sholat hari ini?", "today"),
        ("jadwal salat sekarang", "today"),
        ("waktu sholat hari ini dong", "today"),
        ("tolong tampilkan jadwal shalat besok", "tomorrow"),
        ("jadwal solat", "today"),
    ],
)
def test_plain_schedule_requests_are_answered_directly(message, expected):
    assert detect_schedule_request(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "jadwal sholat surabaya",  # a different city must not get the saved location's times
        "jadwal sholat subuh",
        "jam berapa maghrib",
        "hadits tentang sholat",
        "jadwal sholat jumat bulan depan",
        "",
        "jadwal " + "sholat " * 20,
    ],
)
def test_other_messages_are_not_plain_schedule_requests(message):
    assert detect_schedule_request(message) is None


@pytest.mark.parametrize(
    "message",
    [
        "maghrib jam berapa?",
        "kapan adzan subuh besok",
        "sebentar lagi sholat apa?",
        "jam berapa waktu isya",
        "kapan buka puasa hari ini",
        "jadwal sholat surabaya",
        "imsak jam berapa",
    ],
)
def test_prayer_time_questions(message):
    assert is_prayer_related(message)


@pytest.mark.parametrize(
    "message",
    [
        "hadits tentang keutamaan sholat subuh",
        "doa setelah sholat",
        "ayat tentang sholat",
        "bagaimana cara sholat tahajud",
        "doa buka puasa",
        "terima kasih",
    ],
)
def test_dalil_questions_about_prayer_are_not_prayer_time_questions(message):
    assert not is_prayer_related(message)


def test_follow_ups_keep_the_prayer_times_until_the_topic_changes():
    chain = ["maghrib jam berapa?", "kalau besok?"]
    assert wants_prayer("dan isya?", chain)
    assert wants_prayer("terima kasih", chain[0])
    assert not wants_prayer("ayat tentang sabar", chain)
    assert not wants_prayer("terima kasih", ["hadits tentang niat"])
    assert not wants_prayer("tolong jelaskan " + "panjang " * 20, chain)


def test_build_prayer_context_has_everything_the_llm_needs(fixed_now):
    client = StubClient()
    text = run(prayer_times.build_prayer_context(client, BANDUNG, "maghrib jam berapa?"))

    assert text.startswith("JADWAL SHOLAT")
    assert "Lokasi tersimpan pengguna: Bandung, Jawa Barat (waktu setempat WIB)" in text
    assert "Sekarang: Sabtu, 17 Oktober 2026 pukul 14:30 WIB" in text
    assert "Sholat wajib berikutnya: Ashar pukul 14:47 hari ini (17 menit lagi)" in text
    assert "Hari ini (Sabtu, 17 Oktober 2026): Imsak 04:02; Subuh 04:12; Terbit 05:31; Dzuhur 11:38" in text
    assert "Besok (Ahad, 18 Oktober 2026)" in text
    assert prayer_times.METHOD_NOTE in text


def test_build_prayer_context_after_isya_points_to_tomorrow(fixed_now, monkeypatch):
    late = datetime(2026, 10, 17, 21, 0, tzinfo=JAKARTA)
    monkeypatch.setattr(prayer_times, "now_in", lambda tz_name: late)
    text = run(prayer_times.build_prayer_context(StubClient(), BANDUNG, "subuh jam berapa?"))
    assert "Sholat wajib berikutnya: Subuh pukul 04:12 besok" in text


def test_build_prayer_context_only_for_prayer_questions_and_survives_errors(fixed_now):
    client = StubClient()
    assert run(prayer_times.build_prayer_context(client, BANDUNG, "hadits tentang niat")) is None
    assert client.calls == []

    class Broken(StubClient):
        async def timings(self, *args):
            raise RuntimeError("boom")

    assert run(prayer_times.build_prayer_context(Broken(), BANDUNG, "maghrib jam berapa?")) is None

    class Down(StubClient):
        async def timings(self, *args):
            return None

    assert run(prayer_times.build_prayer_context(Down(), BANDUNG, "maghrib jam berapa?")) is None


def test_build_prayer_context_marks_locally_computed_times(fixed_now):
    client = StubClient(fixed_timings(source=SOURCE_LOCAL))
    text = run(prayer_times.build_prayer_context(client, BANDUNG, "maghrib jam berapa?"))
    assert f"Catatan: {prayer_times.LOCAL_NOTE}" in text


# --- the saved location ----------------------------------------------------------------------

LOCATION = dict(latitude=-6.9222, longitude=107.6069, timezone="Asia/Jakarta", location_label="Bandung, Jawa Barat")


def test_location_is_stored_rounded_and_read_back():
    store = SessionStore(max_turns=5)
    prefs = store.set_prefs("tg:1", **LOCATION)

    assert (prefs.latitude, prefs.longitude) == (-6.92, 107.61)  # about 1 km: not a street address
    assert store.get_prefs("tg:1").location_label == "Bandung, Jawa Barat"
    saved = prayer_times.location_from_prefs(store.get_prefs("tg:1"))
    assert saved == SavedLocation(-6.92, 107.61, "Asia/Jakarta", "Bandung, Jawa Barat")
    assert prayer_times.location_from_prefs(store.get_prefs("tg:2")) is None


def test_clearing_the_coordinates_clears_the_whole_location():
    store = SessionStore(max_turns=5)
    store.set_prefs("tg:1", style="ringkas", **LOCATION)

    prefs = store.set_prefs("tg:1", latitude=None, longitude=None)

    assert (prefs.latitude, prefs.longitude, prefs.timezone, prefs.location_label) == (None, None, None, None)
    assert prefs.style == "ringkas"  # other settings survive


@pytest.mark.parametrize(
    "bad",
    [
        {"latitude": -6.9},  # half a location
        {"longitude": 107.6},
        {"latitude": 91, "longitude": 10, "timezone": "Asia/Jakarta"},
        {"latitude": 10, "longitude": 181, "timezone": "Asia/Jakarta"},
        {"latitude": float("nan"), "longitude": 10, "timezone": "Asia/Jakarta"},
        {"latitude": True, "longitude": 10, "timezone": "Asia/Jakarta"},
        {"latitude": "x", "longitude": 10, "timezone": "Asia/Jakarta"},
        {"latitude": -6.9, "longitude": 107.6},  # no time zone
        {"latitude": -6.9, "longitude": 107.6, "timezone": "Mars/Olympus"},
        {"latitude": -6.9, "longitude": 107.6, "timezone": "Asia/Jakarta", "location_label": "x" * 81},
        {"timezone": "Asia/Jakarta"},  # a zone without coordinates
        {"location_label": "Bandung"},
    ],
)
def test_invalid_locations_are_rejected_and_nothing_is_saved(bad):
    store = SessionStore(max_turns=5)
    with pytest.raises(ValueError):
        store.set_prefs("tg:1", **bad)
    assert store.get_prefs("tg:1").latitude is None


def test_old_databases_gain_the_location_columns(tmp_path: Path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        create table sessions (id text primary key, created_at text not null, updated_at text not null);
        create table user_prefs (session_id text primary key, language text, style text, note text,
                                 history_turns integer, updated_at text not null);
        insert into sessions values ('tg:1', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
        insert into user_prefs values ('tg:1', 'en', 'ringkas', 'halo', 3, '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    conn.close()

    store = SessionStore(max_turns=5, db_path=path)

    old = store.get_prefs("tg:1")
    assert (old.language, old.style, old.history_turns, old.latitude) == ("en", "ringkas", 3, None)
    assert store.set_prefs("tg:1", **LOCATION).location_label == "Bandung, Jawa Barat"
    store.close()
    assert SessionStore(max_turns=5, db_path=path).get_prefs("tg:1").latitude == -6.92  # reopening is harmless


def test_forgetting_prefs_removes_the_location():
    store = SessionStore(max_turns=5)
    store.set_prefs("tg:1", **LOCATION)
    store.clear_prefs("tg:1")
    assert store.get_prefs("tg:1").latitude is None
