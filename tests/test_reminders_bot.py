from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from telegram import ReplyKeyboardMarkup
from telegram.error import BadRequest, Forbidden, NetworkError

from muslim_bot import api, telegram_bot
from muslim_bot.config import load_settings
from muslim_bot.rate_limit import RateLimiter
from muslim_bot.session_store import REMINDER_PRAYERS, SessionStore
from tests.test_api import _settings
from tests.test_prayer import LOCATION, StubClient, fixed_timings

JAKARTA = ZoneInfo("Asia/Jakarta")
UTC = timezone.utc
ADMIN = {"x-admin-token": "admin-test"}


def run(coro):
    return asyncio.run(coro)


def at(hour: int, minute: int, second: int = 0, tz=JAKARTA) -> datetime:
    return datetime(2026, 10, 17, hour, minute, second, tzinfo=tz)


# --- API ------------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    settings = _settings(tmp_path)
    store = SessionStore(max_turns=5)
    limiter = RateLimiter(1000)
    api.app.dependency_overrides[api.get_settings] = lambda: settings
    api.app.dependency_overrides[api.get_session_store] = lambda: store
    api.app.dependency_overrides[api.get_rate_limiter] = lambda: limiter
    yield TestClient(api.app), store
    api.app.dependency_overrides.clear()


def test_reminders_need_a_location_first(client):
    http, store = client
    response = http.put("/sessions/tg:1/reminders", json={"enabled": True})
    assert response.status_code == 422 and "location" in response.json()["detail"]

    store.set_prefs("tg:1", **LOCATION)
    body = http.put("/sessions/tg:1/reminders", json={"enabled": True}).json()
    assert body["enabled"] is True and body["prayers"] == list(REMINDER_PRAYERS)


def test_prefs_show_the_reminder_state_and_losing_the_location_turns_it_off(client):
    http, store = client
    store.set_prefs("tg:1", **LOCATION)
    http.put("/sessions/tg:1/reminders", json={"enabled": True, "prayers": ["Fajr", "Isha"]})

    prefs = http.get("/sessions/tg:1/prefs").json()
    assert prefs["reminders_enabled"] is True and prefs["reminder_prayers"] == ["Fajr", "Isha"]

    cleared = http.put("/sessions/tg:1/prefs", json={"latitude": None, "longitude": None}).json()
    assert cleared["reminders_enabled"] is False
    assert http.get("/sessions/tg:1/reminders").json()["enabled"] is False


@pytest.mark.parametrize("prayers", [[], ["Duha"], ["fajr"], ["Fajr", 5]])
def test_bad_prayer_choices_are_rejected(client, prayers):
    http, store = client
    store.set_prefs("tg:1", **LOCATION)
    assert http.put("/sessions/tg:1/reminders", json={"prayers": prayers}).status_code == 422


@pytest.mark.parametrize("session_id,status", [("tg:1", 200), ("tg:-1001234", 200), ("web-abc", 422), ("tg:abc", 422), ("tg:", 422)])
def test_only_telegram_sessions_can_have_reminders(client, session_id, status):
    http, store = client
    store.set_prefs(session_id, **LOCATION)
    response = http.put(f"/sessions/{session_id}/reminders", json={"enabled": True})
    assert response.status_code == status
    if status == 422:
        assert "Telegram" in response.json()["detail"]


def subscribe(store: SessionStore, session_id: str = "tg:1", **kwargs) -> None:
    store.set_prefs(session_id, **LOCATION)
    store.set_reminders(session_id, enabled=True, **kwargs)


def test_subscription_list_is_admin_only(client):
    http, store = client
    subscribe(store)
    assert http.get("/api/reminders").status_code == 401  # a list of where people are
    assert http.get("/api/reminders", headers={"x-admin-token": "wrong"}).status_code == 401

    [sub] = http.get("/api/reminders", headers=ADMIN).json()
    assert sub["session_id"] == "tg:1" and sub["timezone"] == "Asia/Jakarta"
    assert (sub["latitude"], sub["longitude"]) == (-6.92, 107.61)
    assert sub["prayers"] == list(REMINDER_PRAYERS) and sub["last_sent"] is not None


def test_subscription_list_is_closed_when_no_admin_token_is_configured(client, tmp_path):
    http, store = client
    subscribe(store)
    api.app.dependency_overrides[api.get_settings] = lambda: replace(_settings(tmp_path), admin_token=None)
    assert http.get("/api/reminders", headers=ADMIN).status_code == 503


def test_admin_marks_a_reminder_sent_and_disables_one(client):
    http, store = client
    subscribe(store)
    sent = (datetime.now(UTC) + timedelta(hours=1)).replace(microsecond=0)

    assert http.post("/api/reminders/tg:1/sent", json={"sent_at": sent.isoformat()}).status_code == 401
    ok = http.post("/api/reminders/tg:1/sent", json={"sent_at": sent.isoformat()}, headers=ADMIN)
    assert ok.status_code == 204
    assert store.get_reminders("tg:1").last_sent == sent

    naive = http.post("/api/reminders/tg:1/sent", json={"sent_at": (sent + timedelta(hours=1)).replace(tzinfo=None).isoformat()}, headers=ADMIN)
    assert naive.status_code == 204 and store.get_reminders("tg:1").last_sent == sent + timedelta(hours=1)  # read as UTC
    assert http.post("/api/reminders/tg:1/sent", json={"sent_at": "yesterday"}, headers=ADMIN).status_code == 422

    assert http.post("/api/reminders/tg:1/disable", headers=ADMIN).status_code == 204
    assert store.reminder_subscriptions() == []
    assert http.post("/api/reminders/tg:1/disable").status_code == 401


# --- /pengingat ---------------------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, chat_id: int = 1) -> None:
        self.chat_id = chat_id
        self.sent: list[dict] = []

    async def reply_text(self, text, parse_mode=None, reply_markup=None):
        self.sent.append({"text": text, "markup": reply_markup})


def context(*args, admin_token="secret"):
    return SimpleNamespace(
        args=list(args),
        bot_data={"settings": SimpleNamespace(admin_token=admin_token, telegram_api_base="http://api"), "prayer": StubClient()},
    )


@pytest.fixture
def backend(monkeypatch):
    """The bot's calls to the API, answered by a real store through the real response builders."""
    store = SessionStore(max_turns=5)

    async def fake_backend_call(settings, method, path, chat_id, json=None):
        session_id = f"tg:{chat_id}"
        try:
            if path == "reminders" and method == "PUT":
                store.set_reminders(session_id, **json)
                return api._reminders_response(session_id, store).model_dump()
            if path == "prefs" and method == "PUT":
                store.set_prefs(session_id, **json)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc  # what the real call does for a 422
        assert path == "prefs"
        return api._prefs_response(session_id, store.get_prefs(session_id), store).model_dump()

    monkeypatch.setattr(telegram_bot, "backend_call", fake_backend_call)
    return store


def command(backend, *args, admin_token="secret", chat_id=1) -> list[dict]:
    message = FakeMessage(chat_id)
    run(telegram_bot.reminder_command(SimpleNamespace(message=message), context(*args, admin_token=admin_token)))
    return message.sent


def test_pengingat_without_arguments_shows_the_state_and_how_to_use_it(backend):
    [reply] = command(backend)
    assert "Pengingat sholat: mati" in reply["text"]
    assert "Waktu: Imsak, Subuh, Terbit, Dzuhur, Ashar, Maghrib, Isya" in reply["text"]
    assert "/pengingat on" in reply["text"] and "tepat waktu" in reply["text"]


def test_on_without_a_location_asks_for_one_and_changes_nothing(backend):
    [reply] = command(backend, "on")
    assert "butuh lokasi" in reply["text"]
    assert isinstance(reply["markup"], ReplyKeyboardMarkup) and reply["markup"].keyboard[0][0].request_location
    assert backend.get_reminders("tg:1").enabled is False


def test_on_with_a_location_turns_every_time_on(backend):
    backend.set_prefs("tg:1", **LOCATION)
    [reply] = command(backend, "on")

    assert "dinyalakan ✅" in reply["text"]
    assert "Imsak, Subuh, Terbit, Dzuhur, Ashar, Maghrib, Isya" in reply["text"]
    state = backend.get_reminders("tg:1")
    assert state.enabled and state.prayers == REMINDER_PRAYERS


def test_on_is_refused_when_the_bot_could_not_deliver(backend):
    backend.set_prefs("tg:1", **LOCATION)
    [reply] = command(backend, "on", admin_token=None)  # without the token it cannot read who is subscribed
    assert "belum tersedia" in reply["text"]
    assert backend.get_reminders("tg:1").enabled is False  # never promise a reminder that will not arrive


def test_off_turns_everything_off(backend):
    backend.set_prefs("tg:1", **LOCATION)
    command(backend, "on")
    [reply] = command(backend, "off")
    assert reply["text"] == "Pengingat sholat dimatikan."
    assert backend.get_reminders("tg:1").enabled is False
    assert [reply["text"] for reply in command(backend, "mati")] == ["Pengingat sholat dimatikan."]


def test_one_time_can_be_switched_off_and_on_again(backend):
    backend.set_prefs("tg:1", **LOCATION)
    command(backend, "on")

    [off] = command(backend, "imsak", "off")
    assert "Pengingat Imsak dimatikan" in off["text"]
    assert "Subuh, Terbit, Dzuhur, Ashar, Maghrib, Isya" in off["text"] and "Imsak," not in off["text"].split("aktif:")[1]
    assert backend.get_reminders("tg:1").prayers == tuple(k for k in REMINDER_PRAYERS if k != "Imsak")
    assert backend.get_reminders("tg:1").enabled  # still on

    [on] = command(backend, "imsak", "on")
    assert "Pengingat Imsak dinyalakan" in on["text"]
    assert backend.get_reminders("tg:1").prayers == REMINDER_PRAYERS

    command(backend, "terbit", "off")
    assert "Terbit" not in command(backend)[0]["text"].split("Waktu:")[1].split("\n")[0]


def test_every_spelling_of_a_prayer_works(backend):
    backend.set_prefs("tg:1", **LOCATION)
    command(backend, "on")
    for word, key in [("zuhur", "Dhuhr"), ("shubuh", "Fajr"), ("asar", "Asr"), ("isyak", "Isha")]:
        command(backend, word, "off")
        assert key not in backend.get_reminders("tg:1").prayers
        command(backend, word, "on")
        assert key in backend.get_reminders("tg:1").prayers


def test_the_last_time_cannot_be_switched_off_one_by_one(backend):
    backend.set_prefs("tg:1", **LOCATION)
    backend.set_reminders("tg:1", enabled=True, prayers=["Maghrib"])
    [reply] = command(backend, "maghrib", "off")
    assert "Minimal satu waktu" in reply["text"] and "/pengingat off" in reply["text"]
    assert backend.get_reminders("tg:1").prayers == ("Maghrib",)


def test_choosing_times_while_off_is_remembered_and_explained(backend):
    backend.set_prefs("tg:1", **LOCATION)
    [reply] = command(backend, "imsak", "off")
    assert "masih mati" in reply["text"]
    assert backend.get_reminders("tg:1").enabled is False and "Imsak" not in backend.get_reminders("tg:1").prayers


@pytest.mark.parametrize("args", [("dhuha", "off"), ("subuh",), ("subuh", "mungkin"), ("banyak", "kata", "ngawur")])
def test_unclear_commands_show_the_usage(backend, args):
    backend.set_prefs("tg:1", **LOCATION)
    [reply] = command(backend, *args)
    assert reply["text"].startswith("Cara pakai:")
    assert backend.get_reminders("tg:1").prayers == REMINDER_PRAYERS


def test_removing_the_location_switches_reminders_off(backend):
    backend.set_prefs("tg:1", **LOCATION)
    command(backend, "on")
    backend.set_prefs("tg:1", latitude=None, longitude=None)  # what /lokasi hapus does
    assert backend.get_reminders("tg:1").enabled is False
    assert "mati" in command(backend)[0]["text"]


def test_settings_summary_mentions_reminders():
    assert telegram_bot.format_reminder_summary({}) == "mati"
    assert telegram_bot.format_reminder_summary({"reminders_enabled": True, "reminder_prayers": ["Fajr", "Isha"]}) == "aktif (2 waktu)"
    assert "• Pengingat: mati" in telegram_bot.format_prefs({})


# --- delivering ---------------------------------------------------------------------------------------------


class FakeBot:
    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list[tuple[int, str]] = []
        self.error = error

    async def send_message(self, chat_id, text):
        if self.error is not None:
            raise self.error
        self.sent.append((chat_id, text))


def sub(session_id="tg:1", tz="Asia/Jakarta", prayers=None, last_sent=None, label="Bandung, Jawa Barat") -> dict:
    return {
        "session_id": session_id,
        "latitude": -6.92,
        "longitude": 107.61,
        "timezone": tz,
        "location_label": label,
        "prayers": list(prayers or REMINDER_PRAYERS),
        "last_sent": last_sent,
    }


@pytest.fixture
def server(monkeypatch):
    """The API as the delivery loop sees it: a list of subscriptions in, bookkeeping posts out."""
    state = {"subs": [], "posts": [], "post_error": None}

    async def fake_fetch(settings):
        return state["subs"]

    async def fake_post(settings, path, json=None):
        if state["post_error"] is not None:
            raise state["post_error"]
        state["posts"].append((path, json))

    monkeypatch.setattr(telegram_bot, "fetch_subscriptions", fake_fetch)
    monkeypatch.setattr(telegram_bot, "admin_post", fake_post)
    return state


def deliver(bot, now, cache=None, prayer=None):
    cache = {} if cache is None else cache
    count = run(telegram_bot.send_due_reminders(bot, SimpleNamespace(), prayer or StubClient(), now.astimezone(UTC), cache, delay=0))
    return count, cache


def test_a_due_reminder_is_sent_to_its_chat_and_recorded(server):
    server["subs"] = [sub("tg:42")]
    bot = FakeBot()

    count, cache = deliver(bot, at(4, 12, 3))

    assert count == 1
    assert bot.sent == [(42, "🕌 Waktunya sholat Subuh (04:12 WIB) — Bandung, Jawa Barat.")]
    assert server["posts"] == [("/api/reminders/tg:42/sent", {"sent_at": at(4, 12).astimezone(UTC).isoformat()})]
    assert cache["tg:42"] == at(4, 12)


def test_nothing_is_sent_between_prayer_times(server):
    server["subs"] = [sub()]
    bot = FakeBot()
    assert deliver(bot, at(9, 0))[0] == 0
    assert bot.sent == [] and server["posts"] == []


def test_the_same_minute_is_never_sent_twice(server):
    server["subs"] = [sub()]
    bot = FakeBot()
    count, cache = deliver(bot, at(4, 12, 3))
    again, _ = deliver(bot, at(4, 13, 3), cache)  # next tick, the server still says last_sent is empty
    assert (count, again) == (1, 0) and len(bot.sent) == 1


def test_a_reminder_the_server_already_recorded_is_not_repeated_after_a_restart(server):
    server["subs"] = [sub(last_sent=at(4, 12).astimezone(UTC).isoformat())]
    bot = FakeBot()
    assert deliver(bot, at(4, 12, 30))[0] == 0 and bot.sent == []


def test_failed_bookkeeping_neither_stops_nor_repeats_reminders(server):
    server["subs"] = [sub()]
    server["post_error"] = RuntimeError("api down")
    bot = FakeBot()
    count, cache = deliver(bot, at(4, 12, 3))
    again, _ = deliver(bot, at(4, 13, 3), cache)
    assert (count, again) == (1, 0) and len(bot.sent) == 1


def test_each_place_is_reminded_in_its_own_time_zone(server):
    server["subs"] = [sub("tg:1", "Asia/Jakarta"), sub("tg:2", "Asia/Makassar", label="Makassar, Sulawesi Selatan"), sub("tg:3", "Asia/Jayapura", label="Jayapura, Papua")]
    bot = FakeBot()

    deliver(bot, at(4, 12, 3, JAKARTA))  # 04:12 in Jakarta is 05:12 in Makassar and 06:12 in Jayapura
    assert [chat for chat, _ in bot.sent] == [1]
    deliver(bot, at(4, 12, 3, ZoneInfo("Asia/Makassar")))
    assert [chat for chat, _ in bot.sent] == [1, 2]
    deliver(bot, at(4, 12, 3, ZoneInfo("Asia/Jayapura")))
    assert [chat for chat, _ in bot.sent] == [1, 2, 3]
    assert "(04:12 WITA)" in bot.sent[1][1] and "(04:12 WIT)" in bot.sent[2][1]


def test_only_the_chosen_times_are_sent(server):
    server["subs"] = [sub("tg:1", prayers=["Dhuhr"]), sub("tg:2")]
    bot = FakeBot()
    deliver(bot, at(11, 38, 2))
    deliver(bot, at(5, 31, 2))  # Terbit: only the one that kept every time hears about it
    assert [(chat, text.split("(")[0]) for chat, text in bot.sent] == [
        (1, "🕌 Waktunya sholat Dzuhur "), (2, "🕌 Waktunya sholat Dzuhur "), (2, "🌅 Waktu Terbit ")
    ]


def test_all_seven_times_of_the_day_are_sent_once(server):
    server["subs"] = [sub()]
    bot, cache = FakeBot(), {}
    for hour, minute in [(4, 2), (4, 12), (5, 31), (11, 38), (14, 47), (17, 46), (18, 56)]:
        deliver(bot, at(hour, minute, 2), cache)
    kinds = [text.split("\n")[0].split(" (")[0] for _, text in bot.sent]
    assert kinds == [
        "🌙 Waktu Imsak", "🕌 Waktunya sholat Subuh", "🌅 Waktu Terbit", "🕌 Waktunya sholat Dzuhur",
        "🕌 Waktunya sholat Ashar", "🕌 Waktunya sholat Maghrib", "🕌 Waktunya sholat Isya",
    ]


@pytest.mark.parametrize("error", [Forbidden("Forbidden: bot was blocked by the user"), BadRequest("Chat not found")])
def test_a_chat_that_cannot_be_reached_has_its_reminders_turned_off(server, error):
    server["subs"] = [sub("tg:9")]
    count, cache = deliver(FakeBot(error), at(4, 12, 3))
    assert count == 0 and cache == {}
    assert server["posts"] == [("/api/reminders/tg:9/disable", None)]


def test_a_temporary_failure_is_retried_on_the_next_minute(server):
    server["subs"] = [sub()]
    count, cache = deliver(FakeBot(NetworkError("timeout")), at(4, 12, 3))
    assert count == 0 and cache == {} and server["posts"] == []  # nothing recorded: still due

    bot = FakeBot()
    assert deliver(bot, at(4, 13, 3), cache)[0] == 1  # within the 5 minute grace
    assert len(bot.sent) == 1


def test_one_failing_chat_does_not_stop_the_others(server):
    server["subs"] = [sub("tg:1"), sub("tg:2")]

    class Selective(FakeBot):
        async def send_message(self, chat_id, text):
            if chat_id == 1:
                raise Forbidden("blocked")
            await super().send_message(chat_id, text)

    bot = Selective()
    assert deliver(bot, at(4, 12, 3))[0] == 1
    assert [chat for chat, _ in bot.sent] == [2]


def test_odd_subscriptions_are_skipped(server):
    class NoTimings(StubClient):
        async def timings(self, *args):
            return None

    server["subs"] = [sub("web-abc"), sub("tg:oops")]
    bot = FakeBot()
    assert deliver(bot, at(4, 12, 3))[0] == 0
    server["subs"] = [sub("tg:1")]
    assert deliver(bot, at(4, 12, 3), prayer=NoTimings())[0] == 0  # no source can compute the day
    assert bot.sent == []


def test_old_entries_leave_the_memory_cache(server):
    server["subs"] = []
    cache = {"tg:old": at(4, 12) - timedelta(days=2), "tg:recent": at(4, 12) - timedelta(hours=3)}
    deliver(FakeBot(), at(4, 12, 3), cache)
    assert list(cache) == ["tg:recent"]


def test_chat_ids():
    assert telegram_bot.chat_id_of("tg:123") == 123
    assert telegram_bot.chat_id_of("tg:-1001") == -1001
    assert telegram_bot.chat_id_of("web:123") is None and telegram_bot.chat_id_of("tg:x") is None


# --- the loop ----------------------------------------------------------------------------------------------


def test_the_loop_survives_failures_and_wakes_just_after_each_minute(monkeypatch):
    calls: list[int] = []
    sleeps: list[float] = []

    async def flaky(bot, settings, prayer, now, cache, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("api unreachable")
        return 0

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(telegram_bot, "send_due_reminders", flaky)
    monkeypatch.setattr(telegram_bot.asyncio, "sleep", fake_sleep)
    application = SimpleNamespace(bot=object(), bot_data={"settings": SimpleNamespace(), "prayer": StubClient()})

    with pytest.raises(asyncio.CancelledError):
        run(telegram_bot.reminder_loop(application))

    assert len(calls) == 3  # one failure did not end it
    assert all(1.0 <= seconds <= telegram_bot.REMINDER_INTERVAL_SECONDS + 1 for seconds in sleeps)


def test_the_loop_starts_only_with_an_admin_token_and_stops_cleanly(monkeypatch):
    async def idle(application):
        await asyncio.sleep(3600)

    monkeypatch.setattr(telegram_bot, "reminder_loop", idle)

    async def scenario(token):
        application = SimpleNamespace(bot_data={"settings": SimpleNamespace(admin_token=token)})
        await telegram_bot.start_reminders(application)
        task = application.bot_data.get(telegram_bot.REMINDER_TASK_KEY)
        started = task is not None and not task.done()
        await telegram_bot.stop_reminders(application)
        return started, task is None or task.cancelled(), telegram_bot.REMINDER_TASK_KEY in application.bot_data

    assert run(scenario(None)) == (False, True, False)
    assert run(scenario("secret")) == (True, True, False)


def test_the_application_wires_the_command_and_the_loop():
    application = telegram_bot.build_application(replace(load_settings(env_file=None), telegram_bot_token="123:TEST"))

    callbacks = [handler.callback for handler in application.handlers[0]]
    assert telegram_bot.reminder_command in callbacks
    assert application.post_init is telegram_bot.start_reminders
    assert application.post_shutdown is telegram_bot.stop_reminders


# --- the bot and the API together --------------------------------------------------------------------------


@pytest.fixture
def wired(client, monkeypatch):
    """Bot code talking to the real API app over (in-process) HTTP, so the two sides must agree on the wire."""
    import httpx

    http, store = client
    real = httpx.AsyncClient
    monkeypatch.setattr(
        telegram_bot.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.ASGITransport(app=api.app), base_url="http://testserver", **kw),
    )
    settings = SimpleNamespace(telegram_api_base="http://testserver", admin_token="admin-test")
    subscribe(store, "tg:42")
    store._conn.execute("update reminders set last_sent = '2000-01-01T00:00:00+00:00'")  # as if enabled long ago
    store._conn.commit()
    return store, settings


def deliver_over_http(settings, bot, now):
    return run(telegram_bot.send_due_reminders(bot, settings, StubClient(), now.astimezone(UTC), {}, delay=0))


def test_bot_and_api_agree_on_listing_sending_and_recording(wired):
    store, settings = wired
    bot = FakeBot()

    assert deliver_over_http(settings, bot, at(4, 12, 3)) == 1
    assert bot.sent == [(42, "🕌 Waktunya sholat Subuh (04:12 WIB) — Bandung, Jawa Barat.")]
    assert store.get_reminders("tg:42").last_sent == at(4, 12).astimezone(UTC)  # recorded through the API

    # A new process (empty memory cache) must not repeat it: the server remembers.
    assert deliver_over_http(settings, bot, at(4, 12, 40)) == 0
    assert len(bot.sent) == 1
    assert deliver_over_http(settings, bot, at(5, 31, 3)) == 1  # and the next one still goes out
    assert "Terbit" in bot.sent[1][1]


def test_bot_and_api_agree_on_turning_off_an_unreachable_chat(wired):
    store, settings = wired
    assert deliver_over_http(settings, FakeBot(Forbidden("bot was blocked by the user")), at(4, 12, 3)) == 0
    assert store.get_reminders("tg:42").enabled is False
    assert store.reminder_subscriptions() == []


def test_the_api_refuses_the_bot_without_the_right_token(wired):
    store, settings = wired
    settings.admin_token = "not-the-token"
    with pytest.raises(Exception):
        deliver_over_http(settings, FakeBot(), at(4, 12, 3))
