from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove

from muslim_bot import prayer_times, telegram_bot
from muslim_bot.agent import AgentReply
from muslim_bot.config import load_settings
from tests.test_hijri import chat_client  # noqa: F401 (fixture)
from tests.test_prayer import NOW, StubClient, fixed_now  # noqa: F401 (fixture)

LOCATION_JSON = {
    "latitude": -6.9222,
    "longitude": 107.6069,
    "timezone": "Asia/Jakarta",
    "location_label": "Bandung, Jawa Barat",
}


def run(coro):
    return asyncio.run(coro)


# --- chat endpoints ------------------------------------------------------------------------


class CapturingRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ask(self, message, *, session_id, history_prompt=None, language="id", recent_turns=None):
        self.calls.append({"message": message, "history_prompt": history_prompt or ""})
        return AgentReply(answer="Jawaban dari LLM.", used_tools=[])

    def try_fast_path(self, message, *, language="id"):
        return None


@pytest.fixture
def prayer_chat(chat_client, fixed_now):  # noqa: F811
    client, api, monkeypatch = chat_client
    runner = CapturingRunner()
    api.app.dependency_overrides[api.get_runner] = lambda: runner
    monkeypatch.setattr(api, "get_prayer_client", lambda: StubClient())

    def set_location(session_id: str) -> None:
        response = client.put(f"/sessions/{session_id}/prefs", json=LOCATION_JSON)
        assert response.status_code == 200

    return client, runner, set_location


def test_plain_schedule_question_is_answered_without_the_llm(prayer_chat):
    client, runner, set_location = prayer_chat
    set_location("s1")

    body = client.post("/chat", json={"message": "jadwal sholat hari ini", "session_id": "s1"}).json()

    assert "Jadwal sholat — Bandung, Jawa Barat" in body["answer"]
    assert "Ashar: 14:47" in body["answer"] and "Berikutnya: **Ashar**" in body["answer"]
    assert body["used_tools"] == ["prayer_times"]
    assert runner.calls == []


def test_tomorrows_schedule(prayer_chat):
    client, runner, set_location = prayer_chat
    set_location("s1")

    body = client.post("/chat", json={"message": "jadwal sholat besok", "session_id": "s1"}).json()

    assert "Ahad, 18 Oktober 2026" in body["answer"]
    assert runner.calls == []


def test_schedule_question_without_a_location_explains_how_to_set_one(prayer_chat):
    client, runner, _ = prayer_chat

    body = client.post("/chat", json={"message": "jadwal sholat", "session_id": "nobody"}).json()

    assert body["answer"] == prayer_times.NO_LOCATION_MESSAGE
    assert "/lokasi" in body["answer"]
    assert runner.calls == []


def test_schedule_stream_is_instant_too(prayer_chat):
    client, runner, set_location = prayer_chat
    set_location("s1")

    response = client.post("/chat/stream", json={"message": "jadwal sholat", "session_id": "s1"})

    assert response.status_code == 200
    assert "Jadwal sholat" in response.text and "prayer_times" in response.text
    assert runner.calls == []


def test_specific_question_gives_the_llm_the_times(prayer_chat):
    client, runner, set_location = prayer_chat
    set_location("s1")

    body = client.post("/chat", json={"message": "maghrib jam berapa?", "session_id": "s1"}).json()

    assert body["answer"] == "Jawaban dari LLM."
    [call] = runner.calls
    prompt = call["history_prompt"]
    assert "JADWAL SHOLAT" in prompt
    assert "Sholat wajib berikutnya: Ashar pukul 14:47 hari ini (17 menit lagi)" in prompt
    assert "Maghrib 17:46" in prompt
    assert prompt.rstrip().endswith("maghrib jam berapa?")  # the question itself is still last


def test_question_about_another_city_still_reaches_the_llm_with_the_saved_location(prayer_chat):
    client, runner, set_location = prayer_chat
    set_location("s1")

    client.post("/chat", json={"message": "jadwal sholat surabaya", "session_id": "s1"})

    [call] = runner.calls  # not answered with Bandung's times as if they were Surabaya's
    assert "Lokasi tersimpan pengguna: Bandung, Jawa Barat" in call["history_prompt"]


def test_no_location_means_no_prayer_block_for_the_llm(prayer_chat):
    client, runner, _ = prayer_chat

    client.post("/chat", json={"message": "maghrib jam berapa?", "session_id": "nobody"})

    assert "JADWAL SHOLAT" not in runner.calls[0]["history_prompt"]


def test_ordinary_questions_do_not_get_the_block(prayer_chat):
    client, runner, set_location = prayer_chat
    set_location("s1")

    client.post("/chat", json={"message": "hadits tentang sholat subuh", "session_id": "s1"})

    assert "JADWAL SHOLAT" not in runner.calls[0]["history_prompt"]


def test_follow_up_keeps_the_prayer_block(prayer_chat):
    client, runner, set_location = prayer_chat
    set_location("s1")

    client.post("/chat", json={"message": "maghrib jam berapa?", "session_id": "s1"})
    client.post("/chat", json={"message": "kalau besok?", "session_id": "s1"})
    client.post("/chat", json={"message": "ayat tentang sabar", "session_id": "s1"})

    assert "JADWAL SHOLAT" in runner.calls[1]["history_prompt"]
    assert "JADWAL SHOLAT" not in runner.calls[2]["history_prompt"]


def test_prayer_answers_are_never_cached(prayer_chat):
    client, runner, set_location = prayer_chat
    for session in ("a", "b", "c"):
        set_location(session)
        client.post("/chat", json={"message": "maghrib jam berapa?", "session_id": session})
    assert len(runner.calls) == 3  # times pass, so yesterday's answer must not be reused


def test_prefs_endpoints_round_trip_the_location(prayer_chat):
    client, _, set_location = prayer_chat
    set_location("tg:5")

    body = client.get("/sessions/tg:5/prefs").json()
    assert (body["latitude"], body["longitude"]) == (-6.92, 107.61)  # stored to about 1 km
    assert (body["timezone"], body["location_label"]) == ("Asia/Jakarta", "Bandung, Jawa Barat")

    cleared = client.put("/sessions/tg:5/prefs", json={"latitude": None, "longitude": None}).json()
    assert cleared["latitude"] is None and cleared["timezone"] is None and cleared["location_label"] is None

    set_location("tg:5")
    assert client.delete("/sessions/tg:5/prefs").json()["latitude"] is None


@pytest.mark.parametrize(
    "bad",
    [
        {"latitude": -6.9},
        {"latitude": 200, "longitude": 10, "timezone": "Asia/Jakarta"},
        {"latitude": -6.9, "longitude": 107.6, "timezone": "Mars/Olympus"},
        {"latitude": -6.9, "longitude": 107.6},
    ],
)
def test_prefs_endpoint_rejects_bad_locations(prayer_chat, bad):
    client, _, _ = prayer_chat
    assert client.put("/sessions/tg:6/prefs", json=bad).status_code == 422
    assert client.get("/sessions/tg:6/prefs").json()["latitude"] is None


# --- Telegram -----------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, chat_id: int = 7, location=None) -> None:
        self.chat_id = chat_id
        self.location = location
        self.sent: list[dict] = []

    async def reply_text(self, text, parse_mode=None, reply_markup=None):
        self.sent.append({"text": text, "parse_mode": parse_mode, "markup": reply_markup})


class FakeUpdate:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeContext:
    def __init__(self, args=None, prayer=None) -> None:
        self.args = args or []
        self.bot_data = {"settings": object(), "prayer": prayer or StubClient()}


@pytest.fixture
def backend(monkeypatch):
    """Replace the HTTP calls to the API with an in-memory prefs dict."""
    state = {"prefs": {}, "calls": [], "error": None}

    async def fake_backend_call(settings, method, path, chat_id, json=None):
        state["calls"].append((method, path, json))
        if state["error"] is not None:
            raise state["error"]
        if method == "PUT":
            state["prefs"].update(json or {})
        return dict(state["prefs"])

    monkeypatch.setattr(telegram_bot, "backend_call", fake_backend_call)
    return state


def keyboard_requests_location(markup) -> bool:
    return isinstance(markup, ReplyKeyboardMarkup) and markup.keyboard[0][0].request_location is True


def test_lokasi_without_arguments_offers_the_share_button(backend):
    message = FakeMessage()
    run(telegram_bot.location_command(FakeUpdate(message), FakeContext()))

    [reply] = message.sent
    assert reply["text"] == telegram_bot.LOCATION_HELP and "/lokasi Bandung" in reply["text"]
    assert keyboard_requests_location(reply["markup"])
    assert backend["calls"] == []


def test_lokasi_city_saves_it_and_shows_todays_schedule(backend, fixed_now):
    message = FakeMessage()
    run(telegram_bot.location_command(FakeUpdate(message), FakeContext(["Kota", "Bandung"])))

    method, path, payload = backend["calls"][0]
    assert (method, path) == ("PUT", "prefs")
    assert payload["location_label"] == "Bandung, Jawa Barat" and payload["timezone"] == "Asia/Jakarta"
    assert (payload["latitude"], payload["longitude"]) == pytest.approx((-6.9222, 107.6069))

    confirmation, schedule = message.sent
    assert "Lokasi diatur ke: Bandung, Jawa Barat (WIB)" in confirmation["text"]
    assert "bukan alamat" in confirmation["text"]
    assert isinstance(confirmation["markup"], ReplyKeyboardRemove)  # the share button goes away
    assert "<b>🕌 Jadwal sholat — Bandung, Jawa Barat</b>" in schedule["text"]
    assert "• Ashar: 14:47" in schedule["text"]


def test_lokasi_with_province_picks_the_right_namesake(backend, fixed_now):
    message = FakeMessage()
    run(telegram_bot.location_command(FakeUpdate(message), FakeContext(["Depok,", "Jawa", "Barat"])))

    assert backend["calls"][0][2]["location_label"] == "Depok, Jawa Barat"


def test_lokasi_ambiguous_city_asks_which_one_and_saves_nothing(backend):
    message = FakeMessage()
    run(telegram_bot.location_command(FakeUpdate(message), FakeContext(["Depok"])))

    [reply] = message.sent
    assert "Depok, Jawa Barat" in reply["text"] and "Depok, DI Yogyakarta" in reply["text"]
    assert "/lokasi Depok, " in reply["text"]
    assert backend["calls"] == []


def test_lokasi_unknown_city_offers_the_share_button(backend):
    message = FakeMessage()
    run(telegram_bot.location_command(FakeUpdate(message), FakeContext(["Xqzvwk"])))

    [reply] = message.sent
    assert "tidak menemukan" in reply["text"]
    assert keyboard_requests_location(reply["markup"])
    assert backend["calls"] == []


def test_lokasi_hapus_clears_the_location(backend):
    backend["prefs"].update(LOCATION_JSON)
    message = FakeMessage()
    run(telegram_bot.location_command(FakeUpdate(message), FakeContext(["hapus"])))

    assert backend["calls"] == [("PUT", "prefs", {"latitude": None, "longitude": None})]
    assert "sudah dihapus" in message.sent[0]["text"]
    assert isinstance(message.sent[0]["markup"], ReplyKeyboardRemove)


def test_lokasi_reports_backend_failures_politely(backend, fixed_now):
    backend["error"] = RuntimeError("down")
    message = FakeMessage()
    run(telegram_bot.location_command(FakeUpdate(message), FakeContext(["Bandung"])))

    assert message.sent[0]["text"] == telegram_bot.ERROR_REPLY


def test_location_pin_is_saved_with_the_resolved_zone_and_nearest_city(backend, fixed_now):
    message = FakeMessage(location=SimpleNamespace(latitude=-6.93, longitude=107.62))
    run(telegram_bot.location_message_handler(FakeUpdate(message), FakeContext()))

    payload = backend["calls"][0][2]
    assert payload["location_label"] == "sekitar Bandung, Jawa Barat"
    assert (payload["latitude"], payload["longitude"], payload["timezone"]) == (-6.93, 107.62, "Asia/Jakarta")
    assert len(message.sent) == 2 and "Jadwal sholat" in message.sent[1]["text"]


def test_location_pin_far_from_any_listed_city_is_labelled_with_coordinates(backend, fixed_now):
    class Riyadh(StubClient):
        async def resolve_timezone(self, latitude, longitude):
            return "Asia/Riyadh"

    message = FakeMessage(location=SimpleNamespace(latitude=24.7136, longitude=46.6753))
    run(telegram_bot.location_message_handler(FakeUpdate(message), FakeContext(prayer=Riyadh())))

    payload = backend["calls"][0][2]
    assert payload["location_label"] == "24.71, 46.68" and payload["timezone"] == "Asia/Riyadh"


def test_location_pin_without_a_resolvable_zone_saves_nothing(backend):
    class NoZone(StubClient):
        async def resolve_timezone(self, latitude, longitude):
            return None

    message = FakeMessage(location=SimpleNamespace(latitude=0.0, longitude=-30.0))
    run(telegram_bot.location_message_handler(FakeUpdate(message), FakeContext(prayer=NoZone())))

    assert backend["calls"] == []
    assert "zona waktu" in message.sent[0]["text"]
    assert isinstance(message.sent[0]["markup"], ReplyKeyboardRemove)


def test_jadwal_without_a_location_asks_for_one(backend):
    message = FakeMessage()
    run(telegram_bot.schedule_command(FakeUpdate(message), FakeContext()))

    [reply] = message.sent
    assert reply["text"] == telegram_bot.NO_LOCATION_REPLY
    assert keyboard_requests_location(reply["markup"])


def test_jadwal_shows_today_and_tomorrow(backend, fixed_now):
    backend["prefs"].update(LOCATION_JSON)

    today = FakeMessage()
    run(telegram_bot.schedule_command(FakeUpdate(today), FakeContext()))
    tomorrow = FakeMessage()
    run(telegram_bot.schedule_command(FakeUpdate(tomorrow), FakeContext(["besok"])))

    assert "Sabtu, 17 Oktober 2026" in today.sent[0]["text"] and "Berikutnya" in today.sent[0]["text"]
    assert "Ahad, 18 Oktober 2026" in tomorrow.sent[0]["text"]


def test_settings_lists_the_location():
    assert "• Lokasi: belum diatur" in telegram_bot.format_prefs({})
    assert "• Lokasi: Bandung, Jawa Barat" in telegram_bot.format_prefs(LOCATION_JSON)


def test_application_handles_location_pins_before_the_text_only_catch_all():
    settings = replace(load_settings(env_file=None), telegram_bot_token="123:TEST")
    application = telegram_bot.build_application(settings)

    callbacks = [handler.callback for handler in application.handlers[0]]
    assert callbacks.index(telegram_bot.location_message_handler) < callbacks.index(telegram_bot.non_text_handler)
    assert telegram_bot.location_command in callbacks and telegram_bot.schedule_command in callbacks
    assert "prayer" in application.bot_data and "hijri" in application.bot_data
