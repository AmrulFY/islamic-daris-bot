from __future__ import annotations

import asyncio
from datetime import date

import httpx
import pytest

from muslim_bot import hijri
from muslim_bot.hijri import (
    SOURCE_COMPUTED,
    SOURCE_FALLBACK,
    SOURCE_TABLE,
    HijriCalendar,
    calculate_hijri,
    parse_api_response,
)

TABLE_PAYLOAD = {
    "input": {"date": "2026-10-17", "calendar": "gregorian", "tz": "Asia/Jakarta"},
    "output": {
        "date": "1448-05-06",
        "calendar": "hijri",
        "day": 6,
        "month": 5,
        "month_name": "Jumadil Awal",
        "year": 1448,
        "weekday": "Sabtu",
    },
    "source": "mabims",
    "warnings": [],
}


def computed_payload(month: int = 9, name: str = "Ramadhan") -> dict:
    return {
        "output": {"day": 1, "month": month, "month_name": name, "year": 1448, "weekday": "Senin"},
        "source": "mabims-computed",
        "warnings": ["computed"],
    }


def make_calendar(handler, **kwargs) -> tuple[HijriCalendar, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    return HijriCalendar("https://example.test/api/v1/", client=client, **kwargs), requests


def run(coro):
    return asyncio.run(coro)


def test_parse_table_response_is_exact():
    result = parse_api_response(TABLE_PAYLOAD)
    assert (result.year, result.month, result.day) == (1448, 5, 6)
    assert result.source == SOURCE_TABLE
    assert not result.approximate
    assert result.notes == ()
    assert result.format() == "Sabtu, 6 Jumadil Awal 1448 H"


def test_parse_computed_response_is_marked_approximate():
    result = parse_api_response(computed_payload(month=2, name="Safar"))
    assert result.source == SOURCE_COMPUTED
    assert result.approximate
    assert hijri.APPROXIMATE_NOTE in result.notes
    assert hijri.ISBAT_NOTE not in result.notes


@pytest.mark.parametrize("month,name", [(9, "Ramadhan"), (10, "Syawal"), (12, "Dzulhijjah")])
def test_isbat_note_for_key_months(month, name):
    payload = {**TABLE_PAYLOAD, "output": {**TABLE_PAYLOAD["output"], "month": month, "month_name": name}}
    assert hijri.ISBAT_NOTE in parse_api_response(payload).notes


@pytest.mark.parametrize("payload", [{}, {"source": "mabims"}, {"output": {"year": 1448}, "source": "mabims"}])
def test_parse_rejects_malformed_response(payload):
    with pytest.raises(ValueError):
        parse_api_response(payload)


def test_calculate_hijri_matches_known_dates():
    result = calculate_hijri(date(2026, 10, 17))
    assert (result.year, result.month, result.day) == (1448, 5, 6)
    assert result.month_name == "Jumadil Awal"
    assert result.weekday == "Sabtu"
    assert result.source == SOURCE_FALLBACK
    assert result.approximate
    assert hijri.FALLBACK_NOTE in result.notes
    assert calculate_hijri(date(2027, 2, 8)).weekday == "Senin"
    assert calculate_hijri(date(2026, 10, 18)).weekday == "Ahad"


def test_calculate_hijri_out_of_range_returns_none():
    assert calculate_hijri(date(2200, 1, 1)) is None
    assert calculate_hijri(date(1800, 1, 1)) is None


def test_api_result_is_used_and_cached():
    calendar, requests = make_calendar(lambda request: httpx.Response(200, json=TABLE_PAYLOAD))
    day = date(2026, 10, 17)
    first = run(calendar.convert(day))
    second = run(calendar.convert(day))
    assert first == second
    assert first.source == SOURCE_TABLE
    assert len(requests) == 1
    assert str(requests[0].url) == "https://example.test/api/v1/today/2026-10-17"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"unexpected": True}),
    ],
)
def test_falls_back_to_local_calculation_when_api_misbehaves(response):
    calendar, _ = make_calendar(lambda request: response)
    result = run(calendar.convert(date(2026, 10, 17)))
    assert result.source == SOURCE_FALLBACK
    assert (result.year, result.month, result.day) == (1448, 5, 6)


def test_falls_back_when_api_unreachable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    calendar, _ = make_calendar(refuse)
    assert run(calendar.convert(date(2026, 10, 17))).source == SOURCE_FALLBACK


def test_cooldown_skips_api_after_failure_then_retries(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(hijri.time, "monotonic", lambda: now[0])
    state = {"healthy": False}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=TABLE_PAYLOAD) if state["healthy"] else httpx.Response(500)

    calendar, requests = make_calendar(handler)
    day = date(2026, 10, 17)
    assert run(calendar.convert(day)).source == SOURCE_FALLBACK
    assert run(calendar.convert(day)).source == SOURCE_FALLBACK
    assert len(requests) == 1  # second call skipped the API during the cooldown

    state["healthy"] = True
    now[0] += hijri.FAILURE_COOLDOWN_SECONDS + 1
    assert run(calendar.convert(day)).source == SOURCE_TABLE
    assert len(requests) == 2


def test_cache_expires(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(hijri.time, "monotonic", lambda: now[0])
    calendar, requests = make_calendar(lambda request: httpx.Response(200, json=TABLE_PAYLOAD))
    day = date(2026, 10, 17)
    run(calendar.convert(day))
    now[0] += hijri.CACHE_TTL_SECONDS + 1
    run(calendar.convert(day))
    assert len(requests) == 2


def test_disabled_api_never_called():
    calendar, requests = make_calendar(lambda request: httpx.Response(200, json=TABLE_PAYLOAD), enabled=False)
    assert run(calendar.convert(date(2026, 10, 17))).source == SOURCE_FALLBACK
    assert requests == []


def test_returns_none_when_api_down_and_date_out_of_range():
    calendar, _ = make_calendar(lambda request: httpx.Response(500))
    assert run(calendar.convert(date(2200, 1, 1))) is None


def test_settings_defaults(monkeypatch):
    from muslim_bot.config import load_settings

    for name in ("HIJRI_API_ENABLED", "HIJRI_API_BASE", "HIJRI_API_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(env_file=None)
    assert settings.hijri_api_enabled is True
    assert settings.hijri_api_base == "https://api.mabims.dev/api/v1"
    assert settings.hijri_api_timeout_seconds == 3.0


# --- "tanggal hijriyah hari ini" shortcut ---------------------------------------------

TODAY = date(2026, 10, 17)


@pytest.mark.parametrize(
    "message",
    [
        "tanggal hijriyah hari ini",
        "Hari ini tanggal berapa hijriyah?",
        "tanggal hijriah sekarang",
        "hari ini tanggal berapa hijri",
        "tanggal hijriyah",
        "hijriyah",
        "kalender islam hari ini",
        "Tanggal hijriyah hari ini berapa ya?",
    ],
)
def test_detects_todays_hijri_date_queries(message):
    assert hijri.detect_hijri_query(message)


@pytest.mark.parametrize(
    "message",
    [
        "",
        "hari ini tanggal berapa",  # no Hijri cue: not ours
        "apa itu kalender hijriyah",
        "kapan 1 ramadan 1448 hijriyah",
        "tanggal hijriyah besok",
        "tanggal hijriyah 17 oktober 2026",
        "jelaskan sejarah kalender hijriyah",
        "tanggal hijriyah bulan ramadhan",
        "hadits tentang hijrah ke madinah hari ini",
        "tanggal hijriyah hari ini " + "x" * 80,
    ],
)
def test_ignores_other_questions(message):
    assert not hijri.detect_hijri_query(message)


def test_format_today_message_exact_date_has_no_notes():
    message = hijri.format_today_message(parse_api_response(TABLE_PAYLOAD), TODAY)
    assert "**Sabtu, 6 Jumadil Awal 1448 H**" in message
    assert "Masehi: Sabtu, 17 Oktober 2026" in message
    assert ">" not in message


def test_format_today_message_shows_fallback_notice():
    message = hijri.format_today_message(calculate_hijri(TODAY), TODAY)
    assert f"> {hijri.FALLBACK_NOTE}" in message


class StubCalendar:
    def __init__(self, result):
        self.result = result
        self.days: list[date] = []

    async def convert(self, day=None):
        self.days.append(day)
        return self.result


@pytest.fixture
def chat_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from muslim_bot import api
    from tests.test_api import FakeRunner, _settings
    from muslim_bot.answer_cache import AnswerCache
    from muslim_bot.rate_limit import RateLimiter
    from muslim_bot.runtime_config import RuntimeConfigStore
    from muslim_bot.session_store import SessionStore

    settings = _settings(tmp_path)
    store = SessionStore(max_turns=3)
    # Own limiter: the app-wide one is shared by the whole test session and would run out.
    limiter = RateLimiter(1000)
    api.app.dependency_overrides[api.get_rate_limiter] = lambda: limiter
    api.app.dependency_overrides[api.get_settings] = lambda: settings
    api.app.dependency_overrides[api.get_config_store] = lambda: RuntimeConfigStore(settings)
    api.app.dependency_overrides[api.get_session_store] = lambda: store
    api.app.dependency_overrides[api.get_runner] = lambda: FakeRunner()
    api.app.dependency_overrides[api.get_answer_cache] = lambda: AnswerCache(
        settings.session_db_path, ttl_days=7, enabled=True
    )
    monkeypatch.setattr(api, "today_in", lambda: TODAY)
    yield TestClient(api.app), api, monkeypatch
    api.app.dependency_overrides.clear()


def test_chat_answers_hijri_date_without_llm(chat_client):
    client, api, monkeypatch = chat_client
    calendar = StubCalendar(parse_api_response(TABLE_PAYLOAD))
    monkeypatch.setattr(api, "get_hijri_calendar", lambda: calendar)

    response = client.post("/chat", json={"message": "tanggal hijriyah hari ini"})

    assert response.status_code == 200
    body = response.json()
    assert "6 Jumadil Awal 1448 H" in body["answer"]
    assert body["used_tools"] == ["hijri_calendar"]
    assert calendar.days == [TODAY]


def test_chat_stream_answers_hijri_date_without_llm(chat_client):
    client, api, monkeypatch = chat_client
    monkeypatch.setattr(api, "get_hijri_calendar", lambda: StubCalendar(parse_api_response(TABLE_PAYLOAD)))

    response = client.post("/chat/stream", json={"message": "tanggal hijriyah hari ini"})

    assert response.status_code == 200
    assert "6 Jumadil Awal 1448 H" in response.text
    assert "hijri_calendar" in response.text


def test_chat_reports_when_no_calendar_source_works(chat_client):
    client, api, monkeypatch = chat_client
    monkeypatch.setattr(api, "get_hijri_calendar", lambda: StubCalendar(None))

    body = client.post("/chat", json={"message": "tanggal hijriyah hari ini"}).json()

    assert body["answer"] == hijri.UNAVAILABLE_MESSAGE


def test_chat_other_messages_still_reach_the_agent(chat_client):
    client, api, monkeypatch = chat_client
    calendar = StubCalendar(parse_api_response(TABLE_PAYLOAD))
    monkeypatch.setattr(api, "get_hijri_calendar", lambda: calendar)

    body = client.post("/chat", json={"message": "hadis tentang niat"}).json()

    assert body["answer"].startswith("Amal")
    assert calendar.days == []


class FakeTelegramMessage:
    def __init__(self):
        self.sent: list[str] = []

    async def reply_text(self, text, parse_mode=None):
        self.sent.append(text)


class FakeUpdate:
    def __init__(self):
        self.message = FakeTelegramMessage()


class FakeContext:
    def __init__(self, calendar):
        self.bot_data = {"hijri": calendar}


def test_telegram_hijri_command_replies_with_formatted_date(monkeypatch):
    from muslim_bot import telegram_bot

    monkeypatch.setattr(telegram_bot, "today_in", lambda: TODAY)
    update = FakeUpdate()
    run(telegram_bot.hijri_command(update, FakeContext(StubCalendar(parse_api_response(TABLE_PAYLOAD)))))

    [reply] = update.message.sent
    assert "<b>Sabtu, 6 Jumadil Awal 1448 H</b>" in reply
    assert "Masehi: Sabtu, 17 Oktober 2026" in reply


def test_telegram_hijri_command_when_unavailable(monkeypatch):
    from muslim_bot import telegram_bot

    update = FakeUpdate()
    run(telegram_bot.hijri_command(update, FakeContext(StubCalendar(None))))

    assert update.message.sent == [hijri.UNAVAILABLE_MESSAGE]
