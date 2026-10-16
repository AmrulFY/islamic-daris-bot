from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from muslim_bot.agent import AgentReply
from muslim_bot.answer_cache import AnswerCache
from muslim_bot.api import app, get_answer_cache, get_config_store, get_runner, get_session_store, get_settings
from muslim_bot.runtime_config import RuntimeConfigStore
from muslim_bot.session_store import SessionStore
from tests.test_api import _settings


def test_prefs_default_to_empty() -> None:
    store = SessionStore(max_turns=5)

    prefs = store.get_prefs("tg:1")

    assert (prefs.language, prefs.style, prefs.note, prefs.history_turns) == (None, None, None, None)
    assert prefs.prompt_block() == ""
    assert not prefs.customizes_answer


def test_set_prefs_merges_and_none_resets() -> None:
    store = SessionStore(max_turns=5)

    store.set_prefs("tg:1", style="ringkas", language="en")
    prefs = store.set_prefs("tg:1", note="  panggil saya   Kak Ali ")

    assert (prefs.style, prefs.language, prefs.note) == ("ringkas", "en", "panggil saya Kak Ali")
    assert store.set_prefs("tg:1", style=None).style is None
    assert store.get_prefs("tg:1").language == "en"


def test_set_prefs_validates_values() -> None:
    store = SessionStore(max_turns=5)

    for bad in ({"language": "fr"}, {"style": "puitis"}, {"note": "x" * 201}, {"history_turns": 6}, {"history_turns": -1}, {"nope": 1}):
        with pytest.raises(ValueError):
            store.set_prefs("tg:1", **bad)
    assert store.get_prefs("tg:1").language is None


def test_prefs_are_isolated_per_session() -> None:
    store = SessionStore(max_turns=5)

    store.set_prefs("tg:1", style="lengkap")

    assert store.get_prefs("tg:2").style is None


def test_prompt_block_guards_against_overriding_rules() -> None:
    store = SessionStore(max_turns=5)

    block = store.set_prefs("tg:1", style="ringkas", note="pakai bahasa santai").prompt_block()

    assert "ringkas" in block
    assert "pakai bahasa santai" in block
    assert "tidak boleh mengubah aturan sumber" in block


def test_history_turns_pref_limits_history_prompt() -> None:
    store = SessionStore(max_turns=5)
    store.append("tg:1", "lama", "jawab lama")
    store.append("tg:1", "baru", "jawab baru")

    store.set_prefs("tg:1", history_turns=1)
    limited = store.history_prompt("tg:1", "lanjut")
    store.set_prefs("tg:1", history_turns=0)
    none = store.history_prompt("tg:1", "lanjut")

    assert "lama" not in limited and "baru" in limited
    assert none == "lanjut"


def test_clear_history_keeps_prefs() -> None:
    store = SessionStore(max_turns=5)
    store.append("tg:1", "satu", "jawab")
    store.set_prefs("tg:1", style="ringkas")

    assert store.clear_history("tg:1") == 1

    assert store.recent_turns("tg:1") == []
    assert store.get_prefs("tg:1").style == "ringkas"


def test_clear_prefs_and_purge(tmp_path: Path) -> None:
    store = SessionStore(max_turns=5, db_path=tmp_path / "s.db")
    store.set_prefs("tg:1", style="ringkas")
    store.clear_prefs("tg:1")
    assert store.get_prefs("tg:1").style is None

    store.set_prefs("tg:2", note="halo")
    store._conn.execute("update sessions set updated_at = '2000-01-01T00:00:00+00:00'")
    store._conn.commit()
    store.purge_inactive(days=1)

    assert store.get_prefs("tg:2").note is None


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ask(self, message, *, session_id, history_prompt=None, language="id", recent_turns=None):
        self.calls.append({"history_prompt": history_prompt, "language": language})
        return AgentReply(answer="Jawaban.", references=[], used_tools=[])

    def try_fast_path(self, message, *, language="id"):
        return None


@pytest.fixture
def api(tmp_path: Path):
    settings = _settings(tmp_path)
    store = SessionStore(max_turns=3)
    runner = RecordingRunner()
    cache = AnswerCache(settings.session_db_path, ttl_days=7, enabled=True)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: store
    app.dependency_overrides[get_runner] = lambda: runner
    app.dependency_overrides[get_answer_cache] = lambda: cache
    yield TestClient(app), runner
    app.dependency_overrides.clear()


def test_prefs_endpoints_roundtrip(api) -> None:
    client, _ = api

    assert client.get("/sessions/tg:1/prefs").json()["style"] is None
    response = client.put("/sessions/tg:1/prefs", json={"style": "ringkas", "history_turns": 2})
    assert response.status_code == 200
    body = response.json()
    assert (body["style"], body["history_turns"], body["max_history_turns"]) == ("ringkas", 2, 3)

    # fields not in the request are untouched; explicit null resets
    client.put("/sessions/tg:1/prefs", json={"note": "halo"})
    body = client.put("/sessions/tg:1/prefs", json={"style": None}).json()
    assert (body["style"], body["note"], body["history_turns"]) == (None, "halo", 2)

    assert client.delete("/sessions/tg:1/prefs").json()["note"] is None


def test_prefs_endpoint_rejects_invalid_values(api) -> None:
    client, _ = api

    assert client.put("/sessions/tg:1/prefs", json={"language": "fr"}).status_code == 422
    assert client.put("/sessions/tg:1/prefs", json={"history_turns": 99}).status_code == 422


def test_history_delete_endpoint(api) -> None:
    client, _ = api
    client.post("/chat", json={"message": "satu", "session_id": "tg:1"})

    assert client.delete("/sessions/tg:1/history").json()["deleted_turns"] == 1
    assert client.delete("/sessions/tg:1/history").json()["deleted_turns"] == 0


def test_chat_applies_session_prefs(api) -> None:
    client, runner = api
    client.put("/sessions/tg:1/prefs", json={"language": "en", "style": "ringkas", "note": "sapa saya Kak Ali"})

    client.post("/chat", json={"message": "hadis niat", "session_id": "tg:1", "language": "id"})

    call = runner.calls[0]
    assert call["language"] == "en"
    assert "ringkas" in call["history_prompt"]
    assert "sapa saya Kak Ali" in call["history_prompt"]
    assert call["history_prompt"].endswith("hadis niat")


def test_customized_answers_bypass_shared_cache(api) -> None:
    client, runner = api

    client.post("/chat", json={"message": "hadis niat", "session_id": "plain-a"})
    client.post("/chat", json={"message": "hadis niat", "session_id": "plain-b"})
    assert len(runner.calls) == 1  # second answered from cache

    client.put("/sessions/tg:9/prefs", json={"style": "lengkap"})
    client.post("/chat", json={"message": "hadis niat", "session_id": "tg:9"})
    assert len(runner.calls) == 2  # not served from cache

    client.post("/chat", json={"message": "hadis niat", "session_id": "plain-c"})
    assert len(runner.calls) == 2  # customized answer was not cached for others


def test_parse_pref_args() -> None:
    from muslim_bot.telegram_bot import parse_pref_args

    assert parse_pref_args("language", ["EN"]) == (True, "en")
    assert parse_pref_args("language", []) == (False, None)
    assert parse_pref_args("note", ["hapus"]) == (True, None)
    assert parse_pref_args("note", ["Panggil", "saya", "Ali"]) == (True, "Panggil saya Ali")
    assert parse_pref_args("history_turns", ["3"]) == (True, 3)
    assert parse_pref_args("history_turns", ["tiga"]) == (False, None)


class CommandMessage:
    chat_id = 7

    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply_text(self, text: str, parse_mode=None) -> None:
        self.replies.append(text)


def _run_command(command, message, args, monkeypatch, backend):
    import muslim_bot.telegram_bot as tg_bot

    calls: list[tuple] = []

    async def fake_backend(settings, method, path, chat_id, json=None):
        calls.append((method, path, chat_id, json))
        return backend(method, path, json)

    monkeypatch.setattr(tg_bot, "backend_call", fake_backend)
    context = SimpleNamespace(bot_data={"settings": SimpleNamespace(telegram_api_base="http://x")}, args=args)
    asyncio.run(command(SimpleNamespace(message=message), context))
    return calls


def test_pref_command_sends_field_to_backend(monkeypatch) -> None:
    from muslim_bot.telegram_bot import make_pref_command

    message = CommandMessage()
    calls = _run_command(
        make_pref_command("style", "/gaya ringkas|lengkap", "Gaya jawaban"),
        message,
        ["Ringkas"],
        monkeypatch,
        lambda method, path, json: {"style": "ringkas"},
    )

    assert calls == [("PUT", "prefs", 7, {"style": "ringkas"})]
    assert message.replies == ["Gaya jawaban diatur ke: ringkas"]


def test_pref_command_shows_validation_error(monkeypatch) -> None:
    from muslim_bot.telegram_bot import make_pref_command

    def reject(method, path, json):
        raise ValueError("style must be one of: ringkas, lengkap")

    message = CommandMessage()
    _run_command(make_pref_command("style", "/gaya ringkas|lengkap", "Gaya"), message, ["puitis"], monkeypatch, reject)

    assert "style must be one of" in message.replies[0]
    assert "Cara pakai" in message.replies[0]


def test_reset_and_forget_commands(monkeypatch) -> None:
    from muslim_bot.telegram_bot import forget_command, reset_command

    message = CommandMessage()
    calls = _run_command(reset_command, message, [], monkeypatch, lambda m, p, j: {"deleted_turns": 4})
    assert calls == [("DELETE", "history", 7, None)]
    assert "4 percakapan" in message.replies[0]

    message = CommandMessage()
    calls = _run_command(forget_command, message, [], monkeypatch, lambda m, p, j: {})
    assert [(c[0], c[1]) for c in calls] == [("DELETE", "history"), ("DELETE", "prefs")]


@pytest.mark.parametrize(
    ("text", "field", "value"),
    [
        ("jawab singkat saja", "style", "ringkas"),
        ("Tolong jawabannya ringkas aja ya", "style", "ringkas"),
        ("lengkap", "style", "lengkap"),
        ("jawab yang detail dong!", "style", "lengkap"),
        ("jawab dalam bahasa Inggris", "language", "en"),
        ("pakai bahasa indonesia saja", "language", "id"),
        ("reply in English", "language", "en"),
        ("panggil saya Kak Ali ya", "note", "Panggil pengguna dengan nama Kak Ali"),
        ("ingat bahwa saya pemula dalam tajwid", "note", "saya pemula dalam tajwid"),
    ],
)
def test_detect_pref_intent_matches(text: str, field: str, value: str) -> None:
    from muslim_bot.telegram_bot import detect_pref_intent

    intent = detect_pref_intent(text)

    assert intent is not None
    assert (intent.field, intent.value) == (field, value)


@pytest.mark.parametrize(
    "text",
    [
        "hadis tentang niat",
        "jelaskan niat dengan singkat dan beri dalilnya",
        "bagaimana hukum jawab singkat dalam salat",
        "apa itu bahasa inggris dalam islam",
        "/gaya ringkas",
        "ingat",
        "",
    ],
)
def test_detect_pref_intent_ignores_real_questions(text: str) -> None:
    from muslim_bot.telegram_bot import detect_pref_intent

    assert detect_pref_intent(text) is None


class PrefChatMessage(CommandMessage):
    def __init__(self, text: str = "") -> None:
        super().__init__()
        self.text = text


def test_pref_message_asks_confirmation_then_saves(monkeypatch) -> None:
    import muslim_bot.telegram_bot as tg_bot

    async def fail_post(*args, **kwargs):
        raise AssertionError("a preference message must not reach the LLM backend")

    monkeypatch.setattr(tg_bot, "post_chat", fail_post)
    chat_data: dict = {}
    context = SimpleNamespace(
        bot_data={"settings": SimpleNamespace(telegram_api_base="http://x")}, args=[], chat_data=chat_data
    )

    message = PrefChatMessage("jawab singkat saja")
    asyncio.run(tg_bot.text_handler(SimpleNamespace(message=message), context))
    assert "/ya" in message.replies[0] and "ringkas" in message.replies[0]
    assert tg_bot.PENDING_KEY in chat_data

    calls: list[tuple] = []

    async def fake_backend(settings, method, path, chat_id, json=None):
        calls.append((method, path, json))
        return {}

    monkeypatch.setattr(tg_bot, "backend_call", fake_backend)
    confirm = PrefChatMessage()
    asyncio.run(tg_bot.confirm_pref_command(SimpleNamespace(message=confirm), context))

    assert calls == [("PUT", "prefs", {"style": "ringkas"})]
    assert "Tersimpan" in confirm.replies[0]
    assert tg_bot.PENDING_KEY not in chat_data


def test_pref_message_can_be_cancelled_and_confirm_without_pending(monkeypatch) -> None:
    import muslim_bot.telegram_bot as tg_bot

    chat_data: dict = {tg_bot.PENDING_KEY: tg_bot.PrefIntent("style", "ringkas", "Gaya jawaban: ringkas")}
    context = SimpleNamespace(bot_data={}, args=[], chat_data=chat_data)

    cancel = PrefChatMessage()
    asyncio.run(tg_bot.cancel_pref_command(SimpleNamespace(message=cancel), context))
    assert cancel.replies == ["Dibatalkan."]

    confirm = PrefChatMessage()
    asyncio.run(tg_bot.confirm_pref_command(SimpleNamespace(message=confirm), context))
    assert "Tidak ada pengaturan" in confirm.replies[0]
