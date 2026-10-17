from __future__ import annotations

import asyncio

import pytest

from muslim_bot.telegram_bot import (
    build_reply,
    forward_header_for,
    markdown_to_html,
    split_message,
    telegram_session_id,
)


def test_telegram_session_id_prefixes_chat_id() -> None:
    assert telegram_session_id(12345) == "tg:12345"


def test_forward_header_identifies_telegram_user() -> None:
    assert forward_header_for(12345) == {"x-forwarded-for": "tg:12345"}


def test_markdown_headers_become_bold() -> None:
    assert markdown_to_html("## Hadis tentang Niat") == "<b>Hadis tentang Niat</b>"


def test_markdown_bold_and_code() -> None:
    assert markdown_to_html("Nilai **niat** itu `penting`") == "Nilai <b>niat</b> itu <code>penting</code>"


def test_markdown_single_asterisk_becomes_italic() -> None:
    assert markdown_to_html('*"Wahai orang yang sabar"*') == "<i>&quot;Wahai orang yang sabar&quot;</i>"


def test_markdown_escapes_html() -> None:
    assert markdown_to_html("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_markdown_quote_becomes_italic() -> None:
    assert markdown_to_html("> Semua amal tergantung niat") == "<i>Semua amal tergantung niat</i>"


def test_build_reply_marks_verified_references() -> None:
    reply = build_reply(
        "Jawaban singkat.",
        [
            {"label": "Sahih al-Bukhari #1", "verified": True},
            {"label": "Al-Mulk 67:999", "verified": False},
            {"label": "Al-Baqarah 2:255", "verified": None},
        ],
        [],
    )

    assert "✓ <i>Sahih al-Bukhari #1</i>" in reply
    assert "⚠ <i>Al-Mulk 67:999</i>" in reply
    assert "? <i>Al-Baqarah 2:255</i>" in reply
    assert "<b>Rujukan:</b>" in reply


def test_build_reply_includes_warnings() -> None:
    reply = build_reply("Jawaban.", [], ["Referensi 'X' tidak ditemukan."])

    assert "⚠ Referensi 'X' tidak ditemukan." in reply


def test_split_message_keeps_short_text_whole() -> None:
    assert split_message("halo") == ["halo"]


def test_split_message_splits_on_paragraphs() -> None:
    text = "a" * 4000 + "\n\n" + "b" * 100

    chunks = split_message(text)

    assert len(chunks) == 2
    assert all(len(chunk) <= 4096 for chunk in chunks)


def test_split_message_cuts_oversized_paragraph() -> None:
    chunks = split_message("x" * 9000)

    assert len(chunks) == 3
    assert all(len(chunk) <= 4096 for chunk in chunks)


class FakeMessage:
    def __init__(self, fail_html: bool = False) -> None:
        self.sent: list[tuple[str, object]] = []
        self.fail_html = fail_html
        self.calls = 0

    async def reply_text(self, text: str, parse_mode=None) -> None:
        from telegram.error import BadRequest

        self.calls += 1
        if self.fail_html and parse_mode is not None and self.calls == 1:
            raise BadRequest("can't parse")
        self.sent.append((text, parse_mode))


def test_send_reply_uses_html() -> None:
    from muslim_bot.telegram_bot import send_reply

    message = FakeMessage()
    asyncio.run(send_reply(message, "<b>halo</b>"))

    assert len(message.sent) == 1
    assert message.sent[0][1] is not None


def test_send_reply_falls_back_to_plain_text() -> None:
    from telegram.constants import ParseMode

    from muslim_bot.telegram_bot import send_reply

    message = FakeMessage(fail_html=True)
    asyncio.run(send_reply(message, "<b>halo</b>"))

    assert len(message.sent) == 1
    assert message.sent[0] == ("halo", None)
    assert ParseMode.HTML is not None


def test_build_application_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from muslim_bot.config import Settings
    from muslim_bot.telegram_bot import build_application

    settings = Settings(
        openai_api_key=None,
        openai_model="gpt-test",
        openai_base_url=None,
        openai_organization=None,
        openai_project=None,
        openai_use_responses=None,
        admin_token=None,
        runtime_config_path="runtime.json",
        quran_source_db_path="quran-src.db",
        quran_api_dir=".",
        quran_db_path="quran.db",
        quran_mcp_command="go",
        quran_mcp_args=[],
        quran_mcp_cwd=".",
        hadith_api_dir=".",
        hadith_db_path="hadith.db",
        hadith_mcp_command="go",
        hadith_mcp_args=[],
        hadith_mcp_cwd=".",
        hadith_seeder_command="go",
        hadith_seeder_args=[],
        hadith_seeder_cwd=".",
        mcp_startup_timeout_seconds=1,
        agent_timeout_seconds=1,
        agent_max_turns=1,
        session_max_turns=1,
        session_db_path="sessions.db",
        session_retention_days=7,
        chat_rate_limit_per_minute=0,
        telegram_bot_token=None,
        telegram_api_base="http://127.0.0.1:8021",
        telegram_proxy=None,
        telegram_connect_timeout=10.0,
        telegram_read_timeout=30.0,
        chat_cache_enabled=True,
        answer_cache_ttl_days=7,
    )

    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        build_application(settings)


class FakeChat:
    def __init__(self) -> None:
        self.calls = 0

    async def send_action(self, action) -> None:
        self.calls += 1


def test_typing_keepalive_stops_immediately_when_set() -> None:
    import asyncio

    from muslim_bot.telegram_bot import typing_keepalive

    async def go() -> None:
        chat = FakeChat()
        stop = asyncio.Event()
        stop.set()
        await asyncio.wait_for(typing_keepalive(chat, stop, interval=0.01), timeout=5)
        assert chat.calls == 0

    asyncio.run(go())


def test_typing_keepalive_repeats_until_stopped() -> None:
    import asyncio

    from muslim_bot.telegram_bot import typing_keepalive

    async def go() -> None:
        chat = FakeChat()
        stop = asyncio.Event()
        task = asyncio.create_task(typing_keepalive(chat, stop, interval=0.01))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=5)
        assert chat.calls >= 2

    asyncio.run(go())


def test_error_reply_mapping() -> None:
    from muslim_bot.telegram_bot import (
        ERROR_REPLY,
        LOOP_REPLY,
        RATE_LIMITED_REPLY,
        SERVICE_ERROR_REPLY,
        error_reply_for_status,
    )

    assert error_reply_for_status(508) == LOOP_REPLY
    assert error_reply_for_status(429) == RATE_LIMITED_REPLY
    assert error_reply_for_status(503) == SERVICE_ERROR_REPLY
    assert error_reply_for_status(504) == SERVICE_ERROR_REPLY
    assert error_reply_for_status(500) == ERROR_REPLY
    assert error_reply_for_status(None) == ERROR_REPLY


class ProgressFakeChat:
    async def send_action(self, action) -> None:
        pass


class ProgressFakeMessage:
    def __init__(self, text: str) -> None:
        from types import SimpleNamespace

        self.text = text
        self.chat = ProgressFakeChat()
        self.chat_id = 7
        self.from_user = SimpleNamespace(id=9)
        self.calls: list = []
        self.progress = None

    async def reply_text(self, text: str, parse_mode=None):
        from types import SimpleNamespace

        from muslim_bot.progress_messages import ALL_PROGRESS_TEXTS

        call = SimpleNamespace(text=text, parse_mode=parse_mode, deleted=False)

        async def delete() -> None:
            call.deleted = True

        call.delete = delete
        if text in ALL_PROGRESS_TEXTS and self.progress is None:
            self.progress = call
        else:
            self.calls.append(call)
        return call


def test_text_handler_shows_and_removes_progress(monkeypatch) -> None:
    import asyncio
    from types import SimpleNamespace

    import muslim_bot.telegram_bot as tg_bot

    async def fake_post(client, api_base, message, chat_id, user_id):
        assert api_base == "http://x"
        assert chat_id == 7
        assert user_id == 9
        await asyncio.sleep(0.1)
        return {"answer": "Jawaban.", "references": [], "warnings": []}

    monkeypatch.setattr(tg_bot, "post_chat", fake_post)
    monkeypatch.setattr(tg_bot, "PROGRESS_DELAY_SECONDS", 0.01)

    message = ProgressFakeMessage("hadis tentang niat")
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot_data={"settings": SimpleNamespace(telegram_api_base="http://x")})

    asyncio.run(tg_bot.text_handler(update, context))

    assert message.progress is not None
    assert message.progress.deleted is True
    assert any(call.text == "Jawaban." for call in message.calls)
