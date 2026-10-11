from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from muslim_bot.agent import AGENT_INSTRUCTIONS, AgentReply
from muslim_bot.api import app, get_answer_cache, get_config_store, get_rate_limiter, get_runner, get_session_store, get_settings
from muslim_bot.answer_cache import AnswerCache
from muslim_bot.config import Settings
from muslim_bot.rate_limit import RateLimiter
from muslim_bot.runtime_config import RuntimeConfigStore
from muslim_bot.schemas import ChatRequest, Reference
from muslim_bot.session_store import SessionStore


def test_rate_limiter_allows_within_budget() -> None:
    limiter = RateLimiter(max_per_minute=2)

    limiter.check("ip-1")
    limiter.check("ip-1")


def test_rate_limiter_rejects_over_budget_with_retry_after() -> None:
    limiter = RateLimiter(max_per_minute=1)
    limiter.check("ip-1")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("ip-1")

    assert exc_info.value.status_code == 429
    assert "Retry-After" in (exc_info.value.headers or {})


def test_rate_limiter_tracks_keys_independently() -> None:
    limiter = RateLimiter(max_per_minute=1)
    limiter.check("ip-1")
    limiter.check("ip-2")


def test_rate_limiter_disabled_when_non_positive() -> None:
    limiter = RateLimiter(max_per_minute=0)
    for _ in range(100):
        limiter.check("ip-1")


def test_chat_request_rejects_oversized_message() -> None:
    with pytest.raises(Exception):
        ChatRequest(message="x" * 2001)


class FakeRunner:
    async def ask(self, message: str, *, session_id: str, history_prompt: str | None = None, language: str = "id") -> AgentReply:
        return AgentReply(answer="ok", references=[], used_tools=[])

    def try_fast_path(self, message: str, *, language: str = "id"):
        return None


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="gpt-test",
        openai_base_url=None,
        openai_organization=None,
        openai_project=None,
        openai_use_responses=None,
        admin_token="admin-test",
        runtime_config_path=tmp_path / "runtime_config.json",
        quran_source_db_path=tmp_path / "source-quran.db",
        quran_api_dir=tmp_path / "quran-api-go",
        quran_db_path=tmp_path / "quran.db",
        quran_mcp_command="go",
        quran_mcp_args=["run", "./cmd/mcp"],
        quran_mcp_cwd=tmp_path / "quran-api-go",
        hadith_api_dir=tmp_path / "hadith-api-go-run",
        hadith_db_path=tmp_path / "hadith.db",
        hadith_mcp_command="go",
        hadith_mcp_args=["run", "./cmd/mcp"],
        hadith_mcp_cwd=tmp_path / "hadith-api-go-run",
        hadith_seeder_command="go",
        hadith_seeder_args=["run", "./cmd/seeder"],
        hadith_seeder_cwd=tmp_path / "hadith-api-go-run",
        mcp_startup_timeout_seconds=1,
        agent_timeout_seconds=1,
        agent_max_turns=3,
        session_max_turns=3,
        session_db_path=tmp_path / "sessions.db",
        session_retention_days=7,
        chat_rate_limit_per_minute=30,
        telegram_bot_token=None,
        telegram_api_base="http://127.0.0.1:8000",
        telegram_proxy=None,
        telegram_connect_timeout=10.0,
        telegram_read_timeout=30.0,
        chat_cache_enabled=True,
        answer_cache_ttl_days=7,
    )


def test_chat_returns_429_when_rate_limited(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_turns=3)
    app.dependency_overrides[get_runner] = lambda: FakeRunner()
    app.dependency_overrides[get_answer_cache] = lambda: AnswerCache(settings.session_db_path, ttl_days=settings.answer_cache_ttl_days, enabled=settings.chat_cache_enabled)
    limiter = RateLimiter(max_per_minute=1)
    app.dependency_overrides[get_rate_limiter] = lambda: limiter

    try:
        client = TestClient(app)
        assert client.post("/chat", json={"message": "satu"}).status_code == 200
        response = client.post("/chat", json={"message": "dua"})
        assert response.status_code == 429
        assert "Retry-After" in response.headers
    finally:
        app.dependency_overrides.clear()


def test_chat_rejects_oversized_message_with_422(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_turns=3)
    app.dependency_overrides[get_runner] = lambda: FakeRunner()
    app.dependency_overrides[get_answer_cache] = lambda: AnswerCache(settings.session_db_path, ttl_days=settings.answer_cache_ttl_days, enabled=settings.chat_cache_enabled)

    try:
        client = TestClient(app)
        response = client.post("/chat", json={"message": "x" * 2001})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_agent_instructions_cover_scope_and_jailbreak() -> None:
    assert "koridor edukasi Islam" in AGENT_INSTRUCTIONS
    assert "mengabaikan" in AGENT_INSTRUCTIONS


def test_agent_instructions_cover_identity_and_limits() -> None:
    assert "Islamic Daris Bot" in AGENT_INSTRUCTIONS
    assert "6 kitab induk" in AGENT_INSTRUCTIONS
    assert "TIDAK tersedia" in AGENT_INSTRUCTIONS
    assert "tafsir" in AGENT_INSTRUCTIONS
