from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from muslim_bot.agent import AgentReply
from muslim_bot.api import app, get_answer_cache, get_config_store, get_runner, get_session_store, get_settings
from muslim_bot.answer_cache import AnswerCache
from muslim_bot.config import Settings
from muslim_bot.runtime_config import RuntimeConfigStore
from muslim_bot.schemas import Reference
from muslim_bot.session_store import SessionStore


class FakeRunner:
    async def ask(self, message: str, *, session_id: str, history_prompt: str | None = None, language: str = "id", recent_turns=None) -> AgentReply:
        return AgentReply(
            answer="Amal bergantung pada niat. Rujukan: Sahih al-Bukhari #1.",
            references=[Reference(type="hadith", label="Sahih al-Bukhari #1", source_id="1")],
            used_tools=["search_hadith"],
        )

    def try_fast_path(self, message: str, *, language: str = "id"):
        return None


def _settings(tmp_path: Path, api_key: str | None = "test-key") -> Settings:
    return Settings(
        openai_api_key=api_key,
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
        chat_rate_limit_per_minute=1000,
        telegram_bot_token=None,
        telegram_api_base="http://127.0.0.1:8000",
        telegram_proxy=None,
        telegram_connect_timeout=10.0,
        telegram_read_timeout=30.0,
        chat_cache_enabled=True,
        answer_cache_ttl_days=7,
    )


def test_chat_returns_agent_response(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_turns=3)
    app.dependency_overrides[get_runner] = lambda: FakeRunner()
    app.dependency_overrides[get_answer_cache] = lambda: AnswerCache(settings.session_db_path, ttl_days=settings.answer_cache_ttl_days, enabled=settings.chat_cache_enabled)

    client = TestClient(app)
    response = client.post("/chat", json={"message": "hadis tentang niat"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Amal")
    assert body["references"][0]["type"] == "hadith"
    assert body["used_tools"] == ["search_hadith"]

    app.dependency_overrides.clear()


def test_chat_requires_openai_key(tmp_path: Path) -> None:
    settings = _settings(tmp_path, api_key=None)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_turns=3)
    app.dependency_overrides[get_runner] = lambda: FakeRunner()
    app.dependency_overrides[get_answer_cache] = lambda: AnswerCache(settings.session_db_path, ttl_days=settings.answer_cache_ttl_days, enabled=settings.chat_cache_enabled)

    client = TestClient(app)
    response = client.post("/chat", json={"message": "ayat tentang sabar"})

    assert response.status_code == 503

    app.dependency_overrides.clear()


def test_admin_config_requires_token(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)

    client = TestClient(app)
    response = client.get("/api/config")

    assert response.status_code == 401

    app.dependency_overrides.clear()


def test_admin_config_returns_masked_secret(tmp_path: Path) -> None:
    settings = _settings(tmp_path, api_key="sk-test")
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)

    client = TestClient(app)
    response = client.get("/api/config", headers={"x-admin-token": "admin-test"})

    assert response.status_code == 200
    assert response.json()["config"]["llm"]["api_key"] == "********"

    app.dependency_overrides.clear()


def _wired_client(tmp_path: Path, runner):
    settings = _settings(tmp_path)
    shared_store = SessionStore(max_turns=3)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: shared_store
    app.dependency_overrides[get_runner] = lambda: runner
    app.dependency_overrides[get_answer_cache] = lambda: AnswerCache(
        settings.session_db_path,
        ttl_days=settings.answer_cache_ttl_days,
        enabled=settings.chat_cache_enabled,
    )
    return TestClient(app)


def test_chat_uses_fast_path_without_agent(tmp_path: Path) -> None:
    fast_reply = AgentReply(
        answer="## 📜 Sahih al-Bukhari #1",
        references=[Reference(type="hadith", label="Sahih al-Bukhari #1", source_id="1", verified=True)],
        used_tools=["local_db"],
    )

    class FastRunner:
        async def ask(self, *args, **kwargs):
            raise AssertionError("agent must not run on fast-path hits")

        def try_fast_path(self, message: str, *, language: str = "id"):
            return fast_reply

    client = _wired_client(tmp_path, FastRunner())
    try:
        response = client.post("/chat", json={"message": "Sahih al-Bukhari #1"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "## 📜 Sahih al-Bukhari #1"
        assert body["used_tools"] == ["local_db"]
        assert body["references"][0]["verified"] is True
    finally:
        app.dependency_overrides.clear()


def test_chat_caches_agent_reply_for_new_sessions(tmp_path: Path) -> None:
    calls = []

    class CountingRunner:
        async def ask(self, message: str, **kwargs) -> AgentReply:
            calls.append(message)
            return AgentReply(answer="Jawaban cache.", references=[], used_tools=["search_quran"])

        def try_fast_path(self, message: str, *, language: str = "id"):
            return None

    client = _wired_client(tmp_path, CountingRunner())
    try:
        first = client.post("/chat", json={"message": "pertanyaan unik"})
        second = client.post("/chat", json={"message": "pertanyaan unik"})
        assert first.status_code == 200
        assert second.status_code == 200
        assert calls == ["pertanyaan unik"]
        assert second.json()["answer"] == "Jawaban cache."
    finally:
        app.dependency_overrides.clear()


def test_chat_skips_cache_when_session_has_history(tmp_path: Path) -> None:
    calls = []

    class CountingRunner:
        async def ask(self, message: str, **kwargs) -> AgentReply:
            calls.append(message)
            return AgentReply(answer=f"Jawaban {len(calls)}.", references=[], used_tools=[])

        def try_fast_path(self, message: str, *, language: str = "id"):
            return None

    client = _wired_client(tmp_path, CountingRunner())
    try:
        first = client.post("/chat", json={"message": "halo"})
        session_id = first.json()["session_id"]
        second = client.post("/chat", json={"message": "halo", "session_id": session_id})
        assert calls == ["halo", "halo"]
        assert second.json()["answer"] == "Jawaban 2."
    finally:
        app.dependency_overrides.clear()


def test_package_logging_is_configured_on_import() -> None:
    import logging

    import muslim_bot.api  # noqa: F401 - import configures logging as a side effect

    package_logger = logging.getLogger("muslim_bot")

    assert package_logger.handlers, "muslim_bot logger should have a handler"
    assert package_logger.level <= logging.INFO
