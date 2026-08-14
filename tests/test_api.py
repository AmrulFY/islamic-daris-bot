from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from muslim_bot.agent import AgentReply
from muslim_bot.api import app, get_config_store, get_runner, get_session_store, get_settings
from muslim_bot.config import Settings
from muslim_bot.runtime_config import RuntimeConfigStore
from muslim_bot.schemas import Reference
from muslim_bot.session_store import SessionStore


class FakeRunner:
    async def ask(self, message: str, *, session_id: str, history_prompt: str | None = None, language: str = "id") -> AgentReply:
        return AgentReply(
            answer="Amal bergantung pada niat. Rujukan: Sahih al-Bukhari #1.",
            references=[Reference(type="hadith", label="Sahih al-Bukhari #1", source_id="1")],
            used_tools=["search_hadith"],
        )


def _settings(tmp_path: Path, api_key: str | None = "test-key") -> Settings:
    return Settings(
        openai_api_key=api_key,
        openai_model="gpt-test",
        openai_base_url=None,
        openai_organization=None,
        openai_project=None,
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
        session_max_turns=3,
    )


def test_chat_returns_agent_response(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_turns=3)
    app.dependency_overrides[get_runner] = lambda: FakeRunner()

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
