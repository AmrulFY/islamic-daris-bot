from __future__ import annotations

from pathlib import Path

from muslim_bot.config import Settings
from muslim_bot.runtime_config import RuntimeConfigStore


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="sk-test",
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
        agent_timeout_seconds=9,
        agent_max_turns=5,
        session_max_turns=4,
        session_db_path=tmp_path / "sessions.db",
        session_retention_days=7,
        chat_rate_limit_per_minute=30,
        telegram_bot_token=None,
        telegram_api_base="http://127.0.0.1:8021",
        telegram_proxy=None,
        telegram_connect_timeout=10.0,
        telegram_read_timeout=30.0,
        chat_cache_enabled=True,
        answer_cache_ttl_days=7,
    )


def test_default_runtime_config_from_settings(tmp_path: Path) -> None:
    config = RuntimeConfigStore(_settings(tmp_path)).load()

    assert config.llm.model == "gpt-test"
    assert config.llm.api_key == "sk-test"
    assert [server.id for server in config.mcp_servers] == ["quran", "hadith"]
    assert all(server.timeout_seconds == 1 for server in config.mcp_servers)


def test_preserves_masked_secret_on_save(tmp_path: Path) -> None:
    store = RuntimeConfigStore(_settings(tmp_path))
    original = store.save(store.load())
    redacted = store.load_redacted()

    saved = store.save_preserving_masked_secrets(redacted)

    assert saved.llm.api_key == original.llm.api_key
    assert store.load().llm.api_key == "sk-test"
