from __future__ import annotations

from pathlib import Path

from muslim_bot.config import load_settings


def test_load_settings_defaults(monkeypatch) -> None:
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "QURAN_API_DIR",
        "QURAN_DB_PATH",
        "HADITH_API_DIR",
        "HADITH_DB_PATH",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(env_file=None)

    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-5.6"
    assert settings.quran_api_dir.name == "quran-api-go"
    assert settings.hadith_api_dir.name == "hadith-api-go-run"
    assert settings.quran_mcp_args == ["run", "./cmd/mcp"]
    assert settings.hadith_seeder_args == ["run", "./cmd/seeder"]


def test_load_settings_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("QURAN_DB_PATH", str(tmp_path / "quran.db"))

    settings = load_settings(env_file=None)

    assert settings.has_openai_api_key
    assert settings.quran_db_path == (tmp_path / "quran.db").resolve()


def test_load_settings_args_from_env(monkeypatch) -> None:
    monkeypatch.setenv("QURAN_MCP_ARGS", "--stdio --fast")

    settings = load_settings(env_file=None)

    assert settings.quran_mcp_args == ["--stdio", "--fast"]


def test_load_settings_use_responses_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_USE_RESPONSES", "false")

    assert load_settings(env_file=None).openai_use_responses is False

    monkeypatch.setenv("OPENAI_USE_RESPONSES", "True")

    assert load_settings(env_file=None).openai_use_responses is True


def test_load_settings_use_responses_defaults_to_none(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_USE_RESPONSES", raising=False)

    assert load_settings(env_file=None).openai_use_responses is None

    monkeypatch.setenv("OPENAI_USE_RESPONSES", "maybe")

    assert load_settings(env_file=None).openai_use_responses is None


def test_load_settings_telegram_defaults(monkeypatch) -> None:
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_PROXY", "TELEGRAM_CONNECT_TIMEOUT", "TELEGRAM_READ_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(env_file=None)

    assert settings.telegram_bot_token is None
    assert settings.telegram_api_base == "http://127.0.0.1:8000"
    assert settings.telegram_proxy is None
    assert settings.telegram_connect_timeout == 10.0
    assert settings.telegram_read_timeout == 30.0


def test_load_settings_telegram_proxy_from_env(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_PROXY", "http://127.0.0.1:1080")
    monkeypatch.setenv("TELEGRAM_CONNECT_TIMEOUT", "15")

    settings = load_settings(env_file=None)

    assert settings.telegram_proxy == "http://127.0.0.1:1080"
    assert settings.telegram_connect_timeout == 15.0
