from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError

from muslim_bot.answer_cache import AnswerCache
from muslim_bot.api import app, get_answer_cache, get_config_store, get_runner, get_session_store, get_settings
from muslim_bot.runtime_config import RuntimeConfigStore
from muslim_bot.session_store import SessionStore
from tests.test_api import _settings

REQUEST = httpx.Request("POST", "http://gateway.test/chat/completions")


def make_errors() -> dict[str, tuple[Exception, int]]:
    return {
        "timeout": (APITimeoutError(request=REQUEST), 504),
        "connection": (APIConnectionError(request=REQUEST), 503),
        "status": (APIStatusError("overloaded", response=httpx.Response(529, request=REQUEST), body=None), 502),
    }


class FailingRunner:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def try_fast_path(self, message: str, *, language: str = "id"):
        return None

    async def ask(self, message: str, **kwargs):
        raise self.error

    async def ask_streamed(self, message: str, **kwargs):
        raise self.error
        yield  # pragma: no cover - makes this an async generator


@pytest.fixture
def client_for(tmp_path: Path):
    def build(error: Exception) -> TestClient:
        settings = _settings(tmp_path)
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
        app.dependency_overrides[get_session_store] = lambda: SessionStore(max_turns=3)
        app.dependency_overrides[get_runner] = lambda: FailingRunner(error)
        app.dependency_overrides[get_answer_cache] = lambda: AnswerCache(settings.session_db_path, enabled=False)
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


@pytest.mark.parametrize("name", ["timeout", "connection", "status"])
def test_chat_maps_llm_gateway_errors_instead_of_500(client_for, name: str) -> None:
    error, expected = make_errors()[name]

    response = client_for(error).post("/chat", json={"message": "ayat tentang sabar"})

    assert response.status_code == expected
    assert "LLM" in response.json()["detail"]


def test_stream_reports_llm_gateway_errors_as_error_event(client_for) -> None:
    error, _ = make_errors()["timeout"]

    response = client_for(error).post("/chat/stream", json={"message": "ayat tentang sabar"})

    assert response.status_code == 200
    assert "event: error" in response.text and "LLM request timed out" in response.text
