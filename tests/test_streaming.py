from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from muslim_bot.agent import AgentLoopLimitError, MuslimBotRunner
from muslim_bot.config import Settings


def install_agents_stub(monkeypatch: pytest.MonkeyPatch, events, final_output="", raise_at=None):
    agents = types.ModuleType("agents")

    class FakeAgent:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeRunConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeMaxTurnsExceeded(Exception):
        pass

    class FakeResult:
        def __init__(self) -> None:
            self.final_output = final_output

        async def stream_events(self):
            for index, event in enumerate(events):
                if raise_at is not None and index == raise_at:
                    raise FakeMaxTurnsExceeded("limit")
                yield event

    class FakeRunner:
        seen: dict = {}

        @staticmethod
        def run_streamed(agent, prompt, run_config=None, max_turns=None):
            FakeRunner.seen = {"prompt": prompt, "max_turns": max_turns}
            return FakeResult()

    agents.Agent = FakeAgent
    agents.RunConfig = FakeRunConfig
    agents.ModelSettings = lambda **kwargs: SimpleNamespace(**kwargs)
    agents.Runner = FakeRunner
    agents.MaxTurnsExceeded = FakeMaxTurnsExceeded

    models = types.ModuleType("agents.models")
    provider_mod = types.ModuleType("agents.models.openai_provider")

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    provider_mod.OpenAIProvider = FakeProvider
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.models", models)
    monkeypatch.setitem(sys.modules, "agents.models.openai_provider", provider_mod)
    return FakeRunner


def make_settings(tmp_path: Path, api_key: str | None = "x") -> Settings:
    return Settings(
        openai_api_key=api_key,
        openai_model="gpt-test",
        openai_base_url=None,
        openai_organization=None,
        openai_project=None,
        openai_use_responses=None,
        admin_token=None,
        runtime_config_path=tmp_path / "runtime_config.json",
        quran_source_db_path=tmp_path / "source-quran.db",
        quran_api_dir=tmp_path,
        quran_db_path=tmp_path / "quran.db",
        quran_mcp_command="go",
        quran_mcp_args=[],
        quran_mcp_cwd=tmp_path,
        hadith_api_dir=tmp_path,
        hadith_db_path=tmp_path / "hadith.db",
        hadith_mcp_command="go",
        hadith_mcp_args=[],
        hadith_mcp_cwd=tmp_path,
        hadith_seeder_command="go",
        hadith_seeder_args=[],
        hadith_seeder_cwd=tmp_path,
        mcp_startup_timeout_seconds=1,
        agent_timeout_seconds=5,
        agent_max_turns=4,
        session_max_turns=3,
        session_db_path=tmp_path / "sessions.db",
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


def make_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_key: str | None = "x") -> MuslimBotRunner:
    runner = MuslimBotRunner(settings=make_settings(tmp_path, api_key))

    async def no_mcp(config):
        return []

    monkeypatch.setattr(runner, "load_mcp_servers", no_mcp)
    return runner


def delta(text: str):
    return SimpleNamespace(
        type="raw_response_event",
        data=SimpleNamespace(type="response.output_text.delta", delta=text),
    )


def other_raw():
    return SimpleNamespace(
        type="raw_response_event",
        data=SimpleNamespace(type="response.completed"),
    )


def tool_called(name: str):
    return SimpleNamespace(
        type="run_item_stream_event",
        name="tool_called",
        item=SimpleNamespace(raw_item=SimpleNamespace(name=name)),
    )


def other_item():
    return SimpleNamespace(type="run_item_stream_event", name="message_output_created", item=None)


def collect(coro):
    async def go():
        return [(kind, payload) async for kind, payload in coro]

    return asyncio.run(go())


def test_ask_streamed_yields_deltas_tools_and_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = [delta("Assalamu "), other_raw(), tool_called("search_quran"), tool_called("search_quran"), delta("alaikum."), other_item()]
    FakeRunner = install_agents_stub(monkeypatch, events, final_output="Assalamu alaikum. Al-Baqarah 2:255.")
    runner = make_runner(tmp_path, monkeypatch)

    streamed = collect(runner.ask_streamed("halo", session_id="s1"))

    kinds = [kind for kind, _ in streamed]
    assert kinds == ["delta", "tool", "delta", "done"]
    assert streamed[0][1] == "Assalamu "
    assert streamed[1][1] == "search_quran"
    done = streamed[-1][1]
    assert done.answer == "Assalamu alaikum. Al-Baqarah 2:255."
    assert done.used_tools == ["search_quran"]
    assert done.references[0].label == "Al-Baqarah 2:255"
    assert FakeRunner.seen["max_turns"] == 4
    assert "Session: s1" in FakeRunner.seen["prompt"]


def test_ask_streamed_empty_deltas_are_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_agents_stub(monkeypatch, [delta(""), delta("hi")], final_output="hi")
    runner = make_runner(tmp_path, monkeypatch)

    streamed = collect(runner.ask_streamed("halo", session_id="s1"))

    assert [kind for kind, _ in streamed] == ["delta", "done"]


def test_ask_streamed_raises_loop_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_agents_stub(monkeypatch, [delta("parsial"), delta("lagi")], raise_at=1)
    runner = make_runner(tmp_path, monkeypatch)

    with pytest.raises(AgentLoopLimitError):
        collect(runner.ask_streamed("halo", session_id="s1"))


def test_ask_streamed_requires_llm_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_agents_stub(monkeypatch, [])
    runner = make_runner(tmp_path, monkeypatch, api_key=None)

    with pytest.raises(RuntimeError, match="LLM connection is not configured"):
        collect(runner.ask_streamed("halo", session_id="s1"))


def test_ask_collects_streamed_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_agents_stub(
        monkeypatch,
        [tool_called("search_hadith"), delta("Alhamdulillah.")],
        final_output="Alhamdulillah.",
    )
    runner = make_runner(tmp_path, monkeypatch)

    reply = asyncio.run(runner.ask("syukur", session_id="s9"))

    assert reply.answer == "Alhamdulillah."
    assert reply.used_tools == ["search_hadith"]


def test_ask_maps_stalled_stream_to_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    agents = types.ModuleType("agents")

    class FakeMaxTurnsExceeded(Exception):
        pass

    class SlowResult:
        final_output = ""

        async def stream_events(self):
            await asyncio.sleep(30)
            yield delta("never")

    class SlowRunner:
        @staticmethod
        def run_streamed(*args, **kwargs):
            return SlowResult()

    agents.Agent = lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    agents.RunConfig = lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    agents.ModelSettings = lambda **kwargs: SimpleNamespace(**kwargs)
    agents.Runner = SlowRunner
    agents.MaxTurnsExceeded = FakeMaxTurnsExceeded
    models = types.ModuleType("agents.models")
    provider_mod = types.ModuleType("agents.models.openai_provider")
    provider_mod.OpenAIProvider = lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.models", models)
    monkeypatch.setitem(sys.modules, "agents.models.openai_provider", provider_mod)

    runner = make_runner(tmp_path, monkeypatch)

    async def no_mcp(config):
        return []

    from muslim_bot.runtime_config import LLMConnection, RuntimeConfig

    monkeypatch.setattr(runner, "load_mcp_servers", no_mcp)
    monkeypatch.setattr(
        runner,
        "load_runtime_config",
        lambda: RuntimeConfig(llm=LLMConnection(api_key="x"), agent_timeout_seconds=1),
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(runner.ask("halo", session_id="s1"))


def parse_sse(body: str) -> list[tuple[str, dict]]:
    import json

    events: list[tuple[str, dict]] = []
    for chunk in body.split("\n\n"):
        event = "message"
        data = ""
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data += line[len("data:"):].strip()
        if data:
            events.append((event, json.loads(data)))
    return events


def test_chat_stream_endpoint_streams_and_stores(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from muslim_bot.agent import AgentReply
    from muslim_bot.api import app, get_answer_cache, get_config_store, get_rate_limiter, get_runner, get_session_store, get_settings
    from muslim_bot.answer_cache import AnswerCache
    from muslim_bot.rate_limit import RateLimiter
    from muslim_bot.runtime_config import RuntimeConfigStore
    from muslim_bot.session_store import SessionStore

    async def fake_stream(*args, **kwargs):
        yield ("delta", "halo ")
        yield ("tool", "search_quran")
        yield (
            "done",
            AgentReply(answer="halo dunia", references=[], used_tools=["search_quran"], warnings=[]),
        )

    class FakeRunner:
        def try_fast_path(self, message, *, language='id'):
            return None

        def ask_streamed(self, *args, **kwargs):
            return fake_stream()

    settings = make_settings(tmp_path)
    shared_store = SessionStore(max_turns=3)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: shared_store
    app.dependency_overrides[get_runner] = lambda: FakeRunner()
    app.dependency_overrides[get_answer_cache] = lambda: AnswerCache(settings.session_db_path, ttl_days=settings.answer_cache_ttl_days, enabled=settings.chat_cache_enabled)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_per_minute=0)

    try:
        client = TestClient(app)
        response = client.post("/chat/stream", json={"message": "halo"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = parse_sse(response.text)
        assert [kind for kind, _ in events] == ["delta", "tool", "done"]
        assert events[0][1] == {"text": "halo "}
        assert events[1][1] == {"name": "search_quran"}
        done = events[2][1]
        assert done["answer"] == "halo dunia"
        assert done["used_tools"] == ["search_quran"]

        prompt = shared_store.history_prompt(done["session_id"], "lanjut")
        assert "halo dunia" in prompt
    finally:
        app.dependency_overrides.clear()


def test_chat_stream_endpoint_maps_errors_to_events(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from muslim_bot.api import app, get_answer_cache, get_config_store, get_rate_limiter, get_runner, get_session_store, get_settings
    from muslim_bot.answer_cache import AnswerCache
    from muslim_bot.rate_limit import RateLimiter
    from muslim_bot.runtime_config import RuntimeConfigStore
    from muslim_bot.session_store import SessionStore

    async def failing_stream(*args, **kwargs):
        yield ("delta", "parsial")
        raise RuntimeError("MCP servers unavailable: X")
        yield ("done", None)  # pragma: no cover

    class FakeRunner:
        def try_fast_path(self, message, *, language="id"):
            return None

        def ask_streamed(self, *args, **kwargs):
            return failing_stream()

    settings = make_settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_config_store] = lambda: RuntimeConfigStore(settings)
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_turns=3)
    app.dependency_overrides[get_runner] = lambda: FakeRunner()
    app.dependency_overrides[get_answer_cache] = lambda: AnswerCache(settings.session_db_path, ttl_days=settings.answer_cache_ttl_days, enabled=settings.chat_cache_enabled)
    app.dependency_overrides[get_rate_limiter] = lambda: RateLimiter(max_per_minute=0)

    try:
        client = TestClient(app)
        response = client.post("/chat/stream", json={"message": "halo"})
        assert response.status_code == 200
        events = parse_sse(response.text)
        assert [kind for kind, _ in events] == ["delta", "error"]
        assert "MCP servers unavailable" in events[1][1]["detail"]
    finally:
        app.dependency_overrides.clear()
