from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from muslim_bot.agent import (
    MCPServerManager,
    RunTimings,
    SUBPROCESS_ENV_KEYS,
    extract_references,
    extract_used_tools,
    mcp_server_fingerprint,
    mcp_subprocess_env,
)
from muslim_bot.runtime_config import MCPServerConfig


def test_extract_references_from_answer() -> None:
    answer = "Rujukan: Al-Baqarah 2:255 dan Sahih al-Bukhari #1."

    refs = extract_references(answer)

    assert [ref.type for ref in refs] == ["quran", "hadith"]
    assert refs[0].source_id == "2:255"
    assert refs[1].source_id == "1"


def test_extract_used_tools_is_deduplicated() -> None:
    result = SimpleNamespace(
        new_items=[
            SimpleNamespace(raw_item=SimpleNamespace(name="search_quran")),
            SimpleNamespace(raw_item=SimpleNamespace(tool_name="search_quran")),
            SimpleNamespace(raw_item=SimpleNamespace(name="search_hadith")),
        ]
    )

    assert extract_used_tools(result) == ["search_quran", "search_hadith"]


class FakeServer:
    def __init__(self, *, fail_connect: bool = False, fail_health_check: bool = False) -> None:
        self.fail_connect = fail_connect
        self.fail_health_check = fail_health_check
        self.connect_calls = 0
        self.cleanup_calls = 0
        self.list_tools_calls = 0

    async def connect(self) -> None:
        if self.fail_connect:
            raise RuntimeError("boom")
        self.connect_calls += 1

    async def cleanup(self) -> None:
        self.cleanup_calls += 1

    async def list_tools(self) -> list[str]:
        self.list_tools_calls += 1
        if self.fail_health_check:
            raise RuntimeError("connection died")
        return []


def make_config(server_id: str = "quran", timeout: float = 45) -> MCPServerConfig:
    return MCPServerConfig(
        id=server_id,
        name="Test",
        transport="stdio",
        command="go",
        args=["run", "./cmd/mcp"],
        timeout_seconds=timeout,
    )


def patch_builder(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    async def fake_connect(self, config):
        return factory(config)

    monkeypatch.setattr(MCPServerManager, "_connect", fake_connect)


def test_manager_reuses_cached_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MCPServerManager(health_check_interval=0)
    server = FakeServer()
    build_calls: list[str] = []

    async def fake_connect(self, config):
        build_calls.append(config.id)
        return server

    monkeypatch.setattr(MCPServerManager, "_connect", fake_connect)
    config = make_config()

    servers_1 = asyncio.run(manager.get_servers([config]))
    servers_2 = asyncio.run(manager.get_servers([config]))

    assert servers_1 == servers_2 == [server]
    assert build_calls == ["quran"]
    assert server.list_tools_calls == 1
    asyncio.run(manager.aclose())
    assert server.cleanup_calls == 1


def test_manager_rebuilds_when_config_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MCPServerManager(health_check_interval=0)
    created: list[FakeServer] = []
    patch_builder(monkeypatch, lambda config: created.append(FakeServer()) or created[-1])
    asyncio.run(manager.get_servers([make_config(timeout=45)]))
    asyncio.run(manager.get_servers([make_config(timeout=60)]))

    assert len(created) == 2
    assert created[0].cleanup_calls == 1
    assert created[1].cleanup_calls == 0
    asyncio.run(manager.aclose())


def test_manager_reconnects_dead_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MCPServerManager(health_check_interval=0)
    dead_server = FakeServer(fail_health_check=True)
    fresh_server = FakeServer()
    servers_to_build = iter([dead_server, fresh_server])

    async def fake_connect(self, config):
        return next(servers_to_build)

    monkeypatch.setattr(MCPServerManager, "_connect", fake_connect)
    config = make_config()

    asyncio.run(manager.get_servers([config]))
    servers = asyncio.run(manager.get_servers([config]))

    assert dead_server.cleanup_calls == 1
    assert servers == [fresh_server]
    asyncio.run(manager.aclose())


def test_manager_skips_disabled_and_removes_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MCPServerManager(health_check_interval=0)
    quran_server = FakeServer()
    hadith_server = FakeServer()
    builders = {"quran": quran_server, "hadith": hadith_server}
    patch_builder(monkeypatch, lambda config: builders[config.id])

    enabled_cfg = make_config("quran")
    disabled_cfg = make_config("hadith")
    disabled_cfg.enabled = False

    servers = asyncio.run(manager.get_servers([enabled_cfg, disabled_cfg]))
    assert servers == [quran_server]

    servers = asyncio.run(manager.get_servers([]))
    assert servers == []
    assert quran_server.cleanup_calls == 1
    asyncio.run(manager.aclose())


def test_fingerprint_changes_with_timeout() -> None:
    assert mcp_server_fingerprint(make_config(timeout=45)) != mcp_server_fingerprint(make_config(timeout=60))


def test_subprocess_env_preserves_windows_temp_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("TMP", "TEMP", "USERPROFILE", "LOCALAPPDATA"):
        assert key in SUBPROCESS_ENV_KEYS
    for key in ("SystemRoot", "WINDIR"):
        assert key in SUBPROCESS_ENV_KEYS
    for key in ("GOPROXY", "GOFLAGS", "GOPRIVATE"):
        assert key in SUBPROCESS_ENV_KEYS

    monkeypatch.setenv("TMP", "C:\\Temp")
    monkeypatch.setenv("TEMP", "C:\\Temp")
    monkeypatch.delenv("TMPDIR", raising=False)

    env = mcp_subprocess_env(EXTRA="1")

    assert env["TMP"] == "C:\\Temp"
    assert env["TEMP"] == "C:\\Temp"
    assert env["EXTRA"] == "1"
    assert "TMPDIR" not in env


def test_manager_throttles_health_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = MCPServerManager(health_check_interval=3600)
    server = FakeServer()
    patch_builder(monkeypatch, lambda config: server)
    config = make_config()

    async def run() -> None:
        for _ in range(3):
            await manager.get_servers([config])

    asyncio.run(run())

    # Connect, then one check on first reuse, none afterwards within the interval.
    assert server.list_tools_calls == 1
    manager.invalidate_health()
    asyncio.run(manager.get_servers([config]))
    assert server.list_tools_calls == 2


def test_run_timings_breakdown() -> None:
    now = [0.0]
    timings = RunTimings(clock=lambda: now[0])
    timings.mcp_setup = 0.1
    now[0] = 0.1
    timings.on_raw_event("response.created")
    now[0] = 1.1
    timings.on_raw_event("response.completed")
    timings.on_tool_called("search_quran")
    now[0] = 1.6
    timings.on_tool_output("search_quran")
    timings.on_raw_event("response.created")
    now[0] = 2.0
    timings.on_raw_event("response.output_text.delta")
    now[0] = 3.0
    timings.on_raw_event("response.completed")

    assert timings.llm_calls == [1.0, pytest.approx(1.4)]
    assert timings.tool_calls == [("search_quran", pytest.approx(0.5))]
    assert timings.first_delta == 2.0
    summary = timings.summary()
    assert "llm_calls=2" in summary and "tool_calls=1" in summary and "ttft=2.00s" in summary
