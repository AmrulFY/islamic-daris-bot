from __future__ import annotations

import asyncio
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, load_settings
from .runtime_config import MCPServerConfig, RuntimeConfig, RuntimeConfigStore
from .schemas import Reference


SUBPROCESS_ENV_KEYS = {
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "GOTOOLCHAIN",
    "GOCACHE",
    "GOMODCACHE",
    "GOPATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
}


AGENT_INSTRUCTIONS = """
Anda adalah MuslimBot, asisten edukasi Islam berbahasa Indonesia.

Aturan utama:
- Gunakan tool Quran dan Hadith MCP untuk klaim yang berkaitan dengan dalil.
- Jawab ringkas, hangat, dan mudah dipahami.
- Selalu cantumkan rujukan yang ditemukan, misalnya "Al-Baqarah 2:255" atau "Sahih al-Bukhari #1".
- Jika tool tidak menemukan dalil yang relevan, katakan dengan jujur.
- Jangan mengaku sebagai mufti, ustaz, atau otoritas fatwa.
- Untuk pertanyaan hukum/fatwa pribadi, berikan konteks edukatif dan sarankan bertanya ke ulama yang kompeten.
- Jangan membuat ayat, nomor hadith, atau sanad yang tidak muncul dari tool.
- Gunakan terjemahan Indonesia sebagai default; tambahkan teks Arab bila tool menyediakannya dan relevan.
""".strip()


QURAN_REF_RE = re.compile(
    r"\b(?:QS\.?\s*)?([A-ZÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ' -]+)\s+(\d{1,3}):(\d{1,3})\b"
)
HADITH_REF_RE = re.compile(
    r"\b(Sahih al-Bukhari|Sahih Bukhari|Sahih Muslim|Sunan Abu Dawud|Jami(?:'|`) At-Tirmidhi|Jami At-Tirmidhi|Sunan an-Nasai|Sunan Ibn Majah)\s*(?:#|no\.?|nomor)?\s*(\d{1,5})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AgentReply:
    answer: str
    references: list[Reference] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_references(answer: str) -> list[Reference]:
    refs: list[Reference] = []
    seen: set[tuple[str, str]] = set()

    for match in QURAN_REF_RE.finditer(answer):
        label = f"{match.group(1).strip()} {match.group(2)}:{match.group(3)}"
        key = ("quran", label.lower())
        if key not in seen:
            refs.append(Reference(type="quran", label=label, source_id=f"{match.group(2)}:{match.group(3)}"))
            seen.add(key)

    for match in HADITH_REF_RE.finditer(answer):
        book = " ".join(match.group(1).split())
        number = match.group(2)
        label = f"{book} #{number}"
        key = ("hadith", label.lower())
        if key not in seen:
            refs.append(Reference(type="hadith", label=label, source_id=number))
            seen.add(key)

    return refs


def extract_used_tools(result: Any) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in getattr(result, "new_items", []) or []:
        raw = getattr(item, "raw_item", item)
        candidates = [
            getattr(raw, "name", None),
            getattr(raw, "tool_name", None),
            getattr(raw, "server_label", None),
        ]
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                names.append(candidate)
    return names


def mcp_subprocess_env(**overrides: str) -> dict[str, str]:
    env = {key: os.environ[key] for key in SUBPROCESS_ENV_KEYS if key in os.environ}
    env.update(overrides)
    return env


class MuslimBotRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.config_store = RuntimeConfigStore(self.settings)

    def load_runtime_config(self) -> RuntimeConfig:
        return self.config_store.load()

    async def ask(
        self,
        message: str,
        *,
        session_id: str,
        history_prompt: str | None = None,
        language: str = "id",
    ) -> AgentReply:
        runtime_config = self.load_runtime_config()
        if not runtime_config.llm.is_configured:
            raise RuntimeError("LLM connection is not configured")

        try:
            from agents import Agent, RunConfig, Runner
            from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp
            from agents.models.openai_provider import OpenAIProvider
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `make install` before running the agent") from exc

        if runtime_config.llm.api_key:
            os.environ.setdefault("OPENAI_API_KEY", runtime_config.llm.api_key)

        prompt = history_prompt or message
        prompt = f"Bahasa jawaban: {language}.\nSession: {session_id}\n\n{prompt}"

        async with AsyncExitStack() as stack:
            mcp_servers = []
            for server_config in runtime_config.mcp_servers:
                if not server_config.enabled:
                    continue
                mcp_server = self._build_mcp_server(
                    server_config,
                    MCPServerStdio=MCPServerStdio,
                    MCPServerStreamableHttp=MCPServerStreamableHttp,
                    MCPServerSse=MCPServerSse,
                )
                mcp_servers.append(await stack.enter_async_context(mcp_server))

            agent = Agent(
                name="MuslimBot",
                instructions=AGENT_INSTRUCTIONS,
                model=runtime_config.llm.model,
                mcp_servers=mcp_servers,
            )
            run_config = RunConfig(
                model_provider=OpenAIProvider(
                    api_key=runtime_config.llm.api_key or "not-needed",
                    base_url=runtime_config.llm.base_url,
                    organization=runtime_config.llm.organization,
                    project=runtime_config.llm.project,
                    use_responses=runtime_config.llm.use_responses,
                ),
                workflow_name="MuslimBot",
                tracing_disabled=True,
            )
            result = await asyncio.wait_for(
                Runner.run(agent, prompt, run_config=run_config),
                timeout=runtime_config.agent_timeout_seconds,
            )

        answer = str(getattr(result, "final_output", "")).strip()
        warnings: list[str] = []
        if not answer:
            warnings.append("Agent returned an empty answer.")

        return AgentReply(
            answer=answer,
            references=extract_references(answer),
            used_tools=extract_used_tools(result),
            warnings=warnings,
        )

    def _build_mcp_server(
        self,
        config: MCPServerConfig,
        *,
        MCPServerStdio: Any,
        MCPServerStreamableHttp: Any,
        MCPServerSse: Any,
    ) -> Any:
        common = {
            "name": config.name,
            "client_session_timeout_seconds": config.timeout_seconds,
            "use_structured_content": config.use_structured_content,
            "require_approval": config.require_approval,
        }
        if config.transport == "stdio":
            params: dict[str, Any] = {
                "command": config.command,
                "args": config.args,
                "env": mcp_subprocess_env(**config.env),
            }
            if config.cwd:
                params["cwd"] = config.cwd
            return MCPServerStdio(params=params, **common)

        params = {
            "url": config.url,
            "headers": config.headers,
            "timeout": config.timeout_seconds,
            "sse_read_timeout": config.sse_read_timeout_seconds,
        }
        if config.transport == "streamable_http":
            return MCPServerStreamableHttp(params=params, **common)
        return MCPServerSse(params=params, **common)


async def list_mcp_tools(config: RuntimeConfig) -> list[dict[str, Any]]:
    try:
        from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `make install` before testing MCP servers") from exc

    runner = MuslimBotRunner()
    summaries: list[dict[str, Any]] = []
    for server_config in config.mcp_servers:
        if not server_config.enabled:
            summaries.append({"id": server_config.id, "name": server_config.name, "enabled": False, "tools": [], "error": None})
            continue
        try:
            async with runner._build_mcp_server(
                server_config,
                MCPServerStdio=MCPServerStdio,
                MCPServerStreamableHttp=MCPServerStreamableHttp,
                MCPServerSse=MCPServerSse,
            ) as server:
                tools = await server.list_tools()
            summaries.append(
                {
                    "id": server_config.id,
                    "name": server_config.name,
                    "enabled": True,
                    "tools": [tool.name for tool in tools],
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as an admin diagnostic.
            summaries.append(
                {
                    "id": server_config.id,
                    "name": server_config.name,
                    "enabled": True,
                    "tools": [],
                    "error": str(exc),
                }
            )
    return summaries
