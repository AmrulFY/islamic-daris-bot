from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import Settings


SECRET_PLACEHOLDER = "********"
SECRET_KEY_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTHORIZATION", "COOKIE")


def _is_secret_key(key: str) -> bool:
    upper = key.upper()
    return any(part in upper for part in SECRET_KEY_PARTS)


def _clean_string(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


class LLMConnection(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str = "gpt-5.6"
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    use_responses: bool | None = None

    @field_validator("model")
    @classmethod
    def model_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model is required")
        return value

    @field_validator("api_key", "base_url", "organization", "project", mode="before")
    @classmethod
    def clean_optional_strings(cls, value: str | None) -> str | None:
        return _clean_string(value)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key or self.base_url)


class MCPServerConfig(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_.-]+$")
    name: str
    enabled: bool = True
    transport: Literal["stdio", "streamable_http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 45
    sse_read_timeout_seconds: float = 300
    use_structured_content: bool = True
    require_approval: Literal["never", "always"] = "never"

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name is required")
        return value

    @field_validator("command", "cwd", "url", mode="before")
    @classmethod
    def clean_optional_strings(cls, value: str | None) -> str | None:
        return _clean_string(value)

    @model_validator(mode="after")
    def validate_transport_settings(self) -> MCPServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP servers require command")
        if self.transport in {"streamable_http", "sse"} and not self.url:
            raise ValueError("HTTP MCP servers require url")
        return self


class RuntimeConfig(BaseModel):
    llm: LLMConnection
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    answer_language: Literal["id", "en"] = "id"
    agent_timeout_seconds: int = 300
    agent_max_turns: int = 10
    session_max_turns: int = 12

    @field_validator("agent_timeout_seconds", "agent_max_turns", "session_max_turns")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be positive")
        return value

    @model_validator(mode="after")
    def server_ids_must_be_unique(self) -> RuntimeConfig:
        ids = [server.id for server in self.mcp_servers]
        if len(ids) != len(set(ids)):
            raise ValueError("mcp server ids must be unique")
        return self


def default_runtime_config(settings: Settings) -> RuntimeConfig:
    return RuntimeConfig(
        llm=LLMConnection(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            organization=settings.openai_organization,
            project=settings.openai_project,
            use_responses=settings.openai_use_responses,
        ),
        mcp_servers=[
            MCPServerConfig(
                id="quran",
                name="Quran API",
                command=settings.quran_mcp_command,
                args=settings.quran_mcp_args,
                cwd=str(settings.quran_mcp_cwd),
                env={
                    "DB_PATH": str(settings.quran_db_path),
                    "LOG_LEVEL": "warn",
                },
                timeout_seconds=settings.mcp_startup_timeout_seconds,
            ),
            MCPServerConfig(
                id="hadith",
                name="Hadith API",
                command=settings.hadith_mcp_command,
                args=settings.hadith_mcp_args,
                cwd=str(settings.hadith_mcp_cwd),
                env={
                    "DATABASE_PATH": str(settings.hadith_db_path),
                    "LOG_LEVEL": "silent",
                },
                timeout_seconds=settings.mcp_startup_timeout_seconds,
            ),
        ],
        agent_timeout_seconds=settings.agent_timeout_seconds,
        agent_max_turns=settings.agent_max_turns,
        session_max_turns=settings.session_max_turns,
    )


def redact_runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    data = config.model_dump()
    if data["llm"].get("api_key"):
        data["llm"]["api_key"] = SECRET_PLACEHOLDER
    for server in data["mcp_servers"]:
        for section in ("env", "headers"):
            for key, value in list(server.get(section, {}).items()):
                if value and _is_secret_key(key):
                    server[section][key] = SECRET_PLACEHOLDER
    return RuntimeConfig.model_validate(data)


def preserve_masked_secrets(incoming: RuntimeConfig, existing: RuntimeConfig) -> RuntimeConfig:
    data = incoming.model_dump()
    if data["llm"].get("api_key") == SECRET_PLACEHOLDER:
        data["llm"]["api_key"] = existing.llm.api_key

    existing_by_id = {server.id: server for server in existing.mcp_servers}
    for server_data in data["mcp_servers"]:
        existing_server = existing_by_id.get(server_data["id"])
        if not existing_server:
            continue
        for section in ("env", "headers"):
            existing_values = getattr(existing_server, section)
            for key, value in list(server_data.get(section, {}).items()):
                if value == SECRET_PLACEHOLDER:
                    server_data[section][key] = existing_values.get(key, value)
    return RuntimeConfig.model_validate(data)


class RuntimeConfigStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.runtime_config_path

    def load(self) -> RuntimeConfig:
        if not self.path.exists():
            return default_runtime_config(self.settings)
        with self.path.open("r", encoding="utf-8") as fh:
            return RuntimeConfig.model_validate(json.load(fh))

    def load_redacted(self) -> RuntimeConfig:
        return redact_runtime_config(self.load())

    def save(self, config: RuntimeConfig) -> RuntimeConfig:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(config.model_dump(), fh, indent=2)
            fh.write("\n")
        os.chmod(temp_path, 0o600)
        temp_path.replace(self.path)
        return config

    def save_preserving_masked_secrets(self, config: RuntimeConfig) -> RuntimeConfig:
        return self.save(preserve_masked_secrets(config, self.load()))

    def reset(self) -> RuntimeConfig:
        config = default_runtime_config(self.settings)
        return self.save(config)
